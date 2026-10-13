// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUSharedTexture.h"
#include "WebGPUCompute.h"
#include "WebGPUDevice.h"
#include "WebGPUInteropUtils.h"
#include "RHI.h"
#include "RHICommandList.h"
#include "RHIResources.h"

#if WITH_WEBGPU_D3D12_INTEROP
#include "ID3D12DynamicRHI.h"
#endif

bool WebGPUInterop::GetRHIAdapterLUID(uint32& OutLowPart, int32& OutHighPart)
{
#if WITH_WEBGPU_D3D12_INTEROP
	if (IsRHID3D12())
	{
		if (ID3D12Device* D3DDevice = GetID3D12DynamicRHI()->RHIGetDevice(0))
		{
			const LUID AdapterLUID = D3DDevice->GetAdapterLuid();
			OutLowPart = AdapterLUID.LowPart;
			OutHighPart = AdapterLUID.HighPart;
			return true;
		}
	}
#endif
	return false;
}

struct FWebGPUSharedTexture::FImpl
{
	TSharedPtr<FWebGPUDevice> Device;
	FTextureRHIRef RHITexture;

	WGPUSharedTextureMemory Memory = nullptr;
	WGPUTexture Texture = nullptr;
	WGPUSharedTextureMemoryProperties Properties = {};
	bool bAccessing = false;

#if WITH_WEBGPU_D3D12_INTEROP
	// UE signals, WebGPU waits
	TRefCountPtr<ID3D12Fence> UEFence;
	WGPUSharedFence UEFenceForWebGPU = nullptr;
	uint64 UEFenceValue = 0;

	// WebGPU's queue fences opened on UE's device, keyed by their shared handle
	TMap<void*, TRefCountPtr<ID3D12Fence>> WebGPUFences;
#endif

	~FImpl()
	{
		if (Memory && Texture && bAccessing)
		{
			WGPUSharedTextureMemoryEndAccessState State = {};
			wgpuSharedTextureMemoryEndAccess(Memory, Texture, &State);
			wgpuSharedTextureMemoryEndAccessStateFreeMembers(State);
		}
		if (Texture)
		{
			wgpuTextureRelease(Texture);
		}
		if (Memory)
		{
			wgpuSharedTextureMemoryRelease(Memory);
		}
#if WITH_WEBGPU_D3D12_INTEROP
		if (UEFenceForWebGPU)
		{
			wgpuSharedFenceRelease(UEFenceForWebGPU);
		}
#endif
	}
};

FWebGPUSharedTexture::FWebGPUSharedTexture()
	: Impl(MakeUnique<FImpl>())
{
}

FWebGPUSharedTexture::~FWebGPUSharedTexture() = default;

WGPUTexture FWebGPUSharedTexture::GetTexture() const
{
	return Impl->Texture;
}

WGPUTextureFormat FWebGPUSharedTexture::GetFormat() const
{
	return Impl->Properties.format;
}

WGPUTextureUsage FWebGPUSharedTexture::GetUsage() const
{
	return Impl->Properties.usage;
}

FIntPoint FWebGPUSharedTexture::GetSize() const
{
	return FIntPoint(Impl->Properties.size.width, Impl->Properties.size.height);
}

TSharedPtr<FWebGPUSharedTexture> FWebGPUSharedTexture::Create(const TSharedRef<FWebGPUDevice>& Device, FRHITexture* Texture, FString& OutError)
{
#if WITH_WEBGPU_D3D12_INTEROP
	if (!Texture)
	{
		OutError = TEXT("No texture");
		return nullptr;
	}
	if (!IsRHID3D12())
	{
		OutError = TEXT("Shared textures need the D3D12 RHI");
		return nullptr;
	}
	if (!Device->IsValid() || !Device->IsUsingRHIAdapter())
	{
		OutError = TEXT("WebGPU device is not running on the RHI's adapter");
		return nullptr;
	}
	if (!Device->HasFeature(WGPUFeatureName_SharedTextureMemoryDXGISharedHandle) || !Device->HasFeature(WGPUFeatureName_SharedFenceDXGISharedHandle))
	{
		OutError = TEXT("WebGPU device lacks DXGI shared texture/fence features");
		return nullptr;
	}
	if (!EnumHasAnyFlags(Texture->GetDesc().Flags, ETextureCreateFlags::Shared))
	{
		OutError = TEXT("Texture must be created with ETextureCreateFlags::Shared (bGPUSharedFlag on render targets)");
		return nullptr;
	}

	ID3D12DynamicRHI* D3D12RHI = GetID3D12DynamicRHI();
	ID3D12Device* D3DDevice = D3D12RHI->RHIGetDevice(0);
	ID3D12Resource* Resource = D3D12RHI->RHIGetResource(Texture);

	HANDLE ResourceHandle = nullptr;
	HRESULT Result = D3DDevice->CreateSharedHandle(Resource, nullptr, GENERIC_ALL, nullptr, &ResourceHandle);
	if (FAILED(Result))
	{
		OutError = FString::Printf(TEXT("CreateSharedHandle(texture) failed: 0x%08x"), (uint32)Result);
		return nullptr;
	}

	TSharedPtr<FWebGPUSharedTexture> Shared = MakeShareable(new FWebGPUSharedTexture());
	FImpl& State = *Shared->Impl;
	State.Device = Device;
	State.RHITexture = Texture;

	Device->PushErrorScopes();

	WGPUSharedTextureMemoryDXGISharedHandleDescriptor HandleDesc = {};
	HandleDesc.chain.sType = WGPUSType_SharedTextureMemoryDXGISharedHandleDescriptor;
	HandleDesc.handle = ResourceHandle;
	HandleDesc.useKeyedMutex = false;

	WGPUSharedTextureMemoryDescriptor MemoryDesc = {};
	MemoryDesc.nextInChain = &HandleDesc.chain;
	State.Memory = wgpuDeviceImportSharedTextureMemory(Device->GetDevice(), &MemoryDesc);

	// Dawn opened its own reference to the resource
	CloseHandle(ResourceHandle);

	if (State.Memory)
	{
		wgpuSharedTextureMemoryGetProperties(State.Memory, &State.Properties);
		State.Properties.nextInChain = nullptr;
		State.Texture = wgpuSharedTextureMemoryCreateTexture(State.Memory, nullptr);
	}

	// Fence UE signals and WebGPU waits on
	Result = D3DDevice->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(State.UEFence.GetInitReference()));
	HANDLE FenceHandle = nullptr;
	if (SUCCEEDED(Result))
	{
		Result = D3DDevice->CreateSharedHandle(State.UEFence.GetReference(), nullptr, GENERIC_ALL, nullptr, &FenceHandle);
	}
	if (SUCCEEDED(Result))
	{
		WGPUSharedFenceDXGISharedHandleDescriptor FenceHandleDesc = {};
		FenceHandleDesc.chain.sType = WGPUSType_SharedFenceDXGISharedHandleDescriptor;
		FenceHandleDesc.handle = FenceHandle;

		WGPUSharedFenceDescriptor FenceDesc = {};
		FenceDesc.nextInChain = &FenceHandleDesc.chain;
		State.UEFenceForWebGPU = wgpuDeviceImportSharedFence(Device->GetDevice(), &FenceDesc);
		CloseHandle(FenceHandle);
	}

	FString ScopeError;
	if (!Device->PopErrorScopes(ScopeError) || !State.Memory || !State.Texture || !State.UEFenceForWebGPU || FAILED(Result))
	{
		OutError = FString::Printf(TEXT("Import failed (hr 0x%08x): %s"), (uint32)Result, *ScopeError);
		return nullptr;
	}

	UE_LOG(LogWebGPU, Verbose, TEXT("Shared texture %dx%d format %d usage 0x%llx"),
		State.Properties.size.width, State.Properties.size.height, (int32)State.Properties.format, (uint64)State.Properties.usage);
	return Shared;
#else
	OutError = TEXT("Shared textures are not supported on this platform yet");
	return nullptr;
#endif
}

bool FWebGPUSharedTexture::BeginAccess(FRHICommandListImmediate& RHICmdList, bool bSubmitUEWork)
{
#if WITH_WEBGPU_D3D12_INTEROP
	if (Impl->bAccessing)
	{
		UE_LOG(LogWebGPU, Warning, TEXT("FWebGPUSharedTexture::BeginAccess called twice"));
		return false;
	}

	const uint64 Value = ++Impl->UEFenceValue;
	TRefCountPtr<ID3D12Fence> Fence = Impl->UEFence;
	RHICmdList.EnqueueLambda(TEXT("WebGPUSignal"), [Fence, Value](FRHICommandListImmediate& Cmd)
	{
		GetID3D12DynamicRHI()->RHISignalManualFence(Cmd, Fence.GetReference(), Value);
	});
	if (bSubmitUEWork)
	{
		RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);
	}

	WGPUSharedTextureMemoryBeginAccessDescriptor Desc = WGPU_SHARED_TEXTURE_MEMORY_BEGIN_ACCESS_DESCRIPTOR_INIT;
	Desc.concurrentRead = false;
	Desc.initialized = true;
	Desc.fenceCount = 1;
	Desc.fences = &Impl->UEFenceForWebGPU;
	Desc.signaledValueCount = 1;
	Desc.signaledValues = &Value;
	if (wgpuSharedTextureMemoryBeginAccess(Impl->Memory, Impl->Texture, &Desc) != WGPUStatus_Success)
	{
		UE_LOG(LogWebGPU, Error, TEXT("wgpuSharedTextureMemoryBeginAccess failed"));
		return false;
	}
	Impl->bAccessing = true;
	return true;
#else
	return false;
#endif
}

bool FWebGPUSharedTexture::EndAccess(FRHICommandListImmediate& RHICmdList)
{
#if WITH_WEBGPU_D3D12_INTEROP
	if (!Impl->bAccessing)
	{
		return false;
	}
	Impl->bAccessing = false;

	WGPUSharedTextureMemoryEndAccessState State = {};
	if (wgpuSharedTextureMemoryEndAccess(Impl->Memory, Impl->Texture, &State) != WGPUStatus_Success)
	{
		UE_LOG(LogWebGPU, Error, TEXT("wgpuSharedTextureMemoryEndAccess failed"));
		wgpuSharedTextureMemoryEndAccessStateFreeMembers(State);
		return false;
	}

	ID3D12Device* D3DDevice = GetID3D12DynamicRHI()->RHIGetDevice(0);
	bool bOk = true;
	for (size_t Index = 0; Index < State.fenceCount; Index++)
	{
		WGPUSharedFenceDXGISharedHandleExportInfo HandleInfo = {};
		HandleInfo.chain.sType = WGPUSType_SharedFenceDXGISharedHandleExportInfo;
		WGPUSharedFenceExportInfo ExportInfo = {};
		ExportInfo.nextInChain = &HandleInfo.chain;
		wgpuSharedFenceExportInfo(State.fences[Index], &ExportInfo);

		TRefCountPtr<ID3D12Fence>* Cached = Impl->WebGPUFences.Find(HandleInfo.handle);
		if (!Cached)
		{
			TRefCountPtr<ID3D12Fence> Opened;
			const HRESULT Result = D3DDevice->OpenSharedHandle((HANDLE)HandleInfo.handle, IID_PPV_ARGS(Opened.GetInitReference()));
			if (FAILED(Result))
			{
				UE_LOG(LogWebGPU, Error, TEXT("OpenSharedHandle(WebGPU fence) failed: 0x%08x"), (uint32)Result);
				bOk = false;
				continue;
			}
			Cached = &Impl->WebGPUFences.Add(HandleInfo.handle, Opened);
		}

		TRefCountPtr<ID3D12Fence> Fence = *Cached;
		const uint64 Value = State.signaledValues[Index];
		// WebGPU already submitted the signal, so UE can wait for it without a CPU-side handshake
		RHICmdList.EnqueueLambda(TEXT("WebGPUWait"), [Fence, Value](FRHICommandListImmediate& Cmd)
		{
			GetID3D12DynamicRHI()->RHIWaitManualFence(Cmd, Fence.GetReference(), Value, nullptr);
		});
	}

	wgpuSharedTextureMemoryEndAccessStateFreeMembers(State);
	return bOk;
#else
	return false;
#endif
}
