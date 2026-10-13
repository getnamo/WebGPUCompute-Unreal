// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUTextureInterop.h"
#include "WebGPUCompute.h"
#include "WebGPUContext.h"
#include "WebGPUDevice.h"
#include "HAL/Event.h"
#include "HAL/PlatformProcess.h"
#include "RenderingThread.h"
#include "RHI.h"
#include "RHICommandList.h"
#include "RHIResources.h"

#if WITH_WEBGPU_D3D12_INTEROP
#include "ID3D12DynamicRHI.h"
#endif

struct FWebGPUSharedTextureState
{
	FTextureRHIRef RHITexture;
	EWebGPUTextureFormat Format = EWebGPUTextureFormat::RGBA8Unorm;

	// Game thread
	uint64 SignalValue = 0;

#if WITH_WEBGPU_D3D12_INTEROP
	/** UE signals, WebGPU waits. Lives as long as the texture, across WebGPU device losses. */
	TRefCountPtr<ID3D12Fence> UEFence;

	// Worker thread
	WGPUSharedTextureMemory Memory = nullptr;
	WGPUSharedFence UEFenceForWebGPU = nullptr;
	/** WebGPU's queue fences opened on UE's device, by their shared handle. */
	TMap<void*, TRefCountPtr<ID3D12Fence>> WebGPUFences;
#endif
};

struct FWebGPUTextureAccessResult
{
	FEvent* Done = FPlatformProcess::GetSynchEventFromPool(true);
#if WITH_WEBGPU_D3D12_INTEROP
	TArray<TPair<TRefCountPtr<ID3D12Fence>, uint64>> Waits;
#endif

	~FWebGPUTextureAccessResult()
	{
		FPlatformProcess::ReturnSynchEventToPool(Done);
	}
};

namespace
{
	bool ToTextureFormat(EPixelFormat Format, EWebGPUTextureFormat& Out)
	{
		switch (Format)
		{
		case PF_R8G8B8A8: Out = EWebGPUTextureFormat::RGBA8Unorm; return true;
		case PF_B8G8R8A8: Out = EWebGPUTextureFormat::BGRA8Unorm; return true;
		case PF_FloatRGBA: Out = EWebGPUTextureFormat::RGBA16Float; return true;
		case PF_A32B32G32R32F: Out = EWebGPUTextureFormat::RGBA32Float; return true;
		case PF_R32_FLOAT: Out = EWebGPUTextureFormat::R32Float; return true;
		case PF_G32R32F: Out = EWebGPUTextureFormat::RG32Float; return true;
		case PF_R32_UINT: Out = EWebGPUTextureFormat::R32Uint; return true;
		case PF_R32_SINT: Out = EWebGPUTextureFormat::R32Sint; return true;
		case PF_R16F: Out = EWebGPUTextureFormat::R16Float; return true;
		case PF_G16R16F: Out = EWebGPUTextureFormat::RG16Float; return true;
		case PF_G8: Out = EWebGPUTextureFormat::R8Unorm; return true;
		default: return false;
		}
	}

	WGPUStringView ToView(const char* Str)
	{
		WGPUStringView View = WGPU_STRING_VIEW_INIT;
		View.data = Str;
		return View;
	}
}

namespace WebGPUTextureInterop
{
	TSharedPtr<FWebGPUSharedTextureState> CreateState(FRHITexture* Texture, FIntPoint& OutSize, EWebGPUTextureFormat& OutFormat, EWebGPUTextureUsage& OutUsage, FString& OutError)
	{
#if WITH_WEBGPU_D3D12_INTEROP
		if (!Texture)
		{
			OutError = TEXT("No RHI texture (render targets create theirs on the render thread; flush rendering commands first)");
			return nullptr;
		}
		if (!GDynamicRHI || GDynamicRHI->GetInterfaceType() != ERHIInterfaceType::D3D12)
		{
			OutError = TEXT("Shared textures need the D3D12 RHI");
			return nullptr;
		}
		const FRHITextureDesc& Desc = Texture->GetDesc();
		if (!EnumHasAnyFlags(Desc.Flags, ETextureCreateFlags::Shared))
		{
			OutError = TEXT("The texture must be created with ETextureCreateFlags::Shared (bGPUSharedFlag on render targets)");
			return nullptr;
		}
		if (Desc.Dimension != ETextureDimension::Texture2D || Desc.NumMips != 1 || Desc.NumSamples != 1)
		{
			OutError = TEXT("Only single-mip, single-sample 2D textures can be shared");
			return nullptr;
		}
		if (!ToTextureFormat(Desc.Format, OutFormat))
		{
			OutError = FString::Printf(TEXT("Pixel format %s has no WebGPU equivalent here"), GPixelFormats[Desc.Format].Name);
			return nullptr;
		}

		TSharedRef<FWebGPUSharedTextureState> State = MakeShared<FWebGPUSharedTextureState>();
		ID3D12Device* D3DDevice = GetID3D12DynamicRHI()->RHIGetDevice(0);
		const HRESULT Result = D3DDevice->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(State->UEFence.GetInitReference()));
		if (FAILED(Result))
		{
			OutError = FString::Printf(TEXT("CreateFence failed: 0x%08x"), (uint32)Result);
			return nullptr;
		}

		State->RHITexture = Texture;
		State->Format = OutFormat;
		OutSize = FIntPoint((int32)Desc.Extent.X, (int32)Desc.Extent.Y);
		OutUsage = EWebGPUTextureUsage::None;
		if (EnumHasAnyFlags(Desc.Flags, ETextureCreateFlags::UAV))
		{
			OutUsage |= EWebGPUTextureUsage::Storage;
		}
		if (EnumHasAnyFlags(Desc.Flags, ETextureCreateFlags::ShaderResource))
		{
			OutUsage |= EWebGPUTextureUsage::Sampled;
		}
		return State;
#else
		OutError = TEXT("Shared textures are not supported on this platform yet");
		return nullptr;
#endif
	}

	WGPUTexture Import(FWebGPUSharedTextureState& State, FWebGPUDevice& Device, FString& OutError)
	{
#if WITH_WEBGPU_D3D12_INTEROP
		if (!Device.IsUsingRHIAdapter())
		{
			OutError = TEXT("The WebGPU device is not on the RHI's adapter");
			return nullptr;
		}
		if (!Device.HasFeature(WGPUFeatureName_SharedTextureMemoryDXGISharedHandle) || !Device.HasFeature(WGPUFeatureName_SharedFenceDXGISharedHandle))
		{
			OutError = TEXT("The WebGPU device lacks the DXGI shared texture/fence features");
			return nullptr;
		}

		ID3D12DynamicRHI* D3D12RHI = GetID3D12DynamicRHI();
		ID3D12Device* D3DDevice = D3D12RHI->RHIGetDevice(0);
		ID3D12Resource* Resource = D3D12RHI->RHIGetResource(State.RHITexture);

		HANDLE ResourceHandle = nullptr;
		HRESULT Result = D3DDevice->CreateSharedHandle(Resource, nullptr, GENERIC_ALL, nullptr, &ResourceHandle);
		if (FAILED(Result))
		{
			OutError = FString::Printf(TEXT("CreateSharedHandle(texture) failed: 0x%08x"), (uint32)Result);
			return nullptr;
		}

		Device.PushErrorScopes();

		WGPUSharedTextureMemoryDXGISharedHandleDescriptor HandleDesc = {};
		HandleDesc.chain.sType = WGPUSType_SharedTextureMemoryDXGISharedHandleDescriptor;
		HandleDesc.handle = ResourceHandle;
		HandleDesc.useKeyedMutex = false;
		WGPUSharedTextureMemoryDescriptor MemoryDesc = {};
		MemoryDesc.nextInChain = &HandleDesc.chain;
		MemoryDesc.label = ToView("WebGPUCompute shared texture");
		State.Memory = wgpuDeviceImportSharedTextureMemory(Device.GetDevice(), &MemoryDesc);
		// Dawn holds its own reference to the resource
		CloseHandle(ResourceHandle);

		WGPUTexture Texture = nullptr;
		WGPUSharedTextureMemoryProperties Properties = {};
		if (State.Memory)
		{
			wgpuSharedTextureMemoryGetProperties(State.Memory, &Properties);
			Texture = wgpuSharedTextureMemoryCreateTexture(State.Memory, nullptr);
		}

		HANDLE FenceHandle = nullptr;
		Result = D3DDevice->CreateSharedHandle(State.UEFence.GetReference(), nullptr, GENERIC_ALL, nullptr, &FenceHandle);
		if (SUCCEEDED(Result))
		{
			WGPUSharedFenceDXGISharedHandleDescriptor FenceHandleDesc = {};
			FenceHandleDesc.chain.sType = WGPUSType_SharedFenceDXGISharedHandleDescriptor;
			FenceHandleDesc.handle = FenceHandle;
			WGPUSharedFenceDescriptor FenceDesc = {};
			FenceDesc.nextInChain = &FenceHandleDesc.chain;
			State.UEFenceForWebGPU = wgpuDeviceImportSharedFence(Device.GetDevice(), &FenceDesc);
			CloseHandle(FenceHandle);
		}

		FString ScopeError;
		const bool bScopeOk = Device.PopErrorScopes(ScopeError);
		const WGPUTextureFormat Expected = FWebGPUTexture::ToWGPU(State.Format);
		if (!bScopeOk || !Texture || !State.UEFenceForWebGPU || FAILED(Result) || Properties.format != Expected)
		{
			OutError = Properties.format != Expected && Texture
				? FString::Printf(TEXT("Dawn imported the texture as format %d, expected %d"), (int32)Properties.format, (int32)Expected)
				: FString::Printf(TEXT("Import failed (hr 0x%08x): %s"), (uint32)Result, *ScopeError);
			Release(State, Texture, false);
			if (Texture)
			{
				wgpuTextureRelease(Texture);
			}
			return nullptr;
		}
		return Texture;
#else
		OutError = TEXT("Shared textures are not supported on this platform yet");
		return nullptr;
#endif
	}

	void Release(FWebGPUSharedTextureState& State, WGPUTexture Texture, bool bAccessing)
	{
#if WITH_WEBGPU_D3D12_INTEROP
		if (bAccessing && State.Memory && Texture)
		{
			WGPUSharedTextureMemoryEndAccessState EndState = {};
			wgpuSharedTextureMemoryEndAccess(State.Memory, Texture, &EndState);
			wgpuSharedTextureMemoryEndAccessStateFreeMembers(EndState);
		}
		if (State.UEFenceForWebGPU)
		{
			wgpuSharedFenceRelease(State.UEFenceForWebGPU);
			State.UEFenceForWebGPU = nullptr;
		}
		if (State.Memory)
		{
			wgpuSharedTextureMemoryRelease(State.Memory);
			State.Memory = nullptr;
		}
		State.WebGPUFences.Reset();
#endif
	}

	bool BeginAccess(FWebGPUSharedTextureState& State, WGPUTexture Texture, uint64 SignalValue)
	{
#if WITH_WEBGPU_D3D12_INTEROP
		if (!State.Memory || !Texture)
		{
			return false;
		}
		WGPUSharedTextureMemoryBeginAccessDescriptor Desc = WGPU_SHARED_TEXTURE_MEMORY_BEGIN_ACCESS_DESCRIPTOR_INIT;
		Desc.concurrentRead = false;
		Desc.initialized = true;
		Desc.fenceCount = 1;
		Desc.fences = &State.UEFenceForWebGPU;
		Desc.signaledValueCount = 1;
		Desc.signaledValues = &SignalValue;
		if (wgpuSharedTextureMemoryBeginAccess(State.Memory, Texture, &Desc) != WGPUStatus_Success)
		{
			UE_LOG(LogWebGPU, Error, TEXT("Shared texture: BeginAccess failed"));
			return false;
		}
		return true;
#else
		return false;
#endif
	}

	void EndAccess(FWebGPUSharedTextureState& State, WGPUTexture Texture, FWebGPUTextureAccessResult& Result)
	{
#if WITH_WEBGPU_D3D12_INTEROP
		WGPUSharedTextureMemoryEndAccessState EndState = {};
		if (wgpuSharedTextureMemoryEndAccess(State.Memory, Texture, &EndState) != WGPUStatus_Success)
		{
			UE_LOG(LogWebGPU, Error, TEXT("Shared texture: EndAccess failed"));
			wgpuSharedTextureMemoryEndAccessStateFreeMembers(EndState);
			return;
		}

		// Opened here (D3D12 devices are free-threaded) so the RHI thread only has to wait on them
		ID3D12Device* D3DDevice = GetID3D12DynamicRHI()->RHIGetDevice(0);
		for (size_t Index = 0; Index < EndState.fenceCount; Index++)
		{
			WGPUSharedFenceDXGISharedHandleExportInfo HandleInfo = {};
			HandleInfo.chain.sType = WGPUSType_SharedFenceDXGISharedHandleExportInfo;
			WGPUSharedFenceExportInfo ExportInfo = {};
			ExportInfo.nextInChain = &HandleInfo.chain;
			wgpuSharedFenceExportInfo(EndState.fences[Index], &ExportInfo);

			TRefCountPtr<ID3D12Fence>* Cached = State.WebGPUFences.Find(HandleInfo.handle);
			if (!Cached)
			{
				TRefCountPtr<ID3D12Fence> Opened;
				const HRESULT OpenResult = D3DDevice->OpenSharedHandle((HANDLE)HandleInfo.handle, IID_PPV_ARGS(Opened.GetInitReference()));
				if (FAILED(OpenResult))
				{
					UE_LOG(LogWebGPU, Error, TEXT("Shared texture: OpenSharedHandle(WebGPU fence) failed: 0x%08x"), (uint32)OpenResult);
					continue;
				}
				Cached = &State.WebGPUFences.Add(HandleInfo.handle, Opened);
			}
			Result.Waits.Emplace(*Cached, EndState.signaledValues[Index]);
		}
		wgpuSharedTextureMemoryEndAccessStateFreeMembers(EndState);
#endif
	}

	uint64 NextSignalValue(FWebGPUSharedTextureState& State)
	{
		return ++State.SignalValue;
	}

	void EnqueueSignal(const TSharedRef<FWebGPUSharedTextureState>& State, uint64 SignalValue)
	{
#if WITH_WEBGPU_D3D12_INTEROP
		ENQUEUE_RENDER_COMMAND(WebGPUTextureSignal)([State, SignalValue](FRHICommandListImmediate& RHICmdList)
		{
			TRefCountPtr<ID3D12Fence> Fence = State->UEFence;
			RHICmdList.EnqueueLambda(TEXT("WebGPUTextureSignal"), [Fence, SignalValue](FRHICommandListImmediate& Cmd)
			{
				GetID3D12DynamicRHI()->RHISignalManualFence(Cmd, Fence.GetReference(), SignalValue);
			});
			// Hand UE's work to the GPU now rather than at the end of the frame: WebGPU's queue waits on it
			RHICmdList.ImmediateFlush(EImmediateFlushType::DispatchToRHIThread);
		});
#endif
	}

	void EnqueueWait(const TSharedRef<FWebGPUTextureAccessResult>& Result)
	{
#if WITH_WEBGPU_D3D12_INTEROP
		ENQUEUE_RENDER_COMMAND(WebGPUTextureWait)([Result](FRHICommandListImmediate& RHICmdList)
		{
			RHICmdList.EnqueueLambda(TEXT("WebGPUTextureWait"), [Result](FRHICommandListImmediate& Cmd)
			{
				// The worker never waits on UE, so it gets here; usually long before the RHI thread does
				if (!Result->Done->Wait(5000))
				{
					UE_LOG(LogWebGPU, Error, TEXT("Shared texture: the WebGPU worker did not end the access within 5 s; UE goes on without waiting for it"));
					return;
				}
				for (const TPair<TRefCountPtr<ID3D12Fence>, uint64>& Wait : Result->Waits)
				{
					GetID3D12DynamicRHI()->RHIWaitManualFence(Cmd, Wait.Key.GetReference(), Wait.Value, nullptr);
				}
			});
		});
#endif
	}

	TSharedRef<FWebGPUTextureAccessResult> MakeAccessResult()
	{
		return MakeShared<FWebGPUTextureAccessResult>();
	}

	void CompleteAccess(FWebGPUTextureAccessResult& Result)
	{
		Result.Done->Trigger();
	}
}
