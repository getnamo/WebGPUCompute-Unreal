// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUDevice.h"
#include "WebGPUCompute.h"
#include "WebGPUInteropUtils.h"

namespace
{
	WGPUStringView ToView(const ANSICHAR* Str, int32 Length)
	{
		WGPUStringView View = WGPU_STRING_VIEW_INIT;
		View.data = Str;
		View.length = (size_t)Length;
		return View;
	}

	WGPUStringView ToView(const ANSICHAR* Str)
	{
		WGPUStringView View = WGPU_STRING_VIEW_INIT;
		View.data = Str;
		return View;
	}

	// Mirrors dawn::native::d3d::RequestAdapterOptionsLUID (a C++ ChainedStruct followed by a LUID)
	// so we don't need Dawn's C++ native headers here.
	struct FDawnRequestAdapterOptionsLUID
	{
		WGPUChainedStruct Chain;
		uint32 LowPart;
		int32 HighPart;
	};

	const TCHAR* ErrorTypeToString(WGPUErrorType Type)
	{
		switch (Type)
		{
		case WGPUErrorType_NoError: return TEXT("NoError");
		case WGPUErrorType_Validation: return TEXT("Validation");
		case WGPUErrorType_OutOfMemory: return TEXT("OutOfMemory");
		case WGPUErrorType_Internal: return TEXT("Internal");
		default: return TEXT("Unknown");
		}
	}
}

FString FWebGPUDevice::ToFString(WGPUStringView View)
{
	if (!View.data)
	{
		return FString();
	}
	const int32 Length = View.length == WGPU_STRLEN ? FCStringAnsi::Strlen(View.data) : (int32)View.length;
	return FString(FUTF8ToTCHAR(View.data, Length));
}

FString FWebGPUDevice::FormatCompileMessages(const TArray<FWebGPUCompileMessage>& Messages)
{
	FString Result;
	for (const FWebGPUCompileMessage& Message : Messages)
	{
		Result += FString::Printf(TEXT("%s:%llu:%llu: %s\n"), Message.bIsError ? TEXT("error") : TEXT("warning"), Message.Line, Message.Column, *Message.Message);
	}
	return Result;
}

FWebGPUDevice::~FWebGPUDevice()
{
	Shutdown();
}

bool FWebGPUDevice::Initialize(const FWebGPUDeviceOptions& Options)
{
	if (Device)
	{
		return true;
	}

	// Instance with timed WaitAny so all async calls can be awaited with a timeout
	WGPUInstanceFeatureName InstanceFeatures[] = { WGPUInstanceFeatureName_TimedWaitAny };
	WGPUInstanceDescriptor InstanceDesc = {};
	InstanceDesc.requiredFeatureCount = UE_ARRAY_COUNT(InstanceFeatures);
	InstanceDesc.requiredFeatures = InstanceFeatures;
	Instance = wgpuCreateInstance(&InstanceDesc);
	if (!Instance)
	{
		UE_LOG(LogWebGPU, Error, TEXT("wgpuCreateInstance failed"));
		return false;
	}

	// Adapter
	WGPURequestAdapterOptions AdapterOptions = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
	AdapterOptions.featureLevel = WGPUFeatureLevel_Core;
	AdapterOptions.powerPreference = WGPUPowerPreference_HighPerformance;
#if PLATFORM_WINDOWS
	AdapterOptions.backendType = WGPUBackendType_D3D12;
#endif

	FDawnRequestAdapterOptionsLUID LUIDOptions = {};
	if (Options.bMatchRHIAdapter && WebGPUInterop::GetRHIAdapterLUID(LUIDOptions.LowPart, LUIDOptions.HighPart))
	{
		LUIDOptions.Chain.sType = WGPUSType_RequestAdapterOptionsLUID;
		AdapterOptions.nextInChain = &LUIDOptions.Chain;
		bUsingRHIAdapter = true;
	}

	struct FAdapterResult
	{
		WGPUAdapter Adapter = nullptr;
		FString Message;
	} AdapterResult;

	WGPURequestAdapterCallbackInfo AdapterCallback = {};
	AdapterCallback.mode = WGPUCallbackMode_WaitAnyOnly;
	AdapterCallback.userdata1 = &AdapterResult;
	AdapterCallback.callback = [](WGPURequestAdapterStatus Status, WGPUAdapter InAdapter, WGPUStringView Message, void* UserData1, void*)
	{
		FAdapterResult& Result = *static_cast<FAdapterResult*>(UserData1);
		Result.Adapter = Status == WGPURequestAdapterStatus_Success ? InAdapter : nullptr;
		Result.Message = FWebGPUDevice::ToFString(Message);
	};

	if (!Wait(wgpuInstanceRequestAdapter(Instance, &AdapterOptions, AdapterCallback)) || !AdapterResult.Adapter)
	{
		UE_LOG(LogWebGPU, Error, TEXT("Could not get a WebGPU adapter: %s"), *AdapterResult.Message);
		Shutdown();
		return false;
	}
	Adapter = AdapterResult.Adapter;

	WGPUAdapterInfo Info = {};
	if (wgpuAdapterGetInfo(Adapter, &Info) == WGPUStatus_Success)
	{
		AdapterDescription = FString::Printf(TEXT("%s (%s, backend %d, vendor 0x%04x device 0x%04x)"),
			*ToFString(Info.description), *ToFString(Info.vendor), (int32)Info.backendType, Info.vendorID, Info.deviceID);
		wgpuAdapterInfoFreeMembers(Info);
	}

	// Features: request what we use, if the adapter has it
	TArray<WGPUFeatureName> WantedFeatures = {
		WGPUFeatureName_ImplicitDeviceSynchronization,
		WGPUFeatureName_TimestampQuery,
		WGPUFeatureName_ShaderF16,
		WGPUFeatureName_Subgroups,
	};
	if (Options.bRequestInteropFeatures)
	{
		WantedFeatures.Append({
			WGPUFeatureName_SharedTextureMemoryDXGISharedHandle,
			WGPUFeatureName_SharedFenceDXGISharedHandle,
		});
	}

	TArray<WGPUFeatureName> RequiredFeatures;
	for (WGPUFeatureName Feature : WantedFeatures)
	{
		if (wgpuAdapterHasFeature(Adapter, Feature))
		{
			RequiredFeatures.Add(Feature);
		}
	}

	// Ask for everything the adapter can do (default limits cap storage bindings at 128 MiB)
	WGPULimits Limits = WGPU_LIMITS_INIT;
	const bool bHaveLimits = wgpuAdapterGetLimits(Adapter, &Limits) == WGPUStatus_Success;
	Limits.nextInChain = nullptr;

	// Native use: full-resolution GPU timestamps (Dawn quantizes them by default, as browsers must)
	const char* DisabledToggles[] = { "timestamp_quantization" };
	WGPUDawnTogglesDescriptor Toggles = {};
	Toggles.chain.sType = WGPUSType_DawnTogglesDescriptor;
	Toggles.disabledToggleCount = UE_ARRAY_COUNT(DisabledToggles);
	Toggles.disabledToggles = DisabledToggles;

	WGPUDeviceDescriptor DeviceDesc = WGPU_DEVICE_DESCRIPTOR_INIT;
	DeviceDesc.nextInChain = &Toggles.chain;
	DeviceDesc.label = ToView("WebGPUCompute");
	DeviceDesc.requiredFeatureCount = RequiredFeatures.Num();
	DeviceDesc.requiredFeatures = RequiredFeatures.GetData();
	DeviceDesc.requiredLimits = bHaveLimits ? &Limits : nullptr;

	DeviceDesc.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
	DeviceDesc.deviceLostCallbackInfo.userdata1 = this;
	DeviceDesc.deviceLostCallbackInfo.callback = [](WGPUDevice const*, WGPUDeviceLostReason Reason, WGPUStringView Message, void* UserData1, void*)
	{
		FWebGPUDevice* Self = static_cast<FWebGPUDevice*>(UserData1);
		Self->bLost = true;
		// Destroyed is the expected reason during Shutdown
		if (Reason != WGPUDeviceLostReason_Destroyed && Reason != WGPUDeviceLostReason_CallbackCancelled)
		{
			UE_LOG(LogWebGPU, Error, TEXT("WebGPU device lost (reason %d): %s"), (int32)Reason, *FWebGPUDevice::ToFString(Message));
			if (Self->LostHandler)
			{
				Self->LostHandler();
			}
		}
	};

	DeviceDesc.uncapturedErrorCallbackInfo.callback = [](WGPUDevice const*, WGPUErrorType Type, WGPUStringView Message, void*, void*)
	{
		UE_LOG(LogWebGPU, Error, TEXT("Uncaptured %s error: %s"), ErrorTypeToString(Type), *FWebGPUDevice::ToFString(Message));
	};

	struct FDeviceResult
	{
		WGPUDevice Device = nullptr;
		FString Message;
	} DeviceResult;

	WGPURequestDeviceCallbackInfo DeviceCallback = {};
	DeviceCallback.mode = WGPUCallbackMode_WaitAnyOnly;
	DeviceCallback.userdata1 = &DeviceResult;
	DeviceCallback.callback = [](WGPURequestDeviceStatus Status, WGPUDevice InDevice, WGPUStringView Message, void* UserData1, void*)
	{
		FDeviceResult& Result = *static_cast<FDeviceResult*>(UserData1);
		Result.Device = Status == WGPURequestDeviceStatus_Success ? InDevice : nullptr;
		Result.Message = FWebGPUDevice::ToFString(Message);
	};

	if (!Wait(wgpuAdapterRequestDevice(Adapter, &DeviceDesc, DeviceCallback)) || !DeviceResult.Device)
	{
		// f16/subgroups need DXC (dxcompiler.dll next to this module); retry without them so FXC can be used
		const int32 Removed = RequiredFeatures.RemoveAll([](WGPUFeatureName Feature)
		{
			return Feature == WGPUFeatureName_ShaderF16 || Feature == WGPUFeatureName_Subgroups;
		});
		if (Removed > 0)
		{
			UE_LOG(LogWebGPU, Warning, TEXT("WebGPU device creation failed (%s); retrying without f16/subgroups"), *DeviceResult.Message);
			DeviceDesc.requiredFeatureCount = RequiredFeatures.Num();
			DeviceDesc.requiredFeatures = RequiredFeatures.GetData();
			DeviceResult = FDeviceResult();
			Wait(wgpuAdapterRequestDevice(Adapter, &DeviceDesc, DeviceCallback));
		}
	}
	if (!DeviceResult.Device)
	{
		UE_LOG(LogWebGPU, Error, TEXT("Could not create a WebGPU device: %s"), *DeviceResult.Message);
		Shutdown();
		return false;
	}
	Device = DeviceResult.Device;
	Queue = wgpuDeviceGetQueue(Device);
	bLost = false;

	WGPULoggingCallbackInfo LoggingCallback = {};
	LoggingCallback.callback = [](WGPULoggingType Type, WGPUStringView Message, void*, void*)
	{
		const FString Text = FWebGPUDevice::ToFString(Message);
		switch (Type)
		{
		case WGPULoggingType_Error: UE_LOG(LogWebGPU, Error, TEXT("%s"), *Text); break;
		case WGPULoggingType_Warning: UE_LOG(LogWebGPU, Warning, TEXT("%s"), *Text); break;
		case WGPULoggingType_Info: UE_LOG(LogWebGPU, Log, TEXT("%s"), *Text); break;
		default: UE_LOG(LogWebGPU, Verbose, TEXT("%s"), *Text); break;
		}
	};
	wgpuDeviceSetLoggingCallback(Device, LoggingCallback);

	UE_LOG(LogWebGPU, Log, TEXT("WebGPU (Dawn) device on %s%s; features: thread-safe=%d interop=%d timestamps=%d f16=%d subgroups=%d; maxBuffer=%llu MiB maxStorageBinding=%llu MiB"),
		*AdapterDescription,
		bUsingRHIAdapter ? TEXT(" [matched RHI adapter]") : TEXT(""),
		HasFeature(WGPUFeatureName_ImplicitDeviceSynchronization),
		HasFeature(WGPUFeatureName_SharedTextureMemoryDXGISharedHandle) && HasFeature(WGPUFeatureName_SharedFenceDXGISharedHandle),
		HasFeature(WGPUFeatureName_TimestampQuery),
		HasFeature(WGPUFeatureName_ShaderF16),
		HasFeature(WGPUFeatureName_Subgroups),
		Limits.maxBufferSize >> 20, Limits.maxStorageBufferBindingSize >> 20);

	return true;
}

void FWebGPUDevice::Shutdown()
{
	if (Queue)
	{
		wgpuQueueRelease(Queue);
		Queue = nullptr;
	}
	if (Device)
	{
		wgpuDeviceDestroy(Device);
		wgpuDeviceRelease(Device);
		Device = nullptr;
	}
	if (Adapter)
	{
		wgpuAdapterRelease(Adapter);
		Adapter = nullptr;
	}
	if (Instance)
	{
		// Flush callbacks that are waiting on the instance (e.g. device lost on destroy)
		wgpuInstanceProcessEvents(Instance);
		wgpuInstanceRelease(Instance);
		Instance = nullptr;
	}
}

bool FWebGPUDevice::HasFeature(WGPUFeatureName Feature) const
{
	return Device && wgpuDeviceHasFeature(Device, Feature);
}

bool FWebGPUDevice::Wait(WGPUFuture Future, double TimeoutSeconds)
{
	if (!Instance || Future.id == 0)
	{
		return false;
	}

	WGPUFutureWaitInfo WaitInfo = {};
	WaitInfo.future = Future;
	const uint64 TimeoutNS = (uint64)(TimeoutSeconds * 1e9);
	const WGPUWaitStatus Status = wgpuInstanceWaitAny(Instance, 1, &WaitInfo, TimeoutNS);
	if (Status != WGPUWaitStatus_Success)
	{
		UE_LOG(LogWebGPU, Warning, TEXT("WebGPU wait failed (status %d)"), (int32)Status);
		return false;
	}
	return WaitInfo.completed != 0;
}

bool FWebGPUDevice::WaitForQueueIdle(double TimeoutSeconds)
{
	if (!Queue)
	{
		return false;
	}

	WGPUQueueWorkDoneCallbackInfo Callback = {};
	Callback.mode = WGPUCallbackMode_WaitAnyOnly;
	Callback.callback = [](WGPUQueueWorkDoneStatus, WGPUStringView, void*, void*) {};
	return Wait(wgpuQueueOnSubmittedWorkDone(Queue, Callback), TimeoutSeconds);
}

void FWebGPUDevice::PushErrorScopes()
{
	wgpuDevicePushErrorScope(Device, WGPUErrorFilter_Internal);
	wgpuDevicePushErrorScope(Device, WGPUErrorFilter_OutOfMemory);
	wgpuDevicePushErrorScope(Device, WGPUErrorFilter_Validation);
}

bool FWebGPUDevice::PopErrorScopes(FString& OutError)
{
	bool bOk = true;
	for (int32 Index = 0; Index < 3; Index++)
	{
		struct FScopeResult
		{
			WGPUErrorType Type = WGPUErrorType_NoError;
			FString Message;
		} Result;

		WGPUPopErrorScopeCallbackInfo Callback = {};
		Callback.mode = WGPUCallbackMode_WaitAnyOnly;
		Callback.userdata1 = &Result;
		Callback.callback = [](WGPUPopErrorScopeStatus, WGPUErrorType Type, WGPUStringView Message, void* UserData1, void*)
		{
			FScopeResult& ScopeResult = *static_cast<FScopeResult*>(UserData1);
			ScopeResult.Type = Type;
			ScopeResult.Message = FWebGPUDevice::ToFString(Message);
		};

		Wait(wgpuDevicePopErrorScope(Device, Callback));
		if (Result.Type != WGPUErrorType_NoError)
		{
			bOk = false;
			OutError += FString::Printf(TEXT("[%s] %s\n"), ErrorTypeToString(Result.Type), *Result.Message);
		}
	}
	return bOk;
}

WGPUShaderModule FWebGPUDevice::CreateShaderModule(const FString& Source, const FString& Label, TArray<FWebGPUCompileMessage>& OutMessages)
{
	if (!IsValid())
	{
		return nullptr;
	}

	FTCHARToUTF8 SourceUtf8(*Source);
	FTCHARToUTF8 LabelUtf8(*Label);

	WGPUShaderSourceWGSL WGSL = {};
	WGSL.chain.sType = WGPUSType_ShaderSourceWGSL;
	WGSL.code = ToView(SourceUtf8.Get(), SourceUtf8.Length());

	WGPUShaderModuleDescriptor Desc = {};
	Desc.nextInChain = &WGSL.chain;
	Desc.label = ToView(LabelUtf8.Get(), LabelUtf8.Length());

	PushErrorScopes();
	WGPUShaderModule Module = wgpuDeviceCreateShaderModule(Device, &Desc);

	WGPUCompilationInfoCallbackInfo InfoCallback = {};
	InfoCallback.mode = WGPUCallbackMode_WaitAnyOnly;
	InfoCallback.userdata1 = &OutMessages;
	InfoCallback.callback = [](WGPUCompilationInfoRequestStatus Status, WGPUCompilationInfo const* Info, void* UserData1, void*)
	{
		TArray<FWebGPUCompileMessage>& Messages = *static_cast<TArray<FWebGPUCompileMessage>*>(UserData1);
		if (Status != WGPUCompilationInfoRequestStatus_Success || !Info)
		{
			return;
		}
		for (size_t Index = 0; Index < Info->messageCount; Index++)
		{
			const WGPUCompilationMessage& Source = Info->messages[Index];
			FWebGPUCompileMessage& Message = Messages.AddDefaulted_GetRef();
			Message.bIsError = Source.type == WGPUCompilationMessageType_Error;
			Message.Message = FWebGPUDevice::ToFString(Source.message);
			Message.Line = Source.lineNum;
			Message.Column = Source.linePos;
		}
	};
	if (Module)
	{
		Wait(wgpuShaderModuleGetCompilationInfo(Module, InfoCallback));
	}

	FString ScopeError;
	const bool bScopeOk = PopErrorScopes(ScopeError);
	const bool bHasErrorMessage = OutMessages.ContainsByPredicate([](const FWebGPUCompileMessage& Message) { return Message.bIsError; });

	if (!Module || !bScopeOk || bHasErrorMessage)
	{
		if (!bHasErrorMessage)
		{
			FWebGPUCompileMessage& Message = OutMessages.AddDefaulted_GetRef();
			Message.bIsError = true;
			Message.Message = ScopeError.IsEmpty() ? TEXT("Shader module creation failed") : ScopeError;
		}
		if (Module)
		{
			wgpuShaderModuleRelease(Module);
		}
		return nullptr;
	}
	return Module;
}

bool FWebGPUDevice::RunBufferKernel(const FString& Source, const TArray<uint32>& InData, TArray<uint32>& OutData, FString& OutError)
{
	if (!IsValid())
	{
		OutError = TEXT("WebGPU device is not available");
		return false;
	}
	if (InData.Num() == 0)
	{
		OutData.Reset();
		return true;
	}

	TArray<FWebGPUCompileMessage> Messages;
	WGPUShaderModule Module = CreateShaderModule(Source, TEXT("RunBufferKernel"), Messages);
	if (!Module)
	{
		OutError = FormatCompileMessages(Messages);
		return false;
	}

	const uint64 Size = (uint64)InData.Num() * sizeof(uint32);
	WGPUBuffer StorageBuffer = nullptr;
	WGPUBuffer StagingBuffer = nullptr;
	WGPUComputePipeline Pipeline = nullptr;
	WGPUBindGroupLayout Layout = nullptr;
	WGPUBindGroup BindGroup = nullptr;
	bool bOk = false;

	PushErrorScopes();
	{
		WGPUBufferDescriptor StorageDesc = WGPU_BUFFER_DESCRIPTOR_INIT;
		StorageDesc.label = ToView("storage");
		StorageDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
		StorageDesc.size = Size;
		StorageBuffer = wgpuDeviceCreateBuffer(Device, &StorageDesc);

		WGPUBufferDescriptor StagingDesc = WGPU_BUFFER_DESCRIPTOR_INIT;
		StagingDesc.label = ToView("staging");
		StagingDesc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
		StagingDesc.size = Size;
		StagingBuffer = wgpuDeviceCreateBuffer(Device, &StagingDesc);

		WGPUComputePipelineDescriptor PipelineDesc = {};
		PipelineDesc.label = ToView("RunBufferKernel");
		PipelineDesc.compute = WGPU_COMPUTE_STATE_INIT;
		PipelineDesc.compute.module = Module;
		PipelineDesc.compute.entryPoint = ToView("main");
		Pipeline = wgpuDeviceCreateComputePipeline(Device, &PipelineDesc);
		Layout = wgpuComputePipelineGetBindGroupLayout(Pipeline, 0);

		WGPUBindGroupEntry Entry = WGPU_BIND_GROUP_ENTRY_INIT;
		Entry.binding = 0;
		Entry.buffer = StorageBuffer;
		Entry.size = Size;

		WGPUBindGroupDescriptor BindGroupDesc = {};
		BindGroupDesc.layout = Layout;
		BindGroupDesc.entryCount = 1;
		BindGroupDesc.entries = &Entry;
		BindGroup = wgpuDeviceCreateBindGroup(Device, &BindGroupDesc);

		wgpuQueueWriteBuffer(Queue, StorageBuffer, 0, InData.GetData(), Size);

		WGPUCommandEncoder Encoder = wgpuDeviceCreateCommandEncoder(Device, nullptr);
		WGPUComputePassEncoder Pass = wgpuCommandEncoderBeginComputePass(Encoder, nullptr);
		wgpuComputePassEncoderSetPipeline(Pass, Pipeline);
		wgpuComputePassEncoderSetBindGroup(Pass, 0, BindGroup, 0, nullptr);
		wgpuComputePassEncoderDispatchWorkgroups(Pass, (uint32)InData.Num(), 1, 1);
		wgpuComputePassEncoderEnd(Pass);
		wgpuComputePassEncoderRelease(Pass);

		wgpuCommandEncoderCopyBufferToBuffer(Encoder, StorageBuffer, 0, StagingBuffer, 0, Size);
		WGPUCommandBuffer Commands = wgpuCommandEncoderFinish(Encoder, nullptr);
		wgpuQueueSubmit(Queue, 1, &Commands);
		wgpuCommandBufferRelease(Commands);
		wgpuCommandEncoderRelease(Encoder);
	}
	FString ScopeError;
	if (!PopErrorScopes(ScopeError))
	{
		OutError = ScopeError;
	}
	else
	{
		struct FMapResult
		{
			WGPUMapAsyncStatus Status = WGPUMapAsyncStatus_Error;
			FString Message;
		} MapResult;

		WGPUBufferMapCallbackInfo MapCallback = {};
		MapCallback.mode = WGPUCallbackMode_WaitAnyOnly;
		MapCallback.userdata1 = &MapResult;
		MapCallback.callback = [](WGPUMapAsyncStatus Status, WGPUStringView Message, void* UserData1, void*)
		{
			FMapResult& Result = *static_cast<FMapResult*>(UserData1);
			Result.Status = Status;
			Result.Message = FWebGPUDevice::ToFString(Message);
		};

		if (Wait(wgpuBufferMapAsync(StagingBuffer, WGPUMapMode_Read, 0, Size, MapCallback)) && MapResult.Status == WGPUMapAsyncStatus_Success)
		{
			const uint32* Mapped = static_cast<const uint32*>(wgpuBufferGetConstMappedRange(StagingBuffer, 0, Size));
			OutData.SetNumUninitialized(InData.Num());
			FMemory::Memcpy(OutData.GetData(), Mapped, Size);
			wgpuBufferUnmap(StagingBuffer);
			bOk = true;
		}
		else
		{
			OutError = FString::Printf(TEXT("Buffer map failed: %s"), *MapResult.Message);
		}
	}

	if (BindGroup) wgpuBindGroupRelease(BindGroup);
	if (Layout) wgpuBindGroupLayoutRelease(Layout);
	if (Pipeline) wgpuComputePipelineRelease(Pipeline);
	if (StagingBuffer) wgpuBufferRelease(StagingBuffer);
	if (StorageBuffer) wgpuBufferRelease(StorageBuffer);
	wgpuShaderModuleRelease(Module);
	return bOk;
}
