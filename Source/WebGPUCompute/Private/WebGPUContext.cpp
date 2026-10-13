// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUContext.h"
#include "WebGPUCompute.h"
#include "Async/Async.h"
#include "HAL/Event.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTLS.h"
#include "HAL/RunnableThread.h"
#include "Misc/ScopeLock.h"

namespace
{
	WGPUStringView ToView(const FTCHARToUTF8& Utf8)
	{
		WGPUStringView View = WGPU_STRING_VIEW_INIT;
		View.data = Utf8.Get();
		View.length = (size_t)Utf8.Length();
		return View;
	}

	WGPUStringView ToView(const char* Str)
	{
		WGPUStringView View = WGPU_STRING_VIEW_INIT;
		View.data = Str;
		return View;
	}

	WGPUBufferUsage ToWGPUUsage(EWebGPUBufferUsage Usage)
	{
		WGPUBufferUsage Result = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
		if (EnumHasAnyFlags(Usage, EWebGPUBufferUsage::Storage))
		{
			Result |= WGPUBufferUsage_Storage;
		}
		if (EnumHasAnyFlags(Usage, EWebGPUBufferUsage::Uniform))
		{
			Result |= WGPUBufferUsage_Uniform;
		}
		if (EnumHasAnyFlags(Usage, EWebGPUBufferUsage::Indirect))
		{
			Result |= WGPUBufferUsage_Indirect;
		}
		return Result;
	}

	constexpr uint64 MinStagingSize = 256;
	constexpr int32 MaxPooledStagingPerSize = 4;

	/** Shared between a Flush caller and the worker so a timed-out wait never touches a dead event. */
	struct FFlushSignal
	{
		FEvent* Event = FPlatformProcess::GetSynchEventFromPool(true);
		std::atomic<bool> bOk = false;

		~FFlushSignal()
		{
			FPlatformProcess::ReturnSynchEventToPool(Event);
		}
	};
}

// Async requests, owned by their callback

struct FWebGPUContext::FReadRequest
{
	FWebGPUContext* Context = nullptr;
	WGPUBuffer Staging = nullptr;
	uint64 Capacity = 0;
	uint64 Size = 0;
	uint64 MappedSize = 0;
	uint32 Generation = 0;
	FWebGPUReadCallback Callback;
	EWebGPUCallbackThread Thread = EWebGPUCallbackThread::GameThread;
};

struct FWebGPUContext::FCompileRequest
{
	FWebGPUContext* Context = nullptr;
	TWeakPtr<FWebGPUKernel> Kernel;
	uint32 Serial = 0;
	uint32 Generation = 0;
	TArray<FWebGPUCompileMessage> Messages;
	FWebGPUCompileCallback Callback;
	EWebGPUCallbackThread Thread = EWebGPUCallbackThread::GameThread;
};

struct FWebGPUContext::FBatchScopeRequest
{
	FWebGPUContext* Context = nullptr;
	FString Labels;
};

struct FWebGPUContext::FTimestampRequest
{
	FWebGPUContext* Context = nullptr;
	WGPUBuffer Staging = nullptr;
	uint64 Capacity = 0;
	uint32 Count = 0;
	uint32 Generation = 0;
	TArray<TWeakPtr<FWebGPUKernel>> Kernels;
};

struct FWebGPUContext::FFlushRequest
{
	FWebGPUContext* Context = nullptr;
	TSharedPtr<FFlushSignal> Signal;
};

// FWebGPUBuffer

TSharedPtr<FWebGPUContext> FWebGPUBuffer::GetContext() const
{
	return Context.Pin();
}

FWebGPUBuffer::~FWebGPUBuffer()
{
	if (!Handle)
	{
		return;
	}

	WGPUBuffer ToRelease = Handle;
	auto Release = [ToRelease]()
	{
		// Destroy frees the memory now rather than when Dawn drops its last internal reference
		wgpuBufferDestroy(ToRelease);
		wgpuBufferRelease(ToRelease);
	};

	TSharedPtr<FWebGPUContext> Ctx = Context.Pin();
	if (Ctx && !Ctx->IsInWorkerThread())
	{
		Ctx->Enqueue(MoveTemp(Release));
	}
	else
	{
		Release();
	}
}

// FWebGPUKernel

TSharedPtr<FWebGPUContext> FWebGPUKernel::GetContext() const
{
	return Context.Pin();
}

TArray<FWebGPUCompileMessage> FWebGPUKernel::GetMessages() const
{
	FScopeLock Lock(&MessagesLock);
	return Messages;
}

void FWebGPUKernel::SetMessages(const TArray<FWebGPUCompileMessage>& InMessages)
{
	FScopeLock Lock(&MessagesLock);
	Messages = InMessages;
}

void FWebGPUKernel::ReleaseBindGroups()
{
	for (const TPair<uint32, WGPUBindGroup>& Pair : BindGroups)
	{
		wgpuBindGroupRelease(Pair.Value);
	}
	BindGroups.Reset();
	bBindGroupsDirty = true;
}

void FWebGPUKernel::ReleaseGPU()
{
	ReleaseBindGroups();
	if (Pipeline)
	{
		wgpuComputePipelineRelease(Pipeline);
		Pipeline = nullptr;
	}
	bReady = false;
}

FWebGPUKernel::~FWebGPUKernel()
{
	if (!Pipeline && BindGroups.Num() == 0)
	{
		return;
	}

	WGPUComputePipeline ToRelease = Pipeline;
	TArray<WGPUBindGroup> Groups;
	BindGroups.GenerateValueArray(Groups);
	auto Release = [ToRelease, Groups]()
	{
		for (WGPUBindGroup Group : Groups)
		{
			wgpuBindGroupRelease(Group);
		}
		if (ToRelease)
		{
			wgpuComputePipelineRelease(ToRelease);
		}
	};

	TSharedPtr<FWebGPUContext> Ctx = Context.Pin();
	if (Ctx && !Ctx->IsInWorkerThread())
	{
		Ctx->Enqueue(MoveTemp(Release));
	}
	else
	{
		Release();
	}
}

// FWebGPUContext: lifetime

TSharedPtr<FWebGPUContext> FWebGPUContext::Create(const FWebGPUDeviceOptions& Options)
{
	TSharedPtr<FWebGPUContext> Context = MakeShareable(new FWebGPUContext(Options));
	if (!Context->Start())
	{
		return nullptr;
	}
	return Context;
}

FWebGPUContext::FWebGPUContext(const FWebGPUDeviceOptions& InOptions)
	: Options(InOptions)
{
	WakeEvent = FPlatformProcess::GetSynchEventFromPool(false);
}

FWebGPUContext::~FWebGPUContext()
{
	Shutdown();
	FPlatformProcess::ReturnSynchEventToPool(WakeEvent);
	WakeEvent = nullptr;
}

bool FWebGPUContext::Start()
{
	// The first device is created on the calling thread so a missing GPU is reported synchronously
	TSharedPtr<FWebGPUDevice> NewDevice = MakeShared<FWebGPUDevice>();
	NewDevice->SetLostHandler([this]()
	{
		bDeviceLostSignal = true;
		Wake();
	});
	if (!NewDevice->Initialize(Options))
	{
		return false;
	}

	{
		FScopeLock Lock(&DeviceLock);
		Device = NewDevice;
	}
	WorkerDevice = NewDevice;
	DeviceGeneration = 1;
	CreateDeviceObjects();

	Thread = FRunnableThread::Create(this, TEXT("WebGPUWorker"), 0, TPri_AboveNormal);
	return Thread != nullptr;
}

void FWebGPUContext::Shutdown()
{
	if (!Thread)
	{
		return;
	}
	check(!IsInWorkerThread());

	bStopping = true;
	Wake();
	Thread->WaitForCompletion();
	delete Thread;
	Thread = nullptr;
}

uint32 FWebGPUContext::Run()
{
	WorkerThreadId = FPlatformTLS::GetCurrentThreadId();

	while (!bStopping)
	{
		for (;;)
		{
			TUniqueFunction<void()> Command;
			if (bStopping || !Commands.Dequeue(Command))
			{
				break;
			}
			Command();
		}
		if (bStopping)
		{
			break;
		}

		// The queue ran dry: everything encoded so far goes out as one command buffer
		if (Encoder)
		{
			SubmitBatch();
		}

		if (PendingCallbacks > 0 && WorkerDevice)
		{
			wgpuInstanceProcessEvents(WorkerDevice->GetInstance());
		}

		if ((bDeviceLostSignal || !WorkerDevice || WorkerDevice->IsLost()) && FPlatformTime::Seconds() >= NextRecoveryTime)
		{
			Recover();
		}

		if (Commands.IsEmpty())
		{
			if (PendingCallbacks > 0 && FPlatformTime::Seconds() < BusyPollUntil)
			{
				// Results are due soon after a submit: poll again right away. Sleep(0) only yields to threads of
				// equal priority, so the (above normal) worker isn't descheduled behind busy engine threads.
				FPlatformProcess::SleepNoStats(0.0f);
			}
			else
			{
				// Poll Dawn every millisecond while async work is pending; otherwise sleep until woken
				const uint32 WaitMs = PendingCallbacks > 0 ? 1 : (!HasDevice() ? 1000 : MAX_uint32);
				WakeEvent->Wait(WaitMs);
			}
		}
	}

	ReleaseAllResources();
	return 0;
}

bool FWebGPUContext::IsInWorkerThread() const
{
	const uint32 Id = WorkerThreadId;
	return Id != 0 && FPlatformTLS::GetCurrentThreadId() == Id;
}

void FWebGPUContext::Wake()
{
	if (WakeEvent)
	{
		WakeEvent->Trigger();
	}
}

void FWebGPUContext::Enqueue(TUniqueFunction<void()>&& Command)
{
	if (bStopping)
	{
		return;
	}
	Commands.Enqueue(MoveTemp(Command));
	Wake();
}

void FWebGPUContext::Deliver(EWebGPUCallbackThread Thread, TUniqueFunction<void()>&& Function)
{
	if (Thread == EWebGPUCallbackThread::GameThread)
	{
		AsyncTask(ENamedThreads::GameThread, MoveTemp(Function));
	}
	else
	{
		Function();
	}
}

TSharedPtr<FWebGPUDevice> FWebGPUContext::GetDevice() const
{
	FScopeLock Lock(&DeviceLock);
	return Device;
}

FWebGPUStats FWebGPUContext::GetStats() const
{
	FWebGPUStats Stats;
	Stats.Submits = StatSubmits;
	Stats.Dispatches = StatDispatches;
	Stats.SkippedDispatches = StatSkippedDispatches;
	Stats.FailedSubmits = StatFailedSubmits;
	Stats.BytesWritten = StatBytesWritten;
	Stats.BytesRead = StatBytesRead;
	Stats.Readbacks = StatReadbacks;
	Stats.DeviceGeneration = (int32)DeviceGeneration;
	return Stats;
}

// Public commands

TSharedRef<FWebGPUBuffer> FWebGPUContext::CreateBuffer(uint64 Size, EWebGPUBufferUsage Usage, const FString& Label)
{
	TSharedRef<FWebGPUBuffer> Buffer = MakeShared<FWebGPUBuffer>();
	Buffer->Context = AsShared();
	Buffer->Size = Size;
	Buffer->Usage = Usage;
	Buffer->Label = Label.IsEmpty() ? TEXT("WebGPUBuffer") : Label;

	Enqueue([this, Buffer]()
	{
		LiveBuffers.Add(Buffer);
		CreateBufferHandle(*Buffer);
	});
	return Buffer;
}

TSharedRef<FWebGPUKernel> FWebGPUContext::CreateKernel(const FString& Source, const FString& EntryPoint, const FString& Label,
	FWebGPUCompileCallback OnCompiled, EWebGPUCallbackThread CallbackThread)
{
	TSharedRef<FWebGPUKernel> Kernel = MakeShared<FWebGPUKernel>();
	Kernel->Context = AsShared();
	Kernel->Label = Label.IsEmpty() ? EntryPoint : Label;

	Enqueue([this, Kernel, Source, EntryPoint, OnCompiled = MoveTemp(OnCompiled), CallbackThread]() mutable
	{
		LiveKernels.Add(Kernel);
		Kernel->Source = Source;
		Kernel->EntryPoint = EntryPoint;
		Kernel->Listener = MoveTemp(OnCompiled);
		Kernel->ListenerThread = CallbackThread;
		CompileKernel(Kernel, nullptr, CallbackThread);
	});
	return Kernel;
}

void FWebGPUContext::RecompileKernel(const TSharedRef<FWebGPUKernel>& Kernel, const FString& Source,
	FWebGPUCompileCallback OnCompiled, EWebGPUCallbackThread CallbackThread)
{
	Enqueue([this, Kernel, Source, OnCompiled = MoveTemp(OnCompiled), CallbackThread]() mutable
	{
		Kernel->Source = Source;
		CompileKernel(Kernel, MoveTemp(OnCompiled), CallbackThread);
	});
}

void FWebGPUContext::WriteBuffer(const TSharedRef<FWebGPUBuffer>& Buffer, uint64 Offset, const void* Data, uint64 Size)
{
	TArray<uint8> Copy;
	Copy.Append(static_cast<const uint8*>(Data), (int32)Size);
	WriteBuffer(Buffer, Offset, MoveTemp(Copy));
}

void FWebGPUContext::WriteBuffer(const TSharedRef<FWebGPUBuffer>& Buffer, uint64 Offset, TArray<uint8>&& Data)
{
	const uint64 Size = (uint64)Data.Num();
	if (Size == 0)
	{
		return;
	}
	if (Offset % 4 != 0 || Size % 4 != 0 || Offset + Size > Buffer->GetSize())
	{
		UE_LOG(LogWebGPU, Error, TEXT("WriteBuffer(%s): offset %llu and size %llu must be multiples of 4 within %llu bytes"),
			*Buffer->GetLabel(), Offset, Size, Buffer->GetSize());
		return;
	}

	Enqueue([this, Buffer, Offset, Data = MoveTemp(Data)]()
	{
		if (!HasDevice() || !Buffer->Handle)
		{
			return;
		}
		// Keep order with dispatches already encoded in this batch
		if (Encoder)
		{
			SubmitBatch();
		}
		wgpuQueueWriteBuffer(WorkerDevice->GetQueue(), Buffer->Handle, Offset, Data.GetData(), (size_t)Data.Num());
		StatBytesWritten += Data.Num();
	});
}

void FWebGPUContext::SetBuffer(const TSharedRef<FWebGPUKernel>& Kernel, uint32 Group, uint32 Binding, const TSharedPtr<FWebGPUBuffer>& Buffer, uint64 Offset, uint64 Size)
{
	Enqueue([Kernel, Group, Binding, Buffer, Offset, Size]()
	{
		if (Buffer)
		{
			FWebGPUKernel::FBinding& Entry = Kernel->Bindings.FindOrAdd(Group).FindOrAdd(Binding);
			Entry.Buffer = Buffer;
			Entry.Offset = Offset;
			Entry.Size = Size;
		}
		else if (TSortedMap<uint32, FWebGPUKernel::FBinding>* GroupBindings = Kernel->Bindings.Find(Group))
		{
			GroupBindings->Remove(Binding);
			if (GroupBindings->Num() == 0)
			{
				Kernel->Bindings.Remove(Group);
			}
		}
		Kernel->bBindGroupsDirty = true;
	});
}

void FWebGPUContext::Dispatch(const TSharedRef<FWebGPUKernel>& Kernel, uint32 X, uint32 Y, uint32 Z)
{
	Enqueue([this, Kernel, X, Y, Z]()
	{
		EncodeDispatch(Kernel, X, Y, Z);
	});
}

void FWebGPUContext::SetProfiling(const TSharedRef<FWebGPUKernel>& Kernel, bool bEnabled)
{
	Enqueue([Kernel, bEnabled]()
	{
		Kernel->bProfile = bEnabled;
	});
}

void FWebGPUContext::ReadBuffer(const TSharedRef<FWebGPUBuffer>& Buffer, uint64 Offset, uint64 Size, FWebGPUReadCallback&& Callback, EWebGPUCallbackThread CallbackThread)
{
	if (Size == 0 || Offset % 4 != 0 || Size % 4 != 0 || Offset + Size > Buffer->GetSize())
	{
		UE_LOG(LogWebGPU, Error, TEXT("ReadBuffer(%s): offset %llu and size %llu must be non-empty multiples of 4 within %llu bytes"),
			*Buffer->GetLabel(), Offset, Size, Buffer->GetSize());
		Deliver(CallbackThread, [Callback = MoveTemp(Callback)]() mutable
		{
			if (Callback)
			{
				Callback(false, TArray<uint8>());
			}
		});
		return;
	}

	Enqueue([this, Buffer, Offset, Size, Callback = MoveTemp(Callback), CallbackThread]() mutable
	{
		ReadBufferOnWorker(*Buffer, Offset, Size, MoveTemp(Callback), CallbackThread);
	});
}

bool FWebGPUContext::Flush(double TimeoutSeconds)
{
	check(!IsInWorkerThread());
	if (bStopping)
	{
		return false;
	}

	TSharedPtr<FFlushSignal> Signal = MakeShared<FFlushSignal>();
	Enqueue([this, Signal]()
	{
		if (!HasDevice())
		{
			Signal->Event->Trigger();
			return;
		}
		SubmitBatch();

		FFlushRequest* Request = new FFlushRequest{ this, Signal };
		WGPUQueueWorkDoneCallbackInfo Info = {};
		Info.mode = WGPUCallbackMode_AllowProcessEvents;
		Info.callback = &FWebGPUContext::OnFlushDone;
		Info.userdata1 = Request;
		PendingCallbacks++;
		wgpuQueueOnSubmittedWorkDone(WorkerDevice->GetQueue(), Info);
	});

	return Signal->Event->Wait((uint32)(TimeoutSeconds * 1000.0)) && Signal->bOk;
}

void FWebGPUContext::ForceDeviceLoss()
{
	Enqueue([this]()
	{
		if (HasDevice())
		{
			wgpuDeviceForceLoss(WorkerDevice->GetDevice(), WGPUDeviceLostReason_Unknown, ToView("WebGPUCompute: forced device loss"));
		}
	});
}

// Worker thread

void FWebGPUContext::CreateBufferHandle(FWebGPUBuffer& Buffer)
{
	if (!HasDevice())
	{
		return;
	}

	FTCHARToUTF8 Label(*Buffer.Label);
	WGPUBufferDescriptor Desc = WGPU_BUFFER_DESCRIPTOR_INIT;
	Desc.label = ToView(Label);
	Desc.usage = ToWGPUUsage(Buffer.Usage);
	Desc.size = Align(FMath::Max<uint64>(Buffer.Size, 4), 4);

	WorkerDevice->PushErrorScopes();
	Buffer.Handle = wgpuDeviceCreateBuffer(WorkerDevice->GetDevice(), &Desc);
	FString Error;
	if (!WorkerDevice->PopErrorScopes(Error))
	{
		UE_LOG(LogWebGPU, Error, TEXT("CreateBuffer(%s, %llu bytes) failed: %s"), *Buffer.Label, Buffer.Size, *Error);
		if (Buffer.Handle)
		{
			wgpuBufferRelease(Buffer.Handle);
			Buffer.Handle = nullptr;
		}
	}
}

void FWebGPUContext::CompileKernel(const TSharedRef<FWebGPUKernel>& Kernel, FWebGPUCompileCallback OnCompiled, EWebGPUCallbackThread CallbackThread)
{
	const uint32 Serial = ++Kernel->CompileSerial;

	if (!HasDevice())
	{
		FWebGPUCompileMessage Message;
		Message.bIsError = true;
		Message.Message = TEXT("WebGPU device unavailable; the kernel compiles when it comes back");
		FinishCompile(*Kernel, false, { Message }, MoveTemp(OnCompiled), CallbackThread);
		return;
	}

	TArray<FWebGPUCompileMessage> Messages;
	WGPUShaderModule Module = WorkerDevice->CreateShaderModule(Kernel->Source, Kernel->Label, Messages);
	if (!Module)
	{
		FinishCompile(*Kernel, false, Messages, MoveTemp(OnCompiled), CallbackThread);
		return;
	}

	FTCHARToUTF8 EntryPoint(*Kernel->EntryPoint);
	FTCHARToUTF8 Label(*Kernel->Label);
	WGPUComputePipelineDescriptor Desc = {};
	Desc.label = ToView(Label);
	Desc.compute = WGPU_COMPUTE_STATE_INIT;
	Desc.compute.module = Module;
	Desc.compute.entryPoint = ToView(EntryPoint);

	FCompileRequest* Request = new FCompileRequest();
	Request->Context = this;
	Request->Kernel = Kernel;
	Request->Serial = Serial;
	Request->Generation = DeviceGeneration;
	Request->Messages = MoveTemp(Messages);
	Request->Callback = MoveTemp(OnCompiled);
	Request->Thread = CallbackThread;

	WGPUCreateComputePipelineAsyncCallbackInfo Info = {};
	Info.mode = WGPUCallbackMode_AllowProcessEvents;
	Info.callback = &FWebGPUContext::OnPipelineCreated;
	Info.userdata1 = Request;
	PendingCallbacks++;
	wgpuDeviceCreateComputePipelineAsync(WorkerDevice->GetDevice(), &Desc, Info);
	wgpuShaderModuleRelease(Module);
}

void FWebGPUContext::FinishCompile(FWebGPUKernel& Kernel, bool bSuccess, const TArray<FWebGPUCompileMessage>& Messages,
	FWebGPUCompileCallback&& OnCompiled, EWebGPUCallbackThread CallbackThread)
{
	Kernel.SetMessages(Messages);
	if (OnCompiled)
	{
		Deliver(CallbackThread, [Callback = MoveTemp(OnCompiled), bSuccess, Messages]()
		{
			Callback(bSuccess, Messages);
		});
	}
	if (Kernel.Listener)
	{
		Deliver(Kernel.ListenerThread, [Listener = Kernel.Listener, bSuccess, Messages]()
		{
			Listener(bSuccess, Messages);
		});
	}
}

void FWebGPUContext::OnPipelineCreated(WGPUCreatePipelineAsyncStatus Status, WGPUComputePipeline Pipeline, WGPUStringView Message, void* UserData1, void*)
{
	TUniquePtr<FCompileRequest> Request(static_cast<FCompileRequest*>(UserData1));
	FWebGPUContext& Self = *Request->Context;
	Self.PendingCallbacks--;

	TSharedPtr<FWebGPUKernel> Kernel = Request->Kernel.Pin();
	if (!Kernel || Request->Generation != Self.DeviceGeneration || Self.bStopping || Self.bReleasingDevice)
	{
		// The kernel is gone, or this pipeline belongs to a device that was replaced
		if (Pipeline)
		{
			wgpuComputePipelineRelease(Pipeline);
		}
		return;
	}

	const bool bSuccess = Status == WGPUCreatePipelineAsyncStatus_Success && Pipeline != nullptr;
	if (bSuccess)
	{
		if (Request->Serial > Kernel->AppliedSerial)
		{
			Kernel->ReleaseBindGroups();
			if (Kernel->Pipeline)
			{
				wgpuComputePipelineRelease(Kernel->Pipeline);
			}
			Kernel->Pipeline = Pipeline;
			Kernel->AppliedSerial = Request->Serial;
			Kernel->bReady = true;
			Kernel->bWarnedNotReady = false;
		}
		else
		{
			// A newer compile already landed
			wgpuComputePipelineRelease(Pipeline);
		}
	}
	else
	{
		FWebGPUCompileMessage& Error = Request->Messages.AddDefaulted_GetRef();
		Error.bIsError = true;
		Error.Message = FWebGPUDevice::ToFString(Message);
		if (Pipeline)
		{
			wgpuComputePipelineRelease(Pipeline);
		}
	}

	Self.FinishCompile(*Kernel, bSuccess, Request->Messages, MoveTemp(Request->Callback), Request->Thread);
}

bool FWebGPUContext::UpdateBindGroups(FWebGPUKernel& Kernel)
{
	if (!Kernel.bBindGroupsDirty)
	{
		return true;
	}

	Kernel.ReleaseBindGroups();
	for (const TPair<uint32, TSortedMap<uint32, FWebGPUKernel::FBinding>>& Group : Kernel.Bindings)
	{
		TArray<WGPUBindGroupEntry> Entries;
		for (const TPair<uint32, FWebGPUKernel::FBinding>& Binding : Group.Value)
		{
			const FWebGPUKernel::FBinding& Bound = Binding.Value;
			if (!Bound.Buffer || !Bound.Buffer->Handle)
			{
				UE_LOG(LogWebGPU, Warning, TEXT("Kernel '%s': binding @group(%u) @binding(%u) has no GPU buffer"), *Kernel.Label, Group.Key, Binding.Key);
				Kernel.ReleaseBindGroups();
				return false;
			}
			WGPUBindGroupEntry& Entry = Entries.Add_GetRef(WGPU_BIND_GROUP_ENTRY_INIT);
			Entry.binding = Binding.Key;
			Entry.buffer = Bound.Buffer->Handle;
			Entry.offset = Bound.Offset;
			Entry.size = Bound.Size;
		}

		WGPUBindGroupLayout Layout = wgpuComputePipelineGetBindGroupLayout(Kernel.Pipeline, Group.Key);
		WGPUBindGroupDescriptor Desc = {};
		Desc.layout = Layout;
		Desc.entryCount = Entries.Num();
		Desc.entries = Entries.GetData();
		Kernel.BindGroups.Add(Group.Key, wgpuDeviceCreateBindGroup(WorkerDevice->GetDevice(), &Desc));
		wgpuBindGroupLayoutRelease(Layout);
	}
	Kernel.bBindGroupsDirty = false;
	return true;
}

void FWebGPUContext::EncodeDispatch(const TSharedRef<FWebGPUKernel>& KernelRef, uint32 X, uint32 Y, uint32 Z)
{
	if (!HasDevice())
	{
		return;
	}

	FWebGPUKernel& Kernel = *KernelRef;
	if (!Kernel.Pipeline)
	{
		StatSkippedDispatches++;
		if (!Kernel.bWarnedNotReady)
		{
			UE_LOG(LogWebGPU, Warning, TEXT("Dispatch of '%s' skipped: no compiled pipeline yet"), *Kernel.Label);
			Kernel.bWarnedNotReady = true;
		}
		return;
	}

	// Open the batch (and its error scope) first so bind group errors are attributed to it
	WGPUCommandEncoder BatchEncoder = GetEncoder();
	if (!UpdateBindGroups(Kernel))
	{
		StatSkippedDispatches++;
		return;
	}

	FTCHARToUTF8 Label(*Kernel.Label);
	WGPUComputePassDescriptor PassDesc = {};
	PassDesc.label = ToView(Label);

	WGPUPassTimestampWrites Timestamps = {};
	if (Kernel.bProfile && TimestampQueries && BatchQueryCount + 2 <= MaxTimestampQueries)
	{
		Timestamps.querySet = TimestampQueries;
		Timestamps.beginningOfPassWriteIndex = BatchQueryCount;
		Timestamps.endOfPassWriteIndex = BatchQueryCount + 1;
		BatchQueryCount += 2;
		BatchProfiled.Add(KernelRef);
		PassDesc.timestampWrites = &Timestamps;
	}

	WGPUComputePassEncoder Pass = wgpuCommandEncoderBeginComputePass(BatchEncoder, &PassDesc);
	wgpuComputePassEncoderSetPipeline(Pass, Kernel.Pipeline);
	for (const TPair<uint32, WGPUBindGroup>& Group : Kernel.BindGroups)
	{
		wgpuComputePassEncoderSetBindGroup(Pass, Group.Key, Group.Value, 0, nullptr);
	}
	wgpuComputePassEncoderDispatchWorkgroups(Pass, X, Y, Z);
	wgpuComputePassEncoderEnd(Pass);
	wgpuComputePassEncoderRelease(Pass);

	BatchLabels.AddUnique(Kernel.Label);
	StatDispatches++;
}

void FWebGPUContext::ReadBufferOnWorker(FWebGPUBuffer& Buffer, uint64 Offset, uint64 Size, FWebGPUReadCallback&& Callback, EWebGPUCallbackThread CallbackThread)
{
	if (!HasDevice() || !Buffer.Handle)
	{
		Deliver(CallbackThread, [Callback = MoveTemp(Callback)]() mutable
		{
			if (Callback)
			{
				Callback(false, TArray<uint8>());
			}
		});
		return;
	}

	uint64 Capacity = 0;
	WGPUBuffer Staging = AcquireStaging(Size, Capacity);
	wgpuCommandEncoderCopyBufferToBuffer(GetEncoder(), Buffer.Handle, Offset, Staging, 0, Size);
	SubmitBatch();

	FReadRequest* Request = new FReadRequest();
	Request->Context = this;
	Request->Staging = Staging;
	Request->Capacity = Capacity;
	Request->Size = Size;
	Request->MappedSize = Size;
	Request->Generation = DeviceGeneration;
	Request->Callback = MoveTemp(Callback);
	Request->Thread = CallbackThread;

	WGPUBufferMapCallbackInfo Info = {};
	Info.mode = WGPUCallbackMode_AllowProcessEvents;
	Info.callback = &FWebGPUContext::OnReadMapped;
	Info.userdata1 = Request;
	PendingCallbacks++;
	StatReadbacks++;
	wgpuBufferMapAsync(Staging, WGPUMapMode_Read, 0, (size_t)Size, Info);
}

void FWebGPUContext::OnReadMapped(WGPUMapAsyncStatus Status, WGPUStringView Message, void* UserData1, void*)
{
	TUniquePtr<FReadRequest> Request(static_cast<FReadRequest*>(UserData1));
	FWebGPUContext& Self = *Request->Context;
	Self.PendingCallbacks--;

	TArray<uint8> Data;
	bool bSuccess = Status == WGPUMapAsyncStatus_Success;
	if (bSuccess)
	{
		const void* Mapped = wgpuBufferGetConstMappedRange(Request->Staging, 0, (size_t)Request->MappedSize);
		bSuccess = Mapped != nullptr;
		if (bSuccess)
		{
			Data.SetNumUninitialized((int32)Request->Size);
			FMemory::Memcpy(Data.GetData(), Mapped, Request->Size);
			Self.StatBytesRead += Request->Size;
		}
		wgpuBufferUnmap(Request->Staging);
		Self.ReleaseStaging(Request->Staging, Request->Capacity, Request->Generation);
	}
	else
	{
		UE_LOG(LogWebGPU, Verbose, TEXT("ReadBuffer map failed (%d): %s"), (int32)Status, *FWebGPUDevice::ToFString(Message));
		wgpuBufferRelease(Request->Staging);
	}

	Deliver(Request->Thread, [Callback = MoveTemp(Request->Callback), bSuccess, Data = MoveTemp(Data)]() mutable
	{
		if (Callback)
		{
			Callback(bSuccess, MoveTemp(Data));
		}
	});
}

WGPUCommandEncoder FWebGPUContext::GetEncoder()
{
	if (!Encoder)
	{
		// One validation scope per batch: an invalid command anywhere rejects the whole command buffer
		wgpuDevicePushErrorScope(WorkerDevice->GetDevice(), WGPUErrorFilter_Validation);
		Encoder = wgpuDeviceCreateCommandEncoder(WorkerDevice->GetDevice(), nullptr);
	}
	return Encoder;
}

void FWebGPUContext::SubmitBatch()
{
	if (!Encoder)
	{
		return;
	}

	WGPUDevice Dev = WorkerDevice->GetDevice();

	FTimestampRequest* TimestampRequest = nullptr;
	if (BatchQueryCount > 0)
	{
		const uint64 QueryBytes = (uint64)BatchQueryCount * sizeof(uint64);
		wgpuCommandEncoderResolveQuerySet(Encoder, TimestampQueries, 0, BatchQueryCount, TimestampResolve, 0);

		TimestampRequest = new FTimestampRequest();
		TimestampRequest->Context = this;
		TimestampRequest->Staging = AcquireStaging(QueryBytes, TimestampRequest->Capacity);
		TimestampRequest->Count = BatchQueryCount;
		TimestampRequest->Generation = DeviceGeneration;
		TimestampRequest->Kernels = MoveTemp(BatchProfiled);
		wgpuCommandEncoderCopyBufferToBuffer(Encoder, TimestampResolve, 0, TimestampRequest->Staging, 0, QueryBytes);
	}

	WGPUCommandBuffer CommandBuffer = wgpuCommandEncoderFinish(Encoder, nullptr);
	wgpuQueueSubmit(WorkerDevice->GetQueue(), 1, &CommandBuffer);
	wgpuCommandBufferRelease(CommandBuffer);
	wgpuCommandEncoderRelease(Encoder);
	Encoder = nullptr;
	StatSubmits++;
	BusyPollUntil = FPlatformTime::Seconds() + Options.BusyPollMilliseconds / 1000.0;

	FBatchScopeRequest* ScopeRequest = new FBatchScopeRequest{ this, FString::Join(BatchLabels, TEXT(", ")) };
	WGPUPopErrorScopeCallbackInfo ScopeInfo = {};
	ScopeInfo.mode = WGPUCallbackMode_AllowProcessEvents;
	ScopeInfo.callback = &FWebGPUContext::OnBatchScopePopped;
	ScopeInfo.userdata1 = ScopeRequest;
	PendingCallbacks++;
	wgpuDevicePopErrorScope(Dev, ScopeInfo);

	if (TimestampRequest)
	{
		WGPUBufferMapCallbackInfo MapInfo = {};
		MapInfo.mode = WGPUCallbackMode_AllowProcessEvents;
		MapInfo.callback = &FWebGPUContext::OnTimestampsMapped;
		MapInfo.userdata1 = TimestampRequest;
		PendingCallbacks++;
		wgpuBufferMapAsync(TimestampRequest->Staging, WGPUMapMode_Read, 0, (size_t)TimestampRequest->Count * sizeof(uint64), MapInfo);
	}

	BatchLabels.Reset();
	BatchQueryCount = 0;
	BatchProfiled.Reset();
}

void FWebGPUContext::OnBatchScopePopped(WGPUPopErrorScopeStatus Status, WGPUErrorType Type, WGPUStringView Message, void* UserData1, void*)
{
	TUniquePtr<FBatchScopeRequest> Request(static_cast<FBatchScopeRequest*>(UserData1));
	FWebGPUContext& Self = *Request->Context;
	Self.PendingCallbacks--;

	if (Status == WGPUPopErrorScopeStatus_Success && Type != WGPUErrorType_NoError)
	{
		Self.StatFailedSubmits++;
		UE_LOG(LogWebGPU, Error, TEXT("WebGPU batch rejected (kernels: %s): %s"),
			Request->Labels.IsEmpty() ? TEXT("none") : *Request->Labels, *FWebGPUDevice::ToFString(Message));
	}
}

void FWebGPUContext::OnTimestampsMapped(WGPUMapAsyncStatus Status, WGPUStringView, void* UserData1, void*)
{
	TUniquePtr<FTimestampRequest> Request(static_cast<FTimestampRequest*>(UserData1));
	FWebGPUContext& Self = *Request->Context;
	Self.PendingCallbacks--;

	if (Status != WGPUMapAsyncStatus_Success)
	{
		wgpuBufferRelease(Request->Staging);
		return;
	}

	const uint64* Ticks = static_cast<const uint64*>(wgpuBufferGetConstMappedRange(Request->Staging, 0, (size_t)Request->Count * sizeof(uint64)));
	if (Ticks)
	{
		for (int32 Index = 0; Index < Request->Kernels.Num(); Index++)
		{
			TSharedPtr<FWebGPUKernel> Kernel = Request->Kernels[Index].Pin();
			if (!Kernel)
			{
				continue;
			}
			// WebGPU timestamps are nanoseconds
			const uint64 Begin = Ticks[Index * 2];
			const uint64 End = Ticks[Index * 2 + 1];
			const double Ms = End > Begin ? (double)(End - Begin) / 1.0e6 : 0.0;
			const double Average = Kernel->AverageGpuTimeMs;
			Kernel->LastGpuTimeMs = Ms;
			Kernel->AverageGpuTimeMs = Average > 0.0 ? Average * 0.9 + Ms * 0.1 : Ms;
		}
	}
	wgpuBufferUnmap(Request->Staging);
	Self.ReleaseStaging(Request->Staging, Request->Capacity, Request->Generation);
}

void FWebGPUContext::OnFlushDone(WGPUQueueWorkDoneStatus Status, WGPUStringView, void* UserData1, void*)
{
	TUniquePtr<FFlushRequest> Request(static_cast<FFlushRequest*>(UserData1));
	Request->Context->PendingCallbacks--;
	Request->Signal->bOk = Status == WGPUQueueWorkDoneStatus_Success;
	Request->Signal->Event->Trigger();
}

WGPUBuffer FWebGPUContext::AcquireStaging(uint64 Size, uint64& OutCapacity)
{
	const uint64 Capacity = FMath::RoundUpToPowerOfTwo64(FMath::Max<uint64>(Size, MinStagingSize));
	OutCapacity = Capacity;

	if (TArray<WGPUBuffer>* Free = StagingPool.Find(Capacity))
	{
		if (Free->Num() > 0)
		{
			return Free->Pop(EAllowShrinking::No);
		}
	}

	WGPUBufferDescriptor Desc = WGPU_BUFFER_DESCRIPTOR_INIT;
	Desc.label = ToView("WebGPUCompute readback");
	Desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
	Desc.size = Capacity;
	return wgpuDeviceCreateBuffer(WorkerDevice->GetDevice(), &Desc);
}

void FWebGPUContext::ReleaseStaging(WGPUBuffer Staging, uint64 Capacity, uint32 Generation)
{
	if (bStopping || bReleasingDevice || Generation != DeviceGeneration)
	{
		wgpuBufferRelease(Staging);
		return;
	}

	TArray<WGPUBuffer>& Free = StagingPool.FindOrAdd(Capacity);
	if (Free.Num() < MaxPooledStagingPerSize)
	{
		Free.Add(Staging);
	}
	else
	{
		wgpuBufferRelease(Staging);
	}
}

void FWebGPUContext::CreateDeviceObjects()
{
	if (!WorkerDevice || !WorkerDevice->HasFeature(WGPUFeatureName_TimestampQuery))
	{
		return;
	}

	WGPUQuerySetDescriptor QueryDesc = {};
	QueryDesc.label = ToView("WebGPUCompute timestamps");
	QueryDesc.type = WGPUQueryType_Timestamp;
	QueryDesc.count = MaxTimestampQueries;
	TimestampQueries = wgpuDeviceCreateQuerySet(WorkerDevice->GetDevice(), &QueryDesc);

	WGPUBufferDescriptor ResolveDesc = WGPU_BUFFER_DESCRIPTOR_INIT;
	ResolveDesc.label = ToView("WebGPUCompute timestamp resolve");
	ResolveDesc.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc;
	ResolveDesc.size = MaxTimestampQueries * sizeof(uint64);
	TimestampResolve = wgpuDeviceCreateBuffer(WorkerDevice->GetDevice(), &ResolveDesc);
}

void FWebGPUContext::ReleaseDeviceObjects()
{
	if (Encoder)
	{
		// Abandons the batch; its error scope belongs to the device being released
		wgpuCommandEncoderRelease(Encoder);
		Encoder = nullptr;
	}
	BatchLabels.Reset();
	BatchQueryCount = 0;
	BatchProfiled.Reset();

	for (TPair<uint64, TArray<WGPUBuffer>>& Pair : StagingPool)
	{
		for (WGPUBuffer Staging : Pair.Value)
		{
			wgpuBufferRelease(Staging);
		}
	}
	StagingPool.Reset();

	if (TimestampQueries)
	{
		wgpuQuerySetRelease(TimestampQueries);
		TimestampQueries = nullptr;
	}
	if (TimestampResolve)
	{
		wgpuBufferRelease(TimestampResolve);
		TimestampResolve = nullptr;
	}
}

void FWebGPUContext::Recover()
{
	bDeviceLostSignal = false;

	if (WorkerDevice)
	{
		UE_LOG(LogWebGPU, Warning, TEXT("WebGPU device lost; recreating it and its resources"));

		ReleaseDeviceObjects();
		for (const TWeakPtr<FWebGPUBuffer>& Weak : LiveBuffers)
		{
			if (TSharedPtr<FWebGPUBuffer> Buffer = Weak.Pin())
			{
				if (Buffer->Handle)
				{
					wgpuBufferRelease(Buffer->Handle);
					Buffer->Handle = nullptr;
				}
			}
		}
		for (const TWeakPtr<FWebGPUKernel>& Weak : LiveKernels)
		{
			if (TSharedPtr<FWebGPUKernel> Kernel = Weak.Pin())
			{
				Kernel->ReleaseGPU();
			}
		}

		// Fails whatever is still pending on the old device
		bReleasingDevice = true;
		WorkerDevice->SetLostHandler(nullptr);
		WorkerDevice->Shutdown();
		bReleasingDevice = false;
		ReleaseDeviceObjects();
		{
			FScopeLock Lock(&DeviceLock);
			Device.Reset();
		}
		WorkerDevice.Reset();
		PendingCallbacks = 0;
	}

	TSharedPtr<FWebGPUDevice> NewDevice = MakeShared<FWebGPUDevice>();
	NewDevice->SetLostHandler([this]()
	{
		bDeviceLostSignal = true;
		Wake();
	});
	if (!NewDevice->Initialize(Options))
	{
		UE_LOG(LogWebGPU, Error, TEXT("Could not recreate the WebGPU device; retrying in a second"));
		NextRecoveryTime = FPlatformTime::Seconds() + 1.0;
		return;
	}

	{
		FScopeLock Lock(&DeviceLock);
		Device = NewDevice;
	}
	WorkerDevice = NewDevice;
	const uint32 Generation = ++DeviceGeneration;
	CreateDeviceObjects();

	LiveBuffers.RemoveAll([](const TWeakPtr<FWebGPUBuffer>& Weak) { return !Weak.IsValid(); });
	for (const TWeakPtr<FWebGPUBuffer>& Weak : LiveBuffers)
	{
		if (TSharedPtr<FWebGPUBuffer> Buffer = Weak.Pin())
		{
			CreateBufferHandle(*Buffer);
		}
	}

	LiveKernels.RemoveAll([](const TWeakPtr<FWebGPUKernel>& Weak) { return !Weak.IsValid(); });
	for (const TWeakPtr<FWebGPUKernel>& Weak : LiveKernels)
	{
		if (TSharedPtr<FWebGPUKernel> Kernel = Weak.Pin())
		{
			CompileKernel(Kernel.ToSharedRef(), nullptr, Kernel->ListenerThread);
		}
	}

	UE_LOG(LogWebGPU, Log, TEXT("WebGPU device recreated (generation %u): %d buffers, %d kernels restored (buffer contents are zeroed)"),
		Generation, LiveBuffers.Num(), LiveKernels.Num());

	TWeakPtr<FWebGPUContext> WeakSelf = AsShared();
	Deliver(EWebGPUCallbackThread::GameThread, [WeakSelf, Generation]()
	{
		if (TSharedPtr<FWebGPUContext> Self = WeakSelf.Pin())
		{
			Self->OnDeviceRecovered.Broadcast(Generation);
		}
	});
}

void FWebGPUContext::ReleaseAllResources()
{
	// Drop queued commands without running them (their captures release here, on the worker)
	for (;;)
	{
		TUniqueFunction<void()> Command;
		if (!Commands.Dequeue(Command))
		{
			break;
		}
	}

	ReleaseDeviceObjects();
	for (const TWeakPtr<FWebGPUBuffer>& Weak : LiveBuffers)
	{
		if (TSharedPtr<FWebGPUBuffer> Buffer = Weak.Pin())
		{
			if (Buffer->Handle)
			{
				wgpuBufferRelease(Buffer->Handle);
				Buffer->Handle = nullptr;
			}
		}
	}
	for (const TWeakPtr<FWebGPUKernel>& Weak : LiveKernels)
	{
		if (TSharedPtr<FWebGPUKernel> Kernel = Weak.Pin())
		{
			Kernel->ReleaseGPU();
		}
	}
	LiveBuffers.Reset();
	LiveKernels.Reset();

	if (WorkerDevice)
	{
		bReleasingDevice = true;
		WorkerDevice->SetLostHandler(nullptr);
		WorkerDevice->Shutdown();
		bReleasingDevice = false;
		ReleaseDeviceObjects();
	}
	{
		FScopeLock Lock(&DeviceLock);
		Device.Reset();
	}
	WorkerDevice.Reset();
	PendingCallbacks = 0;
}
