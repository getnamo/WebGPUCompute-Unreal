// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUContext.h"
#include "WebGPUCompute.h"
#include "WebGPUStreams.h"
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

	/** Compile messages refer to preamble + source; report source lines, and flag errors inside the preamble. */
	void RemapPreambleLines(TArray<FWebGPUCompileMessage>& Messages, int32 PreambleLines)
	{
		if (PreambleLines <= 0)
		{
			return;
		}
		for (FWebGPUCompileMessage& Message : Messages)
		{
			if (Message.Line > (uint64)PreambleLines)
			{
				Message.Line -= PreambleLines;
			}
			else if (Message.Line > 0)
			{
				Message.Message = FString::Printf(TEXT("[generated struct preamble, line %llu] %s"), Message.Line, *Message.Message);
				Message.Line = 0;
			}
		}
	}
	constexpr int32 MaxPooledStagingPerSize = 4;

	/** Writes share upload pages of this size; bigger writes get a page of their own (pooled by size). */
	constexpr uint64 UploadPageSize = 1 << 20;
	constexpr int32 MaxPooledUploadPagesPerSize = 4;

	/** A steady stream of commands still gets submitted in batches of at most this many. */
	constexpr int32 MaxBatchCommands = 512;

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

struct FWebGPUContext::FMapRequest
{
	FWebGPUContext* Context = nullptr;
	uint32 Generation = 0;
	TUniqueFunction<void(bool bSuccess)> Handler;
};


struct FWebGPUContext::FCompileRequest
{
	FWebGPUContext* Context = nullptr;
	TWeakPtr<FWebGPUKernel> Kernel;
	uint32 Serial = 0;
	uint32 Generation = 0;
	TArray<FWebGPUCompileMessage> Messages;
	TSharedPtr<const FWebGPUShaderReflection> Reflection;
	FWebGPUCompileCallback Callback;
	EWebGPUCallbackThread Thread = EWebGPUCallbackThread::GameThread;
};

struct FWebGPUContext::FBatchScopeRequest
{
	FWebGPUContext* Context = nullptr;
	uint32 Generation = 0;
	FString Labels;
};

struct FWebGPUContext::FFlushRequest
{
	FWebGPUContext* Context = nullptr;
	uint32 Generation = 0;
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
	TSharedPtr<FWebGPUContext> Ctx = Context.Pin();
	if (!Ctx)
	{
		wgpuBufferRelease(ToRelease);
	}
	else if (Ctx->IsInWorkerThread())
	{
		Ctx->DestroyBufferOnWorker(ToRelease);
	}
	else
	{
		FWebGPUContext* CtxPtr = Ctx.Get();
		Ctx->Enqueue([CtxPtr, ToRelease]()
		{
			CtxPtr->DestroyBufferOnWorker(ToRelease);
		});
	}
}

// FWebGPUKernel

TSharedPtr<FWebGPUContext> FWebGPUKernel::GetContext() const
{
	return Context.Pin();
}

TArray<FWebGPUCompileMessage> FWebGPUKernel::GetMessages() const
{
	FScopeLock Lock(&StateLock);
	return Messages;
}

void FWebGPUKernel::SetMessages(const TArray<FWebGPUCompileMessage>& InMessages)
{
	FScopeLock Lock(&StateLock);
	Messages = InMessages;
}

TSharedPtr<const FWebGPUShaderReflection> FWebGPUKernel::GetReflection() const
{
	FScopeLock Lock(&StateLock);
	return Reflection;
}

bool FWebGPUKernel::GetEntryPointInfo(FWebGPUEntryPointInfo& OutInfo) const
{
	FScopeLock Lock(&StateLock);
	const FWebGPUEntryPointInfo* Info = Reflection ? Reflection->FindEntryPoint(EntryPoint) : nullptr;
	if (Info)
	{
		OutInfo = *Info;
	}
	return Info != nullptr;
}

FString FWebGPUKernel::GetCompiledSource() const
{
	FScopeLock Lock(&StateLock);
	return CompiledSource;
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
			if (Encoder && ++BatchCommands >= MaxBatchCommands)
			{
				SubmitBatch();
			}
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

		if (bHasRetiredDevices)
		{
			ShutdownRetiredDevices(false);
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
	FWebGPUKernelDesc Desc;
	Desc.Source = Source;
	Desc.EntryPoint = EntryPoint;
	Desc.Label = Label;
	return CreateKernel(Desc, MoveTemp(OnCompiled), CallbackThread);
}

TSharedRef<FWebGPUKernel> FWebGPUContext::CreateKernel(const FWebGPUKernelDesc& Desc, FWebGPUCompileCallback OnCompiled, EWebGPUCallbackThread CallbackThread)
{
	TSharedRef<FWebGPUKernel> Kernel = MakeShared<FWebGPUKernel>();
	Kernel->Context = AsShared();
	Kernel->EntryPoint = Desc.EntryPoint;
	Kernel->Label = Desc.Label.IsEmpty() ? Desc.EntryPoint : Desc.Label;

	// UStructs become WGSL declarations ahead of the source (built here: it reads UObject reflection)
	if (Desc.Structs.Num() > 0)
	{
		TArray<TSharedPtr<const FWebGPUStructLayout>> Layouts;
		for (const UScriptStruct* Struct : Desc.Structs)
		{
			FString Error;
			if (TSharedPtr<const FWebGPUStructLayout> Layout = FWebGPUStructLayout::Get(Struct, EWebGPUAddressSpace::Storage, Error))
			{
				Layouts.Add(Layout);
			}
			else
			{
				UE_LOG(LogWebGPU, Error, TEXT("Kernel '%s': struct %s left out of the preamble: %s"), *Kernel->Label, Struct ? *Struct->GetName() : TEXT("null"), *Error);
			}
		}
		Kernel->Preamble = FWebGPUStructLayout::GenerateWGSL(Layouts);
		for (TCHAR Char : Kernel->Preamble)
		{
			Kernel->PreambleLines += Char == TEXT('\n') ? 1 : 0;
		}
	}

	Enqueue([this, Kernel, Source = Desc.Source, OnCompiled = MoveTemp(OnCompiled), CallbackThread]() mutable
	{
		LiveKernels.Add(Kernel);
		Kernel->Source = Source;
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
		UploadOnWorker(*Buffer, Offset, Data.GetData(), (uint64)Data.Num());
	});
}

void FWebGPUContext::SetBuffer(const TSharedRef<FWebGPUKernel>& Kernel, uint32 Group, uint32 Binding, const TSharedPtr<FWebGPUBuffer>& Buffer, uint64 Offset, uint64 Size)
{
	Enqueue([Kernel, Group, Binding, Buffer, Offset, Size]()
	{
		if (Buffer)
		{
			FWebGPUKernel::FBinding& Entry = Kernel->Bindings.FindOrAdd(Group).FindOrAdd(Binding);
			Entry = FWebGPUKernel::FBinding();
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

void FWebGPUContext::SetBufferByName(const TSharedRef<FWebGPUKernel>& Kernel, const FString& Name, const TSharedPtr<FWebGPUBuffer>& Buffer, uint64 Offset, uint64 Size)
{
	Enqueue([Kernel, Name, Buffer, Offset, Size]()
	{
		if (Buffer)
		{
			FWebGPUKernel::FBinding& Entry = Kernel->NamedBindings.FindOrAdd(Name);
			Entry = FWebGPUKernel::FBinding();
			Entry.Buffer = Buffer;
			Entry.Offset = Offset;
			Entry.Size = Size;
		}
		else
		{
			Kernel->NamedBindings.Remove(Name);
		}
		Kernel->bBindGroupsDirty = true;
	});
}

void FWebGPUContext::DispatchForCount(const TSharedRef<FWebGPUKernel>& Kernel, uint32 X, uint32 Y, uint32 Z)
{
	Enqueue([this, Kernel, X, Y, Z]()
	{
		if (X == 0 || Y == 0 || Z == 0)
		{
			return;
		}
		if (!Kernel->EntryInfo)
		{
			// Not compiled yet: counted and warned as a skipped dispatch
			EncodeDispatch(Kernel, 1, 1, 1);
			return;
		}
		const FIntVector WorkgroupSize = Kernel->EntryInfo->WorkgroupSize;
		if (WorkgroupSize.X <= 0 || WorkgroupSize.Y <= 0 || WorkgroupSize.Z <= 0)
		{
			StatSkippedDispatches++;
			WarnOnce(*Kernel, FString::Printf(TEXT("Kernel '%s': DispatchForCount needs a constant @workgroup_size; use Dispatch"), *Kernel->Label));
			return;
		}
		EncodeDispatch(Kernel, FMath::DivideAndRoundUp(X, (uint32)WorkgroupSize.X), FMath::DivideAndRoundUp(Y, (uint32)WorkgroupSize.Y), FMath::DivideAndRoundUp(Z, (uint32)WorkgroupSize.Z));
	});
}

TSharedPtr<FWebGPUBuffer> FWebGPUContext::CreateStructBuffer(const UScriptStruct* Struct, int32 Count, EWebGPUBufferUsage Usage, const FString& Label)
{
	const bool bUniformOnly = EnumHasAnyFlags(Usage, EWebGPUBufferUsage::Uniform) && !EnumHasAnyFlags(Usage, EWebGPUBufferUsage::Storage);
	FString Error;
	TSharedPtr<const FWebGPUStructLayout> Layout = FWebGPUStructLayout::Get(Struct, bUniformOnly ? EWebGPUAddressSpace::Uniform : EWebGPUAddressSpace::Storage, Error);
	if (!Layout || Count <= 0)
	{
		UE_LOG(LogWebGPU, Error, TEXT("CreateStructBuffer(%s, %d): %s"), Struct ? *Struct->GetName() : TEXT("null"), Count, Layout ? TEXT("count must be positive") : *Error);
		return nullptr;
	}

	TSharedRef<FWebGPUBuffer> Buffer = MakeShared<FWebGPUBuffer>();
	Buffer->Context = AsShared();
	Buffer->Size = (uint64)Count * Layout->GetStride();
	Buffer->Usage = Usage;
	Buffer->Label = Label.IsEmpty() ? Struct->GetName() : Label;
	Buffer->ElementLayout = Layout;

	Enqueue([this, Buffer]()
	{
		LiveBuffers.Add(Buffer);
		CreateBufferHandle(*Buffer);
	});
	return Buffer;
}

bool FWebGPUContext::WriteStructs(const TSharedRef<FWebGPUBuffer>& Buffer, const UScriptStruct* Struct, const void* Elements, int32 Count, int32 FirstElement)
{
	const EWebGPUAddressSpace Space = Buffer->ElementLayout ? Buffer->ElementLayout->GetAddressSpace() : EWebGPUAddressSpace::Storage;
	FString Error;
	TSharedPtr<const FWebGPUStructLayout> Layout = FWebGPUStructLayout::Get(Struct, Space, Error);
	if (!Layout)
	{
		UE_LOG(LogWebGPU, Error, TEXT("WriteStructs(%s): %s"), *Buffer->GetLabel(), *Error);
		return false;
	}
	if (Count <= 0)
	{
		return true;
	}
	if (Buffer->ElementLayout && Buffer->ElementLayout->GetStructName() != Layout->GetStructName())
	{
		UE_LOG(LogWebGPU, Warning, TEXT("WriteStructs(%s): writing %s into a buffer of %s"), *Buffer->GetLabel(), *Layout->GetStructName(), *Buffer->ElementLayout->GetStructName());
	}

	const uint64 Offset = (uint64)FMath::Max(FirstElement, 0) * Layout->GetStride();
	const uint64 Size = (uint64)Count * Layout->GetStride();
	if (Offset + Size > Buffer->GetSize())
	{
		UE_LOG(LogWebGPU, Error, TEXT("WriteStructs(%s): elements %d..%d don't fit in %llu bytes (%u per element)"),
			*Buffer->GetLabel(), FirstElement, FirstElement + Count - 1, Buffer->GetSize(), Layout->GetStride());
		return false;
	}
	WriteBuffer(Buffer, Offset, Layout->Pack(Elements, Count));
	return true;
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

		FFlushRequest* Request = new FFlushRequest{ this, DeviceGeneration, Signal };
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

	const FString FullSource = Kernel->Preamble + Kernel->Source;
	{
		FScopeLock Lock(&Kernel->StateLock);
		Kernel->CompiledSource = FullSource;
	}

	TArray<FWebGPUCompileMessage> Messages;
	WGPUShaderModule Module = WorkerDevice->CreateShaderModule(FullSource, Kernel->Label, Messages);
	RemapPreambleLines(Messages, Kernel->PreambleLines);
	if (!Module)
	{
		FinishCompile(*Kernel, false, Messages, MoveTemp(OnCompiled), CallbackThread);
		return;
	}

	// Tint reflection (bindings, workgroup size, struct layouts) of this exact source
	TSharedPtr<FWebGPUShaderReflection> Reflection = MakeShared<FWebGPUShaderReflection>();
	TArray<FWebGPUCompileMessage> ReflectionMessages;
	if (!FWebGPUShaderReflection::Reflect(FullSource, *Reflection, ReflectionMessages))
	{
		Reflection.Reset();
	}
	else
	{
		const FWebGPUEntryPointInfo* Entry = Reflection->FindEntryPoint(Kernel->EntryPoint);
		if (!Entry || !Entry->bCompute)
		{
			TArray<FString> ComputeEntries;
			for (const FWebGPUEntryPointInfo& Info : Reflection->EntryPoints)
			{
				if (Info.bCompute)
				{
					ComputeEntries.Add(Info.Name);
				}
			}
			FWebGPUCompileMessage& Error = Messages.AddDefaulted_GetRef();
			Error.bIsError = true;
			Error.Message = FString::Printf(TEXT("No @compute entry point '%s' (compute entry points: %s)"), *Kernel->EntryPoint,
				ComputeEntries.Num() ? *FString::Join(ComputeEntries, TEXT(", ")) : TEXT("none"));
			wgpuShaderModuleRelease(Module);
			FinishCompile(*Kernel, false, Messages, MoveTemp(OnCompiled), CallbackThread);
			return;
		}
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
	Request->Reflection = Reflection;
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
	Self.CallbackDone(Request->Generation);

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
			Kernel->WorkerReflection = Request->Reflection;
			Kernel->EntryInfo = Request->Reflection ? Request->Reflection->FindEntryPoint(Kernel->EntryPoint) : nullptr;
			Kernel->Warned.Reset();
			{
				FScopeLock Lock(&Kernel->StateLock);
				Kernel->Reflection = Request->Reflection;
			}
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

void FWebGPUContext::WarnOnce(FWebGPUKernel& Kernel, const FString& Message)
{
	if (!Kernel.Warned.Contains(Message))
	{
		Kernel.Warned.Add(Message);
		UE_LOG(LogWebGPU, Warning, TEXT("%s"), *Message);
	}
}

bool FWebGPUContext::UpdateBindGroups(FWebGPUKernel& Kernel)
{
	if (!Kernel.bBindGroupsDirty)
	{
		return Kernel.bBindingsValid;
	}
	Kernel.ReleaseBindGroups();
	Kernel.bBindGroupsDirty = false;
	Kernel.bBindingsValid = false;

	// Explicit (group, binding) bindings, then names resolved through the reflection
	auto Key = [](uint32 Group, uint32 Binding) { return ((uint64)Group << 32) | Binding; };
	TMap<uint64, const FWebGPUKernel::FBinding*> Resolved;
	for (const TPair<uint32, TSortedMap<uint32, FWebGPUKernel::FBinding>>& Group : Kernel.Bindings)
	{
		for (const TPair<uint32, FWebGPUKernel::FBinding>& Binding : Group.Value)
		{
			Resolved.Add(Key(Group.Key, Binding.Key), &Binding.Value);
		}
	}

	const FWebGPUEntryPointInfo* Entry = Kernel.EntryInfo;
	for (const TPair<FString, FWebGPUKernel::FBinding>& Named : Kernel.NamedBindings)
	{
		if (!Entry)
		{
			WarnOnce(Kernel, FString::Printf(TEXT("Kernel '%s': can't resolve binding '%s' without shader reflection"), *Kernel.Label, *Named.Key));
			continue;
		}
		if (const FWebGPUBindingInfo* Info = Entry->FindBinding(Named.Key))
		{
			Resolved.Add(Key(Info->Group, Info->Binding), &Named.Value);
		}
		else
		{
			TArray<FString> Names;
			for (const FWebGPUBindingInfo& Binding : Entry->Bindings)
			{
				Names.Add(Binding.Name);
			}
			WarnOnce(Kernel, FString::Printf(TEXT("Kernel '%s': entry point '%s' uses no binding named '%s' (it uses: %s)"),
				*Kernel.Label, *Entry->Name, *Named.Key, Names.Num() ? *FString::Join(Names, TEXT(", ")) : TEXT("none")));
		}
	}

	TSortedMap<uint32, TArray<WGPUBindGroupEntry>> Groups;
	Kernel.SharedTextures.Reset();
	if (Entry)
	{
		// Exactly the resources the entry point uses: extra bindings are ignored, missing or invalid ones skip the dispatch
		TArray<FString> Problems;
		for (const FWebGPUBindingInfo& Info : Entry->Bindings)
		{
			const FWebGPUKernel::FBinding* const* Found = Resolved.Find(Key(Info.Group, Info.Binding));
			const FWebGPUKernel::FBinding* Bound = Found ? *Found : nullptr;
			WGPUBindGroupEntry BindEntry = WGPU_BIND_GROUP_ENTRY_INIT;
			BindEntry.binding = (uint32)Info.Binding;

			if (Info.Type == EWebGPUBindingType::Sampler)
			{
				const EWebGPUSampler Which = Bound && Bound->Sampler.IsSet() ? Bound->Sampler.GetValue() : EWebGPUSampler::LinearClamp;
				BindEntry.sampler = Samplers[(int32)Which];
			}
			else if (Info.IsTexture())
			{
				if (!Bound || !Bound->Texture)
				{
					Problems.Add(FString::Printf(TEXT("'%s' (@group(%d) @binding(%d)) is not bound to a texture"), *Info.Name, Info.Group, Info.Binding));
					continue;
				}
				const FWebGPUTexture& Texture = *Bound->Texture;
				if (Info.TextureDimension != TEXT("2d"))
				{
					Problems.Add(FString::Printf(TEXT("'%s' is a %s texture; only 2D textures can be bound"), *Info.Name, *Info.TextureDimension));
					continue;
				}
				if (!Texture.View)
				{
					Problems.Add(FString::Printf(TEXT("'%s': texture '%s' has no GPU memory"), *Info.Name, *Texture.Label));
					continue;
				}
				if (Info.Type == EWebGPUBindingType::StorageTexture)
				{
					if (!EnumHasAnyFlags(Texture.Usage, EWebGPUTextureUsage::Storage))
					{
						Problems.Add(FString::Printf(TEXT("'%s' is a storage texture but texture '%s' was not created with the Storage usage"), *Info.Name, *Texture.Label));
					}
					if (Info.TexelFormat != FWebGPUTexture::WgslFormat(Texture.Format))
					{
						Problems.Add(FString::Printf(TEXT("'%s' is texture_storage_2d<%s> but texture '%s' is %s"), *Info.Name, *Info.TexelFormat, *Texture.Label, FWebGPUTexture::WgslFormat(Texture.Format)));
					}
				}
				else if (!EnumHasAnyFlags(Texture.Usage, EWebGPUTextureUsage::Sampled))
				{
					Problems.Add(FString::Printf(TEXT("'%s' is a texture_2d but texture '%s' was not created with the Sampled usage"), *Info.Name, *Texture.Label));
				}
				if (Texture.Shared)
				{
					Kernel.SharedTextures.Add(Bound->Texture);
				}
				BindEntry.textureView = Texture.View;
			}
			else if (Info.IsBuffer())
			{
				if (!Bound || !Bound->Buffer)
				{
					Problems.Add(FString::Printf(TEXT("'%s' (@group(%d) @binding(%d)) is not bound"), *Info.Name, Info.Group, Info.Binding));
					continue;
				}
				const FWebGPUBuffer& Buffer = *Bound->Buffer;
				if (!Buffer.Handle)
				{
					Problems.Add(FString::Printf(TEXT("'%s': buffer '%s' has no GPU memory"), *Info.Name, *Buffer.Label));
					continue;
				}
				const uint64 BoundSize = Bound->Size == WGPU_WHOLE_SIZE ? Buffer.Size - FMath::Min(Bound->Offset, Buffer.Size) : Bound->Size;
				if (BoundSize < (uint64)Info.MinSize)
				{
					Problems.Add(FString::Printf(TEXT("'%s' needs at least %lld bytes, buffer '%s' provides %llu"), *Info.Name, Info.MinSize, *Buffer.Label, BoundSize));
				}
				if (Info.Type == EWebGPUBindingType::UniformBuffer && !EnumHasAnyFlags(Buffer.Usage, EWebGPUBufferUsage::Uniform))
				{
					Problems.Add(FString::Printf(TEXT("'%s' is var<uniform> but buffer '%s' was not created with the Uniform usage"), *Info.Name, *Buffer.Label));
				}
				else if (Info.Type != EWebGPUBindingType::UniformBuffer && !EnumHasAnyFlags(Buffer.Usage, EWebGPUBufferUsage::Storage))
				{
					Problems.Add(FString::Printf(TEXT("'%s' is var<storage> but buffer '%s' was not created with the Storage usage"), *Info.Name, *Buffer.Label));
				}

				// Struct buffers: the UStruct layout should be the shader's (warn only; the bytes still flow)
				if (Buffer.ElementLayout)
				{
					TSharedPtr<const FWebGPUTypeLayout> Element = Info.GetRuntimeArrayElement();
					if (!Element && Info.Layout && Info.Layout->Kind == FWebGPUTypeLayout::EKind::Struct)
					{
						Element = Info.Layout;
					}
					FString Differences;
					if (Element && Element->Kind == FWebGPUTypeLayout::EKind::Struct && !Buffer.ElementLayout->Matches(*Element, Differences))
					{
						WarnOnce(Kernel, FString::Printf(TEXT("Kernel '%s': '%s' holds %s but the shader reads %s: %s"),
							*Kernel.Label, *Info.Name, *Buffer.ElementLayout->GetStructName(), *Element->Name, *Differences));
					}
				}

				BindEntry.buffer = Buffer.Handle;
				BindEntry.offset = Bound->Offset;
				BindEntry.size = Bound->Size;
			}
			else
			{
				Problems.Add(FString::Printf(TEXT("'%s' is a kind of resource that can't be bound yet"), *Info.Name));
				continue;
			}
			Groups.FindOrAdd((uint32)Info.Group).Add(BindEntry);
		}
		if (Problems.Num() > 0)
		{
			WarnOnce(Kernel, FString::Printf(TEXT("Kernel '%s' not dispatched: %s"), *Kernel.Label, *FString::Join(Problems, TEXT("; "))));
			return false;
		}
	}
	else
	{
		// No reflection: bind exactly what was given
		for (const TPair<uint64, const FWebGPUKernel::FBinding*>& Pair : Resolved)
		{
			const FWebGPUKernel::FBinding& Bound = *Pair.Value;
			WGPUBindGroupEntry& BindEntry = Groups.FindOrAdd((uint32)(Pair.Key >> 32)).Add_GetRef(WGPU_BIND_GROUP_ENTRY_INIT);
			BindEntry.binding = (uint32)(Pair.Key & 0xffffffff);
			if (Bound.Sampler.IsSet())
			{
				BindEntry.sampler = Samplers[(int32)Bound.Sampler.GetValue()];
			}
			else if (Bound.Texture && Bound.Texture->View)
			{
				BindEntry.textureView = Bound.Texture->View;
			}
			else if (Bound.Buffer && Bound.Buffer->Handle)
			{
				BindEntry.buffer = Bound.Buffer->Handle;
				BindEntry.offset = Bound.Offset;
				BindEntry.size = Bound.Size;
			}
			else
			{
				WarnOnce(Kernel, FString::Printf(TEXT("Kernel '%s': a bound resource has no GPU memory"), *Kernel.Label));
				return false;
			}
		}
	}

	// WebGPU's own validation (sample types, storage access, formats) is checked here too, so a binding it
	// rejects skips this kernel instead of failing the whole batch. Rebuilt only when bindings change.
	WorkerDevice->PushErrorScopes();
	for (const TPair<uint32, TArray<WGPUBindGroupEntry>>& Group : Groups)
	{
		WGPUBindGroupLayout Layout = wgpuComputePipelineGetBindGroupLayout(Kernel.Pipeline, Group.Key);
		WGPUBindGroupDescriptor Desc = {};
		Desc.layout = Layout;
		Desc.entryCount = Group.Value.Num();
		Desc.entries = Group.Value.GetData();
		Kernel.BindGroups.Add(Group.Key, wgpuDeviceCreateBindGroup(WorkerDevice->GetDevice(), &Desc));
		wgpuBindGroupLayoutRelease(Layout);
	}
	FString BindError;
	if (!WorkerDevice->PopErrorScopes(BindError))
	{
		Kernel.ReleaseBindGroups();
		Kernel.bBindGroupsDirty = false;
		WarnOnce(Kernel, FString::Printf(TEXT("Kernel '%s' not dispatched: WebGPU rejected its bindings: %s"), *Kernel.Label, *BindError));
		return false;
	}
	Kernel.bBindingsValid = true;
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
	for (const TWeakPtr<FWebGPUTexture>& Weak : Kernel.SharedTextures)
	{
		TSharedPtr<FWebGPUTexture> Texture = Weak.Pin();
		if (Texture && !Texture->bAccessing)
		{
			StatSkippedDispatches++;
			WarnOnce(Kernel, FString::Printf(TEXT("Kernel '%s' not dispatched: shared texture '%s' is only usable between BeginTextureAccess and EndTextureAccess"),
				*Kernel.Label, *Texture->GetLabel()));
			return;
		}
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

	// The copy joins the current batch; the staging buffer is mapped once that batch is submitted
	uint64 Capacity = 0;
	WGPUBuffer Staging = AcquireStaging(Size, Capacity);
	wgpuCommandEncoderCopyBufferToBuffer(GetEncoder(), Buffer.Handle, Offset, Staging, 0, Size);
	StatReadbacks++;

	const uint32 Generation = DeviceGeneration;
	AfterSubmit([this, Staging, Capacity, Size, Generation, Callback = MoveTemp(Callback), CallbackThread](bool bSubmitted) mutable
	{
		if (!bSubmitted)
		{
			wgpuBufferRelease(Staging);
			Deliver(CallbackThread, [Callback = MoveTemp(Callback)]() mutable
			{
				if (Callback)
				{
					Callback(false, TArray<uint8>());
				}
			});
			return;
		}

		MapAsync(Staging, WGPUMapMode_Read, 0, Size, [this, Staging, Capacity, Size, Generation, Callback = MoveTemp(Callback), CallbackThread](bool bSuccess) mutable
		{
			TArray<uint8> Data;
			if (bSuccess)
			{
				const void* Mapped = wgpuBufferGetConstMappedRange(Staging, 0, (size_t)Size);
				bSuccess = Mapped != nullptr;
				if (bSuccess)
				{
					Data.SetNumUninitialized((int32)Size);
					FMemory::Memcpy(Data.GetData(), Mapped, Size);
					StatBytesRead += Size;
				}
				wgpuBufferUnmap(Staging);
				ReleaseStaging(Staging, Capacity, Generation);
			}
			else
			{
				wgpuBufferRelease(Staging);
			}

			Deliver(CallbackThread, [Callback = MoveTemp(Callback), bSuccess, Data = MoveTemp(Data)]() mutable
			{
				if (Callback)
				{
					Callback(bSuccess, MoveTemp(Data));
				}
			});
		});
	});
}

void FWebGPUContext::UploadOnWorker(FWebGPUBuffer& Buffer, uint64 Offset, const void* Data, uint64 Size)
{
	uint64 PageOffset = 0;
	if (FUploadPage* Page = AllocateUpload(Size, PageOffset))
	{
		// Staged in upload memory and copied in order with the batch's dispatches (no extra submit)
		FMemory::Memcpy(Page->Mapped + PageOffset, Data, Size);
		wgpuCommandEncoderCopyBufferToBuffer(GetEncoder(), Page->Buffer, PageOffset, Buffer.Handle, Offset, Size);
	}
	else
	{
		// No upload memory: Dawn stages it at the queue's current point, after what is encoded so far
		SubmitBatch();
		wgpuQueueWriteBuffer(WorkerDevice->GetQueue(), Buffer.Handle, Offset, Data, (size_t)Size);
	}
	StatBytesWritten += Size;
}

WGPUBuffer FWebGPUContext::CreateMappableBuffer(WGPUBufferUsage Usage, uint64 Size, bool bMappedAtCreation, const char* Label)
{
	WGPUBufferDescriptor Desc = WGPU_BUFFER_DESCRIPTOR_INIT;
	Desc.label = ToView(Label);
	Desc.usage = Usage;
	Desc.size = Size;
	Desc.mappedAtCreation = bMappedAtCreation ? WGPU_TRUE : WGPU_FALSE;

	WorkerDevice->PushErrorScopes();
	WGPUBuffer Buffer = wgpuDeviceCreateBuffer(WorkerDevice->GetDevice(), &Desc);
	FString Error;
	if (!WorkerDevice->PopErrorScopes(Error))
	{
		UE_LOG(LogWebGPU, Error, TEXT("Creating a %llu-byte staging buffer failed: %s"), Size, *Error);
		if (Buffer)
		{
			wgpuBufferRelease(Buffer);
		}
		return nullptr;
	}
	return Buffer;
}

FWebGPUContext::FUploadPage* FWebGPUContext::AllocateUpload(uint64 Size, uint64& OutOffset, uint64 Alignment)
{
	const uint64 Aligned = Align(Size, (uint64)4);
	if (CurrentUploadPage)
	{
		const uint64 Start = Align(CurrentUploadPage->Used, Alignment);
		if (Start + Aligned <= CurrentUploadPage->Capacity)
		{
			OutOffset = Start;
			CurrentUploadPage->Used = Start + Aligned;
			return CurrentUploadPage;
		}
	}

	const uint64 Capacity = FMath::Max<uint64>(UploadPageSize, FMath::RoundUpToPowerOfTwo64(Aligned));
	FUploadPage* Page = nullptr;
	TArray<FUploadPage*>* Free = UploadPagePool.Find(Capacity);
	if (Free && Free->Num() > 0)
	{
		Page = Free->Pop(EAllowShrinking::No);
	}
	else
	{
		WGPUBuffer Buffer = CreateMappableBuffer(WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc, Capacity, true, "WebGPUCompute upload");
		if (!Buffer)
		{
			return nullptr;
		}
		Page = new FUploadPage();
		Page->Buffer = Buffer;
		Page->Capacity = Capacity;
		Page->Generation = DeviceGeneration;
		Page->Mapped = static_cast<uint8*>(wgpuBufferGetMappedRange(Buffer, 0, (size_t)Capacity));
		if (!Page->Mapped)
		{
			DeleteUploadPage(Page);
			return nullptr;
		}
	}

	Page->Used = Aligned;
	OutOffset = 0;
	BatchUploadPages.Add(Page);
	// Small writes keep filling a shared page; a big write's own page only becomes current if there is none
	if (Capacity == UploadPageSize || !CurrentUploadPage)
	{
		CurrentUploadPage = Page;
	}
	return Page;
}

void FWebGPUContext::RecycleUploadPage(FUploadPage* Page)
{
	MapAsync(Page->Buffer, WGPUMapMode_Write, 0, Page->Capacity, [this, Page](bool bSuccess)
	{
		if (!bSuccess || Page->Generation != DeviceGeneration || bStopping || bReleasingDevice)
		{
			DeleteUploadPage(Page);
			return;
		}
		Page->Mapped = static_cast<uint8*>(wgpuBufferGetMappedRange(Page->Buffer, 0, (size_t)Page->Capacity));
		Page->Used = 0;
		TArray<FUploadPage*>& Free = UploadPagePool.FindOrAdd(Page->Capacity);
		if (Page->Mapped && Free.Num() < MaxPooledUploadPagesPerSize)
		{
			Free.Add(Page);
		}
		else
		{
			DeleteUploadPage(Page);
		}
	});
}

void FWebGPUContext::DeleteUploadPage(FUploadPage* Page)
{
	if (Page->Buffer)
	{
		wgpuBufferRelease(Page->Buffer);
	}
	delete Page;
}

void FWebGPUContext::MapAsync(WGPUBuffer Buffer, WGPUMapMode Mode, uint64 Offset, uint64 Size, TUniqueFunction<void(bool bSuccess)>&& Handler)
{
	FMapRequest* Request = new FMapRequest();
	Request->Context = this;
	Request->Generation = DeviceGeneration;
	Request->Handler = MoveTemp(Handler);

	WGPUBufferMapCallbackInfo Info = {};
	Info.mode = WGPUCallbackMode_AllowProcessEvents;
	Info.callback = &FWebGPUContext::OnBufferMapped;
	Info.userdata1 = Request;
	PendingCallbacks++;
	wgpuBufferMapAsync(Buffer, Mode, (size_t)Offset, (size_t)Size, Info);
}

void FWebGPUContext::OnBufferMapped(WGPUMapAsyncStatus Status, WGPUStringView Message, void* UserData1, void*)
{
	TUniquePtr<FMapRequest> Request(static_cast<FMapRequest*>(UserData1));
	Request->Context->CallbackDone(Request->Generation);
	if (Status != WGPUMapAsyncStatus_Success)
	{
		UE_LOG(LogWebGPU, Verbose, TEXT("Buffer map failed (%d): %s"), (int32)Status, *FWebGPUDevice::ToFString(Message));
	}
	Request->Handler(Status == WGPUMapAsyncStatus_Success);
}

void FWebGPUContext::CallbackDone(uint32 Generation)
{
	// Callbacks of a replaced device don't count: the counter was reset when it was released
	if (Generation == DeviceGeneration && PendingCallbacks > 0)
	{
		PendingCallbacks--;
	}
}

void FWebGPUContext::DestroyBufferOnWorker(WGPUBuffer Buffer)
{
	// Destroy frees the memory now rather than when Dawn drops its last internal reference. Commands of the
	// open batch may still use the buffer, so it waits for that batch's submit (the GPU finishes with it first).
	auto Destroy = [Buffer](bool)
	{
		wgpuBufferDestroy(Buffer);
		wgpuBufferRelease(Buffer);
	};
	if (Encoder)
	{
		AfterSubmit(MoveTemp(Destroy));
	}
	else
	{
		Destroy(true);
	}
}

void FWebGPUContext::AfterSubmit(TUniqueFunction<void(bool bSubmitted)>&& Action)
{
	BatchAfterSubmit.Add(MoveTemp(Action));
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

	// Timestamps of profiled passes are resolved into the batch and mapped after it
	WGPUBuffer TimestampStaging = nullptr;
	uint64 TimestampCapacity = 0;
	const uint32 TimestampCount = BatchQueryCount;
	TArray<TWeakPtr<FWebGPUKernel>> Profiled;
	if (TimestampCount > 0)
	{
		const uint64 QueryBytes = (uint64)TimestampCount * sizeof(uint64);
		wgpuCommandEncoderResolveQuerySet(Encoder, TimestampQueries, 0, TimestampCount, TimestampResolve, 0);
		TimestampStaging = AcquireStaging(QueryBytes, TimestampCapacity);
		wgpuCommandEncoderCopyBufferToBuffer(Encoder, TimestampResolve, 0, TimestampStaging, 0, QueryBytes);
		Profiled = MoveTemp(BatchProfiled);
	}

	// The batch copies from its upload pages: they must be unmapped when it is submitted
	for (FUploadPage* Page : BatchUploadPages)
	{
		wgpuBufferUnmap(Page->Buffer);
		Page->Mapped = nullptr;
	}
	CurrentUploadPage = nullptr;

	WGPUCommandBuffer CommandBuffer = wgpuCommandEncoderFinish(Encoder, nullptr);
	wgpuQueueSubmit(WorkerDevice->GetQueue(), 1, &CommandBuffer);
	wgpuCommandBufferRelease(CommandBuffer);
	wgpuCommandEncoderRelease(Encoder);
	Encoder = nullptr;
	StatSubmits++;
	BusyPollUntil = FPlatformTime::Seconds() + Options.BusyPollMilliseconds / 1000.0;

	FBatchScopeRequest* ScopeRequest = new FBatchScopeRequest{ this, DeviceGeneration, FString::Join(BatchLabels, TEXT(", ")) };
	WGPUPopErrorScopeCallbackInfo ScopeInfo = {};
	ScopeInfo.mode = WGPUCallbackMode_AllowProcessEvents;
	ScopeInfo.callback = &FWebGPUContext::OnBatchScopePopped;
	ScopeInfo.userdata1 = ScopeRequest;
	PendingCallbacks++;
	wgpuDevicePopErrorScope(Dev, ScopeInfo);

	// Reset the batch first: post-submit actions may start new work
	TArray<TUniqueFunction<void(bool)>> Actions = MoveTemp(BatchAfterSubmit);
	TArray<FUploadPage*> Pages = MoveTemp(BatchUploadPages);
	BatchAfterSubmit.Reset();
	BatchUploadPages.Reset();
	BatchLabels.Reset();
	BatchQueryCount = 0;
	BatchProfiled.Reset();
	BatchCommands = 0;

	for (FUploadPage* Page : Pages)
	{
		RecycleUploadPage(Page);
	}
	for (TUniqueFunction<void(bool)>& Action : Actions)
	{
		Action(true);
	}

	if (TimestampStaging)
	{
		const uint32 Generation = DeviceGeneration;
		MapAsync(TimestampStaging, WGPUMapMode_Read, 0, (uint64)TimestampCount * sizeof(uint64),
			[this, TimestampStaging, TimestampCapacity, TimestampCount, Generation, Profiled = MoveTemp(Profiled)](bool bSuccess)
		{
			if (!bSuccess)
			{
				wgpuBufferRelease(TimestampStaging);
				return;
			}
			const uint64* Ticks = static_cast<const uint64*>(wgpuBufferGetConstMappedRange(TimestampStaging, 0, (size_t)TimestampCount * sizeof(uint64)));
			if (Ticks)
			{
				for (int32 Index = 0; Index < Profiled.Num(); Index++)
				{
					TSharedPtr<FWebGPUKernel> Kernel = Profiled[Index].Pin();
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
			wgpuBufferUnmap(TimestampStaging);
			ReleaseStaging(TimestampStaging, TimestampCapacity, Generation);
		});
	}
}

void FWebGPUContext::OnBatchScopePopped(WGPUPopErrorScopeStatus Status, WGPUErrorType Type, WGPUStringView Message, void* UserData1, void*)
{
	TUniquePtr<FBatchScopeRequest> Request(static_cast<FBatchScopeRequest*>(UserData1));
	FWebGPUContext& Self = *Request->Context;
	Self.CallbackDone(Request->Generation);

	if (Status == WGPUPopErrorScopeStatus_Success && Type != WGPUErrorType_NoError)
	{
		Self.StatFailedSubmits++;
		UE_LOG(LogWebGPU, Error, TEXT("WebGPU batch rejected (kernels: %s): %s"),
			Request->Labels.IsEmpty() ? TEXT("none") : *Request->Labels, *FWebGPUDevice::ToFString(Message));
	}
}

void FWebGPUContext::OnFlushDone(WGPUQueueWorkDoneStatus Status, WGPUStringView, void* UserData1, void*)
{
	TUniquePtr<FFlushRequest> Request(static_cast<FFlushRequest*>(UserData1));
	Request->Context->CallbackDone(Request->Generation);
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
	if (!WorkerDevice)
	{
		return;
	}

	// Built-in samplers, indexed by EWebGPUSampler
	for (int32 Index = 0; Index < UE_ARRAY_COUNT(Samplers); Index++)
	{
		const bool bLinear = Index == (int32)EWebGPUSampler::LinearClamp || Index == (int32)EWebGPUSampler::LinearRepeat;
		const bool bRepeat = Index == (int32)EWebGPUSampler::LinearRepeat || Index == (int32)EWebGPUSampler::NearestRepeat;
		WGPUSamplerDescriptor SamplerDesc = WGPU_SAMPLER_DESCRIPTOR_INIT;
		SamplerDesc.addressModeU = SamplerDesc.addressModeV = SamplerDesc.addressModeW = bRepeat ? WGPUAddressMode_Repeat : WGPUAddressMode_ClampToEdge;
		SamplerDesc.magFilter = SamplerDesc.minFilter = bLinear ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
		Samplers[Index] = wgpuDeviceCreateSampler(WorkerDevice->GetDevice(), &SamplerDesc);
	}

	if (!WorkerDevice->HasFeature(WGPUFeatureName_TimestampQuery))
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
	TArray<TUniqueFunction<void(bool)>> Abandoned = MoveTemp(BatchAfterSubmit);
	BatchAfterSubmit.Reset();
	for (TUniqueFunction<void(bool)>& Action : Abandoned)
	{
		Action(false);
	}
	for (FUploadPage* Page : BatchUploadPages)
	{
		DeleteUploadPage(Page);
	}
	BatchUploadPages.Reset();
	CurrentUploadPage = nullptr;
	for (TPair<uint64, TArray<FUploadPage*>>& Pair : UploadPagePool)
	{
		for (FUploadPage* Page : Pair.Value)
		{
			DeleteUploadPage(Page);
		}
	}
	UploadPagePool.Reset();
	BatchLabels.Reset();
	BatchQueryCount = 0;
	BatchProfiled.Reset();
	BatchCommands = 0;

	for (TPair<uint64, TArray<WGPUBuffer>>& Pair : StagingPool)
	{
		for (WGPUBuffer Staging : Pair.Value)
		{
			wgpuBufferRelease(Staging);
		}
	}
	StagingPool.Reset();

	for (WGPUSampler& Sampler : Samplers)
	{
		if (Sampler)
		{
			wgpuSamplerRelease(Sampler);
			Sampler = nullptr;
		}
	}
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
		for (const TWeakPtr<FWebGPUTexture>& Weak : LiveTextures)
		{
			if (TSharedPtr<FWebGPUTexture> Texture = Weak.Pin())
			{
				ReleaseTextureHandle(*Texture);
			}
		}

		// Streams drop their slots, except memory the game still holds (an acquired frame, a slot being written)
		bool bMemoryHeld = false;
		for (const TWeakPtr<FWebGPUReadbackStream>& Weak : LiveReadbackStreams)
		{
			if (TSharedPtr<FWebGPUReadbackStream> Stream = Weak.Pin())
			{
				bMemoryHeld |= Stream->ReleaseForRecovery();
			}
		}
		for (const TWeakPtr<FWebGPUUploadStream>& Weak : LiveUploadStreams)
		{
			if (TSharedPtr<FWebGPUUploadStream> Stream = Weak.Pin())
			{
				bMemoryHeld |= Stream->ReleaseForRecovery();
			}
		}

		// Fails whatever is still pending on the old device. Destroying it would unmap memory the game still
		// reads or writes, so a device with held memory is retired instead and destroyed once that is released.
		bReleasingDevice = true;
		WorkerDevice->SetLostHandler(nullptr);
		if (bMemoryHeld)
		{
			wgpuInstanceProcessEvents(WorkerDevice->GetInstance());
			RetiredDevices.Add(WorkerDevice);
			bHasRetiredDevices = true;
		}
		else
		{
			WorkerDevice->Shutdown();
		}
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

	LiveTextures.RemoveAll([](const TWeakPtr<FWebGPUTexture>& Weak) { return !Weak.IsValid(); });
	for (const TWeakPtr<FWebGPUTexture>& Weak : LiveTextures)
	{
		if (TSharedPtr<FWebGPUTexture> Texture = Weak.Pin())
		{
			CreateTextureHandle(*Texture);
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

	LiveReadbackStreams.RemoveAll([](const TWeakPtr<FWebGPUReadbackStream>& Weak) { return !Weak.IsValid(); });
	LiveUploadStreams.RemoveAll([](const TWeakPtr<FWebGPUUploadStream>& Weak) { return !Weak.IsValid(); });
	for (const TWeakPtr<FWebGPUUploadStream>& Weak : LiveUploadStreams)
	{
		if (TSharedPtr<FWebGPUUploadStream> Stream = Weak.Pin())
		{
			Stream->RecreateAfterRecovery(*this);
		}
	}

	UE_LOG(LogWebGPU, Log, TEXT("WebGPU device recreated (generation %u): %d buffers, %d textures, %d kernels restored (buffer and texture contents are zeroed; shared textures are imported again)"),
		Generation, LiveBuffers.Num(), LiveTextures.Num(), LiveKernels.Num());

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
	for (const TWeakPtr<FWebGPUReadbackStream>& Weak : LiveReadbackStreams)
	{
		if (TSharedPtr<FWebGPUReadbackStream> Stream = Weak.Pin())
		{
			Stream->ReleaseForRecovery();
		}
	}
	for (const TWeakPtr<FWebGPUUploadStream>& Weak : LiveUploadStreams)
	{
		if (TSharedPtr<FWebGPUUploadStream> Stream = Weak.Pin())
		{
			Stream->ReleaseForRecovery();
		}
	}
	for (const TWeakPtr<FWebGPUTexture>& Weak : LiveTextures)
	{
		if (TSharedPtr<FWebGPUTexture> Texture = Weak.Pin())
		{
			ReleaseTextureHandle(*Texture);
		}
	}
	LiveBuffers.Reset();
	LiveKernels.Reset();
	LiveTextures.Reset();
	LiveReadbackStreams.Reset();
	LiveUploadStreams.Reset();
	ShutdownRetiredDevices(true);

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

void FWebGPUContext::ShutdownRetiredDevices(bool bForce)
{
	for (const TSharedPtr<FWebGPUDevice>& Retired : RetiredDevices)
	{
		wgpuInstanceProcessEvents(Retired->GetInstance());
	}

	if (!bForce)
	{
		const uint32 Generation = DeviceGeneration;
		for (const TWeakPtr<FWebGPUReadbackStream>& Weak : LiveReadbackStreams)
		{
			TSharedPtr<FWebGPUReadbackStream> Stream = Weak.Pin();
			if (Stream && Stream->HoldsRetiredMemory(Generation))
			{
				return;
			}
		}
		for (const TWeakPtr<FWebGPUUploadStream>& Weak : LiveUploadStreams)
		{
			TSharedPtr<FWebGPUUploadStream> Stream = Weak.Pin();
			if (Stream && Stream->HoldsRetiredMemory(Generation))
			{
				return;
			}
		}
	}

	bReleasingDevice = true;
	for (const TSharedPtr<FWebGPUDevice>& Retired : RetiredDevices)
	{
		Retired->Shutdown();
	}
	bReleasingDevice = false;
	if (RetiredDevices.Num() > 0)
	{
		UE_LOG(LogWebGPU, Log, TEXT("Released the lost WebGPU device: nothing holds its mapped memory anymore"));
	}
	RetiredDevices.Reset();
	bHasRetiredDevices = false;
}
