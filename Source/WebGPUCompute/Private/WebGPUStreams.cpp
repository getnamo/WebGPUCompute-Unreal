// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUStreams.h"
#include "WebGPUCompute.h"
#include "HAL/Event.h"
#include "HAL/PlatformProcess.h"
#include "Misc/ScopeLock.h"

namespace
{
	constexpr int32 MaxStreamSlots = 64;

	/** FWebGPUStructMirror's scatter: src = [count, words, indices[count], elements[count * words]]. */
	const TCHAR* ScatterSource = TEXT(R"(
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
	let count = src[0];
	let words = src[1];
	let total = count * words;
	let stride = groups.x * 64u;
	for (var w = gid.x; w < total; w += stride) {
		let element = w / words;
		let word = w - element * words;
		dst[src[2u + element] * words + word] = src[2u + count + w];
	}
}
)");

	/** Waits on an event for up to the remaining time, in short steps (several threads may wait on it). */
	bool WaitStep(FEvent* Event, double End)
	{
		const double Remaining = End - FPlatformTime::Seconds();
		if (Remaining <= 0.0)
		{
			return false;
		}
		Event->Wait(FMath::Clamp((uint32)(Remaining * 1000.0), 1u, 5u));
		return true;
	}

	void UpdateLatency(float& Average, double Ms)
	{
		Average = Average > 0.f ? Average * 0.9f + (float)Ms * 0.1f : (float)Ms;
	}

	/** Releases Dawn buffers on the worker (or right here when that is the worker, or it is gone). */
	void ReleaseBuffersOnWorker(const TWeakPtr<FWebGPUContext>& Context, TArray<WGPUBuffer>&& Buffers)
	{
		if (Buffers.Num() == 0)
		{
			return;
		}
		auto Release = [Buffers = MoveTemp(Buffers)]()
		{
			for (WGPUBuffer Buffer : Buffers)
			{
				wgpuBufferRelease(Buffer);
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
}

// FWebGPUContext stream factories

TSharedRef<FWebGPUReadbackStream> FWebGPUContext::CreateReadbackStream(uint64 SlotBytes, int32 Slots, EWebGPUStreamPolicy Policy, const FString& Label)
{
	TSharedRef<FWebGPUReadbackStream> Stream = MakeShareable(new FWebGPUReadbackStream());
	Stream->Context = AsShared();
	Stream->Label = Label.IsEmpty() ? TEXT("WebGPUReadbackStream") : Label;
	Stream->SlotBytes = Align(FMath::Max<uint64>(SlotBytes, 4), (uint64)4);
	Stream->SlotCount = FMath::Clamp(Slots, 1, MaxStreamSlots);
	Stream->Policy = Policy;
	Stream->Slots.SetNum(Stream->SlotCount);
	Stream->ReadyEvent = FPlatformProcess::GetSynchEventFromPool(false);

	Enqueue([this, Weak = TWeakPtr<FWebGPUReadbackStream>(Stream)]()
	{
		LiveReadbackStreams.Add(Weak);
	});
	return Stream;
}

TSharedRef<FWebGPUUploadStream> FWebGPUContext::CreateUploadStream(uint64 SlotBytes, int32 Slots, const FString& Label)
{
	TSharedRef<FWebGPUUploadStream> Stream = MakeShareable(new FWebGPUUploadStream());
	Stream->Context = AsShared();
	Stream->Label = Label.IsEmpty() ? TEXT("WebGPUUploadStream") : Label;
	Stream->SlotBytes = Align(FMath::Max<uint64>(SlotBytes, 4), (uint64)4);
	Stream->SlotCount = FMath::Clamp(Slots, 1, MaxStreamSlots);
	Stream->Slots.SetNum(Stream->SlotCount);
	Stream->SlotFreed = FPlatformProcess::GetSynchEventFromPool(false);

	Enqueue([this, Stream]()
	{
		LiveUploadStreams.Add(Stream);
		for (int32 Index = 0; Index < Stream->SlotCount; Index++)
		{
			Stream->CreateSlotBuffer(*this, Index);
		}
	});
	return Stream;
}

// FWebGPUReadbackFrame

FWebGPUReadbackFrame::FWebGPUReadbackFrame(FWebGPUReadbackFrame&& Other)
{
	*this = MoveTemp(Other);
}

FWebGPUReadbackFrame& FWebGPUReadbackFrame::operator=(FWebGPUReadbackFrame&& Other)
{
	if (this != &Other)
	{
		Release();
		Stream = MoveTemp(Other.Stream);
		Slot = Other.Slot;
		Data = Other.Data;
		Size = Other.Size;
		Sequence = Other.Sequence;
		Width = Other.Width;
		Height = Other.Height;
		RowBytes = Other.RowBytes;
		RowPitch = Other.RowPitch;
		CaptureFrame = Other.CaptureFrame;
		CaptureTime = Other.CaptureTime;
		ReadyTime = Other.ReadyTime;
		Other.Slot = INDEX_NONE;
		Other.Data = nullptr;
		Other.Size = 0;
	}
	return *this;
}

FWebGPUReadbackFrame::~FWebGPUReadbackFrame()
{
	Release();
}

void FWebGPUReadbackFrame::CopyTo(TArray<uint8>& OutBytes) const
{
	if (!Data)
	{
		OutBytes.Reset();
		return;
	}
	if (RowPitch == 0)
	{
		OutBytes.SetNumUninitialized((int32)Size);
		FMemory::Memcpy(OutBytes.GetData(), Data, Size);
		return;
	}
	// Texture rows: drop the 256-byte alignment padding
	OutBytes.SetNumUninitialized((int32)((uint64)RowBytes * Height));
	for (uint32 Y = 0; Y < Height; Y++)
	{
		FMemory::Memcpy(OutBytes.GetData() + (uint64)Y * RowBytes, Data + (uint64)Y * RowPitch, RowBytes);
	}
}

void FWebGPUReadbackFrame::Release()
{
	if (Stream)
	{
		Stream->ReleaseSlot(Slot, Sequence);
		Stream.Reset();
	}
	Slot = INDEX_NONE;
	Data = nullptr;
	Size = 0;
}

// FWebGPUReadbackStream

FWebGPUReadbackStream::~FWebGPUReadbackStream()
{
	TArray<WGPUBuffer> Buffers;
	for (const FSlot& Slot : Slots)
	{
		if (Slot.Buffer)
		{
			Buffers.Add(Slot.Buffer);
		}
	}
	ReleaseBuffersOnWorker(Context, MoveTemp(Buffers));
	if (ReadyEvent)
	{
		FPlatformProcess::ReturnSynchEventToPool(ReadyEvent);
		ReadyEvent = nullptr;
	}
}

bool FWebGPUReadbackStream::Capture(const TSharedRef<FWebGPUBuffer>& Source, uint64 Offset, uint64 Size)
{
	TSharedPtr<FWebGPUContext> Ctx = Context.Pin();
	if (!Ctx)
	{
		return false;
	}
	check(!Ctx->IsInWorkerThread());

	if (Size == WGPU_WHOLE_SIZE)
	{
		Size = Source->GetSize() > Offset ? Source->GetSize() - Offset : 0;
	}
	if (Size == 0 || Offset % 4 != 0 || Size % 4 != 0 || Offset + Size > Source->GetSize() || Size > SlotBytes)
	{
		UE_LOG(LogWebGPU, Error, TEXT("Readback stream '%s': capturing %llu bytes at %llu of '%s' (%llu bytes): needs a non-empty multiple of 4 that fits a %llu-byte slot"),
			*Label, Size, Offset, *Source->GetLabel(), Source->GetSize(), SlotBytes);
		return false;
	}

	uint64 Sequence = 0;
	const int32 SlotIndex = ReserveSlot(Size, 0, 0, 0, 0, Sequence);
	if (SlotIndex == INDEX_NONE)
	{
		return false;
	}
	FWebGPUContext* CtxPtr = Ctx.Get();
	return FinishCapture(*Ctx, SlotIndex, Sequence, [CtxPtr, Self = AsShared(), SlotIndex, Sequence, Source, Offset, Size]()
	{
		Self->CaptureOnWorker(*CtxPtr, SlotIndex, Sequence, *Source, Offset, Size);
	});
}

bool FWebGPUReadbackStream::CaptureTexture(const TSharedRef<FWebGPUTexture>& Source)
{
	TSharedPtr<FWebGPUContext> Ctx = Context.Pin();
	if (!Ctx)
	{
		return false;
	}
	check(!Ctx->IsInWorkerThread());

	const uint32 RowPitch = Source->GetCopyBytesPerRow();
	const uint64 Size = (uint64)RowPitch * Source->GetHeight();
	if (Size > SlotBytes)
	{
		UE_LOG(LogWebGPU, Error, TEXT("Readback stream '%s': texture '%s' (%ux%u) needs %llu-byte slots (rows are 256-byte aligned), these are %llu"),
			*Label, *Source->GetLabel(), Source->GetWidth(), Source->GetHeight(), Size, SlotBytes);
		return false;
	}

	uint64 Sequence = 0;
	const int32 SlotIndex = ReserveSlot(Size, Source->GetWidth(), Source->GetHeight(), Source->GetWidth() * Source->GetBytesPerPixel(), RowPitch, Sequence);
	if (SlotIndex == INDEX_NONE)
	{
		return false;
	}
	FWebGPUContext* CtxPtr = Ctx.Get();
	return FinishCapture(*Ctx, SlotIndex, Sequence, [CtxPtr, Self = AsShared(), SlotIndex, Sequence, Source]()
	{
		Self->CaptureTextureOnWorker(*CtxPtr, SlotIndex, Sequence, *Source);
	});
}

int32 FWebGPUReadbackStream::ReserveSlot(uint64 Size, uint32 Width, uint32 Height, uint32 RowBytes, uint32 RowPitch, uint64& OutSequence)
{
	FScopeLock ScopeLock(&Lock);
	int32 SlotIndex = Slots.IndexOfByPredicate([](const FSlot& Slot) { return Slot.State == ESlot::Free; });
	if (SlotIndex == INDEX_NONE && Policy == EWebGPUStreamPolicy::Latest)
	{
		// Recycle the oldest frame nobody read: newer data is on its way
		uint64 Oldest = MAX_uint64;
		for (int32 Index = 0; Index < Slots.Num(); Index++)
		{
			if (Slots[Index].State == ESlot::Ready && Slots[Index].Sequence < Oldest)
			{
				Oldest = Slots[Index].Sequence;
				SlotIndex = Index;
			}
		}
		if (SlotIndex != INDEX_NONE)
		{
			Stats.Skipped++;
		}
	}
	if (SlotIndex == INDEX_NONE)
	{
		Stats.Dropped++;
		return INDEX_NONE;
	}

	FSlot& Slot = Slots[SlotIndex];
	Slot.State = ESlot::Reserved;
	Slot.Sequence = OutSequence = ++NextSequence;
	Slot.Size = Size;
	Slot.Width = Width;
	Slot.Height = Height;
	Slot.RowBytes = RowBytes;
	Slot.RowPitch = RowPitch;
	Slot.CaptureFrame = GFrameCounter;
	Slot.CaptureTime = FPlatformTime::Seconds();
	Slot.ReadyTime = 0.0;
	Slot.Mapped = nullptr;
	return SlotIndex;
}

bool FWebGPUReadbackStream::FinishCapture(FWebGPUContext& Ctx, int32 SlotIndex, uint64 Sequence, TUniqueFunction<void()>&& Command)
{
	Ctx.Enqueue(MoveTemp(Command));
	if (Policy == EWebGPUStreamPolicy::Lockstep)
	{
		return WaitForSequence(SlotIndex, Sequence, LockstepTimeoutSeconds);
	}
	return true;
}

bool FWebGPUReadbackStream::WaitForSequence(int32 SlotIndex, uint64 Sequence, double TimeoutSeconds)
{
	const double End = FPlatformTime::Seconds() + TimeoutSeconds;
	for (;;)
	{
		{
			FScopeLock ScopeLock(&Lock);
			const FSlot& Slot = Slots[SlotIndex];
			if (Slot.Sequence != Sequence || Slot.State == ESlot::Free)
			{
				// Failed, or already recycled
				return false;
			}
			if (Slot.State == ESlot::Ready || Slot.State == ESlot::Held)
			{
				return true;
			}
		}
		if (!WaitStep(ReadyEvent, End))
		{
			UE_LOG(LogWebGPU, Warning, TEXT("Readback stream '%s': lockstep capture %llu not back within %.1f s"), *Label, Sequence, TimeoutSeconds);
			return false;
		}
	}
}

bool FWebGPUReadbackStream::Acquire(FWebGPUReadbackFrame& OutFrame)
{
	OutFrame.Release();

	FScopeLock ScopeLock(&Lock);
	int32 Chosen = INDEX_NONE;
	if (Policy == EWebGPUStreamPolicy::Latest)
	{
		for (int32 Index = 0; Index < Slots.Num(); Index++)
		{
			if (Slots[Index].State == ESlot::Ready && (Chosen == INDEX_NONE || Slots[Index].Sequence > Slots[Chosen].Sequence))
			{
				Chosen = Index;
			}
		}
		if (Chosen != INDEX_NONE)
		{
			for (int32 Index = 0; Index < Slots.Num(); Index++)
			{
				if (Index != Chosen && Slots[Index].State == ESlot::Ready)
				{
					// Superseded; still mapped, unmapped when the slot is reused
					Slots[Index].State = ESlot::Free;
					Stats.Skipped++;
				}
			}
		}
	}
	else
	{
		// In capture order: the oldest outstanding frame, once it is ready
		for (int32 Index = 0; Index < Slots.Num(); Index++)
		{
			const ESlot State = Slots[Index].State;
			if ((State == ESlot::Reserved || State == ESlot::Pending || State == ESlot::Ready) && (Chosen == INDEX_NONE || Slots[Index].Sequence < Slots[Chosen].Sequence))
			{
				Chosen = Index;
			}
		}
		if (Chosen != INDEX_NONE && Slots[Chosen].State != ESlot::Ready)
		{
			Chosen = INDEX_NONE;
		}
	}
	if (Chosen == INDEX_NONE)
	{
		return false;
	}

	FSlot& Slot = Slots[Chosen];
	Slot.State = ESlot::Held;
	Stats.Delivered++;

	OutFrame.Stream = AsShared();
	OutFrame.Slot = Chosen;
	OutFrame.Data = Slot.Mapped;
	OutFrame.Size = Slot.Size;
	OutFrame.Sequence = Slot.Sequence;
	OutFrame.CaptureFrame = Slot.CaptureFrame;
	OutFrame.CaptureTime = Slot.CaptureTime;
	OutFrame.ReadyTime = Slot.ReadyTime;
	OutFrame.Width = Slot.Width;
	OutFrame.Height = Slot.Height;
	OutFrame.RowBytes = Slot.RowBytes;
	OutFrame.RowPitch = Slot.RowPitch;
	return true;
}

bool FWebGPUReadbackStream::WaitForFrame(double TimeoutSeconds)
{
	TSharedPtr<FWebGPUContext> Ctx = Context.Pin();
	check(!Ctx || !Ctx->IsInWorkerThread());

	const double End = FPlatformTime::Seconds() + TimeoutSeconds;
	for (;;)
	{
		{
			FScopeLock ScopeLock(&Lock);
			if (Policy == EWebGPUStreamPolicy::Latest)
			{
				if (Slots.ContainsByPredicate([](const FSlot& Slot) { return Slot.State == ESlot::Ready; }))
				{
					return true;
				}
			}
			else
			{
				const FSlot* Oldest = nullptr;
				for (const FSlot& Slot : Slots)
				{
					if ((Slot.State == ESlot::Reserved || Slot.State == ESlot::Pending || Slot.State == ESlot::Ready) && (!Oldest || Slot.Sequence < Oldest->Sequence))
					{
						Oldest = &Slot;
					}
				}
				if (Oldest && Oldest->State == ESlot::Ready)
				{
					return true;
				}
			}
		}
		if (!WaitStep(ReadyEvent, End))
		{
			return false;
		}
	}
}

int32 FWebGPUReadbackStream::GetReadyCount() const
{
	FScopeLock ScopeLock(&Lock);
	int32 Count = 0;
	for (const FSlot& Slot : Slots)
	{
		Count += Slot.State == ESlot::Ready ? 1 : 0;
	}
	return Count;
}

FWebGPUStreamStats FWebGPUReadbackStream::GetStats() const
{
	FScopeLock ScopeLock(&Lock);
	FWebGPUStreamStats Result = Stats;
	Result.InUse = 0;
	for (const FSlot& Slot : Slots)
	{
		Result.InUse += Slot.State != ESlot::Free ? 1 : 0;
	}
	return Result;
}

void FWebGPUReadbackStream::ReleaseSlot(int32 SlotIndex, uint64 Sequence)
{
	{
		FScopeLock ScopeLock(&Lock);
		if (!Slots.IsValidIndex(SlotIndex))
		{
			return;
		}
		FSlot& Slot = Slots[SlotIndex];
		if (Slot.Sequence == Sequence && Slot.State == ESlot::Held)
		{
			// Stays mapped until the slot is reused
			Slot.State = ESlot::Free;
		}
	}

	// A frame of a lost device may have been the last thing keeping that device alive
	TSharedPtr<FWebGPUContext> Ctx = Context.Pin();
	if (Ctx && Ctx->bHasRetiredDevices)
	{
		Ctx->Wake();
	}
}

WGPUBuffer FWebGPUReadbackStream::PrepareSlot(FWebGPUContext& Ctx, int32 SlotIndex, uint64 Sequence)
{
	FSlot& Slot = Slots[SlotIndex];
	{
		FScopeLock ScopeLock(&Lock);
		if (Slot.Sequence != Sequence || Slot.State != ESlot::Reserved)
		{
			return nullptr;
		}
	}
	if (!Ctx.HasDevice())
	{
		FailSlot(SlotIndex, Sequence);
		return nullptr;
	}

	// A buffer of the current device, not mapped (it may still hold an older frame)
	if (Slot.Buffer && Slot.Generation != Ctx.DeviceGeneration)
	{
		wgpuBufferRelease(Slot.Buffer);
		Slot.Buffer = nullptr;
		Slot.bMapped = false;
	}
	if (!Slot.Buffer)
	{
		Slot.Buffer = Ctx.CreateMappableBuffer(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, SlotBytes, false, "WebGPUCompute readback stream");
		Slot.Generation = Ctx.DeviceGeneration;
		Slot.bMapped = false;
		if (!Slot.Buffer)
		{
			FailSlot(SlotIndex, Sequence);
			return nullptr;
		}
	}
	else if (Slot.bMapped)
	{
		wgpuBufferUnmap(Slot.Buffer);
		Slot.bMapped = false;
	}
	return Slot.Buffer;
}

void FWebGPUReadbackStream::SubmitSlot(FWebGPUContext& Ctx, int32 SlotIndex, uint64 Sequence, uint64 Size)
{
	{
		FScopeLock ScopeLock(&Lock);
		Slots[SlotIndex].State = ESlot::Pending;
		Stats.Committed++;
	}
	Ctx.StatReadbacks++;

	FWebGPUContext* CtxPtr = &Ctx;
	const uint32 Generation = Ctx.DeviceGeneration;
	WGPUBuffer Buffer = Slots[SlotIndex].Buffer;
	Ctx.AfterSubmit([Self = AsShared(), CtxPtr, SlotIndex, Sequence, Generation, Buffer, Size](bool bSubmitted)
	{
		if (!bSubmitted)
		{
			Self->FailSlot(SlotIndex, Sequence);
			return;
		}
		CtxPtr->MapAsync(Buffer, WGPUMapMode_Read, 0, Size, [Self, CtxPtr, SlotIndex, Sequence, Generation](bool bSuccess)
		{
			Self->FinishMap(*CtxPtr, SlotIndex, Sequence, Generation, bSuccess);
		});
	});
}

void FWebGPUReadbackStream::CaptureOnWorker(FWebGPUContext& Ctx, int32 SlotIndex, uint64 Sequence, FWebGPUBuffer& Source, uint64 Offset, uint64 Size)
{
	if (!Source.GetHandle())
	{
		FailSlot(SlotIndex, Sequence);
		return;
	}
	WGPUBuffer Buffer = PrepareSlot(Ctx, SlotIndex, Sequence);
	if (!Buffer)
	{
		return;
	}
	wgpuCommandEncoderCopyBufferToBuffer(Ctx.GetEncoder(), Source.GetHandle(), Offset, Buffer, 0, Size);
	SubmitSlot(Ctx, SlotIndex, Sequence, Size);
}

void FWebGPUReadbackStream::CaptureTextureOnWorker(FWebGPUContext& Ctx, int32 SlotIndex, uint64 Sequence, FWebGPUTexture& Source)
{
	if (!Ctx.CanUseTexture(Source, TEXT("CaptureTexture")))
	{
		FailSlot(SlotIndex, Sequence);
		return;
	}
	WGPUBuffer Buffer = PrepareSlot(Ctx, SlotIndex, Sequence);
	if (!Buffer)
	{
		return;
	}

	WGPUTexelCopyTextureInfo From = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
	From.texture = Source.GetHandle();
	WGPUTexelCopyBufferInfo To = {};
	To.layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
	To.layout.bytesPerRow = Source.GetCopyBytesPerRow();
	To.layout.rowsPerImage = Source.GetHeight();
	To.buffer = Buffer;
	const WGPUExtent3D Extent = { Source.GetWidth(), Source.GetHeight(), 1 };
	wgpuCommandEncoderCopyTextureToBuffer(Ctx.GetEncoder(), &From, &To, &Extent);
	SubmitSlot(Ctx, SlotIndex, Sequence, (uint64)Source.GetCopyBytesPerRow() * Source.GetHeight());
}

void FWebGPUReadbackStream::FinishMap(FWebGPUContext& Ctx, int32 SlotIndex, uint64 Sequence, uint32 Generation, bool bSuccess)
{
	{
		FScopeLock ScopeLock(&Lock);
		FSlot& Slot = Slots[SlotIndex];
		if (Slot.Sequence != Sequence || Slot.State != ESlot::Pending || Slot.Generation != Generation
			|| Generation != Ctx.DeviceGeneration || Ctx.bReleasingDevice)
		{
			// Recycled meanwhile (device recovery); its buffer is gone
			return;
		}

		if (bSuccess)
		{
			Slot.bMapped = true;
			Slot.Mapped = static_cast<const uint8*>(wgpuBufferGetConstMappedRange(Slot.Buffer, 0, (size_t)Slot.Size));
		}
		if (Slot.Mapped)
		{
			Slot.State = ESlot::Ready;
			Slot.ReadyTime = FPlatformTime::Seconds();
			UpdateLatency(Stats.LatencyMs, (Slot.ReadyTime - Slot.CaptureTime) * 1000.0);
			Ctx.StatBytesRead += Slot.Size;
		}
		else
		{
			Slot.State = ESlot::Free;
			Stats.Dropped++;
		}
	}
	ReadyEvent->Trigger();
}

void FWebGPUReadbackStream::FailSlot(int32 SlotIndex, uint64 Sequence)
{
	{
		FScopeLock ScopeLock(&Lock);
		FSlot& Slot = Slots[SlotIndex];
		if (Slot.Sequence == Sequence && (Slot.State == ESlot::Reserved || Slot.State == ESlot::Pending))
		{
			Slot.State = ESlot::Free;
			Stats.Dropped++;
		}
	}
	ReadyEvent->Trigger();
}

bool FWebGPUReadbackStream::ReleaseForRecovery()
{
	bool bHeld = false;
	{
		FScopeLock ScopeLock(&Lock);
		for (FSlot& Slot : Slots)
		{
			if (Slot.State == ESlot::Held)
			{
				// The consumer reads this memory: it stays mapped on the retired device until released
				bHeld = true;
				continue;
			}
			if (Slot.Buffer)
			{
				wgpuBufferRelease(Slot.Buffer);
				Slot.Buffer = nullptr;
			}
			Slot.bMapped = false;
			Slot.Mapped = nullptr;
			// Reserved slots keep their queued capture, which runs on the new device
			if (Slot.State == ESlot::Pending || Slot.State == ESlot::Ready)
			{
				Slot.State = ESlot::Free;
				Stats.Dropped++;
			}
		}
	}
	ReadyEvent->Trigger();
	return bHeld;
}

bool FWebGPUReadbackStream::HoldsRetiredMemory(uint32 CurrentGeneration) const
{
	FScopeLock ScopeLock(&Lock);
	return Slots.ContainsByPredicate([CurrentGeneration](const FSlot& Slot)
	{
		return Slot.State == ESlot::Held && Slot.Buffer && Slot.Generation != CurrentGeneration;
	});
}

// FWebGPUUploadStream

FWebGPUUploadStream::~FWebGPUUploadStream()
{
	TArray<WGPUBuffer> Buffers;
	for (const FSlot& Slot : Slots)
	{
		if (Slot.Buffer)
		{
			Buffers.Add(Slot.Buffer);
		}
	}
	ReleaseBuffersOnWorker(Context, MoveTemp(Buffers));
	if (SlotFreed)
	{
		FPlatformProcess::ReturnSynchEventToPool(SlotFreed);
		SlotFreed = nullptr;
	}
}

int32 FWebGPUUploadStream::AcquireSlot(double WaitSeconds)
{
	const double End = FPlatformTime::Seconds() + WaitSeconds;
	for (;;)
	{
		{
			FScopeLock ScopeLock(&Lock);
			for (int32 Index = 0; Index < Slots.Num(); Index++)
			{
				FSlot& Slot = Slots[Index];
				if (Slot.State == ESlot::Free && Slot.Mapped)
				{
					Slot.State = ESlot::Writing;
					Slot.Serial++;
					return Index;
				}
			}
		}
		if (WaitSeconds <= 0.0)
		{
			return INDEX_NONE;
		}
		TSharedPtr<FWebGPUContext> Ctx = Context.Pin();
		check(!Ctx || !Ctx->IsInWorkerThread());
		if (!WaitStep(SlotFreed, End))
		{
			return INDEX_NONE;
		}
	}
}

uint8* FWebGPUUploadStream::GetSlotData(int32 Slot) const
{
	FScopeLock ScopeLock(&Lock);
	return Slots.IsValidIndex(Slot) && Slots[Slot].State == ESlot::Writing ? Slots[Slot].Mapped : nullptr;
}

void FWebGPUUploadStream::Commit(int32 Slot, const TSharedRef<FWebGPUBuffer>& Destination, uint64 DestOffset, uint64 Size)
{
	TArray<FWebGPUCopyRegion> Regions;
	Regions.Add({ 0, DestOffset, Size });
	Commit(Slot, Destination, MoveTemp(Regions));
}

void FWebGPUUploadStream::Commit(int32 SlotIndex, const TSharedRef<FWebGPUBuffer>& Destination, TArray<FWebGPUCopyRegion>&& Regions)
{
	bool bValid = Regions.Num() > 0;
	for (const FWebGPUCopyRegion& Region : Regions)
	{
		bValid &= Region.Size > 0 && Region.SourceOffset % 4 == 0 && Region.DestOffset % 4 == 0 && Region.Size % 4 == 0
			&& Region.SourceOffset + Region.Size <= SlotBytes && Region.DestOffset + Region.Size <= Destination->GetSize();
	}
	if (!bValid)
	{
		UE_LOG(LogWebGPU, Error, TEXT("Upload stream '%s': copies into '%s' (%llu bytes) need non-empty multiples of 4 within the %llu-byte slot and the buffer"),
			*Label, *Destination->GetLabel(), Destination->GetSize(), SlotBytes);
		Cancel(SlotIndex);
		return;
	}

	TSharedPtr<FWebGPUContext> Ctx = Context.Pin();
	uint64 Serial = 0;
	{
		FScopeLock ScopeLock(&Lock);
		if (!Slots.IsValidIndex(SlotIndex) || Slots[SlotIndex].State != ESlot::Writing)
		{
			UE_LOG(LogWebGPU, Error, TEXT("Upload stream '%s': commit of slot %d, which was not acquired"), *Label, SlotIndex);
			return;
		}
		FSlot& Slot = Slots[SlotIndex];
		if (!Ctx)
		{
			Slot.State = ESlot::Free;
			return;
		}
		Slot.State = ESlot::Pending;
		Slot.CommitTime = FPlatformTime::Seconds();
		Serial = Slot.Serial;
		Stats.Committed++;
	}

	FWebGPUContext* CtxPtr = Ctx.Get();
	Ctx->Enqueue([CtxPtr, Self = AsShared(), SlotIndex, Serial, Destination, Regions = MoveTemp(Regions)]()
	{
		Self->CommitOnWorker(*CtxPtr, SlotIndex, Serial, *Destination, Regions);
	});
}

void FWebGPUUploadStream::Cancel(int32 SlotIndex)
{
	TSharedPtr<FWebGPUContext> Ctx = Context.Pin();
	uint64 Serial = 0;
	{
		FScopeLock ScopeLock(&Lock);
		if (!Slots.IsValidIndex(SlotIndex) || Slots[SlotIndex].State != ESlot::Writing)
		{
			return;
		}
		FSlot& Slot = Slots[SlotIndex];
		if (!Ctx)
		{
			Slot.State = ESlot::Free;
			return;
		}
		// The worker checks the slot still belongs to the current device before handing it out again
		Slot.State = ESlot::Pending;
		Serial = Slot.Serial;
	}
	FWebGPUContext* CtxPtr = Ctx.Get();
	Ctx->Enqueue([CtxPtr, Self = AsShared(), SlotIndex, Serial]()
	{
		Self->CancelOnWorker(*CtxPtr, SlotIndex, Serial);
	});
}

void FWebGPUUploadStream::Write(const TSharedRef<FWebGPUBuffer>& Destination, uint64 DestOffset, const void* Data, uint64 Size)
{
	TSharedPtr<FWebGPUContext> Ctx = Context.Pin();
	if (!Ctx || Size == 0)
	{
		return;
	}
	if (Size <= SlotBytes && Size % 4 == 0 && DestOffset % 4 == 0)
	{
		const int32 Slot = AcquireSlot(0.0);
		if (Slot != INDEX_NONE)
		{
			FMemory::Memcpy(GetSlotData(Slot), Data, Size);
			Commit(Slot, Destination, DestOffset, Size);
			return;
		}
	}
	CountFallback();
	Ctx->WriteBuffer(Destination, DestOffset, Data, Size);
}

void FWebGPUUploadStream::CountFallback()
{
	FScopeLock ScopeLock(&Lock);
	Stats.Fallbacks++;
}

FWebGPUStreamStats FWebGPUUploadStream::GetStats() const
{
	FScopeLock ScopeLock(&Lock);
	FWebGPUStreamStats Result = Stats;
	Result.InUse = 0;
	for (const FSlot& Slot : Slots)
	{
		Result.InUse += Slot.State == ESlot::Writing || Slot.State == ESlot::Pending ? 1 : 0;
	}
	return Result;
}

void FWebGPUUploadStream::CreateSlotBuffer(FWebGPUContext& Ctx, int32 SlotIndex)
{
	FSlot& Slot = Slots[SlotIndex];
	if (Slot.Buffer)
	{
		wgpuBufferRelease(Slot.Buffer);
		Slot.Buffer = nullptr;
	}

	uint8* Mapped = nullptr;
	if (Ctx.HasDevice())
	{
		Slot.Buffer = Ctx.CreateMappableBuffer(WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc, SlotBytes, true, "WebGPUCompute upload stream");
		Slot.Generation = Ctx.DeviceGeneration;
		Mapped = Slot.Buffer ? static_cast<uint8*>(wgpuBufferGetMappedRange(Slot.Buffer, 0, (size_t)SlotBytes)) : nullptr;
	}
	{
		FScopeLock ScopeLock(&Lock);
		Slot.Mapped = Mapped;
		Slot.State = Mapped ? ESlot::Free : ESlot::Unavailable;
	}
	if (Mapped)
	{
		SlotFreed->Trigger();
	}
}

void FWebGPUUploadStream::CommitOnWorker(FWebGPUContext& Ctx, int32 SlotIndex, uint64 Serial, FWebGPUBuffer& Destination, const TArray<FWebGPUCopyRegion>& Regions)
{
	FSlot& Slot = Slots[SlotIndex];
	{
		FScopeLock ScopeLock(&Lock);
		if (Slot.Serial != Serial || Slot.State != ESlot::Pending)
		{
			return;
		}
	}

	const bool bCurrentSlot = Ctx.HasDevice() && Slot.Buffer && Slot.Generation == Ctx.DeviceGeneration;
	if (!bCurrentSlot || !Destination.GetHandle())
	{
		// Written for a device that is gone (or there is nothing to copy into): the data is dropped
		{
			FScopeLock ScopeLock(&Lock);
			Stats.Dropped++;
		}
		if (bCurrentSlot)
		{
			FScopeLock ScopeLock(&Lock);
			Slot.State = ESlot::Free;
		}
		else
		{
			CreateSlotBuffer(Ctx, SlotIndex);
		}
		SlotFreed->Trigger();
		return;
	}

	wgpuBufferUnmap(Slot.Buffer);
	{
		FScopeLock ScopeLock(&Lock);
		Slot.Mapped = nullptr;
	}
	WGPUCommandEncoder Encoder = Ctx.GetEncoder();
	uint64 Bytes = 0;
	for (const FWebGPUCopyRegion& Region : Regions)
	{
		wgpuCommandEncoderCopyBufferToBuffer(Encoder, Slot.Buffer, Region.SourceOffset, Destination.GetHandle(), Region.DestOffset, Region.Size);
		Bytes += Region.Size;
	}
	Ctx.StatBytesWritten += Bytes;

	FWebGPUContext* CtxPtr = &Ctx;
	const uint32 Generation = Ctx.DeviceGeneration;
	WGPUBuffer Buffer = Slot.Buffer;
	const uint64 MapBytes = SlotBytes;
	Ctx.AfterSubmit([Self = AsShared(), CtxPtr, SlotIndex, Serial, Generation, Buffer, MapBytes](bool bSubmitted)
	{
		if (!bSubmitted)
		{
			Self->FinishRemap(*CtxPtr, SlotIndex, Serial, Generation, false);
			return;
		}
		// Writable again once the GPU has consumed the copy
		CtxPtr->MapAsync(Buffer, WGPUMapMode_Write, 0, MapBytes, [Self, CtxPtr, SlotIndex, Serial, Generation](bool bSuccess)
		{
			Self->FinishRemap(*CtxPtr, SlotIndex, Serial, Generation, bSuccess);
		});
	});
}

void FWebGPUUploadStream::CancelOnWorker(FWebGPUContext& Ctx, int32 SlotIndex, uint64 Serial)
{
	FSlot& Slot = Slots[SlotIndex];
	{
		FScopeLock ScopeLock(&Lock);
		if (Slot.Serial != Serial || Slot.State != ESlot::Pending)
		{
			return;
		}
		if (Ctx.HasDevice() && Slot.Buffer && Slot.Generation == Ctx.DeviceGeneration)
		{
			Slot.State = ESlot::Free;
			SlotFreed->Trigger();
			return;
		}
	}
	CreateSlotBuffer(Ctx, SlotIndex);
}

void FWebGPUUploadStream::FinishRemap(FWebGPUContext& Ctx, int32 SlotIndex, uint64 Serial, uint32 Generation, bool bSuccess)
{
	FSlot& Slot = Slots[SlotIndex];
	{
		FScopeLock ScopeLock(&Lock);
		if (Slot.Serial != Serial || Slot.State != ESlot::Pending || Slot.Generation != Generation)
		{
			// Recycled meanwhile (device recovery)
			return;
		}
	}
	if (Ctx.bReleasingDevice || Ctx.bStopping)
	{
		FScopeLock ScopeLock(&Lock);
		Slot.State = ESlot::Unavailable;
		return;
	}
	if (!bSuccess || Generation != Ctx.DeviceGeneration)
	{
		CreateSlotBuffer(Ctx, SlotIndex);
		return;
	}

	uint8* Mapped = static_cast<uint8*>(wgpuBufferGetMappedRange(Slot.Buffer, 0, (size_t)SlotBytes));
	{
		FScopeLock ScopeLock(&Lock);
		Slot.Mapped = Mapped;
		Slot.State = Mapped ? ESlot::Free : ESlot::Unavailable;
		UpdateLatency(Stats.LatencyMs, (FPlatformTime::Seconds() - Slot.CommitTime) * 1000.0);
	}
	if (Mapped)
	{
		SlotFreed->Trigger();
	}
}

bool FWebGPUUploadStream::ReleaseForRecovery()
{
	bool bHeld = false;
	FScopeLock ScopeLock(&Lock);
	for (FSlot& Slot : Slots)
	{
		if (Slot.State == ESlot::Writing)
		{
			// The producer writes this memory: it stays mapped on the retired device until committed or cancelled
			bHeld = true;
			continue;
		}
		if (Slot.Buffer)
		{
			wgpuBufferRelease(Slot.Buffer);
			Slot.Buffer = nullptr;
		}
		Slot.Mapped = nullptr;
		Slot.State = ESlot::Unavailable;
		// Invalidates queued commits and remaps of this slot
		Slot.Serial++;
	}
	return bHeld;
}

void FWebGPUUploadStream::RecreateAfterRecovery(FWebGPUContext& Ctx)
{
	for (int32 Index = 0; Index < Slots.Num(); Index++)
	{
		bool bUnavailable = false;
		{
			FScopeLock ScopeLock(&Lock);
			bUnavailable = Slots[Index].State == ESlot::Unavailable;
		}
		if (bUnavailable)
		{
			CreateSlotBuffer(Ctx, Index);
		}
	}
}

bool FWebGPUUploadStream::HoldsRetiredMemory(uint32 CurrentGeneration) const
{
	FScopeLock ScopeLock(&Lock);
	return Slots.ContainsByPredicate([CurrentGeneration](const FSlot& Slot)
	{
		return Slot.State == ESlot::Writing && Slot.Buffer && Slot.Generation != CurrentGeneration;
	});
}

// FWebGPUStructMirror

TSharedPtr<FWebGPUStructMirror> FWebGPUStructMirror::Create(const TSharedRef<FWebGPUContext>& Context, const UScriptStruct* Struct, int32 Count,
	EWebGPUBufferUsage Usage, const FString& Label)
{
	if (!Struct || Count <= 0)
	{
		UE_LOG(LogWebGPU, Error, TEXT("FWebGPUStructMirror: needs a struct and a positive count (got %s, %d)"), Struct ? *Struct->GetName() : TEXT("null"), Count);
		return nullptr;
	}
	TSharedPtr<FWebGPUBuffer> Buffer = Context->CreateStructBuffer(Struct, Count, Usage, Label);
	if (!Buffer)
	{
		return nullptr;
	}

	TSharedPtr<FWebGPUStructMirror> Mirror = MakeShareable(new FWebGPUStructMirror(Buffer.ToSharedRef()));
	Mirror->Context = Context;
	Mirror->Struct.Reset(const_cast<UScriptStruct*>(Struct));
	Mirror->Layout = Buffer->GetElementLayout();
	Mirror->Count = Count;
	Mirror->Elements = static_cast<uint8*>(FMemory::Malloc((SIZE_T)Mirror->Layout->GetSourceStride() * Count, FMath::Max(Struct->GetMinAlignment(), 16)));
	Struct->InitializeStruct(Mirror->Elements, Count);

	// The GPU copy starts zeroed: the first Sync uploads everything
	Mirror->Dirty.Init(true, Count);
	Mirror->DirtyCount = Count;
	Mirror->SyncedGeneration = Context->GetDeviceGeneration();

	// Upload slots of up to 4 MB; a bigger sync spills into the copying path
	const uint64 Stride = Mirror->Layout->GetStride();
	const uint64 SlotBytes = FMath::Min<uint64>(Buffer->GetSize(), FMath::Max<uint64>(4ull << 20, Stride));
	Mirror->Upload = Context->CreateUploadStream(SlotBytes, 3, Buffer->GetLabel() + TEXT(" upload"));

	// Scattered edits: a kernel writes uploaded elements to their indices (needs a storage buffer)
	if (EnumHasAnyFlags(Usage, EWebGPUBufferUsage::Storage))
	{
		Mirror->ScatterKernel = Context->CreateKernel(ScatterSource, TEXT("main"), Buffer->GetLabel() + TEXT(" scatter"), nullptr, EWebGPUCallbackThread::WorkerThread);
		Context->SetBufferByName(Mirror->ScatterKernel.ToSharedRef(), TEXT("dst"), Buffer);
	}
	return Mirror;
}

FWebGPUStructMirror::~FWebGPUStructMirror()
{
	if (Elements)
	{
		if (Struct)
		{
			Struct->DestroyStruct(Elements, Count);
		}
		FMemory::Free(Elements);
		Elements = nullptr;
	}
}

const void* FWebGPUStructMirror::GetElement(int32 Index) const
{
	check(Index >= 0 && Index < Count);
	return Elements + (SIZE_T)Index * Layout->GetSourceStride();
}

void* FWebGPUStructMirror::EditElement(int32 Index)
{
	check(Index >= 0 && Index < Count);
	if (!Dirty[Index])
	{
		Dirty[Index] = true;
		DirtyCount++;
	}
	return Elements + (SIZE_T)Index * Layout->GetSourceStride();
}

void FWebGPUStructMirror::MarkDirty(int32 First, int32 Num)
{
	const int32 Begin = FMath::Clamp(First, 0, Count);
	const int32 End = FMath::Clamp(First + Num, Begin, Count);
	for (int32 Index = Begin; Index < End; Index++)
	{
		if (!Dirty[Index])
		{
			Dirty[Index] = true;
			DirtyCount++;
		}
	}
}

void FWebGPUStructMirror::MarkAllDirty()
{
	Dirty.Init(true, Count);
	DirtyCount = Count;
}

uint64 FWebGPUStructMirror::Sync()
{
	LastSync = FWebGPUMirrorSyncStats();
	TSharedPtr<FWebGPUContext> Ctx = Context.Pin();
	if (!Ctx)
	{
		return 0;
	}

	// After a device loss the buffer came back zeroed: upload everything
	const uint32 Generation = Ctx->GetDeviceGeneration();
	if (Generation != SyncedGeneration)
	{
		SyncedGeneration = Generation;
		MarkAllDirty();
	}
	if (DirtyCount == 0)
	{
		return 0;
	}
	LastSync.Elements = DirtyCount;

	// Runs of dirty elements [First, End), merged across small clean gaps
	const int32 MaxGap = (int32)(MaxGapBytes / Layout->GetStride());
	TArray<TPair<int32, int32>> Ranges;
	for (TConstSetBitIterator<> It(Dirty); It; ++It)
	{
		const int32 Index = It.GetIndex();
		if (Ranges.Num() > 0 && Index - Ranges.Last().Value <= MaxGap)
		{
			Ranges.Last().Value = Index + 1;
		}
		else
		{
			Ranges.Emplace(Index, Index + 1);
		}
	}

	if (Ranges.Num() > FMath::Max(MaxCopyRegions, 1) && ScatterKernel && ScatterKernel->IsReady())
	{
		SyncScatter(*Ctx);
	}
	else
	{
		SyncCopies(*Ctx, Ranges, MaxGap);
	}

	Dirty.Init(false, Count);
	DirtyCount = 0;
	return LastSync.Bytes;
}

void FWebGPUStructMirror::SyncCopies(FWebGPUContext& Ctx, TArray<TPair<int32, int32>>& Ranges, int32 MaxGap)
{
	// Too many copies: widen the gap until they fit (uploading some clean elements beats many tiny copies)
	while (Ranges.Num() > FMath::Max(MaxCopyRegions, 1))
	{
		MaxGap = MaxGap * 2 + 1;
		int32 Out = 0;
		for (int32 In = 1; In < Ranges.Num(); In++)
		{
			if (Ranges[In].Key - Ranges[Out].Value <= MaxGap)
			{
				Ranges[Out].Value = Ranges[In].Value;
			}
			else
			{
				Ranges[++Out] = Ranges[In];
			}
		}
		Ranges.SetNum(Out + 1, EAllowShrinking::No);
	}

	// Pack straight into upload slots: as many ranges (or parts of one) as fit per slot
	const uint32 Stride = Layout->GetStride();
	const uint32 SourceStride = Layout->GetSourceStride();
	int32 RangeIndex = 0;
	int32 Next = Ranges[0].Key;
	const int32 SlotElements = (int32)(Upload->GetSlotBytes() / Stride);
	while (RangeIndex < Ranges.Num() && SlotElements > 0)
	{
		const int32 Slot = Upload->AcquireSlot(0.0);
		if (Slot == INDEX_NONE)
		{
			break;
		}
		uint8* SlotData = Upload->GetSlotData(Slot);
		int32 Used = 0;
		TArray<FWebGPUCopyRegion> Regions;
		while (RangeIndex < Ranges.Num() && Used < SlotElements)
		{
			const int32 First = Next;
			const int32 Take = FMath::Min(Ranges[RangeIndex].Value - First, SlotElements - Used);
			Layout->Pack(Elements + (SIZE_T)First * SourceStride, Take, SlotData + (SIZE_T)Used * Stride);
			Regions.Add({ (uint64)Used * Stride, (uint64)First * Stride, (uint64)Take * Stride });
			Used += Take;
			Next = First + Take;
			if (Next >= Ranges[RangeIndex].Value && ++RangeIndex < Ranges.Num())
			{
				Next = Ranges[RangeIndex].Key;
			}
		}
		LastSync.Bytes += (uint64)Used * Stride;
		LastSync.Copies += Regions.Num();
		Upload->Commit(Slot, Buffer, MoveTemp(Regions));
	}

	// Out of slots (the GPU is behind): the rest takes the copying path
	while (RangeIndex < Ranges.Num())
	{
		const int32 First = Next;
		const int32 Num = Ranges[RangeIndex].Value - First;
		Ctx.WriteBuffer(Buffer, (uint64)First * Stride, Layout->Pack(Elements + (SIZE_T)First * SourceStride, Num));
		Upload->CountFallback();
		LastSync.Bytes += (uint64)Num * Stride;
		LastSync.Copies++;
		LastSync.bFallback = true;
		if (++RangeIndex < Ranges.Num())
		{
			Next = Ranges[RangeIndex].Key;
		}
	}
}

void FWebGPUStructMirror::SyncScatter(FWebGPUContext& Ctx)
{
	// Payload per chunk: [count, words per element, indices[count], packed elements[count]] as u32 words
	const uint32 Stride = Layout->GetStride();
	const uint32 SourceStride = Layout->GetSourceStride();
	const uint32 Words = Stride / 4;
	const uint64 PerElement = 4 + (uint64)Stride;
	const int32 ChunkElements = (int32)FMath::Max<uint64>((Upload->GetSlotBytes() - 8) / PerElement, 1);
	const uint64 ScratchBytes = 8 + ChunkElements * PerElement;
	if (!ScatterScratch || ScatterScratch->GetSize() < ScratchBytes)
	{
		ScatterScratch = Ctx.CreateBuffer(ScratchBytes, EWebGPUBufferUsage::Storage, Buffer->GetLabel() + TEXT(" scatter"));
		Ctx.SetBufferByName(ScatterKernel.ToSharedRef(), TEXT("src"), ScatterScratch);
	}

	TArray<uint8> Fallback;
	TConstSetBitIterator<> It(Dirty);
	int32 Remaining = DirtyCount;
	while (Remaining > 0)
	{
		const int32 Chunk = FMath::Min(Remaining, ChunkElements);
		const uint64 Bytes = 8 + Chunk * PerElement;

		const int32 Slot = Upload->AcquireSlot(0.0);
		uint8* Data = nullptr;
		if (Slot != INDEX_NONE)
		{
			Data = Upload->GetSlotData(Slot);
		}
		else
		{
			Fallback.SetNumUninitialized((int32)Bytes);
			Data = Fallback.GetData();
		}

		uint32* Header = reinterpret_cast<uint32*>(Data);
		Header[0] = (uint32)Chunk;
		Header[1] = Words;
		uint32* Indices = Header + 2;
		uint8* Values = Data + 8 + (SIZE_T)Chunk * 4;
		for (int32 Item = 0; Item < Chunk; Item++, ++It)
		{
			const int32 Index = It.GetIndex();
			Indices[Item] = (uint32)Index;
			Layout->Pack(Elements + (SIZE_T)Index * SourceStride, 1, Values + (SIZE_T)Item * Stride);
		}

		if (Slot != INDEX_NONE)
		{
			Upload->Commit(Slot, ScatterScratch.ToSharedRef(), 0, Bytes);
		}
		else
		{
			Ctx.WriteBuffer(ScatterScratch.ToSharedRef(), 0, MoveTemp(Fallback));
			Upload->CountFallback();
			LastSync.bFallback = true;
		}
		// Grid-stride loop in the kernel: a bounded number of workgroups covers any chunk
		const uint32 Groups = (uint32)FMath::Clamp<uint64>(FMath::DivideAndRoundUp<uint64>((uint64)Chunk * Words, 64), 1, 4096);
		Ctx.Dispatch(ScatterKernel.ToSharedRef(), Groups);

		LastSync.Bytes += Bytes;
		LastSync.Scatters++;
		Remaining -= Chunk;
	}
}
