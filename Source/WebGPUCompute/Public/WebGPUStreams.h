// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/StrongObjectPtr.h"
#include "webgpu/webgpu.h"
#include "WebGPUContext.h"
#include "WebGPUStructLayout.h"
#include "WebGPUTypes.h"
#include <atomic>
#include "WebGPUStreams.generated.h"

/** How a readback stream behaves when the consumer is slower than the producer. */
UENUM(BlueprintType)
enum class EWebGPUStreamPolicy : uint8
{
	/**
	 * Live data: Acquire returns the newest frame and skips older ones, and a capture with every slot busy
	 * recycles the oldest unread frame. Nothing ever waits; captures drop only while every slot is on the GPU.
	 */
	Latest,
	/** Every captured frame is delivered, in order. A capture with no free slot is dropped (and counted). */
	Queue,
	/**
	 * Like Queue, but Capture blocks until its frame is back on the CPU, so it can be acquired in the same
	 * tick. Deterministic, at the cost of a GPU round trip (typically ~1 ms) on the capturing thread.
	 */
	Lockstep,
};

/** Counters of a readback or upload stream. */
USTRUCT(BlueprintType)
struct WEBGPUCOMPUTE_API FWebGPUStreamStats
{
	GENERATED_BODY()

	/** Frames sent to the GPU: captures (readback) or slot uploads. */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU|Stream")
	int64 Committed = 0;

	/** Captures that found no slot or failed (device lost); uploads that failed. */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU|Stream")
	int64 Dropped = 0;

	/** Readback: frames the consumer acquired. */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU|Stream")
	int64 Delivered = 0;

	/** Readback (Latest): frames superseded before anyone acquired them. */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU|Stream")
	int64 Skipped = 0;

	/** Upload: writes that found no free slot (or were bigger than one) and took the copying path instead. */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU|Stream")
	int64 Fallbacks = 0;

	/** Slots currently away from the producer: on the GPU, waiting to be read, or held. */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU|Stream")
	int32 InUse = 0;

	/** Average time from commit until the data is on the CPU (readback) or the slot is writable again (upload). */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU|Stream")
	float LatencyMs = 0.f;
};

class FWebGPUReadbackStream;

/**
 * A frame from an FWebGPUReadbackStream. Its data is the mapped GPU readback memory itself (no copy), valid
 * until Release() or the handle's destruction. Move-only. Hold it briefly: the slot is unavailable meanwhile.
 */
class WEBGPUCOMPUTE_API FWebGPUReadbackFrame
{
public:
	FWebGPUReadbackFrame() = default;
	FWebGPUReadbackFrame(FWebGPUReadbackFrame&& Other);
	FWebGPUReadbackFrame& operator=(FWebGPUReadbackFrame&& Other);
	FWebGPUReadbackFrame(const FWebGPUReadbackFrame&) = delete;
	FWebGPUReadbackFrame& operator=(const FWebGPUReadbackFrame&) = delete;
	~FWebGPUReadbackFrame();

	bool IsValid() const { return Data != nullptr; }
	const uint8* GetData() const { return Data; }
	uint64 GetSize() const { return Size; }

	/** The capture's number in its stream, from 1. */
	uint64 GetSequence() const { return Sequence; }
	/** GFrameCounter when Capture was called. */
	uint64 GetCaptureFrame() const { return CaptureFrame; }
	/** FPlatformTime::Seconds() at Capture and when the data reached the CPU. */
	double GetCaptureTime() const { return CaptureTime; }
	double GetReadyTime() const { return ReadyTime; }

	/** Texture captures: size, bytes per row of pixels, and bytes between rows in GetData() (256-byte aligned). 0 for buffers. */
	uint32 GetWidth() const { return Width; }
	uint32 GetHeight() const { return Height; }
	uint32 GetRowBytes() const { return RowBytes; }
	uint32 GetRowPitch() const { return RowPitch; }

	/** The data with texture rows tightly packed (buffers: as is). */
	void CopyTo(TArray<uint8>& OutBytes) const;

	/** The bytes as an array of T (plain types: float, uint32, FVector4f, ...). */
	template <typename T>
	TConstArrayView<T> View() const
	{
		return TConstArrayView<T>(reinterpret_cast<const T*>(Data), (int32)(Size / sizeof(T)));
	}

	/** Unpacks the frame as UStructs in their WGSL storage layout. The first use of a struct type must be on the game thread. */
	template <typename T>
	bool UnpackStructs(TArray<T>& OutItems) const
	{
		TSharedPtr<const FWebGPUStructLayout> Layout = FWebGPUStructLayout::Get(T::StaticStruct());
		if (!Layout || !Data)
		{
			return false;
		}
		const int32 Count = (int32)(Size / Layout->GetStride());
		OutItems.SetNum(Count);
		Layout->Unpack(Data, Count, OutItems.GetData());
		return true;
	}

	/** Gives the slot back to the stream. */
	void Release();

private:
	friend class FWebGPUReadbackStream;

	TSharedPtr<FWebGPUReadbackStream> Stream;
	int32 Slot = INDEX_NONE;
	const uint8* Data = nullptr;
	uint64 Size = 0;
	uint64 Sequence = 0;
	uint64 CaptureFrame = 0;
	double CaptureTime = 0.0;
	double ReadyTime = 0.0;
	uint32 Width = 0;
	uint32 Height = 0;
	uint32 RowBytes = 0;
	uint32 RowPitch = 0;
};

/**
 * Per-frame GPU -> CPU results: a ring of readback slots, each a credit (the PythonCLI stream model).
 *
 * Capture copies a buffer range into a free slot after all earlier commands (it joins the current batch, so
 * dispatch + capture is one submit); the slot comes back mapped on the CPU, and the consumer Acquires it,
 * reads the memory in place, and releases it. Create with FWebGPUContext::CreateReadbackStream.
 * Capture from any thread but the worker; acquire from one consumer thread (typically the game thread).
 */
class WEBGPUCOMPUTE_API FWebGPUReadbackStream : public TSharedFromThis<FWebGPUReadbackStream>
{
public:
	~FWebGPUReadbackStream();

	/**
	 * Copies [Offset, Offset + Size) of Source (WGPU_WHOLE_SIZE: to its end; at most GetSlotBytes()) into a slot.
	 * False when the capture was dropped (see EWebGPUStreamPolicy) or failed. Never call from the worker thread.
	 */
	bool Capture(const TSharedRef<FWebGPUBuffer>& Source, uint64 Offset = 0, uint64 Size = WGPU_WHOLE_SIZE);

	/**
	 * Copies a whole texture into a slot. Its rows are 256-byte aligned there (see the frame's GetRowPitch,
	 * or CopyTo for tight rows). Shared textures must be inside BeginTextureAccess / EndTextureAccess.
	 */
	bool CaptureTexture(const TSharedRef<FWebGPUTexture>& Source);

	/** Latest: the newest ready frame (older ready frames are skipped). Queue and Lockstep: the oldest one, in order. */
	bool Acquire(FWebGPUReadbackFrame& OutFrame);

	/** Blocks until a frame can be acquired or the timeout passes. Never call from the worker thread. */
	bool WaitForFrame(double TimeoutSeconds = 1.0);

	/** Frames ready to acquire. */
	int32 GetReadyCount() const;

	FWebGPUStreamStats GetStats() const;
	uint64 GetSlotBytes() const { return SlotBytes; }
	int32 GetSlotCount() const { return SlotCount; }
	EWebGPUStreamPolicy GetPolicy() const { return Policy; }
	const FString& GetLabel() const { return Label; }

	/** Lockstep: how long Capture waits for its frame before giving up. */
	double LockstepTimeoutSeconds = 5.0;

private:
	friend class FWebGPUContext;
	friend class FWebGPUReadbackFrame;

	enum class ESlot : uint8
	{
		Free,
		/** Claimed by Capture; the copy is queued on the worker. */
		Reserved,
		/** Copied and waiting for the map to finish. */
		Pending,
		/** Mapped, waiting for the consumer. */
		Ready,
		/** Acquired by the consumer. */
		Held,
	};

	struct FSlot
	{
		ESlot State = ESlot::Free;
		uint64 Sequence = 0;
		uint64 Size = 0;
		uint64 CaptureFrame = 0;
		double CaptureTime = 0.0;
		double ReadyTime = 0.0;
		const uint8* Mapped = nullptr;
		uint32 Width = 0;
		uint32 Height = 0;
		uint32 RowBytes = 0;
		uint32 RowPitch = 0;

		// Worker thread only
		WGPUBuffer Buffer = nullptr;
		uint32 Generation = 0;
		bool bMapped = false;
	};

	FWebGPUReadbackStream() = default;
	/** Claims a slot for a capture, or INDEX_NONE (counted as dropped). */
	int32 ReserveSlot(uint64 Size, uint32 Width, uint32 Height, uint32 RowBytes, uint32 RowPitch, uint64& OutSequence);
	bool FinishCapture(FWebGPUContext& Ctx, int32 Slot, uint64 Sequence, TUniqueFunction<void()>&& Command);
	void ReleaseSlot(int32 Slot, uint64 Sequence);
	bool WaitForSequence(int32 Slot, uint64 Sequence, double TimeoutSeconds);

	// Worker thread
	/** The slot's buffer, of the current device and unmapped, or null (the slot failed). */
	WGPUBuffer PrepareSlot(FWebGPUContext& Ctx, int32 Slot, uint64 Sequence);
	void SubmitSlot(FWebGPUContext& Ctx, int32 Slot, uint64 Sequence, uint64 Size);
	void CaptureOnWorker(FWebGPUContext& Ctx, int32 Slot, uint64 Sequence, FWebGPUBuffer& Source, uint64 Offset, uint64 Size);
	void CaptureTextureOnWorker(FWebGPUContext& Ctx, int32 Slot, uint64 Sequence, FWebGPUTexture& Source);
	void FinishMap(FWebGPUContext& Ctx, int32 Slot, uint64 Sequence, uint32 Generation, bool bSuccess);
	void FailSlot(int32 Slot, uint64 Sequence);
	/** Releases the slots' buffers; true if the consumer still holds mapped memory of the lost device. */
	bool ReleaseForRecovery();
	bool HoldsRetiredMemory(uint32 CurrentGeneration) const;

	TWeakPtr<FWebGPUContext> Context;
	FString Label;
	uint64 SlotBytes = 0;
	int32 SlotCount = 0;
	EWebGPUStreamPolicy Policy = EWebGPUStreamPolicy::Latest;

	mutable FCriticalSection Lock;
	TArray<FSlot> Slots;
	FEvent* ReadyEvent = nullptr;
	uint64 NextSequence = 0;
	FWebGPUStreamStats Stats;
};

/** One copy from an upload slot into a buffer. Offsets and sizes are multiples of 4. */
struct FWebGPUCopyRegion
{
	uint64 SourceOffset = 0;
	uint64 DestOffset = 0;
	uint64 Size = 0;
};

/**
 * Per-frame CPU -> GPU data without intermediate copies: a ring of slots that stay mapped for writing.
 *
 * AcquireSlot hands out mapped upload memory, the producer writes (or packs structs) straight into it, and
 * Commit copies it into a buffer in order with other commands (in the current batch). The slot is writable
 * again once the GPU has consumed it. Create with FWebGPUContext::CreateUploadStream. Any thread but the worker.
 */
class WEBGPUCOMPUTE_API FWebGPUUploadStream : public TSharedFromThis<FWebGPUUploadStream>
{
public:
	~FWebGPUUploadStream();

	/** A slot mapped for writing (GetSlotBytes() long), or INDEX_NONE if none frees up within WaitSeconds. */
	int32 AcquireSlot(double WaitSeconds = 0.0);

	/** The slot's memory between AcquireSlot and Commit/Cancel. Write-combined: write it sequentially, never read it. */
	uint8* GetSlotData(int32 Slot) const;

	/** Copies Size bytes from the start of the slot to Destination at DestOffset. */
	void Commit(int32 Slot, const TSharedRef<FWebGPUBuffer>& Destination, uint64 DestOffset, uint64 Size);

	/** Copies several regions of the slot (e.g. dirty ranges) into Destination. */
	void Commit(int32 Slot, const TSharedRef<FWebGPUBuffer>& Destination, TArray<FWebGPUCopyRegion>&& Regions);

	/** Returns the slot unused. */
	void Cancel(int32 Slot);

	/** AcquireSlot + memcpy + Commit; falls back to FWebGPUContext::WriteBuffer (a copy) when no slot is free or Size exceeds a slot. */
	void Write(const TSharedRef<FWebGPUBuffer>& Destination, uint64 DestOffset, const void* Data, uint64 Size);

	FWebGPUStreamStats GetStats() const;
	uint64 GetSlotBytes() const { return SlotBytes; }
	int32 GetSlotCount() const { return SlotCount; }
	const FString& GetLabel() const { return Label; }

private:
	friend class FWebGPUContext;
	friend class FWebGPUStructMirror;

	enum class ESlot : uint8
	{
		/** No mapped memory yet (being created or remapped). */
		Unavailable,
		Free,
		Writing,
		/** Committed: queued, on the GPU, or being remapped. */
		Pending,
	};

	struct FSlot
	{
		ESlot State = ESlot::Unavailable;
		uint8* Mapped = nullptr;
		uint64 Serial = 0;
		double CommitTime = 0.0;

		// Worker thread only
		WGPUBuffer Buffer = nullptr;
		uint32 Generation = 0;
	};

	FWebGPUUploadStream() = default;
	void CountFallback();

	// Worker thread
	void CreateSlotBuffer(FWebGPUContext& Ctx, int32 Slot);
	void CommitOnWorker(FWebGPUContext& Ctx, int32 Slot, uint64 Serial, FWebGPUBuffer& Destination, const TArray<FWebGPUCopyRegion>& Regions);
	void CancelOnWorker(FWebGPUContext& Ctx, int32 Slot, uint64 Serial);
	void FinishRemap(FWebGPUContext& Ctx, int32 Slot, uint64 Serial, uint32 Generation, bool bSuccess);
	bool ReleaseForRecovery();
	void RecreateAfterRecovery(FWebGPUContext& Ctx);
	bool HoldsRetiredMemory(uint32 CurrentGeneration) const;

	TWeakPtr<FWebGPUContext> Context;
	FString Label;
	uint64 SlotBytes = 0;
	int32 SlotCount = 0;

	mutable FCriticalSection Lock;
	TArray<FSlot> Slots;
	FEvent* SlotFreed = nullptr;
	FWebGPUStreamStats Stats;
};

/** What the last FWebGPUStructMirror::Sync did. */
struct FWebGPUMirrorSyncStats
{
	/** Bytes uploaded (element data, plus indices when scattering). */
	uint64 Bytes = 0;
	/** Dirty elements written. */
	int32 Elements = 0;
	/** Buffer copies (copy mode). */
	int32 Copies = 0;
	/** Scatter dispatches (scatter mode). */
	int32 Scatters = 0;
	/** Some data took the copying path because no upload slot was free. */
	bool bFallback = false;
};

/**
 * A struct array resident on the GPU with a CPU copy: edit elements on the CPU and Sync() uploads only what
 * changed. For data the CPU owns that kernels read every frame: world grids, nav data, agent goals and
 * parameters. (Keep state the GPU writes, such as integrated positions, in a separate buffer: a Sync would
 * overwrite it with the CPU copy.)
 *
 * Dirty runs are merged across small clean gaps and copied straight from upload slots. When the edits are
 * too scattered for that (more than MaxCopyRegions runs), only the dirty elements and their indices are
 * uploaded and a small kernel scatters them into place. After a device loss the next Sync re-uploads everything.
 * Not thread-safe: use one mirror from one thread at a time.
 */
class WEBGPUCOMPUTE_API FWebGPUStructMirror : public TSharedFromThis<FWebGPUStructMirror>
{
public:
	/** Game thread (reads UObject reflection). Null if the struct has no WGSL layout (see FWebGPUStructLayout). */
	static TSharedPtr<FWebGPUStructMirror> Create(const TSharedRef<FWebGPUContext>& Context, const UScriptStruct* Struct, int32 Count,
		EWebGPUBufferUsage Usage = EWebGPUBufferUsage::Storage, const FString& Label = FString());
	~FWebGPUStructMirror();

	int32 Num() const { return Count; }
	const UScriptStruct* GetStruct() const { return Struct.Get(); }
	const FWebGPUStructLayout& GetLayout() const { return *Layout; }
	const TSharedRef<FWebGPUBuffer>& GetBuffer() const { return Buffer; }

	const void* GetElement(int32 Index) const;
	/** Marks the element dirty. */
	void* EditElement(int32 Index);
	void MarkDirty(int32 First, int32 Num = 1);
	void MarkAllDirty();
	int32 GetDirtyCount() const { return DirtyCount; }

	/** Uploads the dirty elements and clears them. Returns the bytes uploaded. */
	uint64 Sync();

	const FWebGPUMirrorSyncStats& GetLastSync() const { return LastSync; }

	/** True once the scatter kernel compiled (until then scattered edits are merged into copies). */
	bool IsScatterReady() const { return ScatterKernel && ScatterKernel->IsReady(); }

	/** Clean gaps up to this size are uploaded with their neighbours (fewer, larger copies). */
	uint32 MaxGapBytes = 1024;
	/** More dirty runs than this are scattered by a kernel (or, without Storage usage, merged into fewer copies). */
	int32 MaxCopyRegions = 256;

private:
	FWebGPUStructMirror(const TSharedRef<FWebGPUBuffer>& InBuffer) : Buffer(InBuffer) {}

	void SyncCopies(FWebGPUContext& Ctx, TArray<TPair<int32, int32>>& Ranges, int32 MaxGap);
	void SyncScatter(FWebGPUContext& Ctx);

	TWeakPtr<FWebGPUContext> Context;
	TStrongObjectPtr<UScriptStruct> Struct;
	TSharedPtr<const FWebGPUStructLayout> Layout;
	TSharedRef<FWebGPUBuffer> Buffer;
	TSharedPtr<FWebGPUUploadStream> Upload;
	TSharedPtr<FWebGPUKernel> ScatterKernel;
	TSharedPtr<FWebGPUBuffer> ScatterScratch;
	FWebGPUMirrorSyncStats LastSync;
	int32 Count = 0;
	uint8* Elements = nullptr;
	TBitArray<> Dirty;
	int32 DirtyCount = 0;
	uint32 SyncedGeneration = 0;
};

/** Typed FWebGPUStructMirror: Edit(i) returns T& and marks it dirty. */
template <typename T>
class TWebGPUStructMirror
{
public:
	TWebGPUStructMirror() = default;

	static TWebGPUStructMirror Create(const TSharedRef<FWebGPUContext>& Context, int32 Count,
		EWebGPUBufferUsage Usage = EWebGPUBufferUsage::Storage, const FString& Label = FString())
	{
		TWebGPUStructMirror Result;
		Result.Mirror = FWebGPUStructMirror::Create(Context, T::StaticStruct(), Count, Usage, Label);
		return Result;
	}

	bool IsValid() const { return Mirror.IsValid(); }
	int32 Num() const { return Mirror ? Mirror->Num() : 0; }
	const T& operator[](int32 Index) const { return *static_cast<const T*>(Mirror->GetElement(Index)); }
	T& Edit(int32 Index) { return *static_cast<T*>(Mirror->EditElement(Index)); }
	void MarkDirty(int32 First, int32 Num = 1) { Mirror->MarkDirty(First, Num); }
	void MarkAllDirty() { Mirror->MarkAllDirty(); }
	uint64 Sync() { return Mirror->Sync(); }
	const TSharedRef<FWebGPUBuffer>& GetBuffer() const { return Mirror->GetBuffer(); }
	const TSharedPtr<FWebGPUStructMirror>& Get() const { return Mirror; }

private:
	TSharedPtr<FWebGPUStructMirror> Mirror;
};
