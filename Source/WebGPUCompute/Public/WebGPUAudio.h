// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "WebGPUStreams.h"
#include <atomic>
#include "WebGPUAudio.generated.h"

class ISubmixBufferListener;

/** The block a generator kernel fills; in WGSL "WebGPUAudioBlock" (see FWebGPUAudioGenerator). */
USTRUCT()
struct FWebGPUAudioBlock
{
	GENERATED_BODY()

	/** Index of the block's first frame since the generator started (wraps at 2^32). */
	UPROPERTY()
	uint32 FirstFrame = 0;

	/** Frames in the block. */
	UPROPERTY()
	uint32 Frames = 0;

	/** Interleaved channels per frame. */
	UPROPERTY()
	uint32 Channels = 0;

	UPROPERTY()
	float SampleRate = 0.f;
};

/** Where the newest audio is in an FWebGPUAudioInput ring; in WGSL "WebGPUAudioRing". */
USTRUCT()
struct FWebGPUAudioRing
{
	GENERATED_BODY()

	/** Frames written so far (wraps at 2^32). The newest frame is at (FramesWritten - 1) % RingFrames. */
	UPROPERTY()
	uint32 FramesWritten = 0;

	UPROPERTY()
	uint32 RingFrames = 0;

	/** Interleaved channels per frame. */
	UPROPERTY()
	uint32 Channels = 0;

	UPROPERTY()
	float SampleRate = 0.f;
};

/** Counters of an FWebGPUAudioGenerator. */
USTRUCT(BlueprintType)
struct WEBGPUCOMPUTE_API FWebGPUAudioStats
{
	GENERATED_BODY()

	/** Blocks the kernel generated and the audio thread received. */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU|Audio")
	int64 Blocks = 0;

	/** Samples played as silence because the GPU had not delivered them yet (after the first block arrived). */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU|Audio")
	int64 UnderrunSamples = 0;

	/** Average time from queueing a block until it was back on the CPU. */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU|Audio")
	float RoundTripMs = 0.f;

	/** Generate calls (audio callbacks) and the samples they asked for. */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU|Audio")
	int64 Callbacks = 0;

	UPROPERTY(BlueprintReadOnly, Category = "WebGPU|Audio")
	int64 RequestedSamples = 0;
};

/**
 * Audio into a GPU ring buffer. Interleaved float frames pushed from any thread (the audio render thread, a
 * capture callback, MakeSubmixListener) are collected into chunks and written into a storage buffer of
 * RingFrames frames through an upload stream: no extra copies and the producer never waits. A small cursor
 * buffer (WebGPUAudioRing) says where the newest frames are, so kernels can analyse the latest audio.
 */
class WEBGPUCOMPUTE_API FWebGPUAudioInput : public TSharedFromThis<FWebGPUAudioInput>
{
public:
	/** Game thread. NumChannels is fixed: input with more channels is mixed down (mono) or truncated, with fewer is repeated. */
	static TSharedPtr<FWebGPUAudioInput> Create(const TSharedRef<FWebGPUContext>& Context, int32 NumChannels, int32 RingFrames,
		int32 ChunkFrames = 512, const FString& Label = FString());

	/** Interleaved samples, from any thread (one producer at a time). Complete chunks go to the GPU right away. */
	void Push(const float* Interleaved, int32 NumFrames, int32 InChannels, int32 SampleRate);

	/** array<f32> of RingFrames * NumChannels interleaved samples. */
	const TSharedRef<FWebGPUBuffer>& GetRing() const { return Ring; }
	/** One WebGPUAudioRing (uniform and storage usage), updated in order with each chunk. */
	const TSharedRef<FWebGPUBuffer>& GetCursor() const { return Cursor; }

	int32 GetNumChannels() const { return NumChannels; }
	int32 GetRingFrames() const { return RingFrames; }
	int32 GetChunkFrames() const { return ChunkFrames; }
	/** Sample rate of the last push. */
	int32 GetSampleRate() const { return SampleRate; }
	/** Frames written into the ring so far (whole chunks). */
	uint64 GetFramesWritten() const { return FramesWritten; }
	FWebGPUStreamStats GetUploadStats() const { return Upload->GetStats(); }

	/**
	 * Called on the pushing thread after each chunk was enqueued, with its first frame and length. Commands
	 * enqueued from here (dispatches over the new chunk, captures) run after the chunk landed. Set it before
	 * audio starts flowing.
	 */
	TFunction<void(uint64 FirstFrame, int32 NumFrames)> OnChunk;

	/** A listener for FAudioDevice::RegisterSubmixBufferListener that pushes what the submix renders. */
	TSharedRef<ISubmixBufferListener, ESPMode::ThreadSafe> MakeSubmixListener();

private:
	FWebGPUAudioInput(const TSharedRef<FWebGPUBuffer>& InRing, const TSharedRef<FWebGPUBuffer>& InCursor) : Ring(InRing), Cursor(InCursor) {}
	void WriteChunk(FWebGPUContext& Ctx, const float* Samples);

	TWeakPtr<FWebGPUContext> Context;
	TSharedRef<FWebGPUBuffer> Ring;
	TSharedRef<FWebGPUBuffer> Cursor;
	TSharedPtr<FWebGPUUploadStream> Upload;
	int32 NumChannels = 1;
	int32 RingFrames = 0;
	int32 ChunkFrames = 512;
	std::atomic<int32> SampleRate = 0;
	std::atomic<uint64> FramesWritten = 0;

	FCriticalSection Lock;
	TArray<float> Pending;
};

/**
 * Audio from a kernel, for a synth or any audio callback. The kernel fills blocks of BlockFrames interleaved
 * frames; they come back through a Queue readback stream into a FIFO that Generate plays from. Generate
 * keeps enough blocks queued on the GPU to cover the largest request it has seen plus BlocksAhead more, so
 * the audio thread drives everything and nothing depends on the game thread.
 *
 * The kernel binds "block" (var<uniform> WebGPUAudioBlock) and "samples" (var<storage, read_write> array<f32>,
 * BlockFrames * Channels interleaved). Create it with Structs = { FWebGPUAudioBlock::StaticStruct() } to get
 * the struct declared, e.g. samples[i * block.Channels + c] = sin(TAU * 440.0 * f32(block.FirstFrame + i) / block.SampleRate).
 */
class WEBGPUCOMPUTE_API FWebGPUAudioGenerator : public TSharedFromThis<FWebGPUAudioGenerator>
{
public:
	/** MaxRequestFrames: the largest callback expected (sizes the readback ring); larger ones still work, with fewer blocks ahead. */
	static TSharedPtr<FWebGPUAudioGenerator> Create(const TSharedRef<FWebGPUContext>& Context, const TSharedRef<FWebGPUKernel>& Kernel,
		int32 NumChannels, int32 SampleRate, int32 BlockFrames = 512, int32 BlocksAhead = 2, int32 MaxRequestFrames = 0);

	/** Audio thread: NumSamples interleaved samples. Where the GPU fell behind it plays silence (counted). Returns NumSamples. */
	int32 Generate(float* OutAudio, int32 NumSamples);

	/** Queues the first blocks, so playback starts without a gap. Same thread as Generate (or before it runs). */
	void Prime();

	FWebGPUAudioStats GetStats() const;
	int32 GetNumChannels() const { return NumChannels; }
	int32 GetSampleRate() const { return SampleRate; }
	int32 GetBlockFrames() const { return BlockFrames; }
	const TSharedRef<FWebGPUKernel>& GetKernel() const { return Kernel; }

private:
	FWebGPUAudioGenerator(const TSharedRef<FWebGPUKernel>& InKernel, const TSharedRef<FWebGPUBuffer>& InBlock, const TSharedRef<FWebGPUBuffer>& InSamples,
		const TSharedRef<FWebGPUReadbackStream>& InStream)
		: Kernel(InKernel), Block(InBlock), Samples(InSamples), Stream(InStream) {}
	void Collect();
	void QueueBlocks();

	TWeakPtr<FWebGPUContext> Context;
	TSharedRef<FWebGPUKernel> Kernel;
	TSharedRef<FWebGPUBuffer> Block;
	TSharedRef<FWebGPUBuffer> Samples;
	TSharedRef<FWebGPUReadbackStream> Stream;
	int32 NumChannels = 2;
	int32 SampleRate = 48000;
	int32 BlockFrames = 512;
	int32 BlocksAhead = 2;

	// Generating thread only
	TArray<float> Fifo;
	int32 FifoHead = 0;
	uint32 NextFrame = 0;
	int32 InFlight = 0;
	int32 LargestRequest = 0;

	std::atomic<int64> Blocks = 0;
	std::atomic<int64> UnderrunSamples = 0;
	std::atomic<int64> Callbacks = 0;
	std::atomic<int64> RequestedSamples = 0;
};
