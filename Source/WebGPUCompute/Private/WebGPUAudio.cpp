// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUAudio.h"
#include "WebGPUCompute.h"
#include "ISubmixBufferListener.h"

namespace
{
	/** Pushes what a submix renders into an FWebGPUAudioInput (audio render thread). */
	class FWebGPUSubmixListener : public ISubmixBufferListener
	{
	public:
		explicit FWebGPUSubmixListener(const TSharedRef<FWebGPUAudioInput>& InInput) : Input(InInput) {}

		virtual void OnNewSubmixBuffer(const USoundSubmix* OwningSubmix, float* AudioData, int32 NumSamples, int32 NumChannels, const int32 SampleRate, double AudioClock) override
		{
			TSharedPtr<FWebGPUAudioInput> Target = Input.Pin();
			if (Target && NumChannels > 0)
			{
				Target->Push(AudioData, NumSamples / NumChannels, NumChannels, SampleRate);
			}
		}

		/** Keep the submix rendering while listening, even when nothing plays (silence is data too). */
		virtual bool IsRenderingAudio() const override
		{
			return true;
		}

		virtual const FString& GetListenerName() const override
		{
			static const FString ListenerName = TEXT("WebGPUAudioInput");
			return ListenerName;
		}

	private:
		TWeakPtr<FWebGPUAudioInput> Input;
	};
}

// FWebGPUAudioInput

TSharedPtr<FWebGPUAudioInput> FWebGPUAudioInput::Create(const TSharedRef<FWebGPUContext>& Context, int32 NumChannels, int32 RingFrames, int32 ChunkFrames, const FString& Label)
{
	if (NumChannels <= 0 || RingFrames <= 0 || ChunkFrames <= 0 || ChunkFrames > RingFrames)
	{
		UE_LOG(LogWebGPU, Error, TEXT("FWebGPUAudioInput: needs channels > 0 and 0 < chunk frames (%d) <= ring frames (%d)"), ChunkFrames, RingFrames);
		return nullptr;
	}
	const FString Name = Label.IsEmpty() ? TEXT("WebGPUAudioInput") : Label;
	TSharedRef<FWebGPUBuffer> Ring = Context->CreateBuffer((uint64)RingFrames * NumChannels * sizeof(float), EWebGPUBufferUsage::Storage, Name + TEXT(" ring"));
	TSharedPtr<FWebGPUBuffer> Cursor = Context->CreateStructBuffer(FWebGPUAudioRing::StaticStruct(), 1, EWebGPUBufferUsage::Uniform | EWebGPUBufferUsage::Storage, Name + TEXT(" cursor"));
	if (!Cursor)
	{
		return nullptr;
	}

	TSharedPtr<FWebGPUAudioInput> Input = MakeShareable(new FWebGPUAudioInput(Ring, Cursor.ToSharedRef()));
	Input->Context = Context;
	Input->NumChannels = NumChannels;
	Input->RingFrames = RingFrames;
	Input->ChunkFrames = ChunkFrames;
	// Enough slots for ~16 chunks in flight; a chunk split at the ring's end takes two
	Input->Upload = Context->CreateUploadStream((uint64)ChunkFrames * NumChannels * sizeof(float), 16, Name + TEXT(" upload"));
	return Input;
}

void FWebGPUAudioInput::Push(const float* Interleaved, int32 NumFrames, int32 InChannels, int32 InSampleRate)
{
	TSharedPtr<FWebGPUContext> Ctx = Context.Pin();
	if (!Ctx || !Interleaved || NumFrames <= 0 || InChannels <= 0)
	{
		return;
	}

	FScopeLock ScopeLock(&Lock);
	if (InSampleRate != SampleRate)
	{
		// A new format: what was collected belongs to the old one
		Pending.Reset();
		SampleRate = InSampleRate;
	}

	const int32 Start = Pending.Num();
	Pending.AddUninitialized(NumFrames * NumChannels);
	float* Out = Pending.GetData() + Start;
	for (int32 Frame = 0; Frame < NumFrames; Frame++)
	{
		const float* In = Interleaved + (int64)Frame * InChannels;
		if (NumChannels == 1 && InChannels > 1)
		{
			float Sum = 0.f;
			for (int32 Channel = 0; Channel < InChannels; Channel++)
			{
				Sum += In[Channel];
			}
			*Out++ = Sum / InChannels;
		}
		else
		{
			for (int32 Channel = 0; Channel < NumChannels; Channel++)
			{
				*Out++ = In[FMath::Min(Channel, InChannels - 1)];
			}
		}
	}

	const int32 ChunkSamples = ChunkFrames * NumChannels;
	int32 Consumed = 0;
	while (Pending.Num() - Consumed >= ChunkSamples)
	{
		WriteChunk(*Ctx, Pending.GetData() + Consumed);
		Consumed += ChunkSamples;
	}
	if (Consumed > 0)
	{
		Pending.RemoveAt(0, Consumed, EAllowShrinking::No);
	}
}

void FWebGPUAudioInput::WriteChunk(FWebGPUContext& Ctx, const float* Chunk)
{
	const uint64 First = FramesWritten;
	const int32 Offset = (int32)(First % (uint64)RingFrames);

	// The ring's end splits a chunk in two
	const int32 FirstPart = FMath::Min(ChunkFrames, RingFrames - Offset);
	const uint64 FrameBytes = (uint64)NumChannels * sizeof(float);
	Upload->Write(Ring, (uint64)Offset * FrameBytes, Chunk, (uint64)FirstPart * FrameBytes);
	if (FirstPart < ChunkFrames)
	{
		Upload->Write(Ring, 0, Chunk + (int64)FirstPart * NumChannels, (uint64)(ChunkFrames - FirstPart) * FrameBytes);
	}

	FramesWritten = First + ChunkFrames;
	FWebGPUAudioRing State;
	State.FramesWritten = (uint32)FramesWritten;
	State.RingFrames = (uint32)RingFrames;
	State.Channels = (uint32)NumChannels;
	State.SampleRate = (float)SampleRate;
	Ctx.WriteBuffer(Cursor, 0, &State, sizeof(State));

	if (OnChunk)
	{
		OnChunk(First, ChunkFrames);
	}
}

TSharedRef<ISubmixBufferListener, ESPMode::ThreadSafe> FWebGPUAudioInput::MakeSubmixListener()
{
	return MakeShared<FWebGPUSubmixListener, ESPMode::ThreadSafe>(AsShared());
}

// FWebGPUAudioGenerator

TSharedPtr<FWebGPUAudioGenerator> FWebGPUAudioGenerator::Create(const TSharedRef<FWebGPUContext>& Context, const TSharedRef<FWebGPUKernel>& Kernel,
	int32 NumChannels, int32 SampleRate, int32 BlockFrames, int32 BlocksAhead, int32 MaxRequestFrames)
{
	if (NumChannels <= 0 || SampleRate <= 0 || BlockFrames <= 0)
	{
		UE_LOG(LogWebGPU, Error, TEXT("FWebGPUAudioGenerator: needs positive channels (%d), sample rate (%d) and block frames (%d)"), NumChannels, SampleRate, BlockFrames);
		return nullptr;
	}
	BlocksAhead = FMath::Clamp(BlocksAhead, 1, 16);
	const uint64 BlockBytes = (uint64)BlockFrames * NumChannels * sizeof(float);
	TSharedRef<FWebGPUBuffer> Block = Context->CreateBuffer(sizeof(FWebGPUAudioBlock), EWebGPUBufferUsage::Uniform, Kernel->GetLabel() + TEXT(" block"));
	TSharedRef<FWebGPUBuffer> Samples = Context->CreateBuffer(BlockBytes, EWebGPUBufferUsage::Storage, Kernel->GetLabel() + TEXT(" samples"));
	// Slots for a whole request's worth of blocks plus the ones ahead of it
	const int32 Slots = BlocksAhead + FMath::DivideAndRoundUp(FMath::Max(MaxRequestFrames, BlockFrames), BlockFrames) + 2;
	TSharedRef<FWebGPUReadbackStream> Stream = Context->CreateReadbackStream(BlockBytes, Slots, EWebGPUStreamPolicy::Queue, Kernel->GetLabel() + TEXT(" audio"));
	Context->SetBufferByName(Kernel, TEXT("block"), Block);
	Context->SetBufferByName(Kernel, TEXT("samples"), Samples);

	TSharedPtr<FWebGPUAudioGenerator> Generator = MakeShareable(new FWebGPUAudioGenerator(Kernel, Block, Samples, Stream));
	Generator->Context = Context;
	Generator->NumChannels = NumChannels;
	Generator->SampleRate = SampleRate;
	Generator->BlockFrames = BlockFrames;
	Generator->BlocksAhead = BlocksAhead;
	return Generator;
}

void FWebGPUAudioGenerator::Collect()
{
	FWebGPUReadbackFrame Frame;
	while (Stream->Acquire(Frame))
	{
		if (FifoHead > 0 && FifoHead == Fifo.Num())
		{
			Fifo.Reset();
			FifoHead = 0;
		}
		Fifo.Append(Frame.View<float>());
		Frame.Release();
		InFlight = FMath::Max(InFlight - 1, 0);
		Blocks++;
	}
}

void FWebGPUAudioGenerator::QueueBlocks()
{
	TSharedPtr<FWebGPUContext> Ctx = Context.Pin();
	if (!Ctx || !Kernel->IsReady())
	{
		return;
	}
	// Enough for the largest callback so far, plus BlocksAhead blocks of slack for the GPU round trip
	const int32 BlockSamples = BlockFrames * NumChannels;
	const int32 Target = BlocksAhead * BlockSamples + LargestRequest;
	while ((Fifo.Num() - FifoHead) + InFlight * BlockSamples < Target)
	{
		FWebGPUAudioBlock Params;
		Params.FirstFrame = NextFrame;
		Params.Frames = (uint32)BlockFrames;
		Params.Channels = (uint32)NumChannels;
		Params.SampleRate = (float)SampleRate;
		Ctx->WriteBuffer(Block, 0, &Params, sizeof(Params));
		Ctx->DispatchForCount(Kernel, (uint32)BlockFrames);
		if (!Stream->Capture(Samples))
		{
			break;
		}
		NextFrame += (uint32)BlockFrames;
		InFlight++;
	}
}

void FWebGPUAudioGenerator::Prime()
{
	QueueBlocks();
}

int32 FWebGPUAudioGenerator::Generate(float* OutAudio, int32 NumSamples)
{
	Callbacks++;
	RequestedSamples += NumSamples;
	LargestRequest = FMath::Max(LargestRequest, NumSamples);
	// Only at startup: give the first blocks a moment rather than opening with silence
	if (Blocks == 0 && InFlight > 0 && Fifo.Num() - FifoHead < NumSamples)
	{
		Stream->WaitForFrame(0.005);
	}
	Collect();

	const int32 Available = Fifo.Num() - FifoHead;
	const int32 Played = FMath::Min(Available, NumSamples);
	if (Played > 0)
	{
		FMemory::Memcpy(OutAudio, Fifo.GetData() + FifoHead, Played * sizeof(float));
		FifoHead += Played;
	}
	if (Played < NumSamples)
	{
		FMemory::Memzero(OutAudio + Played, (NumSamples - Played) * sizeof(float));
		// Silence while the kernel compiles or the first block is on its way isn't an underrun
		if (Blocks > 0)
		{
			UnderrunSamples += NumSamples - Played;
		}
	}
	if (FifoHead > 0 && FifoHead >= Fifo.Num() / 2)
	{
		Fifo.RemoveAt(0, FifoHead, EAllowShrinking::No);
		FifoHead = 0;
	}

	QueueBlocks();
	return NumSamples;
}

FWebGPUAudioStats FWebGPUAudioGenerator::GetStats() const
{
	FWebGPUAudioStats Stats;
	Stats.Blocks = Blocks;
	Stats.UnderrunSamples = UnderrunSamples;
	Stats.RoundTripMs = Stream->GetStats().LatencyMs;
	Stats.Callbacks = Callbacks;
	Stats.RequestedSamples = RequestedSamples;
	return Stats;
}
