// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUAudioComponents.h"
#include "WebGPUBuffer.h"
#include "WebGPUCompute.h"
#include "WebGPUKernel.h"
#include "WebGPUSubsystem.h"
#include "AudioDevice.h"
#include "Engine/Engine.h"
#include "Engine/World.h"
#include "ISubmixBufferListener.h"
#include "Sound/SoundGenerator.h"
#include "Sound/SoundSubmix.h"

namespace
{
	constexpr int32 FallbackSampleRate = 48000;

	/** Hands the audio render thread's callbacks to an FWebGPUAudioGenerator. */
	class FWebGPUSoundGenerator : public ISoundGenerator
	{
	public:
		FWebGPUSoundGenerator(const TSharedRef<FWebGPUAudioGenerator>& InGenerator, int32 InSamplesPerCallback)
			: Generator(InGenerator), SamplesPerCallback(InSamplesPerCallback)
		{
		}

		virtual int32 OnGenerateAudio(float* OutAudio, int32 NumSamples) override
		{
			return Generator->Generate(OutAudio, NumSamples);
		}

		virtual void OnBeginGenerate() override
		{
			Generator->Prime();
		}

		virtual int32 GetDesiredNumSamplesToRenderPerCallback() const override
		{
			return SamplesPerCallback;
		}

		virtual int32 GetNumChannels() const override
		{
			return Generator->GetNumChannels();
		}

	private:
		TSharedRef<FWebGPUAudioGenerator> Generator;
		int32 SamplesPerCallback = 1024;
	};
}

// UWebGPUAudioInputComponent

UWebGPUAudioInputComponent::UWebGPUAudioInputComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
}

int32 UWebGPUAudioInputComponent::DeviceSampleRate() const
{
	UWorld* World = GetWorld();
	FAudioDeviceHandle Device = World ? World->GetAudioDevice() : FAudioDeviceHandle();
	if (!Device.IsValid() && GEngine)
	{
		Device = GEngine->GetMainAudioDevice();
	}
	return Device.IsValid() ? (int32)Device->GetSampleRate() : FallbackSampleRate;
}

TSharedPtr<FWebGPUAudioInput> UWebGPUAudioInputComponent::GetInput()
{
	if (!Input)
	{
		TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
		if (!Context)
		{
			return nullptr;
		}
		const int32 RingFrames = FMath::Max(FMath::CeilToInt(RingSeconds * DeviceSampleRate()), ChunkFrames);
		Input = FWebGPUAudioInput::Create(Context.ToSharedRef(), NumChannels, RingFrames, ChunkFrames, GetName());
		NotifiedFrames = 0;
	}
	return Input;
}

void UWebGPUAudioInputComponent::BeginPlay()
{
	Super::BeginPlay();
	if (bCaptureOnBeginPlay && Source == EWebGPUAudioSource::Submix)
	{
		StartCapture();
	}
}

void UWebGPUAudioInputComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	StopCapture();
	Super::EndPlay(EndPlayReason);
}

bool UWebGPUAudioInputComponent::StartCapture()
{
	if (Listener.IsValid())
	{
		return true;
	}
	if (Source != EWebGPUAudioSource::Submix)
	{
		return GetInput().IsValid();
	}

	UWorld* World = GetWorld();
	FAudioDeviceHandle Device = World ? World->GetAudioDevice() : FAudioDeviceHandle();
	if (!Device.IsValid() && GEngine)
	{
		Device = GEngine->GetMainAudioDevice();
	}
	TSharedPtr<FWebGPUAudioInput> Target = GetInput();
	if (!Device.IsValid() || !Target)
	{
		UE_LOG(LogWebGPU, Warning, TEXT("%s: %s"), *GetName(), Target ? TEXT("no audio device to listen to") : TEXT("WebGPU is unavailable"));
		return false;
	}

	USoundSubmix* Listened = Submix ? Submix.Get() : &Device->GetMainSubmixObject();
	Listener = Target->MakeSubmixListener();
	Device->RegisterSubmixBufferListener(Listener.ToSharedRef(), *Listened);
	ListeningTo = Listened;
	return true;
}

void UWebGPUAudioInputComponent::StopCapture()
{
	if (!Listener.IsValid())
	{
		return;
	}
	UWorld* World = GetWorld();
	FAudioDeviceHandle Device = World ? World->GetAudioDevice() : FAudioDeviceHandle();
	if (!Device.IsValid() && GEngine)
	{
		Device = GEngine->GetMainAudioDevice();
	}
	if (Device.IsValid() && ListeningTo.IsValid())
	{
		Device->UnregisterSubmixBufferListener(Listener.ToSharedRef(), *ListeningTo.Get());
	}
	Listener.Reset();
	ListeningTo.Reset();
}

void UWebGPUAudioInputComponent::PushAudio(const TArray<float>& Samples, int32 InChannels, int32 SampleRate)
{
	if (TSharedPtr<FWebGPUAudioInput> Target = GetInput(); Target && InChannels > 0)
	{
		Target->Push(Samples.GetData(), Samples.Num() / InChannels, InChannels, SampleRate);
	}
}

UWebGPUBuffer* UWebGPUAudioInputComponent::GetRingBuffer()
{
	if (!RingObject && GetInput())
	{
		RingObject = NewObject<UWebGPUBuffer>(this);
		RingObject->Initialize(Input->GetRing());
	}
	return RingObject;
}

UWebGPUBuffer* UWebGPUAudioInputComponent::GetCursorBuffer()
{
	if (!CursorObject && GetInput())
	{
		CursorObject = NewObject<UWebGPUBuffer>(this);
		CursorObject->Initialize(Input->GetCursor());
	}
	return CursorObject;
}

int64 UWebGPUAudioInputComponent::GetFramesWritten() const
{
	return Input ? (int64)Input->GetFramesWritten() : 0;
}

int32 UWebGPUAudioInputComponent::GetSampleRate() const
{
	return Input ? Input->GetSampleRate() : 0;
}

void UWebGPUAudioInputComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	if (!Input)
	{
		return;
	}
	const uint64 Written = Input->GetFramesWritten();
	if (Written > NotifiedFrames)
	{
		const uint64 First = NotifiedFrames;
		NotifiedFrames = Written;
		OnAudio.Broadcast((int64)First, (int32)FMath::Min<uint64>(Written - First, MAX_int32));
	}
}

// UWebGPUSynthComponent

UWebGPUSynthComponent::UWebGPUSynthComponent(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	NumChannels = Channels;
}

FString UWebGPUSynthComponent::DefaultKernelSource()
{
	return TEXT(R"(
@group(0) @binding(0) var<uniform> block: WebGPUAudioBlock;
@group(0) @binding(1) var<storage, read_write> samples: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	if (id.x >= block.Frames) { return; }
	// Whole seconds hold whole periods of 440 Hz, so the phase stays exact however long it plays
	let frame = (block.FirstFrame + id.x) % u32(block.SampleRate);
	let value = 0.2 * sin(6.2831853 * 440.0 * f32(frame) / block.SampleRate);
	for (var c = 0u; c < block.Channels; c++) {
		samples[id.x * block.Channels + c] = value;
	}
}
)");
}

bool UWebGPUSynthComponent::Init(int32& SampleRate)
{
	NumChannels = Channels;
	if (KernelObject)
	{
		return true;
	}
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!Context)
	{
		UE_LOG(LogWebGPU, Warning, TEXT("%s: WebGPU is unavailable; the synth plays silence"), *GetName());
		return true;
	}

	// Game thread: the WebGPUAudioBlock declaration comes from UObject reflection
	FWebGPUKernelDesc Desc;
	Desc.Source = KernelSource.IsEmpty() ? DefaultKernelSource() : KernelSource;
	Desc.Label = GetName();
	Desc.Structs = { FWebGPUAudioBlock::StaticStruct() };
	UWebGPUKernel* Kernel = NewObject<UWebGPUKernel>(this);
	TWeakObjectPtr<UWebGPUKernel> WeakKernel = Kernel;
	Kernel->Initialize(Context->CreateKernel(Desc, [WeakKernel](bool bSuccess, const TArray<FWebGPUCompileMessage>& Messages)
	{
		if (UWebGPUKernel* Alive = WeakKernel.Get())
		{
			Alive->HandleCompiled(bSuccess, Messages);
		}
	}, EWebGPUCallbackThread::GameThread));
	KernelObject = Kernel;
	return true;
}

ISoundGeneratorPtr UWebGPUSynthComponent::CreateSoundGenerator(const FSoundGeneratorInitParams& InParams)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!Context || !KernelObject || !KernelObject->GetNative())
	{
		return nullptr;
	}
	const int32 GeneratorChannels = InParams.NumChannels > 0 ? InParams.NumChannels : Channels;
	const int32 Rate = InParams.SampleRate > 0.f ? (int32)InParams.SampleRate : FallbackSampleRate;
	const int32 FramesPerCallback = InParams.NumFramesPerCallback > 0 ? InParams.NumFramesPerCallback : BlockFrames;
	TSharedPtr<FWebGPUAudioGenerator> NewGenerator = FWebGPUAudioGenerator::Create(Context.ToSharedRef(), KernelObject->GetNative().ToSharedRef(),
		GeneratorChannels, Rate, BlockFrames, BlocksAhead, FramesPerCallback);
	if (!NewGenerator)
	{
		return nullptr;
	}
	{
		FScopeLock Lock(&GeneratorLock);
		Generator = NewGenerator;
	}
	// Ask for block-sized callbacks: less audio buffered ahead means less latency
	return MakeShared<FWebGPUSoundGenerator, ESPMode::ThreadSafe>(NewGenerator.ToSharedRef(), BlockFrames * GeneratorChannels);
}

void UWebGPUSynthComponent::SetKernelSource(const FString& Source)
{
	KernelSource = Source;
	if (KernelObject)
	{
		KernelObject->Recompile(Source.IsEmpty() ? DefaultKernelSource() : Source);
	}
}

FWebGPUAudioStats UWebGPUSynthComponent::GetAudioStats() const
{
	TSharedPtr<FWebGPUAudioGenerator> Current = GetGenerator();
	return Current ? Current->GetStats() : FWebGPUAudioStats();
}

TSharedPtr<FWebGPUAudioGenerator> UWebGPUSynthComponent::GetGenerator() const
{
	FScopeLock Lock(&GeneratorLock);
	return Generator;
}
