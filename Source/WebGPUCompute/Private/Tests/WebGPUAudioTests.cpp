// Copyright 2025-current Getnamo. All Rights Reserved.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "WebGPUAudio.h"
#include "WebGPUAudioComponents.h"
#include "WebGPUCompute.h"
#include "WebGPUContext.h"
#include "WebGPUKernel.h"
#include "WebGPUStreams.h"
#include "WebGPUSubsystem.h"
#include "Async/Async.h"
#include "AudioDevice.h"
#include "Engine/Engine.h"
#include "Engine/World.h"
#include "GameFramework/Actor.h"
#include "HAL/PlatformProcess.h"
#include "ISubmixBufferListener.h"
#include "Misc/CommandLine.h"
#include "Misc/ScopeExit.h"
#include "Sound/SoundGenerator.h"
#include "Tests/AutomationCommon.h"

namespace WebGPUAudioTests
{
	constexpr EAutomationTestFlags Flags = EAutomationTestFlags::EditorContext | EAutomationTestFlags::ClientContext | EAutomationTestFlags::ProductFilter;
	constexpr double Tau = 6.283185307179586;

	bool WaitUntil(TFunctionRef<bool()> Condition, double TimeoutSeconds = 10.0)
	{
		const double End = FPlatformTime::Seconds() + TimeoutSeconds;
		while (!Condition())
		{
			if (FPlatformTime::Seconds() > End)
			{
				return false;
			}
			FPlatformProcess::Sleep(0.0002f);
		}
		return true;
	}

	bool ReadBufferNow(FWebGPUContext& Context, const TSharedRef<FWebGPUBuffer>& Buffer, TArray<uint8>& OutData)
	{
		struct FResult
		{
			std::atomic<bool> bDone = false;
			std::atomic<bool> bSuccess = false;
			TArray<uint8> Data;
		};
		TSharedRef<FResult> Result = MakeShared<FResult>();
		Context.ReadBuffer(Buffer, 0, Buffer->GetSize(), [Result](bool bSuccess, TArray<uint8>&& Data)
		{
			Result->Data = MoveTemp(Data);
			Result->bSuccess = bSuccess;
			Result->bDone = true;
		}, EWebGPUCallbackThread::WorkerThread);
		if (!WaitUntil([&] { return Result->bDone.load(); }) || !Result->bSuccess)
		{
			return false;
		}
		OutData = MoveTemp(Result->Data);
		return true;
	}

	/** The default synth kernel's sample for an absolute frame. */
	float SineSample(uint64 Frame, double Frequency, int32 SampleRate, float Amplitude)
	{
		const uint64 Wrapped = Frame % (uint64)SampleRate;
		return Amplitude * (float)FMath::Sin(Tau * Frequency * (double)Wrapped / SampleRate);
	}

	/** Calls a generator the way the audio render thread does: fixed blocks at the real-time rate. */
	TArray<float> RunAudioThread(TFunction<int32(float*, int32)> Generate, int32 Callbacks, int32 SamplesPerCallback, int32 Channels, int32 SampleRate,
		TFunction<void(int32)> OnCallback = nullptr)
	{
		TArray<float> Output;
		Output.SetNumZeroed(Callbacks * SamplesPerCallback);
		const double Period = (double)SamplesPerCallback / Channels / SampleRate;
		TFuture<void> Thread = Async(EAsyncExecution::Thread, [&]()
		{
			const double Start = FPlatformTime::Seconds();
			for (int32 Callback = 0; Callback < Callbacks; Callback++)
			{
				while (FPlatformTime::Seconds() < Start + Callback * Period)
				{
					FPlatformProcess::SleepNoStats(0.0005f);
				}
				Generate(Output.GetData() + (int64)Callback * SamplesPerCallback, SamplesPerCallback);
				if (OnCallback)
				{
					OnCallback(Callback);
				}
			}
		});
		Thread.Wait();
		return Output;
	}
}

using namespace WebGPUAudioTests;

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUAudioInputTest, "WebGPU.Audio.InputRing", WebGPUAudioTests::Flags)
bool FWebGPUAudioInputTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}

	// Mono ring of 8192 frames, 512-frame chunks; stereo input arrives in 480-frame callbacks (10 ms)
	const int32 RingFrames = 8192;
	const int32 ChunkFrames = 512;
	const int32 Rate = 48000;
	TSharedPtr<FWebGPUAudioInput> Input = FWebGPUAudioInput::Create(Context.ToSharedRef(), 1, RingFrames, ChunkFrames, TEXT("test-input"));
	if (!TestTrue(TEXT("Input"), Input.IsValid()))
	{
		return false;
	}

	// Per chunk, from the pushing thread: RMS of the newest chunk, located through the cursor
	FWebGPUKernelDesc Desc;
	Desc.Label = TEXT("chunk-rms");
	Desc.Structs = { FWebGPUAudioRing::StaticStruct() };
	Desc.Source = TEXT(R"(
@group(0) @binding(0) var<storage, read> ring: array<f32>;
@group(0) @binding(1) var<uniform> cursor: WebGPUAudioRing;
@group(0) @binding(2) var<storage, read_write> rms: array<f32>;
@compute @workgroup_size(1)
fn main() {
	var sum = 0.0;
	for (var i = 0u; i < 512u; i++) {
		let frame = (cursor.FramesWritten - 512u + i) % cursor.RingFrames;
		let v = ring[frame * cursor.Channels];
		sum += v * v;
	}
	rms[0] = sqrt(sum / 512.0);
}
)");
	TSharedRef<FWebGPUKernel> Rms = Context->CreateKernel(Desc);
	TSharedRef<FWebGPUBuffer> RmsOut = Context->CreateBuffer(4, EWebGPUBufferUsage::Storage, TEXT("rms"));
	Context->SetBufferByName(Rms, TEXT("ring"), Input->GetRing());
	Context->SetBufferByName(Rms, TEXT("cursor"), Input->GetCursor());
	Context->SetBufferByName(Rms, TEXT("rms"), RmsOut);
	TSharedRef<FWebGPUReadbackStream> RmsStream = Context->CreateReadbackStream(4, 64, EWebGPUStreamPolicy::Queue, TEXT("rms values"));
	if (!TestTrue(TEXT("Compiled"), WaitUntil([&] { return Rms->IsReady(); })))
	{
		AddError(FWebGPUDevice::FormatCompileMessages(Rms->GetMessages()));
		return false;
	}
	std::atomic<int32> Chunks = 0;
	Input->OnChunk = [&Context, Rms, RmsOut, RmsStream, &Chunks](uint64 FirstFrame, int32 NumFrames)
	{
		Context->Dispatch(Rms, 1);
		RmsStream->Capture(RmsOut);
		Chunks++;
	};

	// One second of a 440 Hz sine at amplitude 0.5 in both channels, pushed from another thread at 4x real time
	const int32 TotalFrames = Rate;
	const float Amplitude = 0.5f;
	TArray<float> RmsValues;
	auto DrainRms = [&]()
	{
		FWebGPUReadbackFrame Frame;
		while (RmsStream->Acquire(Frame))
		{
			RmsValues.Add(Frame.View<float>()[0]);
		}
	};
	TFuture<void> Producer = Async(EAsyncExecution::Thread, [&]()
	{
		TArray<float> Callback;
		const double Start = FPlatformTime::Seconds();
		for (int32 Frame = 0; Frame < TotalFrames; Frame += 480)
		{
			while (FPlatformTime::Seconds() < Start + Frame / (4.0 * Rate))
			{
				FPlatformProcess::SleepNoStats(0.0002f);
			}
			const int32 Frames = FMath::Min(480, TotalFrames - Frame);
			Callback.SetNumUninitialized(Frames * 2);
			for (int32 Index = 0; Index < Frames; Index++)
			{
				Callback[Index * 2] = Callback[Index * 2 + 1] = SineSample(Frame + Index, 440.0, Rate, Amplitude);
			}
			Input->Push(Callback.GetData(), Frames, 2, Rate);
		}
	});
	while (!Producer.IsReady())
	{
		DrainRms();
		FPlatformProcess::Sleep(0.001f);
	}
	Input->OnChunk = nullptr;

	const int32 ExpectedChunks = TotalFrames / ChunkFrames;
	TestEqual(TEXT("Whole chunks written"), (int64)Input->GetFramesWritten(), (int64)ExpectedChunks * ChunkFrames);
	TestEqual(TEXT("One callback per chunk"), Chunks.load(), ExpectedChunks);
	TestEqual(TEXT("Sample rate taken from the input"), Input->GetSampleRate(), Rate);

	// The ring holds the newest RingFrames frames, mixed down to mono, at their absolute positions
	TArray<uint8> RingBytes;
	if (!TestTrue(TEXT("Ring read back"), ReadBufferNow(*Context, Input->GetRing(), RingBytes)))
	{
		return false;
	}
	const float* RingSamples = reinterpret_cast<const float*>(RingBytes.GetData());
	const uint64 Written = Input->GetFramesWritten();
	int32 Bad = 0;
	for (uint64 Frame = Written - RingFrames; Frame < Written; Frame++)
	{
		Bad += FMath::IsNearlyEqual(RingSamples[Frame % RingFrames], SineSample(Frame, 440.0, Rate, Amplitude), 1e-5f) ? 0 : 1;
	}
	TestEqual(TEXT("Ring holds the newest audio"), Bad, 0);

	TArray<uint8> CursorBytes;
	TestTrue(TEXT("Cursor read back"), ReadBufferNow(*Context, Input->GetCursor(), CursorBytes));
	const FWebGPUAudioRing* Cursor = reinterpret_cast<const FWebGPUAudioRing*>(CursorBytes.GetData());
	TestTrue(TEXT("Cursor says where the newest frames are"), CursorBytes.Num() == sizeof(FWebGPUAudioRing) && Cursor->FramesWritten == (uint32)Written
		&& Cursor->RingFrames == (uint32)RingFrames && Cursor->Channels == 1 && Cursor->SampleRate == (float)Rate);

	// Every chunk's RMS, computed on the GPU right after the chunk landed
	TestTrue(TEXT("RMS per chunk"), WaitUntil([&] { DrainRms(); return RmsValues.Num() == ExpectedChunks; }));
	int32 BadRms = 0;
	for (const float Value : RmsValues)
	{
		BadRms += FMath::IsNearlyEqual(Value, Amplitude / UE_SQRT_2, 0.01f) ? 0 : 1;
	}
	TestEqual(TEXT("Chunk RMS = amplitude / sqrt(2)"), BadRms, 0);

	const FWebGPUStreamStats Upload = Input->GetUploadStats();
	AddInfo(FString::Printf(TEXT("Audio input: %d chunks of %d frames, %lld through upload slots, %lld through the copying path; RMS on the GPU for each"),
		ExpectedChunks, ChunkFrames, Upload.Committed, Upload.Fallbacks));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUAudioGeneratorTest, "WebGPU.Audio.Generator", WebGPUAudioTests::Flags)
bool FWebGPUAudioGeneratorTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}

	// The synth component's own wiring, driven like the audio render thread drives it
	UWebGPUSynthComponent* Synth = NewObject<UWebGPUSynthComponent>();
	Synth->AddToRoot();
	ON_SCOPE_EXIT { Synth->RemoveFromRoot(); };
	int32 Rate = 48000;
	Synth->Init(Rate);
	if (!TestNotNull(TEXT("Kernel"), Synth->GetKernel()) || !TestTrue(TEXT("Compiled"), WaitUntil([&] { return Synth->GetKernel()->IsReady(); })))
	{
		return false;
	}
	FSoundGeneratorInitParams Params;
	Params.SampleRate = (float)Rate;
	Params.NumChannels = 2;
	Params.NumFramesPerCallback = 512;
	ISoundGeneratorPtr Generator = Synth->CreateSoundGenerator(Params);
	if (!TestTrue(TEXT("Sound generator"), Generator.IsValid()))
	{
		return false;
	}
	TestEqual(TEXT("Samples per callback"), Generator->GetDesiredNumSamplesToRenderPerCallback(), 1024);

	// One second of callbacks; halfway, the kernel is swapped for an 880 Hz one while playing
	const int32 Callbacks = 94;
	const FString Swapped = UWebGPUSynthComponent::DefaultKernelSource().Replace(TEXT("440.0"), TEXT("880.0"));
	TSharedRef<FWebGPUKernel> Native = Synth->GetKernel()->GetNative().ToSharedRef();
	Generator->OnBeginGenerate();
	const TArray<float> Output = RunAudioThread([&Generator](float* Out, int32 Num) { return Generator->OnGenerateAudio(Out, Num); }, Callbacks, 1024, 2, Rate,
		[&](int32 Callback)
		{
			// (The game thread is busy waiting for this loop, so recompile the native kernel directly)
			if (Callback == Callbacks / 2)
			{
				Context->RecompileKernel(Native, Swapped);
			}
		});

	// Before the swap: the 440 Hz sine, continuous, both channels; nothing missing
	const FWebGPUAudioStats Stats = Synth->GetAudioStats();
	TestEqual(TEXT("No underruns"), Stats.UnderrunSamples, (int64)0);
	int32 Bad = 0;
	const int32 CheckedFrames = (Callbacks / 2) * 512;
	for (int32 Frame = 0; Frame < CheckedFrames; Frame++)
	{
		const float Expected = SineSample(Frame, 440.0, Rate, 0.2f);
		Bad += FMath::IsNearlyEqual(Output[Frame * 2], Expected, 1e-4f) && Output[Frame * 2] == Output[Frame * 2 + 1] ? 0 : 1;
	}
	TestEqual(TEXT("Continuous 440 Hz before the swap"), Bad, 0);

	// After the swap compiled: 880 Hz from some block on, still continuous in frame numbering
	const int32 LastFrame = Callbacks * 512 - 1;
	TestTrue(TEXT("The new kernel took over"), FMath::IsNearlyEqual(Output[LastFrame * 2], SineSample(LastFrame, 880.0, Rate, 0.2f), 1e-4f));

	AddInfo(FString::Printf(TEXT("GPU synth: %lld blocks of 512 frames, 2 blocks ahead (21 ms), block round trip %.3f ms, %lld underrun samples, kernel swapped mid-playback"),
		Stats.Blocks, Stats.RoundTripMs, Stats.UnderrunSamples));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUAudioDeviceTest, "WebGPU.Audio.DeviceRoundTrip", WebGPUAudioTests::Flags)
bool FWebGPUAudioDeviceTest::RunTest(const FString& Parameters)
{
	// Plays a quiet tone through the real audio device, so it only runs when asked for
	if (!FParse::Param(FCommandLine::Get(), TEXT("WebGPUAudioDeviceTest")))
	{
		AddInfo(TEXT("Skipped: plays sound through the audio device; run with -WebGPUAudioDeviceTest"));
		return true;
	}
	FAudioDeviceHandle Device = GEngine ? GEngine->GetMainAudioDevice() : FAudioDeviceHandle();
	UWorld* World = GWorld;
	if (!Device.IsValid() || !World)
	{
		AddInfo(TEXT("Skipped: no audio device or world"));
		return true;
	}

	// An unfocused editor mutes all output unless background audio is allowed (an editor setting; set through
	// reflection so this runtime module doesn't link the editor)
	UClass* MiscSettings = FindObject<UClass>(nullptr, TEXT("/Script/UnrealEd.LevelEditorMiscSettings"));
	FBoolProperty* Background = MiscSettings ? FindFProperty<FBoolProperty>(MiscSettings, TEXT("bAllowBackgroundAudio")) : nullptr;
	UObject* MiscDefaults = MiscSettings ? MiscSettings->GetDefaultObject() : nullptr;
	const bool bBackgroundBefore = Background && MiscDefaults ? Background->GetPropertyValue_InContainer(MiscDefaults) : false;
	if (Background && MiscDefaults)
	{
		Background->SetPropertyValue_InContainer(MiscDefaults, true);
	}

	// GPU synth -> UE's mixer -> main submix -> GPU input ring
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	TSharedPtr<FWebGPUAudioInput> Input = FWebGPUAudioInput::Create(Context.ToSharedRef(), 1, (int32)Device->GetSampleRate(), 512, TEXT("device-input"));
	TSharedRef<ISubmixBufferListener, ESPMode::ThreadSafe> Listener = Input->MakeSubmixListener();
	Device->RegisterSubmixBufferListener(Listener, Device->GetMainSubmixObject());

	// A headless editor world has no viewport, hence no listener, and sounds don't start without one
	Device->SetListener(World, 0, FTransform::Identity, 0.f);
	AActor* Actor = World->SpawnActor<AActor>();
	UWebGPUSynthComponent* Synth = NewObject<UWebGPUSynthComponent>(Actor);
	Synth->RegisterComponent();
	Synth->SetVolumeMultiplier(0.05f);
	// Editor worlds keep game sounds paused while no game is running; UI sounds play
	Synth->bIsUISound = true;
	Synth->Start();

	struct FState
	{
		double Deadline = 0.0;
	};
	TSharedRef<FState> State = MakeShared<FState>();
	State->Deadline = FPlatformTime::Seconds() + 3.0;
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([this, State, Input, Listener, Synth, Actor, Context, Background, MiscDefaults, bBackgroundBefore]() -> bool
	{
		const bool bEnough = Input->GetFramesWritten() >= (uint64)Input->GetRingFrames() && Synth->GetAudioStats().Blocks > 40;
		if (!bEnough && FPlatformTime::Seconds() < State->Deadline)
		{
			return false;
		}

		TArray<uint8> Bytes;
		ReadBufferNow(*Context, Input->GetRing(), Bytes);
		const float* Samples = reinterpret_cast<const float*>(Bytes.GetData());
		double Sum = 0.0;
		for (int32 Index = 0; Index < Bytes.Num() / 4; Index++)
		{
			Sum += Samples[Index] * Samples[Index];
		}
		const double Rms = Bytes.Num() > 0 ? FMath::Sqrt(Sum / (Bytes.Num() / 4)) : 0.0;
		const FWebGPUAudioStats Stats = Synth->GetAudioStats();
		TestTrue(TEXT("The device played GPU-generated audio"), Stats.Blocks > 40);
		TestTrue(TEXT("The submix fed the GPU ring"), Input->GetFramesWritten() >= (uint64)Input->GetRingFrames());
		TestTrue(TEXT("The synth's tone reached the ring"), Rms > 1e-4);
		TestTrue(TEXT("The GPU kept up with the mixer's callbacks"), Stats.UnderrunSamples * 10 < Stats.RequestedSamples);
		AddInfo(FString::Printf(TEXT("Device round trip: %lld synth blocks, %lld callbacks asking %lld samples (%lld underrun), %llu frames into the ring, RMS %.5f"),
			Stats.Blocks, Stats.Callbacks, Stats.RequestedSamples, Stats.UnderrunSamples, Input->GetFramesWritten(), Rms));

		Synth->Stop();
		if (Background && MiscDefaults)
		{
			Background->SetPropertyValue_InContainer(MiscDefaults, bBackgroundBefore);
		}
		if (FAudioDeviceHandle Main = GEngine->GetMainAudioDevice(); Main.IsValid())
		{
			Main->UnregisterSubmixBufferListener(Listener, Main->GetMainSubmixObject());
		}
		Actor->Destroy();
		return true;
	}));
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
