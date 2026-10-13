// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Components/SynthComponent.h"
#include "WebGPUAudio.h"
#include "WebGPUAudioComponents.generated.h"

class ISubmixBufferListener;
class USoundSubmix;
class UWebGPUBuffer;
class UWebGPUKernel;

UENUM(BlueprintType)
enum class EWebGPUAudioSource : uint8
{
	/** What a submix renders (default: the main output). Route a microphone through a submix to analyse it. */
	Submix,
	/** Only what you pass to Push Audio. */
	Manual,
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FWebGPUAudioArrivedSignature, int64, FirstFrame, int32, NumFrames);

/**
 * Feeds audio into a GPU ring buffer for kernels (spectra, onsets, ML on sound). Get Ring Buffer holds the
 * last Ring Seconds of audio as array<f32> (interleaved), and Get Cursor Buffer says where the newest frames
 * are (WGSL struct WebGPUAudioRing). On Audio fires on the game thread with the frames that arrived since
 * the last tick; dispatch analysis kernels from there.
 */
UCLASS(ClassGroup = "WebGPU", meta = (BlueprintSpawnableComponent))
class WEBGPUCOMPUTE_API UWebGPUAudioInputComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UWebGPUAudioInputComponent();

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "WebGPU|Audio")
	EWebGPUAudioSource Source = EWebGPUAudioSource::Submix;

	/** Submix source: the submix to listen to. Unset: the main output. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "WebGPU|Audio")
	TObjectPtr<USoundSubmix> Submix;

	/** Channels in the ring; 1 mixes the input down to mono. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "WebGPU|Audio", meta = (ClampMin = "1", ClampMax = "8"))
	int32 NumChannels = 1;

	/** Audio the ring holds. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "WebGPU|Audio", meta = (ClampMin = "0.05"))
	float RingSeconds = 2.f;

	/** Frames per upload to the GPU (the granularity of On Audio). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "WebGPU|Audio", meta = (ClampMin = "64"))
	int32 ChunkFrames = 512;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "WebGPU|Audio")
	bool bCaptureOnBeginPlay = true;

	/** New audio is in the ring: FirstFrame counts frames since capture started. */
	UPROPERTY(BlueprintAssignable, Category = "WebGPU|Audio")
	FWebGPUAudioArrivedSignature OnAudio;

	/** Starts listening to the submix (Submix source). The ring is created on first use with the current settings. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Audio")
	bool StartCapture();

	UFUNCTION(BlueprintCallable, Category = "WebGPU|Audio")
	void StopCapture();

	UFUNCTION(BlueprintPure, Category = "WebGPU|Audio")
	bool IsCapturing() const { return Listener.IsValid(); }

	/** Interleaved samples into the ring (any source). */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Audio")
	void PushAudio(const TArray<float>& Samples, int32 InChannels = 1, int32 SampleRate = 48000);

	/** array<f32>: the last Ring Seconds of audio, interleaved. */
	UFUNCTION(BlueprintPure, Category = "WebGPU|Audio")
	UWebGPUBuffer* GetRingBuffer();

	/** WebGPUAudioRing (FramesWritten, RingFrames, Channels, SampleRate), bindable as uniform or storage. */
	UFUNCTION(BlueprintPure, Category = "WebGPU|Audio")
	UWebGPUBuffer* GetCursorBuffer();

	UFUNCTION(BlueprintPure, Category = "WebGPU|Audio")
	int64 GetFramesWritten() const;

	UFUNCTION(BlueprintPure, Category = "WebGPU|Audio")
	int32 GetSampleRate() const;

	/** The native input (created on first use). */
	TSharedPtr<FWebGPUAudioInput> GetInput();

	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

private:
	int32 DeviceSampleRate() const;

	TSharedPtr<FWebGPUAudioInput> Input;
	TSharedPtr<ISubmixBufferListener, ESPMode::ThreadSafe> Listener;
	TWeakObjectPtr<USoundSubmix> ListeningTo;
	uint64 NotifiedFrames = 0;

	UPROPERTY()
	TObjectPtr<UWebGPUBuffer> RingObject;

	UPROPERTY()
	TObjectPtr<UWebGPUBuffer> CursorObject;
};

/**
 * Plays audio a WGSL kernel generates on the GPU. The kernel binds "block" (var<uniform> WebGPUAudioBlock:
 * FirstFrame, Frames, Channels, SampleRate, declared for it) and "samples" (var<storage, read_write>
 * array<f32>, interleaved); one invocation per frame works well. Set Kernel Source swaps the kernel while
 * playing; the previous one keeps playing until the new one compiles.
 */
UCLASS(ClassGroup = "WebGPU", meta = (BlueprintSpawnableComponent))
class WEBGPUCOMPUTE_API UWebGPUSynthComponent : public USynthComponent
{
	GENERATED_BODY()

public:
	UWebGPUSynthComponent(const FObjectInitializer& ObjectInitializer);

	/** WGSL; empty plays DefaultKernelSource (a quiet 440 Hz sine). */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "WebGPU|Audio", meta = (MultiLine = true))
	FString KernelSource;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "WebGPU|Audio", meta = (ClampMin = "1", ClampMax = "8"))
	int32 Channels = 2;

	/** Frames per kernel dispatch. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "WebGPU|Audio", meta = (ClampMin = "64"))
	int32 BlockFrames = 512;

	/** Blocks generated ahead of playback: the latency budget for the GPU round trip. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "WebGPU|Audio", meta = (ClampMin = "1", ClampMax = "16"))
	int32 BlocksAhead = 2;

	/** Compiles new WGSL; the current kernel plays until it succeeds. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Audio")
	void SetKernelSource(const FString& Source);

	/** The kernel, e.g. to bind more inputs by name. Valid once the synth was started. */
	UFUNCTION(BlueprintPure, Category = "WebGPU|Audio")
	UWebGPUKernel* GetKernel() const { return KernelObject; }

	UFUNCTION(BlueprintPure, Category = "WebGPU|Audio")
	FWebGPUAudioStats GetAudioStats() const;

	static FString DefaultKernelSource();

	TSharedPtr<FWebGPUAudioGenerator> GetGenerator() const;

	// USynthComponent (public so the generator can also be driven without an audio device)
	virtual bool Init(int32& SampleRate) override;
	virtual ISoundGeneratorPtr CreateSoundGenerator(const FSoundGeneratorInitParams& InParams) override;

private:
	UPROPERTY()
	TObjectPtr<UWebGPUKernel> KernelObject;

	mutable FCriticalSection GeneratorLock;
	TSharedPtr<FWebGPUAudioGenerator> Generator;
};
