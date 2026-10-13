// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "WebGPUStreams.h"
#include "WebGPUReadbackStream.generated.h"

class UWebGPUBuffer;
class UWebGPUTexture;

/** The frame a UWebGPUReadbackStream read last. */
USTRUCT(BlueprintType)
struct WEBGPUCOMPUTE_API FWebGPUFrameInfo
{
	GENERATED_BODY()

	/** The capture's number in its stream, from 1 (0: nothing read yet). */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU|Stream")
	int64 Sequence = 0;

	/** Engine frame (GFrameCounter) of the capture. */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU|Stream")
	int64 CaptureFrame = 0;

	/** Engine frames between the capture and the read. */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU|Stream")
	int32 FramesLate = 0;

	/** Milliseconds from the capture until the data was back on the CPU. */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU|Stream")
	float RoundTripMs = 0.f;

	UPROPERTY(BlueprintReadOnly, Category = "WebGPU|Stream")
	int64 Bytes = 0;
};

/**
 * Per-frame results from the GPU (create with CreateWebGPUReadbackStream): Capture a buffer after
 * dispatching, then Read what came back, typically every tick. EWebGPUStreamPolicy decides which frame
 * a read returns and what happens when slots run out.
 */
UCLASS(BlueprintType)
class WEBGPUCOMPUTE_API UWebGPUReadbackStream : public UObject
{
	GENERATED_BODY()

public:
	/** Copies a byte range of Buffer (Size < 0: to the end) into a free slot, after all earlier commands. False when dropped. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Stream")
	bool Capture(UWebGPUBuffer* Buffer, int64 Offset = 0, int64 Size = -1);

	/** Copies a whole texture into a free slot, after all earlier commands. False when dropped. Reads return its pixels, rows tightly packed. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Stream")
	bool CaptureTexture(UWebGPUTexture* Texture);

	/** The next frame as bytes (Latest: the newest; Queue and Lockstep: the oldest unread). False when nothing new is ready. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Stream")
	bool ReadBytes(TArray<uint8>& OutBytes);

	UFUNCTION(BlueprintCallable, Category = "WebGPU|Stream")
	bool ReadFloats(TArray<float>& OutValues);

	UFUNCTION(BlueprintCallable, Category = "WebGPU|Stream")
	bool ReadInts(TArray<int32>& OutValues);

	/** The next frame unpacked into an array of the connected struct type (WGSL storage layout). */
	UFUNCTION(BlueprintCallable, CustomThunk, Category = "WebGPU|Stream", meta = (ArrayParm = "OutItems"))
	bool ReadStructs(TArray<int32>& OutItems);
	DECLARE_FUNCTION(execReadStructs);

	/** Frames ready to read. */
	UFUNCTION(BlueprintPure, Category = "WebGPU|Stream")
	int32 GetReadyCount() const;

	UFUNCTION(BlueprintPure, Category = "WebGPU|Stream")
	FWebGPUStreamStats GetStats() const;

	/** The frame the last successful read returned. */
	UFUNCTION(BlueprintPure, Category = "WebGPU|Stream")
	FWebGPUFrameInfo GetLastFrameInfo() const { return LastFrame; }

	/** Frees the slots now instead of at garbage collection. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Stream")
	void Release();

	/** Reads the next frame in place, without copying; the frame is released when Reader returns. */
	bool ReadFrame(TFunctionRef<void(const FWebGPUReadbackFrame& Frame)> Reader);

	/** ReadStructs for C++ callers holding an array property. */
	bool GenericReadStructs(const FArrayProperty* ArrayProperty, void* ArrayAddress);

	void Initialize(const TSharedRef<FWebGPUReadbackStream>& InNative) { Native = InNative; }
	const TSharedPtr<FWebGPUReadbackStream>& GetNative() const { return Native; }

	virtual void BeginDestroy() override;

private:
	TSharedPtr<FWebGPUReadbackStream> Native;
	FWebGPUFrameInfo LastFrame;
};
