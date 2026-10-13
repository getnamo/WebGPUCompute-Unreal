// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "WebGPUContext.h"
#include "WebGPUBuffer.generated.h"

/**
 * A GPU buffer for compute kernels (create with UWebGPUBlueprintLibrary::CreateWebGPUBuffer).
 * Writes and reads are queued in order with dispatches. GPU memory is freed on Release or garbage collection.
 */
UCLASS(BlueprintType)
class WEBGPUCOMPUTE_API UWebGPUBuffer : public UObject
{
	GENERATED_BODY()

public:
	/** Size in bytes. */
	UFUNCTION(BlueprintPure, Category = "WebGPU")
	int64 GetSize() const;

	/** Byte offset must be a multiple of 4; the data is padded to a multiple of 4. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU")
	void WriteBytes(const TArray<uint8>& Data, int64 Offset = 0);

	UFUNCTION(BlueprintCallable, Category = "WebGPU")
	void WriteFloats(const TArray<float>& Data, int64 Offset = 0);

	UFUNCTION(BlueprintCallable, Category = "WebGPU")
	void WriteInts(const TArray<int32>& Data, int64 Offset = 0);

	/** Frees the GPU memory now instead of at garbage collection. Kernels still bound to it keep it alive. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU")
	void Release();

	UFUNCTION(BlueprintPure, Category = "WebGPU")
	bool IsBufferValid() const { return Native.IsValid(); }

	/** Reads a byte range back after all earlier commands. Size < 0 reads to the end. */
	void Read(int64 Offset, int64 Size, FWebGPUReadCallback&& Callback, EWebGPUCallbackThread Thread = EWebGPUCallbackThread::GameThread);

	void Initialize(const TSharedRef<FWebGPUBuffer>& InNative) { Native = InNative; }
	const TSharedPtr<FWebGPUBuffer>& GetNative() const { return Native; }

	virtual void BeginDestroy() override;

private:
	void Write(TArray<uint8>&& Bytes, int64 Offset);

	TSharedPtr<FWebGPUBuffer> Native;
};
