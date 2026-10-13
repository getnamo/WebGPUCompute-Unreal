// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "WebGPUTypes.generated.h"

/** How a buffer is bound in shaders. Copy usages are always added so buffers can be written and read back. */
UENUM(BlueprintType, meta = (Bitflags, UseEnumValuesAsMaskValuesInEditor = "true"))
enum class EWebGPUBufferUsage : uint8
{
	None = 0 UMETA(Hidden),
	/** var<storage> */
	Storage = 1 << 0,
	/** var<uniform> */
	Uniform = 1 << 1,
	/** Indirect dispatch arguments */
	Indirect = 1 << 2,
};
ENUM_CLASS_FLAGS(EWebGPUBufferUsage)

/** Where async results are delivered. */
enum class EWebGPUCallbackThread : uint8
{
	/** Via the game thread task queue (default for gameplay code, Blueprint and JS). */
	GameThread,
	/** Directly on the WebGPU worker thread. Keep the callback short; never block on WebGPU from it. */
	WorkerThread,
};

/** Runtime counters, cumulative since the context started. */
USTRUCT(BlueprintType)
struct FWebGPUStats
{
	GENERATED_BODY()

	/** Command buffers submitted (dispatches enqueued together are batched into one). */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU")
	int64 Submits = 0;

	UPROPERTY(BlueprintReadOnly, Category = "WebGPU")
	int64 Dispatches = 0;

	/** Dispatches skipped because the kernel had no compiled pipeline yet. */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU")
	int64 SkippedDispatches = 0;

	/** Submits rejected by validation (every dispatch in the batch is dropped). */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU")
	int64 FailedSubmits = 0;

	UPROPERTY(BlueprintReadOnly, Category = "WebGPU")
	int64 BytesWritten = 0;

	UPROPERTY(BlueprintReadOnly, Category = "WebGPU")
	int64 BytesRead = 0;

	UPROPERTY(BlueprintReadOnly, Category = "WebGPU")
	int64 Readbacks = 0;

	/** Increments every time the device is recreated after a loss. Starts at 1. */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU")
	int32 DeviceGeneration = 0;
};
