// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "Kismet/BlueprintAsyncActionBase.h"
#include "WebGPUTypes.h"
#include "WebGPUBlueprintLibrary.generated.h"

class UWebGPUBuffer;
class UWebGPUKernel;

UCLASS()
class WEBGPUCOMPUTE_API UWebGPUBlueprintLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	/** A GPU buffer of Size bytes (zero-initialized). Null if WebGPU is unavailable. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU")
	static UWebGPUBuffer* CreateWebGPUBuffer(int64 Size,
		UPARAM(meta = (Bitmask, BitmaskEnum = "/Script/WebGPUCompute.EWebGPUBufferUsage")) int32 Usage = 1,
		const FString& Label = TEXT(""));

	/** Compiles WGSL asynchronously; bind OnCompiled or poll IsReady. Null if WebGPU is unavailable. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU")
	static UWebGPUKernel* CreateWebGPUKernel(const FString& Source, const FString& EntryPoint = TEXT("main"), const FString& Label = TEXT(""));

	/** Workgroup count that covers Count items: ceil(Count / WorkgroupSize). */
	UFUNCTION(BlueprintPure, Category = "WebGPU")
	static int32 WorkgroupsForCount(int32 Count, int32 WorkgroupSize = 64);

	UFUNCTION(BlueprintPure, Category = "WebGPU")
	static TArray<float> WebGPUBytesToFloats(const TArray<uint8>& Bytes);

	UFUNCTION(BlueprintPure, Category = "WebGPU")
	static TArray<int32> WebGPUBytesToInts(const TArray<uint8>& Bytes);

	UFUNCTION(BlueprintPure, Category = "WebGPU")
	static FWebGPUStats GetWebGPUStats();
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FWebGPUReadBufferPin, const TArray<uint8>&, Data);

/** Reads a WebGPU buffer back to the CPU after all earlier commands. */
UCLASS()
class WEBGPUCOMPUTE_API UWebGPUReadBufferAction : public UBlueprintAsyncActionBase
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintAssignable)
	FWebGPUReadBufferPin OnCompleted;

	UPROPERTY(BlueprintAssignable)
	FWebGPUReadBufferPin OnFailed;

	/** Offset and Size are in bytes and must be multiples of 4. Size < 0 reads to the end. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU", meta = (BlueprintInternalUseOnly = "true"))
	static UWebGPUReadBufferAction* ReadWebGPUBuffer(UWebGPUBuffer* Buffer, int64 Offset = 0, int64 Size = -1);

	virtual void Activate() override;

private:
	UPROPERTY()
	TObjectPtr<UWebGPUBuffer> Buffer;

	int64 Offset = 0;
	int64 Size = -1;
};
