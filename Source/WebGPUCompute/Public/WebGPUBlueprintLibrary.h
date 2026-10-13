// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "Kismet/BlueprintAsyncActionBase.h"
#include "WebGPUStreams.h"
#include "WebGPUStructLayout.h"
#include "WebGPUTypes.h"
#include "WebGPUBlueprintLibrary.generated.h"

class UWebGPUBuffer;
class UWebGPUKernel;
class UWebGPUReadbackStream;
class UWebGPUStructMirror;
class UWebGPUTexture;
class UTextureRenderTarget2D;

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

	/** A buffer of Count elements of a struct, in the struct's WGSL layout (Uniform-only usage uses uniform rules). */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Structs")
	static UWebGPUBuffer* CreateWebGPUStructBuffer(UScriptStruct* StructType, int32 Count,
		UPARAM(meta = (Bitmask, BitmaskEnum = "/Script/WebGPUCompute.EWebGPUBufferUsage")) int32 Usage = 1,
		const FString& Label = TEXT(""));

	/** Like CreateWebGPUKernel, with the structs declared in WGSL ahead of the source so the shader can use them by name. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Structs")
	static UWebGPUKernel* CreateWebGPUKernelWithStructs(const FString& Source, const TArray<UScriptStruct*>& Structs,
		const FString& EntryPoint = TEXT("main"), const FString& Label = TEXT(""));

	/** The WGSL declaration of a struct (and the structs it nests), e.g. to paste into a shader. Empty if unsupported. */
	UFUNCTION(BlueprintPure, Category = "WebGPU|Structs")
	static FString GetWGSLForStruct(UScriptStruct* StructType, EWebGPUAddressSpace Space = EWebGPUAddressSpace::Storage);

	/** Writes an array of structs (any struct type) into a buffer from element FirstElement on. */
	UFUNCTION(BlueprintCallable, CustomThunk, Category = "WebGPU|Structs", meta = (ArrayParm = "Items"))
	static bool WriteWebGPUStructArray(UWebGPUBuffer* Buffer, const TArray<int32>& Items, int32 FirstElement = 0);
	DECLARE_FUNCTION(execWriteWebGPUStructArray);

	/** Unpacks bytes read from a struct buffer into an array of the connected struct type. */
	UFUNCTION(BlueprintCallable, CustomThunk, Category = "WebGPU|Structs", meta = (ArrayParm = "OutItems"))
	static bool WebGPUBytesToStructArray(const TArray<uint8>& Bytes, TArray<int32>& OutItems);
	DECLARE_FUNCTION(execWebGPUBytesToStructArray);

	/** A 2D texture for kernels (zeroed). Storage use needs a WGSL storage format (RGBA8Unorm, RGBA16Float, RGBA32Float, R32Float, ...). */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Texture")
	static UWebGPUTexture* CreateWebGPUTexture(int32 Width, int32 Height, EWebGPUTextureFormat Format = EWebGPUTextureFormat::RGBA8Unorm,
		UPARAM(meta = (Bitmask, BitmaskEnum = "/Script/WebGPUCompute.EWebGPUTextureUsage")) int32 Usage = 3,
		const FString& Label = TEXT(""));

	/**
	 * Shares a render target with kernels without copies (D3D12 RHI). The render target needs Shared
	 * (bGPUSharedFlag) set, plus Can Create UAV for kernels that write it. Use the texture between its
	 * Begin Access and End Access. Null (with a warning) when it can't be shared.
	 */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Texture")
	static UWebGPUTexture* CreateWebGPUTextureFromRenderTarget(UTextureRenderTarget2D* RenderTarget, const FString& Label = TEXT(""));

	/** A readback stream whose slots hold a whole texture (rows are 256-byte aligned in the slots). */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Stream")
	static UWebGPUReadbackStream* CreateWebGPUReadbackStreamForTexture(UWebGPUTexture* Texture, int32 Slots = 3, EWebGPUStreamPolicy Policy = EWebGPUStreamPolicy::Latest, const FString& Label = TEXT(""));

	/** Per-frame results from the GPU: a ring of Slots readback slots of SlotBytes each (see EWebGPUStreamPolicy). */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Stream")
	static UWebGPUReadbackStream* CreateWebGPUReadbackStream(int64 SlotBytes, int32 Slots = 3, EWebGPUStreamPolicy Policy = EWebGPUStreamPolicy::Latest, const FString& Label = TEXT(""));

	/** A readback stream whose slots hold all of Buffer. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Stream")
	static UWebGPUReadbackStream* CreateWebGPUReadbackStreamForBuffer(UWebGPUBuffer* Buffer, int32 Slots = 3, EWebGPUStreamPolicy Policy = EWebGPUStreamPolicy::Latest, const FString& Label = TEXT(""));

	/** A struct array kept on the GPU and the CPU: set elements, then Sync uploads only what changed. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Mirror")
	static UWebGPUStructMirror* CreateWebGPUStructMirror(UScriptStruct* StructType, int32 Count,
		UPARAM(meta = (Bitmask, BitmaskEnum = "/Script/WebGPUCompute.EWebGPUBufferUsage")) int32 Usage = 1,
		const FString& Label = TEXT(""));

	/** Implementations of the wildcard nodes, for C++ callers holding an array property. */
	static bool GenericWriteStructArray(UWebGPUBuffer* Buffer, const FArrayProperty* ArrayProperty, const void* ArrayAddress, int32 FirstElement);
	static bool GenericBytesToStructArray(const TArray<uint8>& Bytes, const FArrayProperty* ArrayProperty, void* ArrayAddress);
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
