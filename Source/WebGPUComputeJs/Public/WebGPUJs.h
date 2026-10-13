// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "WebGPUJs.generated.h"

class UWebGPUBuffer;
class UWebGPUReadbackStream;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FWebGPUJsReadSignature, bool, bSuccess);

/**
 * A buffer read for scripts (WebGPUJs.ReadBuffer): On Completed fires on the game thread, then
 * memory.access(request.GetMemory()) views the bytes in place until Release.
 */
UCLASS(BlueprintType)
class WEBGPUCOMPUTEJS_API UWebGPUJsRead : public UObject
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintAssignable, Category = "WebGPU|Script")
	FWebGPUJsReadSignature OnCompleted;

	/** The bytes read, as a JavascriptMemoryObject (null before completion or after Release). */
	UFUNCTION(BlueprintPure, Category = "WebGPU|Script")
	UObject* GetMemory() const { return Memory; }

	/** A copy of the bytes, for scripts without the 'memory' global. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Script")
	TArray<uint8> CopyBytes() const { return Bytes; }

	UFUNCTION(BlueprintPure, Category = "WebGPU|Script")
	bool IsDone() const { return bDone; }

	/** Detaches script views and frees the bytes. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Script")
	void Release();

	void Complete(bool bSuccess, TArray<uint8>&& InBytes);
	virtual void BeginDestroy() override;

private:
	TArray<uint8> Bytes;
	bool bDone = false;

	UPROPERTY()
	TObjectPtr<UObject> Memory;
};

/**
 * WebGPU for scripts (UnrealJs), used by Content/Scripts/webgpu.js. Everything else (kernels,
 * bindings, buffers, streams) is the regular Blueprint API, which scripts call directly; these add
 * what scripts need on top: GPU results viewed in place as ArrayBuffers, and writes straight from
 * script memory. Without UnrealJs in the project the functions return nothing and IsAvailable is false.
 */
UCLASS()
class WEBGPUCOMPUTEJS_API UWebGPUJs : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	/** True when the plugin was built with UnrealJs. */
	UFUNCTION(BlueprintPure, Category = "WebGPU|Script")
	static bool IsAvailable();

	/**
	 * Takes the stream's next frame (see the stream's policy) and lends it to scripts as a
	 * JavascriptExternalMemory: memory.access(frame) is an ArrayBuffer over the mapped GPU readback, no
	 * copy. Call frame.Release() when done: the slot goes back to the stream and every view of it is
	 * detached. Null when no frame is ready. The stream's GetLastFrameInfo describes the frame.
	 */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Script")
	static UObject* AcquireFrame(UWebGPUReadbackStream* Stream);

	/**
	 * Inside memory.exec(arrayBuffer, fn): writes Size bytes from SourceOffset of that ArrayBuffer into
	 * Buffer at Offset (all multiples of 4; Size < 0: to the end of the ArrayBuffer). One copy, into the
	 * upload memory of the next submit.
	 */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Script")
	static bool WriteBufferFromScript(UWebGPUBuffer* Buffer, int64 Offset = 0, int64 SourceOffset = 0, int64 Size = -1);

	/** Reads Size bytes (< 0: to the end) after all earlier commands; the request's On Completed fires on the game thread. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Script")
	static UWebGPUJsRead* ReadBuffer(UWebGPUBuffer* Buffer, int64 Offset = 0, int64 Size = -1);
};
