// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "WebGPUStreams.h"
#include "WebGPUStructMirror.generated.h"

class UWebGPUBuffer;

/**
 * A struct array kept on the GPU and the CPU (create with CreateWebGPUStructMirror): set elements, then
 * Sync uploads only what changed. Bind GetBuffer() to kernels. Meant for data the CPU owns (grids, goals,
 * parameters); state a kernel writes belongs in a separate buffer, since a Sync overwrites what it uploads.
 */
UCLASS(BlueprintType)
class WEBGPUCOMPUTE_API UWebGPUStructMirror : public UObject
{
	GENERATED_BODY()

public:
	/** Sets element Index from a struct of the mirror's type. Uploaded by the next Sync. */
	UFUNCTION(BlueprintCallable, CustomThunk, Category = "WebGPU|Mirror", meta = (CustomStructureParam = "Item"))
	bool SetElement(int32 Index, const int32& Item);
	DECLARE_FUNCTION(execSetElement);

	/** Copies element Index into a struct of the mirror's type. */
	UFUNCTION(BlueprintCallable, CustomThunk, Category = "WebGPU|Mirror", meta = (CustomStructureParam = "OutItem"))
	bool GetElement(int32 Index, int32& OutItem);
	DECLARE_FUNCTION(execGetElement);

	/** Sets Items.Num() elements from FirstIndex on. */
	UFUNCTION(BlueprintCallable, CustomThunk, Category = "WebGPU|Mirror", meta = (ArrayParm = "Items"))
	bool SetElements(const TArray<int32>& Items, int32 FirstIndex = 0);
	DECLARE_FUNCTION(execSetElements);

	/** Uploads what changed since the last Sync, in order with other commands. Returns the bytes uploaded. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Mirror")
	int64 Sync();

	/** The GPU buffer, to bind to kernels. */
	UFUNCTION(BlueprintPure, Category = "WebGPU|Mirror")
	UWebGPUBuffer* GetBuffer() const { return Buffer; }

	UFUNCTION(BlueprintPure, Category = "WebGPU|Mirror")
	int32 Num() const;

	/** Elements changed since the last Sync. */
	UFUNCTION(BlueprintPure, Category = "WebGPU|Mirror")
	int32 GetDirtyCount() const;

	UFUNCTION(BlueprintPure, Category = "WebGPU|Mirror")
	UScriptStruct* GetStructType() const;

	/** Frees the CPU and GPU memory now instead of at garbage collection. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Mirror")
	void Release();

	/** Implementations of the wildcard nodes, for C++ callers. */
	bool GenericSetElement(int32 Index, const UScriptStruct* Struct, const void* Data);
	bool GenericGetElement(int32 Index, const UScriptStruct* Struct, void* Data) const;
	bool GenericSetElements(const FArrayProperty* ArrayProperty, const void* ArrayAddress, int32 FirstIndex);

	void Initialize(const TSharedRef<FWebGPUStructMirror>& InNative, UWebGPUBuffer* InBuffer);
	const TSharedPtr<FWebGPUStructMirror>& GetNative() const { return Native; }

	virtual void BeginDestroy() override;

private:
	bool CheckStruct(const UScriptStruct* Struct, const TCHAR* Function) const;

	TSharedPtr<FWebGPUStructMirror> Native;

	UPROPERTY()
	TObjectPtr<UWebGPUBuffer> Buffer;
};
