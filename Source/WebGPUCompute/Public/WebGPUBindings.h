// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "WebGPUContext.h"
#include "WebGPUBindings.generated.h"

class UWebGPUBuffer;
class UWebGPUTexture;

/**
 * Buffers, textures and uniform values shared by several kernels, by WGSL name (create with
 * UWebGPUBlueprintLibrary::CreateWebGPUBindings). A kernel created with (or given) these bindings takes every
 * binding it doesn't set itself from here: one Create Buffer "agents" serves every kernel that declares
 * `var<storage> agents`. Set Float / Int / Vector write members of uniform structs by name ("params.dt");
 * the struct's layout comes from each shader, so editing it during hot reload keeps the values in place.
 */
UCLASS(BlueprintType)
class WEBGPUCOMPUTE_API UWebGPUBindings : public UObject
{
	GENERATED_BODY()

public:
	/** A new zeroed buffer of Count elements of Stride bytes, bound as Name. Replaces an earlier one of that name. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Bindings")
	UWebGPUBuffer* CreateBuffer(const FString& Name, int32 Count, int32 Stride = 4,
		UPARAM(meta = (Bitmask, BitmaskEnum = "/Script/WebGPUCompute.EWebGPUBufferUsage")) int32 Usage = 1);

	/** Binds an existing buffer as Name. None removes it. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Bindings")
	void SetBuffer(const FString& Name, UWebGPUBuffer* Buffer);

	/** Binds a texture (texture_2d or texture_storage_2d) as Name. None removes it. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Bindings")
	void SetTexture(const FString& Name, UWebGPUTexture* Texture);

	/** Picks the built-in sampler for sampler bindings named Name. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Bindings")
	void SetSampler(const FString& Name, EWebGPUSampler Sampler);

	/** Sets a uniform member by path: "params.dt" is member dt of `var<uniform> params`. i32/u32 members get it truncated. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Bindings")
	void SetFloat(const FString& Name, float Value);

	UFUNCTION(BlueprintCallable, Category = "WebGPU|Bindings")
	void SetInt(const FString& Name, int32 Value);

	/** Fills as many components as the member has (vec2/vec3/vec4 of f32, i32 or u32). */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Bindings")
	void SetVector(const FString& Name, FVector4 Value);

	/** The buffer bound as Name through Create Buffer or Set Buffer, if any. */
	UFUNCTION(BlueprintPure, Category = "WebGPU|Bindings")
	UWebGPUBuffer* GetBuffer(const FString& Name) const;

	/** Frees the buffers Create Buffer made and drops every binding now, instead of at garbage collection. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Bindings")
	void Release();

	void Initialize(const TSharedRef<FWebGPUBindingSet>& InNative) { Native = InNative; }
	const TSharedPtr<FWebGPUBindingSet>& GetNative() const { return Native; }

private:
	TSharedPtr<FWebGPUContext> GetContext() const { return Native ? Native->GetContext() : nullptr; }

	TSharedPtr<FWebGPUBindingSet> Native;

	UPROPERTY()
	TMap<FString, TObjectPtr<UWebGPUBuffer>> Buffers;

	UPROPERTY()
	TMap<FString, TObjectPtr<UWebGPUTexture>> Textures;
};
