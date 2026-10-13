// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "WebGPUContext.h"
#include "WebGPUKernel.generated.h"

class UWebGPUBuffer;
class UWebGPUTexture;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FWebGPUKernelCompiledSignature, bool, bSuccess, const FString&, Messages);

/**
 * A WGSL compute kernel (create with UWebGPUBlueprintLibrary::CreateWebGPUKernel). Compiles asynchronously;
 * Recompile keeps the last good pipeline running until the new source compiles.
 */
UCLASS(BlueprintType)
class WEBGPUCOMPUTE_API UWebGPUKernel : public UObject
{
	GENERATED_BODY()

public:
	/** Fires on the game thread after every compile: the first, each Recompile, and after device recovery. */
	UPROPERTY(BlueprintAssignable, Category = "WebGPU")
	FWebGPUKernelCompiledSignature OnCompiled;

	/** Binds a buffer to @group(Group) @binding(Binding). None unbinds. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU")
	void SetBuffer(int32 Binding, UWebGPUBuffer* Buffer, int32 Group = 0);

	/** Binds by WGSL variable name (e.g. "agents"); follows the binding if a shader edit moves it. None unbinds. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU")
	void SetBufferByName(const FString& Name, UWebGPUBuffer* Buffer);

	/** Binds a texture (texture_2d or texture_storage_2d) by WGSL variable name. None unbinds. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU")
	void SetTextureByName(const FString& Name, UWebGPUTexture* Texture);

	/** Picks the built-in sampler for a sampler binding (unset samplers use Linear Clamp). */
	UFUNCTION(BlueprintCallable, Category = "WebGPU")
	void SetSamplerByName(const FString& Name, EWebGPUSampler Sampler);

	/** Dispatches enough workgroups to cover X*Y*Z invocations, using the shader's @workgroup_size. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU")
	void DispatchForCount(int32 X, int32 Y = 1, int32 Z = 1);

	/** Resources the entry point uses (from the last successful compile). */
	UFUNCTION(BlueprintPure, Category = "WebGPU")
	TArray<FWebGPUBindingInfo> GetBindings() const;

	/** The entry point's @workgroup_size (zero before the first successful compile). */
	UFUNCTION(BlueprintPure, Category = "WebGPU")
	FIntVector GetWorkgroupSize() const;

	/** The generated struct preamble plus the source: what was compiled. */
	UFUNCTION(BlueprintPure, Category = "WebGPU")
	FString GetCompiledSource() const;

	/** Workgroup counts per dimension. Skipped until the kernel has compiled once. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU")
	void Dispatch(int32 X, int32 Y = 1, int32 Z = 1);

	UFUNCTION(BlueprintCallable, Category = "WebGPU")
	void Recompile(const FString& Source);

	UFUNCTION(BlueprintPure, Category = "WebGPU")
	bool IsReady() const;

	/** Errors and warnings of the most recent compile, one per line as "error:line:column: message". */
	UFUNCTION(BlueprintPure, Category = "WebGPU")
	FString GetCompileMessages() const;

	/** Times each dispatch on the GPU (see GetLastGpuTimeMs). */
	UFUNCTION(BlueprintCallable, Category = "WebGPU")
	void SetProfiling(bool bEnabled);

	UFUNCTION(BlueprintPure, Category = "WebGPU")
	float GetLastGpuTimeMs() const;

	UFUNCTION(BlueprintPure, Category = "WebGPU")
	float GetAverageGpuTimeMs() const;

	UFUNCTION(BlueprintCallable, Category = "WebGPU")
	void Release();

	void Initialize(const TSharedRef<FWebGPUKernel>& InNative) { Native = InNative; }
	const TSharedPtr<FWebGPUKernel>& GetNative() const { return Native; }

	/** Game thread, from the native kernel's listener. */
	void HandleCompiled(bool bSuccess, const TArray<FWebGPUCompileMessage>& Messages);

	virtual void BeginDestroy() override;

private:
	TSharedPtr<FWebGPUContext> GetContext() const;

	TSharedPtr<FWebGPUKernel> Native;
};
