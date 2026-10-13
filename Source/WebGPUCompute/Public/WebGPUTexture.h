// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "WebGPUContext.h"
#include "WebGPUTexture.generated.h"

/**
 * A 2D texture for compute kernels (create with CreateWebGPUTexture), or a UE render target shared without
 * copies (CreateWebGPUTextureFromRenderTarget, D3D12). Bind it with the kernel's Set Texture By Name.
 */
UCLASS(BlueprintType)
class WEBGPUCOMPUTE_API UWebGPUTexture : public UObject
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintPure, Category = "WebGPU|Texture")
	int32 GetWidth() const;

	UFUNCTION(BlueprintPure, Category = "WebGPU|Texture")
	int32 GetHeight() const;

	UFUNCTION(BlueprintPure, Category = "WebGPU|Texture")
	EWebGPUTextureFormat GetFormat() const;

	/** A UE render target shared without copies: use it between Begin Access and End Access. */
	UFUNCTION(BlueprintPure, Category = "WebGPU|Texture")
	bool IsShared() const;

	/** Shared textures: WebGPU commands after this see UE's rendering so far. No-op for other textures. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Texture")
	void BeginAccess();

	/** Shared textures: UE rendering after this sees the WebGPU commands before it. No-op for other textures. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Texture")
	void EndAccess();

	/** All pixels, rows tightly packed (Width * bytes per pixel each). */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Texture")
	void WriteBytes(const TArray<uint8>& Pixels);

	/** All pixels as floats (float formats: R32Float, RG32Float, RGBA32Float). */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Texture")
	void WriteFloats(const TArray<float>& Values);

	/** Frees the GPU memory (or stops sharing) now instead of at garbage collection. */
	UFUNCTION(BlueprintCallable, Category = "WebGPU|Texture")
	void Release();

	/** Reads all pixels back after earlier commands, rows tightly packed. */
	void Read(FWebGPUReadCallback&& Callback, EWebGPUCallbackThread Thread = EWebGPUCallbackThread::GameThread);

	void Initialize(const TSharedRef<FWebGPUTexture>& InNative) { Native = InNative; }
	const TSharedPtr<FWebGPUTexture>& GetNative() const { return Native; }

	virtual void BeginDestroy() override;

private:
	TSharedPtr<FWebGPUContext> GetContext() const;

	TSharedPtr<FWebGPUTexture> Native;
};
