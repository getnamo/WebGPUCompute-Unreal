// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "webgpu/webgpu.h"

class FRHITexture;
class FRHICommandListImmediate;
class FWebGPUDevice;

/**
 * A UE RHI texture shared with the WebGPU (Dawn) device without copies.
 * D3D12 RHI only: the texture must be created with ETextureCreateFlags::Shared
 * (UTextureRenderTarget2D::bGPUSharedFlag), plus UAV if WebGPU writes it as a storage texture.
 *
 * Sync is GPU-side: BeginAccess makes the WebGPU queue wait for UE work enqueued so far,
 * EndAccess makes UE work enqueued afterwards wait for the WebGPU work. Call everything on the render thread.
 */
class WEBGPUCOMPUTE_API FWebGPUSharedTexture
{
public:
	static TSharedPtr<FWebGPUSharedTexture> Create(const TSharedRef<FWebGPUDevice>& Device, FRHITexture* Texture, FString& OutError);
	~FWebGPUSharedTexture();

	WGPUTexture GetTexture() const;
	WGPUTextureFormat GetFormat() const;
	WGPUTextureUsage GetUsage() const;
	FIntPoint GetSize() const;

	/**
	 * Grants WebGPU access. Its queue waits on the GPU for UE commands enqueued on RHICmdList before this call.
	 * bSubmitUEWork flushes the RHI thread so that UE work reaches the GPU now rather than at the end of the frame.
	 */
	bool BeginAccess(FRHICommandListImmediate& RHICmdList, bool bSubmitUEWork = true);

	/** Ends WebGPU access (after its queue submit). UE commands enqueued after this wait on the GPU for WebGPU. */
	bool EndAccess(FRHICommandListImmediate& RHICmdList);

private:
	FWebGPUSharedTexture();

	struct FImpl;
	TUniquePtr<FImpl> Impl;
};
