// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "webgpu/webgpu.h"
#include "WebGPUTypes.h"

class FRHITexture;
class FWebGPUDevice;
struct FWebGPUSharedTextureState;
struct FWebGPUTextureAccessResult;

/**
 * Sharing UE RHI textures with the runtime's device (FWebGPUContext::ImportTexture). D3D12 only for now.
 *
 * Per access window: UE signals a fence after the rendering enqueued before BeginTextureAccess, the WebGPU
 * queue waits for it on the GPU, and EndTextureAccess hands WebGPU's fences to UE, whose RHI thread waits
 * for them on the GPU before the rendering enqueued afterwards.
 */
namespace WebGPUTextureInterop
{
	/** Game thread: checks the texture can be shared and creates UE's fence. Null with OutError otherwise. */
	TSharedPtr<FWebGPUSharedTextureState> CreateState(FRHITexture* Texture, FIntPoint& OutSize, EWebGPUTextureFormat& OutFormat, EWebGPUTextureUsage& OutUsage, FString& OutError);

	/** Worker: imports the texture and UE's fence into Device. Null with OutError on failure. */
	WGPUTexture Import(FWebGPUSharedTextureState& State, FWebGPUDevice& Device, FString& OutError);

	/** Worker: releases what Import created, ending an access in progress. */
	void Release(FWebGPUSharedTextureState& State, WGPUTexture Texture, bool bAccessing);

	/** Worker: the WebGPU queue will wait for UE's fence to reach SignalValue. */
	bool BeginAccess(FWebGPUSharedTextureState& State, WGPUTexture Texture, uint64 SignalValue);

	/** Worker, after the submit that used the texture: records the fences UE has to wait for. */
	void EndAccess(FWebGPUSharedTextureState& State, WGPUTexture Texture, FWebGPUTextureAccessResult& Result);

	/** Game thread: the next value UE signals. */
	uint64 NextSignalValue(FWebGPUSharedTextureState& State);

	/** Game thread: UE signals SignalValue after the rendering enqueued so far. */
	void EnqueueSignal(const TSharedRef<FWebGPUSharedTextureState>& State, uint64 SignalValue);

	/** Game thread: UE rendering enqueued after this waits on the GPU for the fences the worker puts in Result. */
	void EnqueueWait(const TSharedRef<FWebGPUTextureAccessResult>& Result);

	TSharedRef<FWebGPUTextureAccessResult> MakeAccessResult();

	/** Worker: Result is final (possibly without fences); the RHI thread may go on. */
	void CompleteAccess(FWebGPUTextureAccessResult& Result);
}
