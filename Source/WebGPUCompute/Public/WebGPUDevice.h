// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "webgpu/webgpu.h"
#include <atomic>

struct FWebGPUDeviceOptions
{
	/** Pick the adapter UE renders with (matched by LUID on D3D12) so textures can be shared. */
	bool bMatchRHIAdapter = true;

	/** Request shared texture/fence features when the adapter supports them. */
	bool bRequestInteropFeatures = true;

	/**
	 * FWebGPUContext worker: after a submit with results pending, poll Dawn without sleeping for this long
	 * before falling back to 1 ms sleeps. Trades a little CPU for sub-millisecond readback latency. 0 disables.
	 */
	float BusyPollMilliseconds = 2.0f;
};

struct FWebGPUCompileMessage
{
	bool bIsError = false;
	FString Message;
	uint64 Line = 0;
	uint64 Column = 0;
};

/**
 * Owns one Dawn instance/adapter/device/queue.
 * Calls are thread-safe when ImplicitDeviceSynchronization is available (requested by default).
 */
class WEBGPUCOMPUTE_API FWebGPUDevice
{
public:
	~FWebGPUDevice();

	bool Initialize(const FWebGPUDeviceOptions& Options = FWebGPUDeviceOptions());

	/** Called (from any thread) when the device is lost for a reason other than Shutdown. Set before Initialize. */
	void SetLostHandler(TFunction<void()>&& Handler) { LostHandler = MoveTemp(Handler); }
	void Shutdown();

	bool IsValid() const { return Device != nullptr && !bLost; }

	/** True once the device was lost (TDR, driver reset, ForceLoss). A lost device never recovers; make a new one. */
	bool IsLost() const { return bLost; }

	WGPUInstance GetInstance() const { return Instance; }
	WGPUAdapter GetAdapter() const { return Adapter; }
	WGPUDevice GetDevice() const { return Device; }
	WGPUQueue GetQueue() const { return Queue; }

	bool HasFeature(WGPUFeatureName Feature) const;
	const FString& GetAdapterDescription() const { return AdapterDescription; }
	bool IsUsingRHIAdapter() const { return bUsingRHIAdapter; }

	/** Blocks until the future completes. Returns false on timeout/error. */
	bool Wait(WGPUFuture Future, double TimeoutSeconds = 10.0);

	/** Blocks until all queue work submitted so far has completed on the GPU. */
	bool WaitForQueueIdle(double TimeoutSeconds = 10.0);

	/** Validation + OOM + internal error scopes around a block of calls. */
	void PushErrorScopes();
	/** Returns true when no error was captured since the matching PushErrorScopes. */
	bool PopErrorScopes(FString& OutError);

	/** Compiles WGSL. Returns null on error; messages carry line/column. */
	WGPUShaderModule CreateShaderModule(const FString& Source, const FString& Label, TArray<FWebGPUCompileMessage>& OutMessages);

	/**
	 * Runs a compute shader over one u32 storage buffer at @group(0) @binding(0) with entry point "main",
	 * dispatching one workgroup per element. Kept for the original proof of concept API.
	 */
	bool RunBufferKernel(const FString& Source, const TArray<uint32>& InData, TArray<uint32>& OutData, FString& OutError);

	static FString ToFString(WGPUStringView View);
	static FString FormatCompileMessages(const TArray<FWebGPUCompileMessage>& Messages);

private:
	WGPUInstance Instance = nullptr;
	WGPUAdapter Adapter = nullptr;
	WGPUDevice Device = nullptr;
	WGPUQueue Queue = nullptr;

	FString AdapterDescription;
	bool bUsingRHIAdapter = false;
	std::atomic<bool> bLost = false;
	TFunction<void()> LostHandler;
};
