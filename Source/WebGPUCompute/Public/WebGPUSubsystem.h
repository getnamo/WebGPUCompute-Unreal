// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/EngineSubsystem.h"
#include "WebGPUTypes.h"
#include "WebGPUSubsystem.generated.h"

class FWebGPUContext;
class FWebGPUDevice;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FWebGPUDeviceRecoveredSignature, int32, Generation);

/** Owns the process-wide WebGPU runtime (device + worker thread), created on first use. */
UCLASS()
class WEBGPUCOMPUTE_API UWebGPUSubsystem : public UEngineSubsystem
{
	GENERATED_BODY()

public:
	virtual void Deinitialize() override;

	/** The shared runtime, or null if WebGPU is unavailable. Safe to call from any thread. */
	TSharedPtr<FWebGPUContext> GetContext();

	/** The runtime's current device, for direct Dawn access (e.g. shared textures). Replaced after a device loss. */
	TSharedPtr<FWebGPUDevice> GetDevice();

	/** Convenience accessors through GEngine. */
	static TSharedPtr<FWebGPUContext> GetSharedContext();
	static TSharedPtr<FWebGPUDevice> GetSharedDevice();

	/** Fires on the game thread after the device was recreated; buffer contents are zeroed, so re-upload them. */
	UPROPERTY(BlueprintAssignable, Category = "WebGPU")
	FWebGPUDeviceRecoveredSignature OnDeviceRecovered;

	UFUNCTION(BlueprintPure, Category = "WebGPU")
	FString GetAdapterDescription();

	UFUNCTION(BlueprintPure, Category = "WebGPU")
	FWebGPUStats GetStats();

private:
	void HandleDeviceRecovered(uint32 Generation);

	FCriticalSection ContextLock;
	TSharedPtr<FWebGPUContext> Context;
	bool bInitAttempted = false;
};
