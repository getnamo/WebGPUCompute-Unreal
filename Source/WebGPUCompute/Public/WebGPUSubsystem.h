// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/EngineSubsystem.h"
#include "WebGPUSubsystem.generated.h"

class FWebGPUDevice;

/** Owns the process-wide WebGPU device, created on first use. */
UCLASS()
class WEBGPUCOMPUTE_API UWebGPUSubsystem : public UEngineSubsystem
{
	GENERATED_BODY()

public:
	virtual void Deinitialize() override;

	/** The shared device, or null if WebGPU is unavailable. Safe to call from any thread. */
	TSharedPtr<FWebGPUDevice> GetDevice();

	/** Convenience accessor through GEngine. */
	static TSharedPtr<FWebGPUDevice> GetSharedDevice();

	UFUNCTION(BlueprintPure, Category = "WebGPU")
	FString GetAdapterDescription();

private:
	FCriticalSection DeviceLock;
	TSharedPtr<FWebGPUDevice> Device;
	bool bInitAttempted = false;
};
