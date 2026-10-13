// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUSubsystem.h"
#include "WebGPUDevice.h"
#include "Engine/Engine.h"

void UWebGPUSubsystem::Deinitialize()
{
	FScopeLock Lock(&DeviceLock);
	if (Device)
	{
		Device->Shutdown();
		Device.Reset();
	}
	Super::Deinitialize();
}

TSharedPtr<FWebGPUDevice> UWebGPUSubsystem::GetDevice()
{
	FScopeLock Lock(&DeviceLock);
	if (!bInitAttempted)
	{
		bInitAttempted = true;
		TSharedPtr<FWebGPUDevice> NewDevice = MakeShared<FWebGPUDevice>();
		if (NewDevice->Initialize())
		{
			Device = NewDevice;
		}
	}
	return Device;
}

TSharedPtr<FWebGPUDevice> UWebGPUSubsystem::GetSharedDevice()
{
	UWebGPUSubsystem* Subsystem = GEngine ? GEngine->GetEngineSubsystem<UWebGPUSubsystem>() : nullptr;
	return Subsystem ? Subsystem->GetDevice() : nullptr;
}

FString UWebGPUSubsystem::GetAdapterDescription()
{
	TSharedPtr<FWebGPUDevice> Current = GetDevice();
	return Current ? Current->GetAdapterDescription() : FString();
}
