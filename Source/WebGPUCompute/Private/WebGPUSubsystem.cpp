// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUSubsystem.h"
#include "WebGPUContext.h"
#include "WebGPUDevice.h"
#include "Engine/Engine.h"

void UWebGPUSubsystem::Deinitialize()
{
	TSharedPtr<FWebGPUContext> ToShutdown;
	{
		FScopeLock Lock(&ContextLock);
		ToShutdown = MoveTemp(Context);
	}
	if (ToShutdown)
	{
		ToShutdown->OnDeviceRecovered.RemoveAll(this);
		ToShutdown->Shutdown();
	}
	Super::Deinitialize();
}

TSharedPtr<FWebGPUContext> UWebGPUSubsystem::GetContext()
{
	FScopeLock Lock(&ContextLock);
	if (!bInitAttempted)
	{
		bInitAttempted = true;
		Context = FWebGPUContext::Create();
		if (Context)
		{
			Context->OnDeviceRecovered.AddUObject(this, &UWebGPUSubsystem::HandleDeviceRecovered);
		}
	}
	return Context;
}

TSharedPtr<FWebGPUDevice> UWebGPUSubsystem::GetDevice()
{
	TSharedPtr<FWebGPUContext> Current = GetContext();
	return Current ? Current->GetDevice() : nullptr;
}

TSharedPtr<FWebGPUContext> UWebGPUSubsystem::GetSharedContext()
{
	UWebGPUSubsystem* Subsystem = GEngine ? GEngine->GetEngineSubsystem<UWebGPUSubsystem>() : nullptr;
	return Subsystem ? Subsystem->GetContext() : nullptr;
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

FWebGPUStats UWebGPUSubsystem::GetStats()
{
	TSharedPtr<FWebGPUContext> Current = GetContext();
	return Current ? Current->GetStats() : FWebGPUStats();
}

void UWebGPUSubsystem::HandleDeviceRecovered(uint32 Generation)
{
	OnDeviceRecovered.Broadcast((int32)Generation);
}
