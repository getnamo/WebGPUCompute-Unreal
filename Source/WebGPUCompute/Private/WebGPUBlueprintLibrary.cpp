// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUBlueprintLibrary.h"
#include "WebGPUBuffer.h"
#include "WebGPUCompute.h"
#include "WebGPUContext.h"
#include "WebGPUKernel.h"
#include "WebGPUSubsystem.h"

UWebGPUBuffer* UWebGPUBlueprintLibrary::CreateWebGPUBuffer(int64 Size, int32 Usage, const FString& Label)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!Context || Size <= 0)
	{
		return nullptr;
	}
	const EWebGPUBufferUsage BufferUsage = Usage == 0 ? EWebGPUBufferUsage::Storage : (EWebGPUBufferUsage)Usage;

	UWebGPUBuffer* Buffer = NewObject<UWebGPUBuffer>();
	Buffer->Initialize(Context->CreateBuffer((uint64)Size, BufferUsage, Label));
	return Buffer;
}

UWebGPUKernel* UWebGPUBlueprintLibrary::CreateWebGPUKernel(const FString& Source, const FString& EntryPoint, const FString& Label)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!Context)
	{
		return nullptr;
	}

	UWebGPUKernel* Kernel = NewObject<UWebGPUKernel>();
	TWeakObjectPtr<UWebGPUKernel> WeakKernel = Kernel;
	Kernel->Initialize(Context->CreateKernel(Source, EntryPoint, Label,
		[WeakKernel](bool bSuccess, const TArray<FWebGPUCompileMessage>& Messages)
		{
			if (UWebGPUKernel* Alive = WeakKernel.Get())
			{
				Alive->HandleCompiled(bSuccess, Messages);
			}
		}, EWebGPUCallbackThread::GameThread));
	return Kernel;
}

int32 UWebGPUBlueprintLibrary::WorkgroupsForCount(int32 Count, int32 WorkgroupSize)
{
	return WorkgroupSize > 0 ? FMath::DivideAndRoundUp(FMath::Max(Count, 0), WorkgroupSize) : 0;
}

TArray<float> UWebGPUBlueprintLibrary::WebGPUBytesToFloats(const TArray<uint8>& Bytes)
{
	TArray<float> Result;
	Result.SetNumUninitialized(Bytes.Num() / sizeof(float));
	FMemory::Memcpy(Result.GetData(), Bytes.GetData(), Result.Num() * sizeof(float));
	return Result;
}

TArray<int32> UWebGPUBlueprintLibrary::WebGPUBytesToInts(const TArray<uint8>& Bytes)
{
	TArray<int32> Result;
	Result.SetNumUninitialized(Bytes.Num() / sizeof(int32));
	FMemory::Memcpy(Result.GetData(), Bytes.GetData(), Result.Num() * sizeof(int32));
	return Result;
}

FWebGPUStats UWebGPUBlueprintLibrary::GetWebGPUStats()
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	return Context ? Context->GetStats() : FWebGPUStats();
}

UWebGPUReadBufferAction* UWebGPUReadBufferAction::ReadWebGPUBuffer(UWebGPUBuffer* Buffer, int64 Offset, int64 Size)
{
	UWebGPUReadBufferAction* Action = NewObject<UWebGPUReadBufferAction>();
	Action->Buffer = Buffer;
	Action->Offset = Offset;
	Action->Size = Size;
	return Action;
}

void UWebGPUReadBufferAction::Activate()
{
	if (!Buffer || !Buffer->IsBufferValid())
	{
		OnFailed.Broadcast(TArray<uint8>());
		SetReadyToDestroy();
		return;
	}

	// Stay alive until the read completes; there may be no game instance (editor utilities)
	AddToRoot();
	TWeakObjectPtr<UWebGPUReadBufferAction> WeakThis = this;
	Buffer->Read(Offset, Size, [WeakThis](bool bSuccess, TArray<uint8>&& Data)
	{
		if (UWebGPUReadBufferAction* Action = WeakThis.Get())
		{
			if (bSuccess)
			{
				Action->OnCompleted.Broadcast(Data);
			}
			else
			{
				Action->OnFailed.Broadcast(Data);
			}
			Action->RemoveFromRoot();
			Action->SetReadyToDestroy();
		}
	});
}
