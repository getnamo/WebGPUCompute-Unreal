// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUBlueprintLibrary.h"
#include "WebGPUBuffer.h"
#include "WebGPUCompute.h"
#include "WebGPUContext.h"
#include "WebGPUKernel.h"
#include "WebGPUReadbackStream.h"
#include "WebGPUStructMirror.h"
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

namespace
{
	UWebGPUKernel* MakeKernel(const FWebGPUKernelDesc& Desc)
	{
		TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
		if (!Context)
		{
			return nullptr;
		}

		UWebGPUKernel* Kernel = NewObject<UWebGPUKernel>();
		TWeakObjectPtr<UWebGPUKernel> WeakKernel = Kernel;
		Kernel->Initialize(Context->CreateKernel(Desc,
			[WeakKernel](bool bSuccess, const TArray<FWebGPUCompileMessage>& Messages)
			{
				if (UWebGPUKernel* Alive = WeakKernel.Get())
				{
					Alive->HandleCompiled(bSuccess, Messages);
				}
			}, EWebGPUCallbackThread::GameThread));
		return Kernel;
	}

	const FStructProperty* GetStructInner(const FArrayProperty* ArrayProperty, const TCHAR* Function)
	{
		const FStructProperty* Inner = ArrayProperty ? CastField<FStructProperty>(ArrayProperty->Inner) : nullptr;
		if (!Inner)
		{
			UE_LOG(LogWebGPU, Error, TEXT("%s needs an array of structs"), Function);
		}
		return Inner;
	}
}

UWebGPUKernel* UWebGPUBlueprintLibrary::CreateWebGPUKernel(const FString& Source, const FString& EntryPoint, const FString& Label)
{
	FWebGPUKernelDesc Desc;
	Desc.Source = Source;
	Desc.EntryPoint = EntryPoint;
	Desc.Label = Label;
	return MakeKernel(Desc);
}

UWebGPUKernel* UWebGPUBlueprintLibrary::CreateWebGPUKernelWithStructs(const FString& Source, const TArray<UScriptStruct*>& Structs, const FString& EntryPoint, const FString& Label)
{
	FWebGPUKernelDesc Desc;
	Desc.Source = Source;
	Desc.EntryPoint = EntryPoint;
	Desc.Label = Label;
	for (UScriptStruct* Struct : Structs)
	{
		Desc.Structs.Add(Struct);
	}
	return MakeKernel(Desc);
}

UWebGPUBuffer* UWebGPUBlueprintLibrary::CreateWebGPUStructBuffer(UScriptStruct* StructType, int32 Count, int32 Usage, const FString& Label)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!Context)
	{
		return nullptr;
	}
	const EWebGPUBufferUsage BufferUsage = Usage == 0 ? EWebGPUBufferUsage::Storage : (EWebGPUBufferUsage)Usage;
	TSharedPtr<FWebGPUBuffer> Native = Context->CreateStructBuffer(StructType, Count, BufferUsage, Label);
	if (!Native)
	{
		return nullptr;
	}
	UWebGPUBuffer* Buffer = NewObject<UWebGPUBuffer>();
	Buffer->Initialize(Native.ToSharedRef());
	return Buffer;
}

UWebGPUReadbackStream* UWebGPUBlueprintLibrary::CreateWebGPUReadbackStream(int64 SlotBytes, int32 Slots, EWebGPUStreamPolicy Policy, const FString& Label)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!Context || SlotBytes <= 0)
	{
		return nullptr;
	}
	UWebGPUReadbackStream* Stream = NewObject<UWebGPUReadbackStream>();
	Stream->Initialize(Context->CreateReadbackStream((uint64)SlotBytes, Slots, Policy, Label));
	return Stream;
}

UWebGPUReadbackStream* UWebGPUBlueprintLibrary::CreateWebGPUReadbackStreamForBuffer(UWebGPUBuffer* Buffer, int32 Slots, EWebGPUStreamPolicy Policy, const FString& Label)
{
	if (!Buffer || !Buffer->GetNative())
	{
		UE_LOG(LogWebGPU, Warning, TEXT("CreateWebGPUReadbackStreamForBuffer: no valid buffer"));
		return nullptr;
	}
	return CreateWebGPUReadbackStream(Buffer->GetSize(), Slots, Policy, Label.IsEmpty() ? Buffer->GetNative()->GetLabel() + TEXT(" readback") : Label);
}

UWebGPUStructMirror* UWebGPUBlueprintLibrary::CreateWebGPUStructMirror(UScriptStruct* StructType, int32 Count, int32 Usage, const FString& Label)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!Context)
	{
		return nullptr;
	}
	const EWebGPUBufferUsage BufferUsage = Usage == 0 ? EWebGPUBufferUsage::Storage : (EWebGPUBufferUsage)Usage;
	TSharedPtr<FWebGPUStructMirror> Native = FWebGPUStructMirror::Create(Context.ToSharedRef(), StructType, Count, BufferUsage, Label);
	if (!Native)
	{
		return nullptr;
	}
	UWebGPUBuffer* Buffer = NewObject<UWebGPUBuffer>();
	Buffer->Initialize(Native->GetBuffer());
	UWebGPUStructMirror* Mirror = NewObject<UWebGPUStructMirror>();
	Mirror->Initialize(Native.ToSharedRef(), Buffer);
	return Mirror;
}

FString UWebGPUBlueprintLibrary::GetWGSLForStruct(UScriptStruct* StructType, EWebGPUAddressSpace Space)
{
	FString Error;
	TSharedPtr<const FWebGPUStructLayout> Layout = FWebGPUStructLayout::Get(StructType, Space, Error);
	if (!Layout)
	{
		UE_LOG(LogWebGPU, Warning, TEXT("GetWGSLForStruct: %s"), *Error);
		return FString();
	}
	return Layout->GenerateWGSL();
}

bool UWebGPUBlueprintLibrary::GenericWriteStructArray(UWebGPUBuffer* Buffer, const FArrayProperty* ArrayProperty, const void* ArrayAddress, int32 FirstElement)
{
	const FStructProperty* Inner = GetStructInner(ArrayProperty, TEXT("WriteWebGPUStructArray"));
	if (!Inner || !ArrayAddress)
	{
		return false;
	}
	TSharedPtr<FWebGPUContext> Context = Buffer && Buffer->GetNative() ? Buffer->GetNative()->GetContext() : nullptr;
	if (!Context)
	{
		UE_LOG(LogWebGPU, Warning, TEXT("WriteWebGPUStructArray: no valid buffer"));
		return false;
	}
	FScriptArrayHelper Helper(ArrayProperty, ArrayAddress);
	if (Helper.Num() == 0)
	{
		return true;
	}
	return Context->WriteStructs(Buffer->GetNative().ToSharedRef(), Inner->Struct, Helper.GetRawPtr(0), Helper.Num(), FirstElement);
}

bool UWebGPUBlueprintLibrary::GenericBytesToStructArray(const TArray<uint8>& Bytes, const FArrayProperty* ArrayProperty, void* ArrayAddress)
{
	const FStructProperty* Inner = GetStructInner(ArrayProperty, TEXT("WebGPUBytesToStructArray"));
	if (!Inner || !ArrayAddress)
	{
		return false;
	}
	FString Error;
	TSharedPtr<const FWebGPUStructLayout> Layout = FWebGPUStructLayout::Get(Inner->Struct, EWebGPUAddressSpace::Storage, Error);
	if (!Layout)
	{
		UE_LOG(LogWebGPU, Error, TEXT("WebGPUBytesToStructArray: %s"), *Error);
		return false;
	}
	const int32 Count = Bytes.Num() / (int32)Layout->GetStride();
	FScriptArrayHelper Helper(ArrayProperty, ArrayAddress);
	Helper.Resize(Count);
	if (Count > 0)
	{
		Layout->Unpack(Bytes.GetData(), Count, Helper.GetRawPtr(0));
	}
	return true;
}

DEFINE_FUNCTION(UWebGPUBlueprintLibrary::execWriteWebGPUStructArray)
{
	P_GET_OBJECT(UWebGPUBuffer, Buffer);
	Stack.MostRecentProperty = nullptr;
	Stack.StepCompiledIn<FArrayProperty>(nullptr);
	const void* ArrayAddress = Stack.MostRecentPropertyAddress;
	const FArrayProperty* ArrayProperty = CastField<FArrayProperty>(Stack.MostRecentProperty);
	P_GET_PROPERTY(FIntProperty, FirstElement);
	P_FINISH;

	P_NATIVE_BEGIN;
	*(bool*)RESULT_PARAM = GenericWriteStructArray(Buffer, ArrayProperty, ArrayAddress, FirstElement);
	P_NATIVE_END;
}

DEFINE_FUNCTION(UWebGPUBlueprintLibrary::execWebGPUBytesToStructArray)
{
	P_GET_TARRAY_REF(uint8, Bytes);
	Stack.MostRecentProperty = nullptr;
	Stack.StepCompiledIn<FArrayProperty>(nullptr);
	void* ArrayAddress = Stack.MostRecentPropertyAddress;
	const FArrayProperty* ArrayProperty = CastField<FArrayProperty>(Stack.MostRecentProperty);
	P_FINISH;

	P_NATIVE_BEGIN;
	*(bool*)RESULT_PARAM = GenericBytesToStructArray(Bytes, ArrayProperty, ArrayAddress);
	P_NATIVE_END;
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
