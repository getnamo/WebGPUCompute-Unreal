// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUStructMirror.h"
#include "WebGPUBuffer.h"
#include "WebGPUCompute.h"

void UWebGPUStructMirror::Initialize(const TSharedRef<FWebGPUStructMirror>& InNative, UWebGPUBuffer* InBuffer)
{
	Native = InNative;
	Buffer = InBuffer;
}

bool UWebGPUStructMirror::CheckStruct(const UScriptStruct* Struct, const TCHAR* Function) const
{
	if (!Native)
	{
		UE_LOG(LogWebGPU, Warning, TEXT("StructMirror %s: the mirror was released"), Function);
		return false;
	}
	if (Struct != Native->GetStruct())
	{
		UE_LOG(LogWebGPU, Error, TEXT("StructMirror %s: the mirror holds %s, not %s"), Function, *Native->GetStruct()->GetName(), Struct ? *Struct->GetName() : TEXT("a struct"));
		return false;
	}
	return true;
}

bool UWebGPUStructMirror::GenericSetElement(int32 Index, const UScriptStruct* Struct, const void* Data)
{
	if (!CheckStruct(Struct, TEXT("SetElement")) || !Data)
	{
		return false;
	}
	if (Index < 0 || Index >= Native->Num())
	{
		UE_LOG(LogWebGPU, Error, TEXT("StructMirror SetElement: index %d outside 0..%d"), Index, Native->Num() - 1);
		return false;
	}
	Struct->CopyScriptStruct(Native->EditElement(Index), Data);
	return true;
}

bool UWebGPUStructMirror::GenericGetElement(int32 Index, const UScriptStruct* Struct, void* Data) const
{
	if (!CheckStruct(Struct, TEXT("GetElement")) || !Data)
	{
		return false;
	}
	if (Index < 0 || Index >= Native->Num())
	{
		UE_LOG(LogWebGPU, Error, TEXT("StructMirror GetElement: index %d outside 0..%d"), Index, Native->Num() - 1);
		return false;
	}
	Struct->CopyScriptStruct(Data, Native->GetElement(Index));
	return true;
}

bool UWebGPUStructMirror::GenericSetElements(const FArrayProperty* ArrayProperty, const void* ArrayAddress, int32 FirstIndex)
{
	const FStructProperty* Inner = ArrayProperty ? CastField<FStructProperty>(ArrayProperty->Inner) : nullptr;
	if (!Inner || !ArrayAddress)
	{
		UE_LOG(LogWebGPU, Error, TEXT("StructMirror SetElements needs an array of structs"));
		return false;
	}
	if (!CheckStruct(Inner->Struct, TEXT("SetElements")))
	{
		return false;
	}
	FScriptArrayHelper Helper(ArrayProperty, ArrayAddress);
	if (FirstIndex < 0 || FirstIndex + Helper.Num() > Native->Num())
	{
		UE_LOG(LogWebGPU, Error, TEXT("StructMirror SetElements: elements %d..%d outside 0..%d"), FirstIndex, FirstIndex + Helper.Num() - 1, Native->Num() - 1);
		return false;
	}
	for (int32 Item = 0; Item < Helper.Num(); Item++)
	{
		Inner->Struct->CopyScriptStruct(Native->EditElement(FirstIndex + Item), Helper.GetRawPtr(Item));
	}
	return true;
}

DEFINE_FUNCTION(UWebGPUStructMirror::execSetElement)
{
	P_GET_PROPERTY(FIntProperty, Index);
	Stack.MostRecentProperty = nullptr;
	Stack.StepCompiledIn<FStructProperty>(nullptr);
	const void* Data = Stack.MostRecentPropertyAddress;
	const FStructProperty* Property = CastField<FStructProperty>(Stack.MostRecentProperty);
	P_FINISH;

	P_NATIVE_BEGIN;
	*(bool*)RESULT_PARAM = P_THIS->GenericSetElement(Index, Property ? Property->Struct.Get() : nullptr, Data);
	P_NATIVE_END;
}

DEFINE_FUNCTION(UWebGPUStructMirror::execGetElement)
{
	P_GET_PROPERTY(FIntProperty, Index);
	Stack.MostRecentProperty = nullptr;
	Stack.StepCompiledIn<FStructProperty>(nullptr);
	void* Data = Stack.MostRecentPropertyAddress;
	const FStructProperty* Property = CastField<FStructProperty>(Stack.MostRecentProperty);
	P_FINISH;

	P_NATIVE_BEGIN;
	*(bool*)RESULT_PARAM = P_THIS->GenericGetElement(Index, Property ? Property->Struct.Get() : nullptr, Data);
	P_NATIVE_END;
}

DEFINE_FUNCTION(UWebGPUStructMirror::execSetElements)
{
	Stack.MostRecentProperty = nullptr;
	Stack.StepCompiledIn<FArrayProperty>(nullptr);
	const void* ArrayAddress = Stack.MostRecentPropertyAddress;
	const FArrayProperty* ArrayProperty = CastField<FArrayProperty>(Stack.MostRecentProperty);
	P_GET_PROPERTY(FIntProperty, FirstIndex);
	P_FINISH;

	P_NATIVE_BEGIN;
	*(bool*)RESULT_PARAM = P_THIS->GenericSetElements(ArrayProperty, ArrayAddress, FirstIndex);
	P_NATIVE_END;
}

int64 UWebGPUStructMirror::Sync()
{
	return Native ? (int64)Native->Sync() : 0;
}

int32 UWebGPUStructMirror::Num() const
{
	return Native ? Native->Num() : 0;
}

int32 UWebGPUStructMirror::GetDirtyCount() const
{
	return Native ? Native->GetDirtyCount() : 0;
}

UScriptStruct* UWebGPUStructMirror::GetStructType() const
{
	return Native ? const_cast<UScriptStruct*>(Native->GetStruct()) : nullptr;
}

void UWebGPUStructMirror::Release()
{
	Native.Reset();
	if (Buffer)
	{
		Buffer->Release();
		Buffer = nullptr;
	}
}

void UWebGPUStructMirror::BeginDestroy()
{
	Native.Reset();
	Super::BeginDestroy();
}
