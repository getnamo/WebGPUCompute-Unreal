// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUBuffer.h"
#include "WebGPUCompute.h"

int64 UWebGPUBuffer::GetSize() const
{
	return Native ? (int64)Native->GetSize() : 0;
}

void UWebGPUBuffer::Write(TArray<uint8>&& Bytes, int64 Offset)
{
	TSharedPtr<FWebGPUContext> Context = Native ? Native->GetContext() : nullptr;
	if (!Context)
	{
		UE_LOG(LogWebGPU, Warning, TEXT("Write to a released WebGPU buffer ignored"));
		return;
	}
	// WebGPU writes are 4-byte granular
	Bytes.SetNumZeroed(Align(Bytes.Num(), 4));
	Context->WriteBuffer(Native.ToSharedRef(), (uint64)Offset, MoveTemp(Bytes));
}

void UWebGPUBuffer::WriteBytes(const TArray<uint8>& Data, int64 Offset)
{
	TArray<uint8> Bytes = Data;
	Write(MoveTemp(Bytes), Offset);
}

void UWebGPUBuffer::WriteFloats(const TArray<float>& Data, int64 Offset)
{
	TArray<uint8> Bytes;
	Bytes.Append(reinterpret_cast<const uint8*>(Data.GetData()), Data.Num() * sizeof(float));
	Write(MoveTemp(Bytes), Offset);
}

void UWebGPUBuffer::WriteInts(const TArray<int32>& Data, int64 Offset)
{
	TArray<uint8> Bytes;
	Bytes.Append(reinterpret_cast<const uint8*>(Data.GetData()), Data.Num() * sizeof(int32));
	Write(MoveTemp(Bytes), Offset);
}

void UWebGPUBuffer::Read(int64 Offset, int64 Size, FWebGPUReadCallback&& Callback, EWebGPUCallbackThread Thread)
{
	TSharedPtr<FWebGPUContext> Context = Native ? Native->GetContext() : nullptr;
	if (!Context)
	{
		if (Callback)
		{
			Callback(false, TArray<uint8>());
		}
		return;
	}
	const uint64 ReadSize = Size < 0 ? Native->GetSize() - (uint64)Offset : (uint64)Size;
	Context->ReadBuffer(Native.ToSharedRef(), (uint64)Offset, ReadSize, MoveTemp(Callback), Thread);
}

void UWebGPUBuffer::Release()
{
	Native.Reset();
}

void UWebGPUBuffer::BeginDestroy()
{
	Native.Reset();
	Super::BeginDestroy();
}
