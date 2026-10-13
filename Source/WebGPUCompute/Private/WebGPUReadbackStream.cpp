// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUReadbackStream.h"
#include "WebGPUBuffer.h"
#include "WebGPUCompute.h"

bool UWebGPUReadbackStream::Capture(UWebGPUBuffer* Buffer, int64 Offset, int64 Size)
{
	if (!Native || !Buffer || !Buffer->GetNative())
	{
		UE_LOG(LogWebGPU, Warning, TEXT("ReadbackStream Capture: %s"), Native ? TEXT("no valid buffer") : TEXT("the stream was released"));
		return false;
	}
	return Native->Capture(Buffer->GetNative().ToSharedRef(), (uint64)FMath::Max<int64>(Offset, 0), Size < 0 ? WGPU_WHOLE_SIZE : (uint64)Size);
}

bool UWebGPUReadbackStream::ReadFrame(TFunctionRef<void(const FWebGPUReadbackFrame& Frame)> Reader)
{
	FWebGPUReadbackFrame Frame;
	if (!Native || !Native->Acquire(Frame))
	{
		return false;
	}

	LastFrame.Sequence = (int64)Frame.GetSequence();
	LastFrame.CaptureFrame = (int64)Frame.GetCaptureFrame();
	LastFrame.FramesLate = (int32)(GFrameCounter - Frame.GetCaptureFrame());
	LastFrame.RoundTripMs = (float)((Frame.GetReadyTime() - Frame.GetCaptureTime()) * 1000.0);
	LastFrame.Bytes = (int64)Frame.GetSize();
	Reader(Frame);
	return true;
}

bool UWebGPUReadbackStream::ReadBytes(TArray<uint8>& OutBytes)
{
	return ReadFrame([&OutBytes](const FWebGPUReadbackFrame& Frame)
	{
		OutBytes.SetNumUninitialized((int32)Frame.GetSize());
		FMemory::Memcpy(OutBytes.GetData(), Frame.GetData(), Frame.GetSize());
	});
}

bool UWebGPUReadbackStream::ReadFloats(TArray<float>& OutValues)
{
	return ReadFrame([&OutValues](const FWebGPUReadbackFrame& Frame)
	{
		OutValues = TArray<float>(Frame.View<float>());
	});
}

bool UWebGPUReadbackStream::ReadInts(TArray<int32>& OutValues)
{
	return ReadFrame([&OutValues](const FWebGPUReadbackFrame& Frame)
	{
		OutValues = TArray<int32>(Frame.View<int32>());
	});
}

bool UWebGPUReadbackStream::GenericReadStructs(const FArrayProperty* ArrayProperty, void* ArrayAddress)
{
	const FStructProperty* Inner = ArrayProperty ? CastField<FStructProperty>(ArrayProperty->Inner) : nullptr;
	if (!Inner || !ArrayAddress)
	{
		UE_LOG(LogWebGPU, Error, TEXT("ReadStructs needs an array of structs"));
		return false;
	}
	FString Error;
	TSharedPtr<const FWebGPUStructLayout> Layout = FWebGPUStructLayout::Get(Inner->Struct, EWebGPUAddressSpace::Storage, Error);
	if (!Layout)
	{
		UE_LOG(LogWebGPU, Error, TEXT("ReadStructs: %s"), *Error);
		return false;
	}
	return ReadFrame([&](const FWebGPUReadbackFrame& Frame)
	{
		const int32 Count = (int32)(Frame.GetSize() / Layout->GetStride());
		FScriptArrayHelper Helper(ArrayProperty, ArrayAddress);
		Helper.Resize(Count);
		if (Count > 0)
		{
			Layout->Unpack(Frame.GetData(), Count, Helper.GetRawPtr(0));
		}
	});
}

DEFINE_FUNCTION(UWebGPUReadbackStream::execReadStructs)
{
	Stack.MostRecentProperty = nullptr;
	Stack.StepCompiledIn<FArrayProperty>(nullptr);
	void* ArrayAddress = Stack.MostRecentPropertyAddress;
	const FArrayProperty* ArrayProperty = CastField<FArrayProperty>(Stack.MostRecentProperty);
	P_FINISH;

	P_NATIVE_BEGIN;
	*(bool*)RESULT_PARAM = P_THIS->GenericReadStructs(ArrayProperty, ArrayAddress);
	P_NATIVE_END;
}

int32 UWebGPUReadbackStream::GetReadyCount() const
{
	return Native ? Native->GetReadyCount() : 0;
}

FWebGPUStreamStats UWebGPUReadbackStream::GetStats() const
{
	return Native ? Native->GetStats() : FWebGPUStreamStats();
}

void UWebGPUReadbackStream::Release()
{
	Native.Reset();
}

void UWebGPUReadbackStream::BeginDestroy()
{
	Native.Reset();
	Super::BeginDestroy();
}
