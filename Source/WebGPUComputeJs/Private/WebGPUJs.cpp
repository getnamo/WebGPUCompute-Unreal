// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUJs.h"
#include "WebGPUBuffer.h"
#include "WebGPUCompute.h"
#include "WebGPUContext.h"
#include "WebGPUReadbackStream.h"
#include "WebGPUStreams.h"
#include "Modules/ModuleManager.h"

#if WITH_WEBGPU_UNREALJS
#include "JavascriptContext.h"
#include "JavascriptMemoryObject.h"
#endif

IMPLEMENT_MODULE(FDefaultModuleImpl, WebGPUComputeJs);

void UWebGPUJsRead::Complete(bool bSuccess, TArray<uint8>&& InBytes)
{
	Bytes = MoveTemp(InBytes);
	bDone = true;
#if WITH_WEBGPU_UNREALJS
	if (bSuccess)
	{
		UJavascriptExternalMemory* External = NewObject<UJavascriptExternalMemory>(this);
		External->SetMemory(Bytes.GetData(), Bytes.Num());
		Memory = External;
	}
#endif
	OnCompleted.Broadcast(bSuccess);
}

void UWebGPUJsRead::Release()
{
#if WITH_WEBGPU_UNREALJS
	if (UJavascriptExternalMemory* External = Cast<UJavascriptExternalMemory>(Memory))
	{
		External->Release();
	}
#endif
	Memory = nullptr;
	Bytes.Empty();
}

void UWebGPUJsRead::BeginDestroy()
{
	Release();
	Super::BeginDestroy();
}

bool UWebGPUJs::IsAvailable()
{
	return WITH_WEBGPU_UNREALJS != 0;
}

UObject* UWebGPUJs::AcquireFrame(UWebGPUReadbackStream* Stream)
{
#if WITH_WEBGPU_UNREALJS
	FWebGPUReadbackFrame Frame;
	if (!Stream || !Stream->AcquireFrame(Frame))
	{
		return nullptr;
	}
	void* Data = const_cast<uint8*>(Frame.GetData());
	const int32 Size = (int32)FMath::Min<uint64>(Frame.GetSize(), MAX_int32);
	UJavascriptExternalMemory* External = NewObject<UJavascriptExternalMemory>(Stream);
	// The frame (its mapped slot) lives in the release callback: releasing gives the slot back
	External->SetMemory(Data, Size, [Held = MoveTemp(Frame)]() mutable
	{
		Held.Release();
	});
	return External;
#else
	UE_LOG(LogWebGPU, Warning, TEXT("WebGPUJs.AcquireFrame: built without UnrealJs"));
	return nullptr;
#endif
}

bool UWebGPUJs::WriteBufferFromScript(UWebGPUBuffer* Buffer, int64 Offset, int64 SourceOffset, int64 Size)
{
#if WITH_WEBGPU_UNREALJS
	const TSharedPtr<FWebGPUBuffer>& Native = Buffer ? Buffer->GetNative() : nullptr;
	TSharedPtr<FWebGPUContext> Context = Native ? Native->GetContext() : nullptr;
	const uint8* Source = static_cast<const uint8*>(FArrayBufferAccessor::GetData());
	const int64 Available = FArrayBufferAccessor::GetSize();
	if (!Context || !Source)
	{
		UE_LOG(LogWebGPU, Warning, TEXT("WebGPUJs.WriteBufferFromScript: needs a buffer, and must run inside memory.exec(arrayBuffer, fn)"));
		return false;
	}
	const int64 Bytes = Size < 0 ? Available - SourceOffset : Size;
	if (SourceOffset < 0 || Bytes <= 0 || SourceOffset + Bytes > Available || Offset < 0 || (uint64)(Offset + Bytes) > Native->GetSize())
	{
		UE_LOG(LogWebGPU, Warning, TEXT("WebGPUJs.WriteBufferFromScript(%s): %lld bytes from %lld of a %lld-byte ArrayBuffer don't fit at %lld of %llu bytes"),
			*Native->GetLabel(), Bytes, SourceOffset, Available, Offset, Native->GetSize());
		return false;
	}
	Context->WriteBuffer(Native.ToSharedRef(), (uint64)Offset, Source + SourceOffset, (uint64)Bytes);
	return true;
#else
	UE_LOG(LogWebGPU, Warning, TEXT("WebGPUJs.WriteBufferFromScript: built without UnrealJs"));
	return false;
#endif
}

UWebGPUJsRead* UWebGPUJs::ReadBuffer(UWebGPUBuffer* Buffer, int64 Offset, int64 Size)
{
	if (!Buffer || !Buffer->GetNative())
	{
		UE_LOG(LogWebGPU, Warning, TEXT("WebGPUJs.ReadBuffer: no valid buffer"));
		return nullptr;
	}
	UWebGPUJsRead* Request = NewObject<UWebGPUJsRead>();
	TWeakObjectPtr<UWebGPUJsRead> WeakRequest = Request;
	Buffer->Read(Offset, Size, [WeakRequest](bool bSuccess, TArray<uint8>&& Data)
	{
		if (UWebGPUJsRead* Alive = WeakRequest.Get())
		{
			Alive->Complete(bSuccess, MoveTemp(Data));
		}
	}, EWebGPUCallbackThread::GameThread);
	return Request;
}
