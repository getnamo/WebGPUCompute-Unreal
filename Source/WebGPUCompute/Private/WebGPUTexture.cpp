// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUTexture.h"
#include "WebGPUCompute.h"

TSharedPtr<FWebGPUContext> UWebGPUTexture::GetContext() const
{
	return Native ? Native->GetContext() : nullptr;
}

int32 UWebGPUTexture::GetWidth() const
{
	return Native ? (int32)Native->GetWidth() : 0;
}

int32 UWebGPUTexture::GetHeight() const
{
	return Native ? (int32)Native->GetHeight() : 0;
}

EWebGPUTextureFormat UWebGPUTexture::GetFormat() const
{
	return Native ? Native->GetFormat() : EWebGPUTextureFormat::RGBA8Unorm;
}

bool UWebGPUTexture::IsShared() const
{
	return Native && Native->IsShared();
}

void UWebGPUTexture::BeginAccess()
{
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		Context->BeginTextureAccess(Native.ToSharedRef());
	}
}

void UWebGPUTexture::EndAccess()
{
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		Context->EndTextureAccess(Native.ToSharedRef());
	}
}

void UWebGPUTexture::WriteBytes(const TArray<uint8>& Pixels)
{
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		Context->WriteTexture(Native.ToSharedRef(), Pixels.GetData(), Pixels.Num());
	}
}

void UWebGPUTexture::WriteFloats(const TArray<float>& Values)
{
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		Context->WriteTexture(Native.ToSharedRef(), Values.GetData(), Values.Num() * sizeof(float));
	}
}

void UWebGPUTexture::Read(FWebGPUReadCallback&& Callback, EWebGPUCallbackThread Thread)
{
	TSharedPtr<FWebGPUContext> Context = GetContext();
	if (!Context)
	{
		if (Callback)
		{
			Callback(false, TArray<uint8>());
		}
		return;
	}
	Context->ReadTexture(Native.ToSharedRef(), MoveTemp(Callback), Thread);
}

void UWebGPUTexture::Release()
{
	Native.Reset();
}

void UWebGPUTexture::BeginDestroy()
{
	Native.Reset();
	Super::BeginDestroy();
}
