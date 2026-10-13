// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUBindings.h"
#include "WebGPUBuffer.h"
#include "WebGPUCompute.h"
#include "WebGPUTexture.h"

UWebGPUBuffer* UWebGPUBindings::CreateBuffer(const FString& Name, int32 Count, int32 Stride, int32 Usage)
{
	TSharedPtr<FWebGPUContext> Context = GetContext();
	if (!Context || Name.IsEmpty() || Count <= 0 || Stride <= 0)
	{
		UE_LOG(LogWebGPU, Warning, TEXT("Bindings CreateBuffer(%s, %d x %d bytes): needs a name and a positive count and stride"), *Name, Count, Stride);
		return nullptr;
	}
	const EWebGPUBufferUsage BufferUsage = Usage == 0 ? EWebGPUBufferUsage::Storage : (EWebGPUBufferUsage)Usage;
	UWebGPUBuffer* Buffer = NewObject<UWebGPUBuffer>(this);
	Buffer->Initialize(Context->CreateBuffer((uint64)Count * (uint64)Stride, BufferUsage, Name));
	SetBuffer(Name, Buffer);
	return Buffer;
}

void UWebGPUBindings::SetBuffer(const FString& Name, UWebGPUBuffer* Buffer)
{
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		Context->SetSharedBuffer(Native.ToSharedRef(), Name, Buffer ? Buffer->GetNative() : nullptr);
		if (Buffer)
		{
			Buffers.Add(Name, Buffer);
		}
		else
		{
			Buffers.Remove(Name);
		}
	}
}

void UWebGPUBindings::SetTexture(const FString& Name, UWebGPUTexture* Texture)
{
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		Context->SetSharedTexture(Native.ToSharedRef(), Name, Texture ? Texture->GetNative() : nullptr);
		if (Texture)
		{
			Textures.Add(Name, Texture);
		}
		else
		{
			Textures.Remove(Name);
		}
	}
}

void UWebGPUBindings::SetSampler(const FString& Name, EWebGPUSampler Sampler)
{
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		Context->SetSharedSampler(Native.ToSharedRef(), Name, Sampler);
	}
}

void UWebGPUBindings::SetFloat(const FString& Name, float Value)
{
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		Context->SetUniformValue(Native.ToSharedRef(), Name, FWebGPUUniformValue::Float(Value));
	}
}

void UWebGPUBindings::SetInt(const FString& Name, int32 Value)
{
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		Context->SetUniformValue(Native.ToSharedRef(), Name, FWebGPUUniformValue::Int(Value));
	}
}

void UWebGPUBindings::SetVector(const FString& Name, FVector4 Value)
{
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		Context->SetUniformValue(Native.ToSharedRef(), Name, FWebGPUUniformValue::Vector(FVector4f(Value)));
	}
}

UWebGPUBuffer* UWebGPUBindings::GetBuffer(const FString& Name) const
{
	const TObjectPtr<UWebGPUBuffer>* Found = Buffers.Find(Name);
	return Found ? Found->Get() : nullptr;
}
