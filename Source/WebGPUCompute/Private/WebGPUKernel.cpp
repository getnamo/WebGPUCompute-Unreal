// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUKernel.h"
#include "WebGPUBindings.h"
#include "WebGPUBuffer.h"
#include "WebGPUCompute.h"
#include "WebGPUTexture.h"

TSharedPtr<FWebGPUContext> UWebGPUKernel::GetContext() const
{
	return Native ? Native->GetContext() : nullptr;
}

void UWebGPUKernel::SetBuffer(int32 Binding, UWebGPUBuffer* Buffer, int32 Group)
{
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		Context->SetBuffer(Native.ToSharedRef(), (uint32)Group, (uint32)Binding, Buffer ? Buffer->GetNative() : nullptr);
	}
}

void UWebGPUKernel::Dispatch(int32 X, int32 Y, int32 Z, int32 Repeat)
{
	if (X <= 0 || Y <= 0 || Z <= 0)
	{
		return;
	}
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		for (int32 Pass = 0; Pass < Repeat; Pass++)
		{
			Context->Dispatch(Native.ToSharedRef(), (uint32)X, (uint32)Y, (uint32)Z);
		}
	}
}

void UWebGPUKernel::SetBufferByName(const FString& Name, UWebGPUBuffer* Buffer)
{
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		Context->SetBufferByName(Native.ToSharedRef(), Name, Buffer ? Buffer->GetNative() : nullptr);
	}
}

void UWebGPUKernel::SetTextureByName(const FString& Name, UWebGPUTexture* Texture)
{
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		Context->SetTextureByName(Native.ToSharedRef(), Name, Texture ? Texture->GetNative() : nullptr);
	}
}

void UWebGPUKernel::SetSamplerByName(const FString& Name, EWebGPUSampler Sampler)
{
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		Context->SetSamplerByName(Native.ToSharedRef(), Name, Sampler);
	}
}

void UWebGPUKernel::DispatchForCount(int32 X, int32 Y, int32 Z, int32 Repeat)
{
	if (X <= 0 || Y <= 0 || Z <= 0)
	{
		return;
	}
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		for (int32 Pass = 0; Pass < Repeat; Pass++)
		{
			Context->DispatchForCount(Native.ToSharedRef(), (uint32)X, (uint32)Y, (uint32)Z);
		}
	}
}

void UWebGPUKernel::SetBindings(UWebGPUBindings* InBindings)
{
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		Context->SetKernelBindingSet(Native.ToSharedRef(), InBindings ? InBindings->GetNative() : nullptr);
		Bindings = InBindings;
	}
}

FString UWebGPUKernel::GetSourceFile() const
{
	return Native ? Native->GetSourceFile() : FString();
}

void UWebGPUKernel::SetHotReload(bool bEnabled)
{
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		Context->SetHotReload(Native.ToSharedRef(), bEnabled);
	}
}

bool UWebGPUKernel::IsHotReloading() const
{
	TSharedPtr<FWebGPUContext> Context = GetContext();
	return Context && Context->IsHotReloading(Native.ToSharedRef());
}

TArray<FWebGPUBindingInfo> UWebGPUKernel::GetBindings() const
{
	FWebGPUEntryPointInfo Info;
	return Native && Native->GetEntryPointInfo(Info) ? Info.Bindings : TArray<FWebGPUBindingInfo>();
}

FIntVector UWebGPUKernel::GetWorkgroupSize() const
{
	FWebGPUEntryPointInfo Info;
	return Native && Native->GetEntryPointInfo(Info) ? Info.WorkgroupSize : FIntVector::ZeroValue;
}

FString UWebGPUKernel::GetCompiledSource() const
{
	return Native ? Native->GetCompiledSource() : FString();
}

void UWebGPUKernel::Recompile(const FString& Source)
{
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		// The listener (bound at creation) reports the result
		Context->RecompileKernel(Native.ToSharedRef(), Source);
	}
}

bool UWebGPUKernel::IsReady() const
{
	return Native && Native->IsReady();
}

FString UWebGPUKernel::GetCompileMessages() const
{
	return Native ? FWebGPUDevice::FormatCompileMessages(Native->GetMessages()) : FString();
}

void UWebGPUKernel::SetProfiling(bool bEnabled)
{
	if (TSharedPtr<FWebGPUContext> Context = GetContext())
	{
		Context->SetProfiling(Native.ToSharedRef(), bEnabled);
	}
}

float UWebGPUKernel::GetLastGpuTimeMs() const
{
	return Native ? (float)Native->GetLastGpuTimeMs() : 0.f;
}

float UWebGPUKernel::GetAverageGpuTimeMs() const
{
	return Native ? (float)Native->GetAverageGpuTimeMs() : 0.f;
}

void UWebGPUKernel::HandleCompiled(bool bSuccess, const TArray<FWebGPUCompileMessage>& Messages)
{
	const FString Text = FWebGPUDevice::FormatCompileMessages(Messages);
	if (!bSuccess)
	{
		UE_LOG(LogWebGPU, Warning, TEXT("Kernel '%s' failed to compile:\n%s"), Native ? *Native->GetLabel() : TEXT("?"), *Text);
	}
	OnCompiled.Broadcast(bSuccess, Text);
}

void UWebGPUKernel::Release()
{
	Native.Reset();
}

void UWebGPUKernel::BeginDestroy()
{
	Native.Reset();
	Super::BeginDestroy();
}
