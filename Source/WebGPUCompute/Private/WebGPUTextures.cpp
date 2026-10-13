// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUContext.h"
#include "WebGPUCompute.h"
#include "WebGPUTextureInterop.h"

namespace
{
	WGPUStringView ToView(const FTCHARToUTF8& Utf8)
	{
		WGPUStringView View = WGPU_STRING_VIEW_INIT;
		View.data = Utf8.Get();
		View.length = (size_t)Utf8.Length();
		return View;
	}

	WGPUTextureUsage ToWGPUUsage(EWebGPUTextureUsage Usage)
	{
		WGPUTextureUsage Result = WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst;
		if (EnumHasAnyFlags(Usage, EWebGPUTextureUsage::Storage))
		{
			Result |= WGPUTextureUsage_StorageBinding;
		}
		if (EnumHasAnyFlags(Usage, EWebGPUTextureUsage::Sampled))
		{
			Result |= WGPUTextureUsage_TextureBinding;
		}
		return Result;
	}

	void ReleaseHandles(WGPUTexture Texture, WGPUTextureView View)
	{
		if (View)
		{
			wgpuTextureViewRelease(View);
		}
		if (Texture)
		{
			wgpuTextureRelease(Texture);
		}
	}
}

// FWebGPUTexture

uint32 FWebGPUTexture::BytesPerPixel(EWebGPUTextureFormat Format)
{
	switch (Format)
	{
	case EWebGPUTextureFormat::RGBA8Unorm: return 4;
	case EWebGPUTextureFormat::BGRA8Unorm: return 4;
	case EWebGPUTextureFormat::RGBA16Float: return 8;
	case EWebGPUTextureFormat::RGBA32Float: return 16;
	case EWebGPUTextureFormat::R32Float: return 4;
	case EWebGPUTextureFormat::RG32Float: return 8;
	case EWebGPUTextureFormat::R32Uint: return 4;
	case EWebGPUTextureFormat::R32Sint: return 4;
	case EWebGPUTextureFormat::R16Float: return 2;
	case EWebGPUTextureFormat::RG16Float: return 4;
	case EWebGPUTextureFormat::R8Unorm: return 1;
	default: return 4;
	}
}

const TCHAR* FWebGPUTexture::WgslFormat(EWebGPUTextureFormat Format)
{
	switch (Format)
	{
	case EWebGPUTextureFormat::RGBA8Unorm: return TEXT("rgba8unorm");
	case EWebGPUTextureFormat::BGRA8Unorm: return TEXT("bgra8unorm");
	case EWebGPUTextureFormat::RGBA16Float: return TEXT("rgba16float");
	case EWebGPUTextureFormat::RGBA32Float: return TEXT("rgba32float");
	case EWebGPUTextureFormat::R32Float: return TEXT("r32float");
	case EWebGPUTextureFormat::RG32Float: return TEXT("rg32float");
	case EWebGPUTextureFormat::R32Uint: return TEXT("r32uint");
	case EWebGPUTextureFormat::R32Sint: return TEXT("r32sint");
	case EWebGPUTextureFormat::R16Float: return TEXT("r16float");
	case EWebGPUTextureFormat::RG16Float: return TEXT("rg16float");
	case EWebGPUTextureFormat::R8Unorm: return TEXT("r8unorm");
	default: return TEXT("");
	}
}

WGPUTextureFormat FWebGPUTexture::ToWGPU(EWebGPUTextureFormat Format)
{
	switch (Format)
	{
	case EWebGPUTextureFormat::RGBA8Unorm: return WGPUTextureFormat_RGBA8Unorm;
	case EWebGPUTextureFormat::BGRA8Unorm: return WGPUTextureFormat_BGRA8Unorm;
	case EWebGPUTextureFormat::RGBA16Float: return WGPUTextureFormat_RGBA16Float;
	case EWebGPUTextureFormat::RGBA32Float: return WGPUTextureFormat_RGBA32Float;
	case EWebGPUTextureFormat::R32Float: return WGPUTextureFormat_R32Float;
	case EWebGPUTextureFormat::RG32Float: return WGPUTextureFormat_RG32Float;
	case EWebGPUTextureFormat::R32Uint: return WGPUTextureFormat_R32Uint;
	case EWebGPUTextureFormat::R32Sint: return WGPUTextureFormat_R32Sint;
	case EWebGPUTextureFormat::R16Float: return WGPUTextureFormat_R16Float;
	case EWebGPUTextureFormat::RG16Float: return WGPUTextureFormat_RG16Float;
	case EWebGPUTextureFormat::R8Unorm: return WGPUTextureFormat_R8Unorm;
	default: return WGPUTextureFormat_Undefined;
	}
}

TSharedPtr<FWebGPUContext> FWebGPUTexture::GetContext() const
{
	return Context.Pin();
}

FWebGPUTexture::~FWebGPUTexture()
{
	if (!Handle && !View && !Shared)
	{
		return;
	}

	WGPUTexture TextureHandle = Handle;
	WGPUTextureView TextureView = View;
	TSharedPtr<FWebGPUSharedTextureState> SharedState = Shared;
	const bool bWasAccessing = bAccessing;
	auto Release = [TextureHandle, TextureView, SharedState, bWasAccessing]()
	{
		if (SharedState)
		{
			WebGPUTextureInterop::Release(*SharedState, TextureHandle, bWasAccessing);
		}
		// Released, not destroyed: commands of the open batch may still use it
		ReleaseHandles(TextureHandle, TextureView);
	};

	TSharedPtr<FWebGPUContext> Ctx = Context.Pin();
	if (Ctx && !Ctx->IsInWorkerThread())
	{
		Ctx->Enqueue(MoveTemp(Release));
	}
	else
	{
		Release();
	}
}

// FWebGPUContext: textures

TSharedRef<FWebGPUTexture> FWebGPUContext::CreateTexture(uint32 Width, uint32 Height, EWebGPUTextureFormat Format, EWebGPUTextureUsage Usage, const FString& Label)
{
	TSharedRef<FWebGPUTexture> Texture = MakeShared<FWebGPUTexture>();
	Texture->Context = AsShared();
	Texture->Width = FMath::Max(Width, 1u);
	Texture->Height = FMath::Max(Height, 1u);
	Texture->Format = Format;
	Texture->Usage = Usage;
	Texture->Label = Label.IsEmpty() ? TEXT("WebGPUTexture") : Label;

	Enqueue([this, Texture]()
	{
		LiveTextures.Add(Texture);
		CreateTextureHandle(*Texture);
	});
	return Texture;
}

TSharedPtr<FWebGPUTexture> FWebGPUContext::ImportTexture(FRHITexture* RHITexture, FString& OutError, const FString& Label)
{
	check(IsInGameThread());
	FIntPoint Size;
	EWebGPUTextureFormat Format = EWebGPUTextureFormat::RGBA8Unorm;
	EWebGPUTextureUsage Usage = EWebGPUTextureUsage::None;
	TSharedPtr<FWebGPUSharedTextureState> State = WebGPUTextureInterop::CreateState(RHITexture, Size, Format, Usage, OutError);
	if (!State)
	{
		return nullptr;
	}

	TSharedRef<FWebGPUTexture> Texture = MakeShared<FWebGPUTexture>();
	Texture->Context = AsShared();
	Texture->Width = (uint32)Size.X;
	Texture->Height = (uint32)Size.Y;
	Texture->Format = Format;
	Texture->Usage = Usage;
	Texture->Label = Label.IsEmpty() ? TEXT("WebGPUSharedTexture") : Label;
	Texture->Shared = State;

	Enqueue([this, Texture]()
	{
		LiveTextures.Add(Texture);
		CreateTextureHandle(*Texture);
	});
	return Texture;
}

void FWebGPUContext::SetTexture(const TSharedRef<FWebGPUKernel>& Kernel, uint32 Group, uint32 Binding, const TSharedPtr<FWebGPUTexture>& Texture)
{
	Enqueue([Kernel, Group, Binding, Texture]()
	{
		if (Texture)
		{
			FWebGPUKernel::FBinding& Entry = Kernel->Bindings.FindOrAdd(Group).FindOrAdd(Binding);
			Entry = FWebGPUKernel::FBinding();
			Entry.Texture = Texture;
		}
		else if (TSortedMap<uint32, FWebGPUKernel::FBinding>* GroupBindings = Kernel->Bindings.Find(Group))
		{
			GroupBindings->Remove(Binding);
			if (GroupBindings->Num() == 0)
			{
				Kernel->Bindings.Remove(Group);
			}
		}
		Kernel->bBindGroupsDirty = true;
	});
}

void FWebGPUContext::SetTextureByName(const TSharedRef<FWebGPUKernel>& Kernel, const FString& Name, const TSharedPtr<FWebGPUTexture>& Texture)
{
	Enqueue([Kernel, Name, Texture]()
	{
		if (Texture)
		{
			FWebGPUKernel::FBinding& Entry = Kernel->NamedBindings.FindOrAdd(Name);
			Entry = FWebGPUKernel::FBinding();
			Entry.Texture = Texture;
		}
		else
		{
			Kernel->NamedBindings.Remove(Name);
		}
		Kernel->bBindGroupsDirty = true;
	});
}

void FWebGPUContext::SetSamplerByName(const TSharedRef<FWebGPUKernel>& Kernel, const FString& Name, EWebGPUSampler Sampler)
{
	Enqueue([Kernel, Name, Sampler]()
	{
		FWebGPUKernel::FBinding& Entry = Kernel->NamedBindings.FindOrAdd(Name);
		Entry = FWebGPUKernel::FBinding();
		Entry.Sampler = Sampler;
		Kernel->bBindGroupsDirty = true;
	});
}

void FWebGPUContext::WriteTexture(const TSharedRef<FWebGPUTexture>& Texture, const void* Pixels, uint64 Size)
{
	TArray<uint8> Copy;
	Copy.Append(static_cast<const uint8*>(Pixels), (int32)Size);
	WriteTexture(Texture, MoveTemp(Copy));
}

void FWebGPUContext::WriteTexture(const TSharedRef<FWebGPUTexture>& Texture, TArray<uint8>&& Pixels)
{
	const uint64 Expected = (uint64)Texture->Width * Texture->Height * Texture->GetBytesPerPixel();
	if ((uint64)Pixels.Num() != Expected)
	{
		UE_LOG(LogWebGPU, Error, TEXT("WriteTexture(%s): %d bytes given, a %ux%u %s texture takes %llu (rows tightly packed)"),
			*Texture->Label, Pixels.Num(), Texture->Width, Texture->Height, FWebGPUTexture::WgslFormat(Texture->Format), Expected);
		return;
	}
	Enqueue([this, Texture, Pixels = MoveTemp(Pixels)]()
	{
		WriteTextureOnWorker(*Texture, Pixels.GetData(), (uint64)Pixels.Num());
	});
}

void FWebGPUContext::ReadTexture(const TSharedRef<FWebGPUTexture>& Texture, FWebGPUReadCallback&& Callback, EWebGPUCallbackThread CallbackThread)
{
	Enqueue([this, Texture, Callback = MoveTemp(Callback), CallbackThread]() mutable
	{
		ReadTextureOnWorker(*Texture, MoveTemp(Callback), CallbackThread);
	});
}

void FWebGPUContext::BeginTextureAccess(const TSharedRef<FWebGPUTexture>& Texture)
{
	if (!Texture->Shared)
	{
		return;
	}
	check(IsInGameThread());
	const uint64 Value = WebGPUTextureInterop::NextSignalValue(*Texture->Shared);
	WebGPUTextureInterop::EnqueueSignal(Texture->Shared.ToSharedRef(), Value);
	Enqueue([this, Texture, Value]()
	{
		if (!HasDevice() || !Texture->Handle)
		{
			return;
		}
		if (Texture->bAccessing)
		{
			UE_LOG(LogWebGPU, Warning, TEXT("BeginTextureAccess(%s) called again before EndTextureAccess"), *Texture->Label);
			return;
		}
		Texture->bAccessing = WebGPUTextureInterop::BeginAccess(*Texture->Shared, Texture->Handle, Value);
	});
}

void FWebGPUContext::EndTextureAccess(const TSharedRef<FWebGPUTexture>& Texture)
{
	if (!Texture->Shared)
	{
		return;
	}
	check(IsInGameThread());
	TSharedRef<FWebGPUTextureAccessResult> Result = WebGPUTextureInterop::MakeAccessResult();
	Enqueue([this, Texture, Result]()
	{
		if (Texture->bAccessing && HasDevice())
		{
			// Dawn hands out fences for submitted work only
			SubmitBatch();
			WebGPUTextureInterop::EndAccess(*Texture->Shared, Texture->Handle, *Result);
		}
		Texture->bAccessing = false;
		WebGPUTextureInterop::CompleteAccess(*Result);
	});
	WebGPUTextureInterop::EnqueueWait(Result);
}

// Worker thread

void FWebGPUContext::CreateTextureHandle(FWebGPUTexture& Texture)
{
	if (!HasDevice())
	{
		return;
	}

	if (Texture.Shared)
	{
		FString Error;
		Texture.Handle = WebGPUTextureInterop::Import(*Texture.Shared, *WorkerDevice, Error);
		if (!Texture.Handle)
		{
			UE_LOG(LogWebGPU, Error, TEXT("Sharing texture '%s' with WebGPU failed: %s"), *Texture.Label, *Error);
			return;
		}
	}
	else
	{
		FTCHARToUTF8 Label(*Texture.Label);
		WGPUTextureDescriptor Desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
		Desc.label = ToView(Label);
		Desc.usage = ToWGPUUsage(Texture.Usage);
		Desc.dimension = WGPUTextureDimension_2D;
		Desc.size = { Texture.Width, Texture.Height, 1 };
		Desc.format = FWebGPUTexture::ToWGPU(Texture.Format);

		WorkerDevice->PushErrorScopes();
		Texture.Handle = wgpuDeviceCreateTexture(WorkerDevice->GetDevice(), &Desc);
		FString Error;
		if (!WorkerDevice->PopErrorScopes(Error))
		{
			UE_LOG(LogWebGPU, Error, TEXT("CreateTexture(%s, %ux%u %s) failed: %s"), *Texture.Label, Texture.Width, Texture.Height, FWebGPUTexture::WgslFormat(Texture.Format), *Error);
			ReleaseHandles(Texture.Handle, nullptr);
			Texture.Handle = nullptr;
			return;
		}
	}
	Texture.View = wgpuTextureCreateView(Texture.Handle, nullptr);
}

void FWebGPUContext::ReleaseTextureHandle(FWebGPUTexture& Texture)
{
	if (Texture.Shared)
	{
		WebGPUTextureInterop::Release(*Texture.Shared, Texture.Handle, Texture.bAccessing);
	}
	ReleaseHandles(Texture.Handle, Texture.View);
	Texture.Handle = nullptr;
	Texture.View = nullptr;
	Texture.bAccessing = false;
}

bool FWebGPUContext::CanUseTexture(FWebGPUTexture& Texture, const TCHAR* What)
{
	if (!HasDevice() || !Texture.Handle)
	{
		UE_LOG(LogWebGPU, Warning, TEXT("%s(%s) skipped: the texture has no GPU memory"), What, *Texture.Label);
		return false;
	}
	if (Texture.Shared && !Texture.bAccessing)
	{
		UE_LOG(LogWebGPU, Warning, TEXT("%s(%s) skipped: shared textures are only usable between BeginTextureAccess and EndTextureAccess"), What, *Texture.Label);
		return false;
	}
	return true;
}

void FWebGPUContext::WriteTextureOnWorker(FWebGPUTexture& Texture, const uint8* Pixels, uint64 Size)
{
	if (!CanUseTexture(Texture, TEXT("WriteTexture")))
	{
		return;
	}

	const uint32 Row = Texture.Width * Texture.GetBytesPerPixel();
	const uint32 Pitch = Texture.GetCopyBytesPerRow();
	WGPUTexelCopyTextureInfo Destination = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
	Destination.texture = Texture.Handle;
	const WGPUExtent3D Extent = { Texture.Width, Texture.Height, 1 };

	uint64 PageOffset = 0;
	if (FUploadPage* Page = AllocateUpload((uint64)Pitch * Texture.Height, PageOffset, 256))
	{
		// Buffer -> texture copies need 256-byte aligned rows
		for (uint32 Y = 0; Y < Texture.Height; Y++)
		{
			FMemory::Memcpy(Page->Mapped + PageOffset + (uint64)Y * Pitch, Pixels + (uint64)Y * Row, Row);
		}
		WGPUTexelCopyBufferInfo Source = {};
		Source.layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
		Source.layout.offset = PageOffset;
		Source.layout.bytesPerRow = Pitch;
		Source.layout.rowsPerImage = Texture.Height;
		Source.buffer = Page->Buffer;
		wgpuCommandEncoderCopyBufferToTexture(GetEncoder(), &Source, &Destination, &Extent);
	}
	else
	{
		SubmitBatch();
		WGPUTexelCopyBufferLayout Layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
		Layout.bytesPerRow = Row;
		Layout.rowsPerImage = Texture.Height;
		wgpuQueueWriteTexture(WorkerDevice->GetQueue(), &Destination, Pixels, (size_t)Size, &Layout, &Extent);
	}
	StatBytesWritten += Size;
}

void FWebGPUContext::ReadTextureOnWorker(FWebGPUTexture& Texture, FWebGPUReadCallback&& Callback, EWebGPUCallbackThread CallbackThread)
{
	if (!CanUseTexture(Texture, TEXT("ReadTexture")))
	{
		Deliver(CallbackThread, [Callback = MoveTemp(Callback)]() mutable
		{
			if (Callback)
			{
				Callback(false, TArray<uint8>());
			}
		});
		return;
	}

	const uint32 Row = Texture.Width * Texture.GetBytesPerPixel();
	const uint32 Pitch = Texture.GetCopyBytesPerRow();
	const uint32 Height = Texture.Height;
	const uint64 Size = (uint64)Pitch * Height;

	uint64 Capacity = 0;
	WGPUBuffer Staging = AcquireStaging(Size, Capacity);
	WGPUTexelCopyTextureInfo Source = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
	Source.texture = Texture.Handle;
	WGPUTexelCopyBufferInfo Destination = {};
	Destination.layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
	Destination.layout.bytesPerRow = Pitch;
	Destination.layout.rowsPerImage = Height;
	Destination.buffer = Staging;
	const WGPUExtent3D Extent = { Texture.Width, Height, 1 };
	wgpuCommandEncoderCopyTextureToBuffer(GetEncoder(), &Source, &Destination, &Extent);
	StatReadbacks++;

	const uint32 Generation = DeviceGeneration;
	AfterSubmit([this, Staging, Capacity, Size, Row, Pitch, Height, Generation, Callback = MoveTemp(Callback), CallbackThread](bool bSubmitted) mutable
	{
		if (!bSubmitted)
		{
			wgpuBufferRelease(Staging);
			Deliver(CallbackThread, [Callback = MoveTemp(Callback)]() mutable
			{
				if (Callback)
				{
					Callback(false, TArray<uint8>());
				}
			});
			return;
		}
		MapAsync(Staging, WGPUMapMode_Read, 0, Size, [this, Staging, Capacity, Size, Row, Pitch, Height, Generation, Callback = MoveTemp(Callback), CallbackThread](bool bSuccess) mutable
		{
			TArray<uint8> Pixels;
			if (bSuccess)
			{
				const uint8* Mapped = static_cast<const uint8*>(wgpuBufferGetConstMappedRange(Staging, 0, (size_t)Size));
				bSuccess = Mapped != nullptr;
				if (bSuccess)
				{
					// Back to tightly packed rows
					Pixels.SetNumUninitialized((int32)((uint64)Row * Height));
					for (uint32 Y = 0; Y < Height; Y++)
					{
						FMemory::Memcpy(Pixels.GetData() + (uint64)Y * Row, Mapped + (uint64)Y * Pitch, Row);
					}
					StatBytesRead += Pixels.Num();
				}
				wgpuBufferUnmap(Staging);
				ReleaseStaging(Staging, Capacity, Generation);
			}
			else
			{
				wgpuBufferRelease(Staging);
			}
			Deliver(CallbackThread, [Callback = MoveTemp(Callback), bSuccess, Pixels = MoveTemp(Pixels)]() mutable
			{
				if (Callback)
				{
					Callback(bSuccess, MoveTemp(Pixels));
				}
			});
		});
	});
}
