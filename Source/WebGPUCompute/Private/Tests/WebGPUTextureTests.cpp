// Copyright 2025-current Getnamo. All Rights Reserved.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "WebGPUBlueprintLibrary.h"
#include "WebGPUCompute.h"
#include "WebGPUContext.h"
#include "WebGPUStreams.h"
#include "WebGPUSubsystem.h"
#include "WebGPUTexture.h"
#include "Engine/TextureRenderTarget2D.h"
#include "HAL/PlatformProcess.h"
#include "Misc/ScopeExit.h"
#include "RenderingThread.h"
#include "RHICommandList.h"
#include "TextureResource.h"

namespace WebGPUTextureTests
{
	constexpr EAutomationTestFlags Flags = EAutomationTestFlags::EditorContext | EAutomationTestFlags::ClientContext | EAutomationTestFlags::ProductFilter;

	bool WaitUntil(TFunctionRef<bool()> Condition, double TimeoutSeconds = 10.0)
	{
		const double End = FPlatformTime::Seconds() + TimeoutSeconds;
		while (!Condition())
		{
			if (FPlatformTime::Seconds() > End)
			{
				return false;
			}
			FPlatformProcess::Sleep(0.0002f);
		}
		return true;
	}

	struct FReadResult
	{
		std::atomic<bool> bDone = false;
		std::atomic<bool> bSuccess = false;
		TArray<uint8> Data;
	};

	bool ReadTextureNow(FWebGPUContext& Context, const TSharedRef<FWebGPUTexture>& Texture, TArray<uint8>& OutPixels)
	{
		TSharedRef<FReadResult> Result = MakeShared<FReadResult>();
		Context.ReadTexture(Texture, [Result](bool bSuccess, TArray<uint8>&& Data)
		{
			Result->Data = MoveTemp(Data);
			Result->bSuccess = bSuccess;
			Result->bDone = true;
		}, EWebGPUCallbackThread::WorkerThread);
		if (!WaitUntil([&] { return Result->bDone.load(); }) || !Result->bSuccess)
		{
			return false;
		}
		OutPixels = MoveTemp(Result->Data);
		return true;
	}

	bool ReadBufferNow(FWebGPUContext& Context, const TSharedRef<FWebGPUBuffer>& Buffer, TArray<uint8>& OutData)
	{
		TSharedRef<FReadResult> Result = MakeShared<FReadResult>();
		Context.ReadBuffer(Buffer, 0, Buffer->GetSize(), [Result](bool bSuccess, TArray<uint8>&& Data)
		{
			Result->Data = MoveTemp(Data);
			Result->bSuccess = bSuccess;
			Result->bDone = true;
		}, EWebGPUCallbackThread::WorkerThread);
		if (!WaitUntil([&] { return Result->bDone.load(); }) || !Result->bSuccess)
		{
			return false;
		}
		OutData = MoveTemp(Result->Data);
		return true;
	}

	bool WaitReady(const TSharedRef<FWebGPUKernel>& Kernel)
	{
		return WaitUntil([&] { return Kernel->IsReady() || Kernel->GetMessages().ContainsByPredicate([](const FWebGPUCompileMessage& M) { return M.bIsError; }); }) && Kernel->IsReady();
	}

	bool IsD3D12RHI()
	{
		return GDynamicRHI && GDynamicRHI->GetInterfaceType() == ERHIInterfaceType::D3D12;
	}

	/** A shared, UAV-capable RGBA16F render target, as a project would set one up. */
	UTextureRenderTarget2D* MakeSharedRenderTarget(int32 Size, const FLinearColor& ClearColor)
	{
		UTextureRenderTarget2D* RenderTarget = NewObject<UTextureRenderTarget2D>();
		RenderTarget->RenderTargetFormat = RTF_RGBA16f;
		RenderTarget->bGPUSharedFlag = true;
		RenderTarget->bCanCreateUAV = true;
		RenderTarget->ClearColor = ClearColor;
		RenderTarget->InitAutoFormat(Size, Size);
		RenderTarget->UpdateResourceImmediate(true);
		RenderTarget->AddToRoot();
		return RenderTarget;
	}

	/** Enqueues a clear of the render target to Color (UE GPU work, no CPU sync). */
	void EnqueueClear(UTextureRenderTarget2D* RenderTarget, const FLinearColor& Color)
	{
		FTextureRenderTargetResource* Resource = RenderTarget->GameThread_GetRenderTargetResource();
		ENQUEUE_RENDER_COMMAND(WebGPUTestClear)([Resource, Color](FRHICommandListImmediate& RHICmdList)
		{
			FRHITexture* Texture = Resource->GetRenderTargetTexture();
			FUnorderedAccessViewRHIRef UAV = RHICmdList.CreateUnorderedAccessView(Texture, FRHIViewDesc::CreateTextureUAV().SetDimensionFromTexture(Texture));
			RHICmdList.Transition(FRHITransitionInfo(Texture, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
			RHICmdList.ClearUAVFloat(UAV, FVector4f(Color.R, Color.G, Color.B, Color.A));
			RHICmdList.Transition(FRHITransitionInfo(Texture, ERHIAccess::UAVCompute, ERHIAccess::SRVMask));
		});
	}

	TArray<FFloat16Color> ReadRenderTarget(UTextureRenderTarget2D* RenderTarget)
	{
		TArray<FFloat16Color> Pixels;
		FTextureRenderTargetResource* Resource = RenderTarget->GameThread_GetRenderTargetResource();
		const FIntPoint Size(RenderTarget->SizeX, RenderTarget->SizeY);
		ENQUEUE_RENDER_COMMAND(WebGPUTestRead)([Resource, Size, &Pixels](FRHICommandListImmediate& RHICmdList)
		{
			RHICmdList.ReadSurfaceFloatData(Resource->GetRenderTargetTexture(), FIntRect(0, 0, Size.X, Size.Y), Pixels, CubeFace_PosX, 0, 0);
		});
		FlushRenderingCommands();
		return Pixels;
	}

	FLinearColor FrameColor(int32 Frame)
	{
		return FLinearColor((Frame % 16) * 0.0625f, 0.5f, 1.f - (Frame % 8) * 0.125f, 1.f);
	}
}

using namespace WebGPUTextureTests;

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUTextureRoundTripTest, "WebGPU.Textures.WriteReadRoundTrip", WebGPUTextureTests::Flags)
bool FWebGPUTextureRoundTripTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}

	// Widths whose rows are not 256-byte multiples exercise the copy padding both ways
	struct FCase { uint32 Width; uint32 Height; EWebGPUTextureFormat Format; };
	for (const FCase& Case : { FCase{ 67, 37, EWebGPUTextureFormat::RGBA8Unorm }, FCase{ 33, 20, EWebGPUTextureFormat::RGBA32Float },
		FCase{ 300, 5, EWebGPUTextureFormat::R32Uint }, FCase{ 64, 64, EWebGPUTextureFormat::RGBA16Float } })
	{
		TSharedRef<FWebGPUTexture> Texture = Context->CreateTexture(Case.Width, Case.Height, Case.Format, EWebGPUTextureUsage::Sampled, TEXT("roundtrip"));
		TArray<uint8> Pixels;
		Pixels.SetNumUninitialized(Case.Width * Case.Height * Texture->GetBytesPerPixel());
		for (int32 Index = 0; Index < Pixels.Num(); Index++)
		{
			// Avoid f16/f32 NaN patterns: keep every 2nd/4th byte small
			Pixels[Index] = (uint8)((Index * 7 + Index / 251) & 0x3f);
		}
		Context->WriteTexture(Texture, Pixels.GetData(), Pixels.Num());
		TArray<uint8> Back;
		const bool bRead = ReadTextureNow(*Context, Texture, Back);
		TestTrue(*FString::Printf(TEXT("%ux%u %s round trip"), Case.Width, Case.Height, FWebGPUTexture::WgslFormat(Case.Format)),
			bRead && Back.Num() == Pixels.Num() && FMemory::Memcmp(Back.GetData(), Pixels.GetData(), Pixels.Num()) == 0);
	}

	AddExpectedMessage(TEXT("rows tightly packed"), ELogVerbosity::Error, EAutomationExpectedMessageFlags::Contains, 1, false);
	TSharedRef<FWebGPUTexture> Small = Context->CreateTexture(4, 4, EWebGPUTextureFormat::RGBA8Unorm);
	const uint8 Wrong[10] = {};
	Context->WriteTexture(Small, Wrong, sizeof(Wrong));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUTextureKernelTest, "WebGPU.Textures.Kernels", WebGPUTextureTests::Flags)
bool FWebGPUTextureKernelTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}

	// A kernel writes a storage texture; another reads it with textureLoad and a sampler
	const uint32 Width = 40;
	const uint32 Height = 24;
	TSharedRef<FWebGPUTexture> Image = Context->CreateTexture(Width, Height, EWebGPUTextureFormat::RGBA16Float, EWebGPUTextureUsage::Storage | EWebGPUTextureUsage::Sampled, TEXT("image"));
	TSharedRef<FWebGPUKernel> Paint = Context->CreateKernel(TEXT(R"(
@group(0) @binding(0) var image: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let size = textureDimensions(image);
	if (id.x < size.x && id.y < size.y) {
		textureStore(image, vec2<i32>(id.xy), vec4<f32>(f32(id.x), f32(id.y), f32(id.x + id.y), 1.0));
	}
}
)"), TEXT("main"), TEXT("paint"));
	TSharedRef<FWebGPUKernel> Probe = Context->CreateKernel(TEXT(R"(
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var nearest_sampler: sampler;
@group(0) @binding(3) var<storage, read_write> loaded: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> between: array<vec4<f32>>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let size = textureDimensions(image);
	if (id.x >= size.x || id.y >= size.y) { return; }
	let i = id.y * size.x + id.x;
	loaded[i] = textureLoad(image, vec2<i32>(id.xy), 0);
	// Halfway between texel x and x + 1 on the same row
	let uv = (vec2<f32>(id.xy) + vec2<f32>(1.0, 0.5)) / vec2<f32>(size);
	between[i * 2u] = textureSampleLevel(image, linear_sampler, uv, 0.0);
	between[i * 2u + 1u] = textureSampleLevel(image, nearest_sampler, (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size), 0.0);
}
)"), TEXT("main"), TEXT("probe"));
	TSharedRef<FWebGPUBuffer> Loaded = Context->CreateBuffer(Width * Height * 16, EWebGPUBufferUsage::Storage, TEXT("loaded"));
	TSharedRef<FWebGPUBuffer> Between = Context->CreateBuffer(Width * Height * 32, EWebGPUBufferUsage::Storage, TEXT("between"));
	Context->SetTextureByName(Paint, TEXT("image"), Image);
	Context->SetTextureByName(Probe, TEXT("image"), Image);
	Context->SetSamplerByName(Probe, TEXT("nearest_sampler"), EWebGPUSampler::NearestClamp);
	Context->SetBufferByName(Probe, TEXT("loaded"), Loaded);
	Context->SetBufferByName(Probe, TEXT("between"), Between);
	if (!TestTrue(TEXT("Paint compiled"), WaitReady(Paint)) || !TestTrue(TEXT("Probe compiled"), WaitReady(Probe)))
	{
		AddError(FWebGPUDevice::FormatCompileMessages(Probe->GetMessages()));
		return false;
	}

	TSharedRef<FWebGPUReadbackStream> Stream = Context->CreateReadbackStream((uint64)Image->GetCopyBytesPerRow() * Height, 2, EWebGPUStreamPolicy::Queue, TEXT("image readback"));
	const FWebGPUStats Before = Context->GetStats();
	Context->DispatchForCount(Paint, Width, Height);
	Context->DispatchForCount(Probe, Width, Height);
	TestTrue(TEXT("Capture texture"), Stream->CaptureTexture(Image));

	TArray<uint8> LoadedBytes;
	TArray<uint8> BetweenBytes;
	if (!TestTrue(TEXT("Read back"), ReadBufferNow(*Context, Loaded, LoadedBytes) && ReadBufferNow(*Context, Between, BetweenBytes)))
	{
		return false;
	}
	TestEqual(TEXT("Both kernels ran"), Context->GetStats().SkippedDispatches - Before.SkippedDispatches, (int64)0);

	const FVector4f* Load = reinterpret_cast<const FVector4f*>(LoadedBytes.GetData());
	const FVector4f* Sampled = reinterpret_cast<const FVector4f*>(BetweenBytes.GetData());
	int32 BadLoad = 0;
	int32 BadLinear = 0;
	int32 BadNearest = 0;
	for (uint32 Y = 0; Y < Height; Y++)
	{
		for (uint32 X = 0; X < Width; X++)
		{
			const uint32 Index = Y * Width + X;
			const FVector4f Expected((float)X, (float)Y, (float)(X + Y), 1.f);
			BadLoad += Load[Index].Equals(Expected, 1e-3f) ? 0 : 1;
			BadNearest += Sampled[Index * 2 + 1].Equals(Expected, 1e-3f) ? 0 : 1;
			if (X + 1 < Width)
			{
				// Bilinear halfway between x and x + 1
				BadLinear += FMath::IsNearlyEqual(Sampled[Index * 2].X, X + 0.5f, 2e-2f) ? 0 : 1;
			}
		}
	}
	TestEqual(TEXT("textureLoad sees the storage writes"), BadLoad, 0);
	TestEqual(TEXT("Nearest sampler at texel centres"), BadNearest, 0);
	TestEqual(TEXT("Linear sampler (the default) interpolates"), BadLinear, 0);

	// The capture holds 256-byte aligned rows; CopyTo packs them tightly
	FWebGPUReadbackFrame Frame;
	if (TestTrue(TEXT("Texture frame"), Stream->WaitForFrame(5.0) && Stream->Acquire(Frame)))
	{
		TestEqual(TEXT("Row pitch"), Frame.GetRowPitch(), 512u);
		TestEqual(TEXT("Row bytes"), Frame.GetRowBytes(), Width * 8);
		TArray<uint8> Tight;
		Frame.CopyTo(Tight);
		const FFloat16Color* Pixels = reinterpret_cast<const FFloat16Color*>(Tight.GetData());
		int32 Bad = Tight.Num() == (int32)(Width * Height * 8) ? 0 : 1;
		for (uint32 Index = 0; Bad == 0 && Index < Width * Height; Index++)
		{
			Bad += Pixels[Index].R.GetFloat() == (float)(Index % Width) && Pixels[Index].G.GetFloat() == (float)(Index / Width) ? 0 : 1;
		}
		TestEqual(TEXT("Captured texture matches"), Bad, 0);
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUTextureValidationTest, "WebGPU.Textures.Validation", WebGPUTextureTests::Flags)
bool FWebGPUTextureValidationTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}

	auto Expect = [this](const TCHAR* Pattern) { AddExpectedMessage(Pattern, ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false); };
	auto Skipped = [&Context]() { return Context->GetStats().SkippedDispatches; };
	auto Check = [&](const TCHAR* What, const TSharedRef<FWebGPUKernel>& Kernel)
	{
		const int64 SkippedBefore = Skipped();
		Context->Dispatch(Kernel, 1);
		Context->Flush();
		TestEqual(What, Skipped() - SkippedBefore, (int64)1);
	};
	const FWebGPUStats Before = Context->GetStats();

	TSharedRef<FWebGPUKernel> Storage = Context->CreateKernel(TEXT(R"(
@group(0) @binding(0) var output: texture_storage_2d<rgba32float, write>;
@compute @workgroup_size(1) fn main() { textureStore(output, vec2<i32>(0, 0), vec4<f32>(1.0)); }
)"), TEXT("main"), TEXT("storage-kernel"));
	TSharedRef<FWebGPUKernel> Volume = Context->CreateKernel(TEXT(R"(
@group(0) @binding(0) var volume: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(1) fn main() { out[0] = textureLoad(volume, vec3<i32>(0), 0); }
)"), TEXT("main"), TEXT("volume-kernel"));
	TSharedRef<FWebGPUKernel> Filtered = Context->CreateKernel(TEXT(R"(
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var smoothing: sampler;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(1) fn main() { out[0] = textureSampleLevel(image, smoothing, vec2<f32>(0.5), 0.0); }
)"), TEXT("main"), TEXT("filtered-kernel"));
	if (!TestTrue(TEXT("Compiled"), WaitReady(Storage) && WaitReady(Volume) && WaitReady(Filtered)))
	{
		return false;
	}
	TSharedRef<FWebGPUBuffer> Out = Context->CreateBuffer(16, EWebGPUBufferUsage::Storage, TEXT("out"));
	Context->SetBufferByName(Volume, TEXT("out"), Out);
	Context->SetBufferByName(Filtered, TEXT("out"), Out);

	Expect(TEXT("'output' (@group(0) @binding(0)) is not bound to a texture"));
	Check(TEXT("Unbound texture skips"), Storage);

	Expect(TEXT("is texture_storage_2d<rgba32float> but texture 'r32' is r32float"));
	Context->SetTextureByName(Storage, TEXT("output"), Context->CreateTexture(4, 4, EWebGPUTextureFormat::R32Float, EWebGPUTextureUsage::Storage, TEXT("r32")));
	Check(TEXT("Storage format mismatch skips"), Storage);

	Expect(TEXT("was not created with the Storage usage"));
	Context->SetTextureByName(Storage, TEXT("output"), Context->CreateTexture(4, 4, EWebGPUTextureFormat::RGBA32Float, EWebGPUTextureUsage::Sampled, TEXT("sampled-only")));
	Check(TEXT("Missing Storage usage skips"), Storage);

	Context->SetTextureByName(Storage, TEXT("output"), Context->CreateTexture(4, 4, EWebGPUTextureFormat::RGBA32Float, EWebGPUTextureUsage::Storage, TEXT("ok")));
	const int64 SkippedBefore = Skipped();
	Context->Dispatch(Storage, 1);
	Context->Flush();
	TestEqual(TEXT("Matching storage texture dispatches"), Skipped() - SkippedBefore, (int64)0);

	Expect(TEXT("only 2D textures can be bound"));
	Context->SetTextureByName(Volume, TEXT("volume"), Context->CreateTexture(4, 4, EWebGPUTextureFormat::RGBA8Unorm, EWebGPUTextureUsage::Sampled, TEXT("flat")));
	Check(TEXT("3D texture binding skips"), Volume);

	// Validation only WebGPU knows: rgba32float isn't filterable, so the filtering sampler's layout rejects it
	TSharedPtr<FWebGPUDevice> Device = Context->GetDevice();
	if (Device && !Device->HasFeature(WGPUFeatureName_Float32Filterable))
	{
		Expect(TEXT("WebGPU rejected its bindings"));
		Context->SetTextureByName(Filtered, TEXT("image"), Context->CreateTexture(4, 4, EWebGPUTextureFormat::RGBA32Float, EWebGPUTextureUsage::Sampled, TEXT("unfilterable")));
		Check(TEXT("Binding WebGPU rejects skips the kernel only"), Filtered);
	}

	TestEqual(TEXT("No batch was rejected"), Context->GetStats().FailedSubmits - Before.FailedSubmits, (int64)0);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUSharedRenderTargetTest, "WebGPU.Textures.SharedRenderTarget", WebGPUTextureTests::Flags)
bool FWebGPUSharedRenderTargetTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}
	if (!IsD3D12RHI())
	{
		AddInfo(TEXT("Skipped: shared render targets need the D3D12 RHI (run with -RenderOffscreen)"));
		return true;
	}

	// UE clears Source every frame; a kernel reads it and writes Target, which UE reads back at the end
	const int32 Size = 64;
	UTextureRenderTarget2D* SourceRT = MakeSharedRenderTarget(Size, FLinearColor::Black);
	UTextureRenderTarget2D* TargetRT = MakeSharedRenderTarget(Size, FLinearColor::Black);
	ON_SCOPE_EXIT
	{
		SourceRT->RemoveFromRoot();
		TargetRT->RemoveFromRoot();
	};
	UWebGPUTexture* SourceObject = UWebGPUBlueprintLibrary::CreateWebGPUTextureFromRenderTarget(SourceRT, TEXT("source"));
	UWebGPUTexture* TargetObject = UWebGPUBlueprintLibrary::CreateWebGPUTextureFromRenderTarget(TargetRT, TEXT("target"));
	if (!TestNotNull(TEXT("Source shared"), SourceObject) || !TestNotNull(TEXT("Target shared"), TargetObject))
	{
		return false;
	}
	TSharedRef<FWebGPUTexture> Source = SourceObject->GetNative().ToSharedRef();
	TSharedRef<FWebGPUTexture> Target = TargetObject->GetNative().ToSharedRef();
	TestTrue(TEXT("Shared"), Source->IsShared() && Target->IsShared());
	TestTrue(TEXT("Format from the render target"), Source->GetFormat() == EWebGPUTextureFormat::RGBA16Float && Source->GetWidth() == (uint32)Size);

	TSharedRef<FWebGPUKernel> Kernel = Context->CreateKernel(TEXT(R"(
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var result: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<storage, read_write> probe: array<vec4<f32>>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let size = textureDimensions(result);
	if (id.x >= size.x || id.y >= size.y) { return; }
	let color = textureLoad(source, vec2<i32>(id.xy), 0);
	textureStore(result, vec2<i32>(id.xy), color + vec4<f32>(f32(id.x), f32(id.y), 0.0, 0.0));
	if (id.x == 3u && id.y == 5u) { probe[0] = color; }
}
)"), TEXT("main"), TEXT("shared-kernel"));
	TSharedRef<FWebGPUBuffer> Probe = Context->CreateBuffer(16, EWebGPUBufferUsage::Storage, TEXT("probe"));
	Context->SetTextureByName(Kernel, TEXT("source"), Source);
	Context->SetTextureByName(Kernel, TEXT("result"), Target);
	Context->SetBufferByName(Kernel, TEXT("probe"), Probe);
	if (!TestTrue(TEXT("Compiled"), WaitReady(Kernel)))
	{
		AddError(FWebGPUDevice::FormatCompileMessages(Kernel->GetMessages()));
		return false;
	}

	// Outside an access window the kernel is skipped, not the batch
	AddExpectedMessage(TEXT("is only usable between BeginTextureAccess and EndTextureAccess"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
	const int64 SkippedBefore = Context->GetStats().SkippedDispatches;
	Context->DispatchForCount(Kernel, Size, Size);
	Context->Flush();
	TestEqual(TEXT("Dispatch outside access skipped"), Context->GetStats().SkippedDispatches - SkippedBefore, (int64)1);

	// Frames: UE clear -> WebGPU reads it and writes Target -> captured. Nothing waits on the CPU.
	TSharedRef<FWebGPUReadbackStream> Probes = Context->CreateReadbackStream(16, 8, EWebGPUStreamPolicy::Queue, TEXT("probes"));
	const int32 Frames = 120;
	const FWebGPUStats Before = Context->GetStats();
	double EnqueueSeconds = 0.0;
	int32 Bad = 0;
	int32 Read = 0;
	auto Drain = [&]()
	{
		FWebGPUReadbackFrame Frame;
		while (Probes->Acquire(Frame))
		{
			const FVector4f Color = Frame.View<FVector4f>()[0];
			const FLinearColor Expected = FrameColor((int32)Frame.GetSequence() - 1);
			Bad += Color.Equals(FVector4f(Expected.R, Expected.G, Expected.B, Expected.A), 2e-3f) ? 0 : 1;
			Read++;
			Frame.Release();
		}
	};
	const double Start = FPlatformTime::Seconds();
	for (int32 Frame = 0; Frame < Frames; Frame++)
	{
		EnqueueClear(SourceRT, FrameColor(Frame));
		const double EnqueueStart = FPlatformTime::Seconds();
		Context->BeginTextureAccess(Source);
		Context->BeginTextureAccess(Target);
		Context->DispatchForCount(Kernel, Size, Size);
		Context->EndTextureAccess(Target);
		Context->EndTextureAccess(Source);
		EnqueueSeconds += FPlatformTime::Seconds() - EnqueueStart;
		const double Deadline = FPlatformTime::Seconds() + 10.0;
		while (!Probes->Capture(Probe))
		{
			Drain();
			if (FPlatformTime::Seconds() > Deadline)
			{
				AddError(TEXT("Probe captures stopped coming back (GPU stalled?)"));
				FlushRenderingCommands();
				return false;
			}
			FPlatformProcess::Sleep(0.0001f);
		}
		Drain();
		// Keep the render thread within a few frames, as the engine's frame pacing would
		if (Frame % 4 == 3)
		{
			FlushRenderingCommands();
		}
	}
	FlushRenderingCommands();
	TestTrue(TEXT("Every frame read back"), WaitUntil([&] { Drain(); return Read == Frames; }));
	const double FrameMs = (FPlatformTime::Seconds() - Start) * 1000.0 / Frames;
	TestEqual(TEXT("WebGPU saw each frame's UE clear"), Bad, 0);
	TestEqual(TEXT("No dispatch skipped"), Context->GetStats().SkippedDispatches - Before.SkippedDispatches, (int64)0);
	TestEqual(TEXT("No batch rejected"), Context->GetStats().FailedSubmits - Before.FailedSubmits, (int64)0);

	// UE reads what WebGPU wrote in the last frame
	const TArray<FFloat16Color> Pixels = ReadRenderTarget(TargetRT);
	const FLinearColor Last = FrameColor(Frames - 1);
	int32 BadPixels = Pixels.Num() == Size * Size ? 0 : 1;
	for (int32 Index = 0; BadPixels == 0 && Index < Pixels.Num(); Index++)
	{
		const FLinearColor Pixel = FLinearColor(Pixels[Index]);
		const int32 X = Index % Size;
		const int32 Y = Index / Size;
		BadPixels += FMath::IsNearlyEqual(Pixel.R, Last.R + X, 0.05f) && FMath::IsNearlyEqual(Pixel.G, Last.G + Y, 0.05f) && FMath::IsNearlyEqual(Pixel.B, Last.B, 2e-3f) ? 0 : 1;
	}
	TestEqual(TEXT("UE sees WebGPU's writes to the shared target"), BadPixels, 0);
	AddInfo(FString::Printf(TEXT("Shared 64x64 RGBA16F render targets, UE clear -> WebGPU read + write -> probe capture, %d frames: %.3f ms per frame including render-thread flushes every 4 frames, %.1f us/frame to enqueue the access windows and dispatch"),
		Frames, FrameMs, EnqueueSeconds * 1.0e6 / Frames));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUSharedRecoveryTest, "WebGPU.Textures.SharedRecovery", WebGPUTextureTests::Flags)
bool FWebGPUSharedRecoveryTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}
	if (!IsD3D12RHI())
	{
		AddInfo(TEXT("Skipped: shared render targets need the D3D12 RHI (run with -RenderOffscreen)"));
		return true;
	}

	UTextureRenderTarget2D* RenderTarget = MakeSharedRenderTarget(32, FLinearColor::Black);
	ON_SCOPE_EXIT { RenderTarget->RemoveFromRoot(); };
	FString Error;
	FlushRenderingCommands();
	TSharedPtr<FWebGPUTexture> Texture = Context->ImportTexture(RenderTarget->GameThread_GetRenderTargetResource()->GetRenderTargetTexture(), Error, TEXT("recovered"));
	if (!TestTrue(FString::Printf(TEXT("Imported: %s"), *Error), Texture.IsValid()))
	{
		return false;
	}
	TSharedRef<FWebGPUKernel> Kernel = Context->CreateKernel(TEXT(R"(
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> probe: array<vec4<f32>>;
@compute @workgroup_size(1) fn main() { probe[0] = textureLoad(image, vec2<i32>(1, 1), 0); }
)"), TEXT("main"), TEXT("recovery-kernel"));
	TSharedRef<FWebGPUBuffer> Probe = Context->CreateBuffer(16, EWebGPUBufferUsage::Storage, TEXT("probe"));
	Context->SetTextureByName(Kernel, TEXT("image"), Texture);
	Context->SetBufferByName(Kernel, TEXT("probe"), Probe);

	auto Sample = [&](const FLinearColor& Color, FVector4f& OutValue)
	{
		EnqueueClear(RenderTarget, Color);
		Context->BeginTextureAccess(Texture.ToSharedRef());
		Context->Dispatch(Kernel, 1);
		Context->EndTextureAccess(Texture.ToSharedRef());
		FlushRenderingCommands();
		TArray<uint8> Bytes;
		if (!ReadBufferNow(*Context, Probe, Bytes))
		{
			return false;
		}
		OutValue = *reinterpret_cast<const FVector4f*>(Bytes.GetData());
		return true;
	};

	FVector4f Value;
	if (!TestTrue(TEXT("Compiled"), WaitReady(Kernel)))
	{
		return false;
	}
	TestTrue(TEXT("Shared read before the loss"), Sample(FLinearColor(0.25f, 0.5f, 0.75f, 1.f), Value) && Value.Equals(FVector4f(0.25f, 0.5f, 0.75f, 1.f), 2e-3f));

	AddExpectedMessage(TEXT("WebGPU device lost"), ELogVerbosity::Error, EAutomationExpectedMessageFlags::Contains, 0, false);
	AddExpectedMessage(TEXT("WebGPU device lost"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 0, false);
	const uint32 Generation = Context->GetDeviceGeneration();
	Context->ForceDeviceLoss();
	if (!TestTrue(TEXT("Device recovered"), WaitUntil([&] { return Context->GetDeviceGeneration() > Generation; })))
	{
		return false;
	}
	TestTrue(TEXT("Kernel recompiled"), WaitUntil([&] { return Kernel->IsReady(); }));
	TestTrue(TEXT("Shared read after the texture was imported again"), Sample(FLinearColor(0.125f, 0.25f, 0.5f, 1.f), Value) && Value.Equals(FVector4f(0.125f, 0.25f, 0.5f, 1.f), 2e-3f));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUSharedBenchmarkTest, "WebGPU.Textures.SharedBenchmark1080p", WebGPUTextureTests::Flags)
bool FWebGPUSharedBenchmarkTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}
	if (!IsD3D12RHI())
	{
		AddInfo(TEXT("Skipped: shared render targets need the D3D12 RHI (run with -RenderOffscreen)"));
		return true;
	}

	// The runtime path at 1080p: game thread enqueues an access window + a full-screen kernel per frame
	UTextureRenderTarget2D* RenderTarget = NewObject<UTextureRenderTarget2D>();
	RenderTarget->RenderTargetFormat = RTF_RGBA16f;
	RenderTarget->bGPUSharedFlag = true;
	RenderTarget->bCanCreateUAV = true;
	RenderTarget->InitAutoFormat(1920, 1080);
	RenderTarget->UpdateResourceImmediate(true);
	RenderTarget->AddToRoot();
	ON_SCOPE_EXIT { RenderTarget->RemoveFromRoot(); };

	UWebGPUTexture* Shared = UWebGPUBlueprintLibrary::CreateWebGPUTextureFromRenderTarget(RenderTarget, TEXT("bench"));
	if (!TestNotNull(TEXT("Shared"), Shared))
	{
		return false;
	}
	TSharedRef<FWebGPUTexture> Texture = Shared->GetNative().ToSharedRef();
	TSharedRef<FWebGPUKernel> Kernel = Context->CreateKernel(TEXT(R"(
@group(0) @binding(0) var image: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> frame: vec4<f32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let size = textureDimensions(image);
	if (id.x >= size.x || id.y >= size.y) { return; }
	let uv = vec2<f32>(id.xy) / vec2<f32>(size);
	textureStore(image, vec2<i32>(id.xy), vec4<f32>(uv, fract(frame.x * 0.01), 1.0));
}
)"), TEXT("main"), TEXT("bench-kernel"));
	TSharedRef<FWebGPUBuffer> Frame = Context->CreateBuffer(16, EWebGPUBufferUsage::Uniform, TEXT("bench.frame"));
	Context->SetTextureByName(Kernel, TEXT("image"), Texture);
	Context->SetBufferByName(Kernel, TEXT("frame"), Frame);
	if (!TestTrue(TEXT("Compiled"), WaitReady(Kernel)))
	{
		return false;
	}
	Context->SetProfiling(Kernel, true);

	const int32 Frames = 240;
	double EnqueueSeconds = 0.0;
	const double Start = FPlatformTime::Seconds();
	for (int32 Index = 0; Index < Frames; Index++)
	{
		const double EnqueueStart = FPlatformTime::Seconds();
		const FVector4f Value((float)Index, 0.f, 0.f, 0.f);
		Context->WriteBuffer(Frame, 0, &Value, sizeof(Value));
		Context->BeginTextureAccess(Texture);
		Context->DispatchForCount(Kernel, 1920, 1080);
		Context->EndTextureAccess(Texture);
		EnqueueSeconds += FPlatformTime::Seconds() - EnqueueStart;
		if (Index % 4 == 3)
		{
			FlushRenderingCommands();
		}
	}
	FlushRenderingCommands();
	TestTrue(TEXT("Flush"), Context->Flush());
	const double FrameMs = (FPlatformTime::Seconds() - Start) * 1000.0 / Frames;

	const TArray<FFloat16Color> Pixels = ReadRenderTarget(RenderTarget);
	const FLinearColor Corner = Pixels.Num() == 1920 * 1080 ? FLinearColor(Pixels[1080 / 2 * 1920 + 1920 / 2]) : FLinearColor::Black;
	TestTrue(TEXT("UE sees the last frame"), FMath::IsNearlyEqual(Corner.R, 0.5f, 2e-3f) && FMath::IsNearlyEqual(Corner.G, 0.5f, 2e-3f) && FMath::IsNearlyEqual(Corner.B, FMath::Frac((Frames - 1) * 0.01f), 2e-3f));
	AddInfo(FString::Printf(TEXT("Shared 1080p RGBA16F through the runtime, %d frames: %.3f ms/frame wall (render-thread flush every 4 frames), %.1f us/frame on the game thread, kernel %.3f ms on the GPU"),
		Frames, FrameMs, EnqueueSeconds * 1.0e6 / Frames, Kernel->GetAverageGpuTimeMs()));
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
