// Copyright 2025-current Getnamo. All Rights Reserved.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "WebGPUCompute.h"
#include "WebGPUDevice.h"
#include "WebGPUSharedTexture.h"
#include "WebGPUSubsystem.h"
#include "RenderingThread.h"
#include "RHI.h"
#include "RHICommandList.h"
#include "RHIResources.h"
#include "RHIUtilities.h"

namespace WebGPUTests
{
	constexpr EAutomationTestFlags Flags = EAutomationTestFlags::EditorContext | EAutomationTestFlags::ClientContext | EAutomationTestFlags::ProductFilter;

	bool IsD3D12RHI()
	{
		return GDynamicRHI && GDynamicRHI->GetInterfaceType() == ERHIInterfaceType::D3D12;
	}

	WGPUStringView View(const char* Str)
	{
		WGPUStringView Result = WGPU_STRING_VIEW_INIT;
		Result.data = Str;
		return Result;
	}

	/** Compute pipeline with an auto layout, plus its group 0 layout. */
	struct FKernel
	{
		WGPUComputePipeline Pipeline = nullptr;
		WGPUBindGroupLayout Layout = nullptr;

		bool Create(FWebGPUDevice& Device, const FString& Source, FString& OutError)
		{
			TArray<FWebGPUCompileMessage> Messages;
			WGPUShaderModule Module = Device.CreateShaderModule(Source, TEXT("test"), Messages);
			if (!Module)
			{
				OutError = FWebGPUDevice::FormatCompileMessages(Messages);
				return false;
			}
			Device.PushErrorScopes();
			WGPUComputePipelineDescriptor Desc = {};
			Desc.compute = WGPU_COMPUTE_STATE_INIT;
			Desc.compute.module = Module;
			Desc.compute.entryPoint = View("main");
			Pipeline = wgpuDeviceCreateComputePipeline(Device.GetDevice(), &Desc);
			Layout = wgpuComputePipelineGetBindGroupLayout(Pipeline, 0);
			wgpuShaderModuleRelease(Module);
			return Device.PopErrorScopes(OutError);
		}

		~FKernel()
		{
			if (Layout) wgpuBindGroupLayoutRelease(Layout);
			if (Pipeline) wgpuComputePipelineRelease(Pipeline);
		}
	};

	WGPUBuffer CreateBuffer(FWebGPUDevice& Device, uint64 Size, WGPUBufferUsage Usage)
	{
		WGPUBufferDescriptor Desc = WGPU_BUFFER_DESCRIPTOR_INIT;
		Desc.usage = Usage;
		Desc.size = Size;
		return wgpuDeviceCreateBuffer(Device.GetDevice(), &Desc);
	}

	/** Maps a MapRead buffer, copies it out and unmaps. Optionally reports the map wait and the copy time. */
	bool ReadBuffer(FWebGPUDevice& Device, WGPUBuffer Buffer, uint64 Size, void* OutData, double* OutMapSeconds = nullptr, double* OutCopySeconds = nullptr)
	{
		WGPUMapAsyncStatus MapStatus = WGPUMapAsyncStatus_Error;
		WGPUBufferMapCallbackInfo Callback = {};
		Callback.mode = WGPUCallbackMode_WaitAnyOnly;
		Callback.userdata1 = &MapStatus;
		Callback.callback = [](WGPUMapAsyncStatus Status, WGPUStringView, void* UserData1, void*)
		{
			*static_cast<WGPUMapAsyncStatus*>(UserData1) = Status;
		};
		double Start = FPlatformTime::Seconds();
		if (!Device.Wait(wgpuBufferMapAsync(Buffer, WGPUMapMode_Read, 0, Size, Callback)) || MapStatus != WGPUMapAsyncStatus_Success)
		{
			return false;
		}
		if (OutMapSeconds)
		{
			*OutMapSeconds += FPlatformTime::Seconds() - Start;
		}
		Start = FPlatformTime::Seconds();
		FMemory::Memcpy(OutData, wgpuBufferGetConstMappedRange(Buffer, 0, Size), Size);
		if (OutCopySeconds)
		{
			*OutCopySeconds += FPlatformTime::Seconds() - Start;
		}
		wgpuBufferUnmap(Buffer);
		return true;
	}

	WGPUBindGroup CreateBindGroup(FWebGPUDevice& Device, WGPUBindGroupLayout Layout, TArrayView<const WGPUBindGroupEntry> Entries)
	{
		WGPUBindGroupDescriptor Desc = {};
		Desc.layout = Layout;
		Desc.entryCount = Entries.Num();
		Desc.entries = Entries.GetData();
		return wgpuDeviceCreateBindGroup(Device.GetDevice(), &Desc);
	}

	void Dispatch(FWebGPUDevice& Device, const FKernel& Kernel, WGPUBindGroup BindGroup, uint32 X, uint32 Y, TFunctionRef<void(WGPUCommandEncoder)> AfterPass)
	{
		WGPUCommandEncoder Encoder = wgpuDeviceCreateCommandEncoder(Device.GetDevice(), nullptr);
		WGPUComputePassEncoder Pass = wgpuCommandEncoderBeginComputePass(Encoder, nullptr);
		wgpuComputePassEncoderSetPipeline(Pass, Kernel.Pipeline);
		wgpuComputePassEncoderSetBindGroup(Pass, 0, BindGroup, 0, nullptr);
		wgpuComputePassEncoderDispatchWorkgroups(Pass, X, Y, 1);
		wgpuComputePassEncoderEnd(Pass);
		wgpuComputePassEncoderRelease(Pass);
		AfterPass(Encoder);
		WGPUCommandBuffer Commands = wgpuCommandEncoderFinish(Encoder, nullptr);
		wgpuQueueSubmit(Device.GetQueue(), 1, &Commands);
		wgpuCommandBufferRelease(Commands);
		wgpuCommandEncoderRelease(Encoder);
	}

	FTextureRHIRef CreateSharedTexture(FRHICommandListImmediate& RHICmdList, const TCHAR* Name, FIntPoint Size, const FLinearColor& ClearColor)
	{
		const FRHITextureCreateDesc Desc = FRHITextureCreateDesc::Create2D(Name, Size, PF_FloatRGBA)
			.SetFlags(ETextureCreateFlags::Shared | ETextureCreateFlags::UAV | ETextureCreateFlags::ShaderResource | ETextureCreateFlags::RenderTargetable)
			.SetClearValue(FClearValueBinding(ClearColor))
			.SetInitialState(ERHIAccess::SRVMask);
		return RHICmdList.CreateTexture(Desc);
	}

	void ClearTexture(FRHICommandListImmediate& RHICmdList, FRHITexture* Texture)
	{
		RHICmdList.Transition(FRHITransitionInfo(Texture, ERHIAccess::Unknown, ERHIAccess::RTV));
		FRHIRenderPassInfo PassInfo(Texture, ERenderTargetActions::Clear_Store);
		RHICmdList.BeginRenderPass(PassInfo, TEXT("WebGPUTestClear"));
		RHICmdList.EndRenderPass();
		RHICmdList.Transition(FRHITransitionInfo(Texture, ERHIAccess::RTV, ERHIAccess::SRVMask));
	}

	template <typename FunctionType>
	void RunOnRenderThread(FunctionType&& Function)
	{
		ENQUEUE_RENDER_COMMAND(WebGPUTest)([&Function](FRHICommandListImmediate& RHICmdList)
		{
			Function(RHICmdList);
		});
		FlushRenderingCommands();
	}

	bool CanShareTextures(FAutomationTestBase& Test, const TSharedPtr<FWebGPUDevice>& Device)
	{
		if (!IsD3D12RHI())
		{
			Test.AddInfo(TEXT("Skipped: needs the D3D12 RHI (run with -RenderOffscreen, not -nullrhi)"));
			return false;
		}
		return Test.TestTrue(TEXT("Device"), Device.IsValid() && Device->IsValid())
			&& Test.TestTrue(TEXT("Device uses the RHI adapter"), Device->IsUsingRHIAdapter())
			&& Test.TestTrue(TEXT("Shared texture feature"), Device->HasFeature(WGPUFeatureName_SharedTextureMemoryDXGISharedHandle))
			&& Test.TestTrue(TEXT("Shared fence feature"), Device->HasFeature(WGPUFeatureName_SharedFenceDXGISharedHandle));
	}

	const TCHAR* CollatzSource = TEXT(R"(
@group(0) @binding(0) var<storage, read_write> v: array<u32>;

fn collatz(n_base: u32) -> u32 {
	var n = n_base;
	var i = 0u;
	loop {
		if (n <= 1u) { break; }
		if (n % 2u == 0u) { n = n / 2u; }
		else {
			if (n >= 1431655765u) { return 4294967295u; }
			n = 3u * n + 1u;
		}
		i = i + 1u;
	}
	return i;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	v[id.x] = collatz(v[id.x]);
}
)");

	uint32 CollatzReference(uint32 N)
	{
		uint32 Steps = 0;
		while (N > 1)
		{
			if (N % 2 == 0)
			{
				N /= 2;
			}
			else
			{
				if (N >= 1431655765u)
				{
					return 4294967295u;
				}
				N = 3 * N + 1;
			}
			Steps++;
		}
		return Steps;
	}
}

using namespace WebGPUTests;

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUDeviceTest, "WebGPU.Device.Init", WebGPUTests::Flags)
bool FWebGPUDeviceTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUDevice> Device = UWebGPUSubsystem::GetSharedDevice();
	if (!TestTrue(TEXT("Device created"), Device.IsValid() && Device->IsValid()))
	{
		return false;
	}
	AddInfo(FString::Printf(TEXT("Adapter: %s (RHI adapter match: %d)"), *Device->GetAdapterDescription(), Device->IsUsingRHIAdapter()));
	TestTrue(TEXT("Thread-safe device"), Device->HasFeature(WGPUFeatureName_ImplicitDeviceSynchronization));
	if (IsD3D12RHI())
	{
		TestTrue(TEXT("Matched the RHI adapter"), Device->IsUsingRHIAdapter());
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUCollatzTest, "WebGPU.Compute.Collatz", WebGPUTests::Flags)
bool FWebGPUCollatzTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUDevice> Device = UWebGPUSubsystem::GetSharedDevice();
	if (!TestTrue(TEXT("Device"), Device.IsValid()))
	{
		return false;
	}

	TArray<uint32> Input;
	for (uint32 Value = 1; Value <= 4096; Value++)
	{
		Input.Add(Value);
	}

	TArray<uint32> Output;
	FString Error;
	if (!TestTrue(FString::Printf(TEXT("RunBufferKernel: %s"), *Error), Device->RunBufferKernel(CollatzSource, Input, Output, Error)))
	{
		return false;
	}
	TestEqual(TEXT("Output size"), Output.Num(), Input.Num());

	int32 Mismatches = 0;
	for (int32 Index = 0; Index < FMath::Min(Input.Num(), Output.Num()); Index++)
	{
		Mismatches += Output[Index] != CollatzReference(Input[Index]) ? 1 : 0;
	}
	TestEqual(TEXT("Mismatches vs CPU reference"), Mismatches, 0);
	TestEqual(TEXT("collatz(3)"), Output[2], 7u);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUCompileErrorTest, "WebGPU.Compute.CompileError", WebGPUTests::Flags)
bool FWebGPUCompileErrorTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUDevice> Device = UWebGPUSubsystem::GetSharedDevice();
	if (!TestTrue(TEXT("Device"), Device.IsValid()))
	{
		return false;
	}

	// Line 4 uses an undeclared identifier
	const FString Broken = TEXT("@group(0) @binding(0) var<storage, read_write> v: array<u32>;\n"
		"@compute @workgroup_size(1)\n"
		"fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
		"\tv[id.x] = not_declared + 1u;\n"
		"}\n");

	TArray<FWebGPUCompileMessage> Messages;
	WGPUShaderModule Module = Device->CreateShaderModule(Broken, TEXT("broken.wgsl"), Messages);
	TestNull(TEXT("Broken shader yields no module"), Module);

	const FWebGPUCompileMessage* Error = Messages.FindByPredicate([](const FWebGPUCompileMessage& Message) { return Message.bIsError; });
	if (TestNotNull(TEXT("Error message"), Error))
	{
		AddInfo(FString::Printf(TEXT("Compile error: %s"), *FWebGPUDevice::FormatCompileMessages(Messages)));
		TestEqual(TEXT("Error line"), Error->Line, (uint64)4);
		TestTrue(TEXT("Error column"), Error->Column > 0);
	}

	// The device keeps working after a failed compile
	TArray<uint32> Output;
	FString RunError;
	TestTrue(TEXT("Device still usable"), Device->RunBufferKernel(CollatzSource, { 6, 7 }, Output, RunError) && Output.Num() == 2 && Output[0] == 8);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUSharedReadTest, "WebGPU.Interop.UEToWebGPU", WebGPUTests::Flags)
bool FWebGPUSharedReadTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUDevice> Device = UWebGPUSubsystem::GetSharedDevice();
	if (!CanShareTextures(*this, Device))
	{
		return true;
	}

	FKernel Kernel;
	FString Error;
	const bool bKernel = Kernel.Create(*Device, TEXT(R"(
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outv: array<vec4<f32>, 3>;
@compute @workgroup_size(1)
fn main() {
	outv[0] = textureLoad(src, vec2<u32>(0u, 0u), 0);
	outv[1] = textureLoad(src, vec2<u32>(128u, 77u), 0);
	outv[2] = textureLoad(src, vec2<u32>(255u, 255u), 0);
}
)"), Error);
	if (!TestTrue(FString::Printf(TEXT("Kernel: %s"), *Error), bKernel))
	{
		return false;
	}

	const FLinearColor ClearColor(0.25f, 0.5f, 0.75f, 1.0f);
	const uint64 OutSize = 3 * 4 * sizeof(float);
	float Texels[12] = {};
	bool bShared = false;
	bool bRead = false;

	RunOnRenderThread([&](FRHICommandListImmediate& RHICmdList)
	{
		FTextureRHIRef Texture = CreateSharedTexture(RHICmdList, TEXT("WebGPUSharedRead"), FIntPoint(256, 256), ClearColor);
		ClearTexture(RHICmdList, Texture);

		TSharedPtr<FWebGPUSharedTexture> Shared = FWebGPUSharedTexture::Create(Device.ToSharedRef(), Texture, Error);
		bShared = Shared.IsValid();
		if (!Shared)
		{
			return;
		}

		WGPUBuffer OutBuffer = CreateBuffer(*Device, OutSize, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
		WGPUBuffer Staging = CreateBuffer(*Device, OutSize, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
		WGPUTextureView TextureView = wgpuTextureCreateView(Shared->GetTexture(), nullptr);

		WGPUBindGroupEntry Entries[2] = { WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT };
		Entries[0].binding = 0;
		Entries[0].textureView = TextureView;
		Entries[1].binding = 1;
		Entries[1].buffer = OutBuffer;
		WGPUBindGroup BindGroup = CreateBindGroup(*Device, Kernel.Layout, Entries);

		Device->PushErrorScopes();
		Shared->BeginAccess(RHICmdList);
		Dispatch(*Device, Kernel, BindGroup, 1, 1, [&](WGPUCommandEncoder Encoder)
		{
			wgpuCommandEncoderCopyBufferToBuffer(Encoder, OutBuffer, 0, Staging, 0, OutSize);
		});
		Shared->EndAccess(RHICmdList);
		FString ScopeError;
		if (Device->PopErrorScopes(ScopeError))
		{
			bRead = ReadBuffer(*Device, Staging, OutSize, Texels);
		}
		else
		{
			Error = ScopeError;
		}

		wgpuBindGroupRelease(BindGroup);
		wgpuTextureViewRelease(TextureView);
		wgpuBufferRelease(Staging);
		wgpuBufferRelease(OutBuffer);
		Shared.Reset();
		RHICmdList.SubmitAndBlockUntilGPUIdle();
	});

	if (!TestTrue(FString::Printf(TEXT("Shared texture: %s"), *Error), bShared) || !TestTrue(FString::Printf(TEXT("Read back: %s"), *Error), bRead))
	{
		return false;
	}

	for (int32 Sample = 0; Sample < 3; Sample++)
	{
		const float* T = &Texels[Sample * 4];
		const bool bMatch = FMath::IsNearlyEqual(T[0], ClearColor.R, 1e-3f) && FMath::IsNearlyEqual(T[1], ClearColor.G, 1e-3f)
			&& FMath::IsNearlyEqual(T[2], ClearColor.B, 1e-3f) && FMath::IsNearlyEqual(T[3], ClearColor.A, 1e-3f);
		const FString Line = FString::Printf(TEXT("WebGPU read texel %d = (%.3f %.3f %.3f %.3f), UE cleared to (%.3f %.3f %.3f %.3f)"),
			Sample, T[0], T[1], T[2], T[3], ClearColor.R, ClearColor.G, ClearColor.B, ClearColor.A);
		TestTrue(Line, bMatch);
		AddInfo(Line);
	}
	return true;
}

static const TCHAR* GradientSource = TEXT(R"(
struct Params { time: f32, width: u32, height: u32, pad: u32 };
@group(0) @binding(0) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> params: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	if (id.x >= params.width || id.y >= params.height) { return; }
	let uv = vec2<f32>(f32(id.x) / f32(params.width - 1u), f32(id.y) / f32(params.height - 1u));
	textureStore(dst, vec2<i32>(id.xy), vec4<f32>(uv.x, uv.y, fract(params.time), 1.0));
}
)");

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUSharedWriteTest, "WebGPU.Interop.WebGPUToUE", WebGPUTests::Flags)
bool FWebGPUSharedWriteTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUDevice> Device = UWebGPUSubsystem::GetSharedDevice();
	if (!CanShareTextures(*this, Device))
	{
		return true;
	}

	FKernel Kernel;
	FString Error;
	if (!TestTrue(FString::Printf(TEXT("Kernel: %s"), *Error), Kernel.Create(*Device, GradientSource, Error)))
	{
		return false;
	}

	const FIntPoint Size(256, 256);
	TArray<FFloat16Color> Pixels;
	bool bShared = false;
	bool bDispatched = false;

	RunOnRenderThread([&](FRHICommandListImmediate& RHICmdList)
	{
		FTextureRHIRef Texture = CreateSharedTexture(RHICmdList, TEXT("WebGPUSharedWrite"), Size, FLinearColor::Black);
		ClearTexture(RHICmdList, Texture);

		TSharedPtr<FWebGPUSharedTexture> Shared = FWebGPUSharedTexture::Create(Device.ToSharedRef(), Texture, Error);
		bShared = Shared.IsValid();
		if (!Shared)
		{
			return;
		}

		const uint32 ParamsData[4] = { 0, (uint32)Size.X, (uint32)Size.Y, 0 }; // time = 0.0f
		WGPUBuffer Params = CreateBuffer(*Device, sizeof(ParamsData), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
		wgpuQueueWriteBuffer(Device->GetQueue(), Params, 0, ParamsData, sizeof(ParamsData));
		WGPUTextureView TextureView = wgpuTextureCreateView(Shared->GetTexture(), nullptr);

		WGPUBindGroupEntry Entries[2] = { WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT };
		Entries[0].binding = 0;
		Entries[0].textureView = TextureView;
		Entries[1].binding = 1;
		Entries[1].buffer = Params;
		WGPUBindGroup BindGroup = CreateBindGroup(*Device, Kernel.Layout, Entries);

		Device->PushErrorScopes();
		Shared->BeginAccess(RHICmdList);
		Dispatch(*Device, Kernel, BindGroup, (Size.X + 7) / 8, (Size.Y + 7) / 8, [](WGPUCommandEncoder) {});
		Shared->EndAccess(RHICmdList);
		bDispatched = Device->PopErrorScopes(Error);

		// UE reads after waiting (on the GPU) for WebGPU's write
		RHICmdList.ReadSurfaceFloatData(Texture, FIntRect(0, 0, Size.X, Size.Y), Pixels, CubeFace_PosX, 0, 0);

		wgpuBindGroupRelease(BindGroup);
		wgpuTextureViewRelease(TextureView);
		wgpuBufferRelease(Params);
		Shared.Reset();
		RHICmdList.SubmitAndBlockUntilGPUIdle();
	});

	if (!TestTrue(FString::Printf(TEXT("Shared texture: %s"), *Error), bShared) || !TestTrue(FString::Printf(TEXT("Dispatch: %s"), *Error), bDispatched)
		|| !TestEqual(TEXT("Pixel count"), Pixels.Num(), Size.X * Size.Y))
	{
		return false;
	}

	const FIntPoint Samples[] = { {0, 0}, {255, 0}, {0, 255}, {100, 200}, {255, 255} };
	for (const FIntPoint& Sample : Samples)
	{
		const FLinearColor Pixel = Pixels[Sample.Y * Size.X + Sample.X].GetFloats();
		const float ExpectedR = Sample.X / 255.0f;
		const float ExpectedG = Sample.Y / 255.0f;
		const bool bMatch = FMath::IsNearlyEqual(Pixel.R, ExpectedR, 2e-3f) && FMath::IsNearlyEqual(Pixel.G, ExpectedG, 2e-3f)
			&& FMath::IsNearlyEqual(Pixel.B, 0.0f, 1e-3f) && FMath::IsNearlyEqual(Pixel.A, 1.0f, 1e-3f);
		const FString Line = FString::Printf(TEXT("UE read pixel (%d,%d) = (%.3f %.3f %.3f %.3f), WebGPU wrote (%.3f %.3f 0 1)"),
			Sample.X, Sample.Y, Pixel.R, Pixel.G, Pixel.B, Pixel.A, ExpectedR, ExpectedG);
		TestTrue(Line, bMatch);
		AddInfo(Line);
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUSharedBenchmark, "WebGPU.Interop.Benchmark1080p", WebGPUTests::Flags)
bool FWebGPUSharedBenchmark::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUDevice> Device = UWebGPUSubsystem::GetSharedDevice();
	if (!CanShareTextures(*this, Device))
	{
		return true;
	}

	FKernel Kernel;
	FString Error;
	if (!TestTrue(FString::Printf(TEXT("Kernel: %s"), *Error), Kernel.Create(*Device, GradientSource, Error)))
	{
		return false;
	}

	const FIntPoint Size(1920, 1080);
	const int32 ZeroCopyFrames = 240;
	const int32 StagedFrames = 60;
	double ZeroCopyLatencyMs = 0, ZeroCopyFrameMs = 0, StagedFrameMs = 0;
	double StagedMapSeconds = 0, StagedCopySeconds = 0, StagedUploadSeconds = 0;
	bool bOk = false;

	RunOnRenderThread([&](FRHICommandListImmediate& RHICmdList)
	{
		FTextureRHIRef Texture = CreateSharedTexture(RHICmdList, TEXT("WebGPUBench"), Size, FLinearColor::Black);
		FTextureRHIRef Consumer = CreateSharedTexture(RHICmdList, TEXT("WebGPUBenchConsumer"), Size, FLinearColor::Black);
		ClearTexture(RHICmdList, Texture);
		TSharedPtr<FWebGPUSharedTexture> Shared = FWebGPUSharedTexture::Create(Device.ToSharedRef(), Texture, Error);
		if (!Shared)
		{
			return;
		}

		WGPUBuffer Params = CreateBuffer(*Device, 16, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
		WGPUTextureView SharedView = wgpuTextureCreateView(Shared->GetTexture(), nullptr);
		WGPUBindGroupEntry Entries[2] = { WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT };
		Entries[0].binding = 0;
		Entries[0].textureView = SharedView;
		Entries[1].binding = 1;
		Entries[1].buffer = Params;
		WGPUBindGroup SharedBindGroup = CreateBindGroup(*Device, Kernel.Layout, Entries);

		auto WriteParams = [&](int32 Frame)
		{
			const float Time = Frame / 60.0f;
			uint32 Data[4] = { 0, (uint32)Size.X, (uint32)Size.Y, 0 };
			FMemory::Memcpy(&Data[0], &Time, sizeof(float));
			wgpuQueueWriteBuffer(Device->GetQueue(), Params, 0, Data, sizeof(Data));
		};

		// UE consumes the result every frame (copy into another texture), as a material/post pass would
		auto ConsumeInUE = [&]()
		{
			RHICmdList.Transition({ FRHITransitionInfo(Texture, ERHIAccess::Unknown, ERHIAccess::CopySrc), FRHITransitionInfo(Consumer, ERHIAccess::Unknown, ERHIAccess::CopyDest) });
			RHICmdList.CopyTexture(Texture, Consumer, FRHICopyTextureInfo());
			RHICmdList.Transition({ FRHITransitionInfo(Texture, ERHIAccess::CopySrc, ERHIAccess::SRVMask), FRHITransitionInfo(Consumer, ERHIAccess::CopyDest, ERHIAccess::SRVMask) });
		};

		auto ZeroCopyFrame = [&](int32 Frame)
		{
			WriteParams(Frame);
			Shared->BeginAccess(RHICmdList, /*bSubmitUEWork*/ false);
			Dispatch(*Device, Kernel, SharedBindGroup, (Size.X + 7) / 8, (Size.Y + 7) / 8, [](WGPUCommandEncoder) {});
			Shared->EndAccess(RHICmdList);
			ConsumeInUE();
			RHICmdList.ImmediateFlush(EImmediateFlushType::DispatchToRHIThread);
		};

		// Warm up, then single-frame latency: dispatch + UE consume, until the GPU is idle
		Device->PushErrorScopes();
		for (int32 Frame = 0; Frame < 10; Frame++)
		{
			ZeroCopyFrame(Frame);
		}
		RHICmdList.SubmitAndBlockUntilGPUIdle();
		Device->WaitForQueueIdle();

		double Start = FPlatformTime::Seconds();
		ZeroCopyFrame(0);
		RHICmdList.SubmitAndBlockUntilGPUIdle();
		ZeroCopyLatencyMs = (FPlatformTime::Seconds() - Start) * 1000.0;

		// Throughput: many frames back to back, one GPU sync at the end
		Start = FPlatformTime::Seconds();
		for (int32 Frame = 0; Frame < ZeroCopyFrames; Frame++)
		{
			ZeroCopyFrame(Frame);
		}
		RHICmdList.SubmitAndBlockUntilGPUIdle();
		Device->WaitForQueueIdle();
		ZeroCopyFrameMs = (FPlatformTime::Seconds() - Start) * 1000.0 / ZeroCopyFrames;

		// CPU-staged comparison: WebGPU-owned texture -> map -> CPU -> UE texture upload -> UE consume
		WGPUTextureDescriptor OwnDesc = WGPU_TEXTURE_DESCRIPTOR_INIT;
		OwnDesc.usage = WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopySrc;
		OwnDesc.size = { (uint32)Size.X, (uint32)Size.Y, 1 };
		OwnDesc.format = WGPUTextureFormat_RGBA16Float;
		WGPUTexture OwnTexture = wgpuDeviceCreateTexture(Device->GetDevice(), &OwnDesc);
		WGPUTextureView OwnView = wgpuTextureCreateView(OwnTexture, nullptr);
		Entries[0].textureView = OwnView;
		WGPUBindGroup OwnBindGroup = CreateBindGroup(*Device, Kernel.Layout, Entries);

		const uint32 BytesPerRow = Size.X * 8; // 15360, already a multiple of 256
		const uint64 ReadbackSize = (uint64)BytesPerRow * Size.Y;
		WGPUBuffer Readback = CreateBuffer(*Device, ReadbackSize, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
		TArray<uint8> CpuPixels;
		CpuPixels.SetNumUninitialized(ReadbackSize);
		FTextureRHIRef UploadTarget = RHICmdList.CreateTexture(FRHITextureCreateDesc::Create2D(TEXT("WebGPUBenchUpload"), Size, PF_FloatRGBA)
			.SetFlags(ETextureCreateFlags::ShaderResource).SetInitialState(ERHIAccess::SRVMask));
		double StagedUpdateCallSeconds = 0;

		Start = FPlatformTime::Seconds();
		for (int32 Frame = 0; Frame < StagedFrames; Frame++)
		{
			WriteParams(Frame);
			Dispatch(*Device, Kernel, OwnBindGroup, (Size.X + 7) / 8, (Size.Y + 7) / 8, [&](WGPUCommandEncoder Encoder)
			{
				WGPUTexelCopyTextureInfo Src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
				Src.texture = OwnTexture;
				WGPUTexelCopyBufferInfo Dst = {};
				Dst.layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
				Dst.layout.bytesPerRow = BytesPerRow;
				Dst.layout.rowsPerImage = Size.Y;
				Dst.buffer = Readback;
				const WGPUExtent3D Extent = { (uint32)Size.X, (uint32)Size.Y, 1 };
				wgpuCommandEncoderCopyTextureToBuffer(Encoder, &Src, &Dst, &Extent);
			});
			ReadBuffer(*Device, Readback, ReadbackSize, CpuPixels.GetData(), &StagedMapSeconds, &StagedCopySeconds);

			const double UploadStart = FPlatformTime::Seconds();
			RHICmdList.Transition(FRHITransitionInfo(UploadTarget, ERHIAccess::Unknown, ERHIAccess::CopyDest));
			const double UpdateStart = FPlatformTime::Seconds();
			RHICmdList.UpdateTexture2D(UploadTarget, 0, FUpdateTextureRegion2D(0, 0, 0, 0, Size.X, Size.Y), BytesPerRow, CpuPixels.GetData());
			StagedUpdateCallSeconds += FPlatformTime::Seconds() - UpdateStart;
			RHICmdList.Transition({ FRHITransitionInfo(UploadTarget, ERHIAccess::CopyDest, ERHIAccess::CopySrc), FRHITransitionInfo(Consumer, ERHIAccess::Unknown, ERHIAccess::CopyDest) });
			RHICmdList.CopyTexture(UploadTarget, Consumer, FRHICopyTextureInfo());
			RHICmdList.Transition({ FRHITransitionInfo(UploadTarget, ERHIAccess::CopySrc, ERHIAccess::SRVMask), FRHITransitionInfo(Consumer, ERHIAccess::CopyDest, ERHIAccess::SRVMask) });
			RHICmdList.SubmitAndBlockUntilGPUIdle();
			StagedUploadSeconds += FPlatformTime::Seconds() - UploadStart;
		}
		StagedFrameMs = (FPlatformTime::Seconds() - Start) * 1000.0 / StagedFrames;
		UE_LOG(LogWebGPU, Display, TEXT("CPU-staged UpdateTexture2D call: %.2f ms/frame"), StagedUpdateCallSeconds * 1000.0 / StagedFrames);

		FString ScopeError;
		bOk = Device->PopErrorScopes(ScopeError);
		Error = ScopeError;

		wgpuBufferRelease(Readback);
		wgpuBindGroupRelease(OwnBindGroup);
		wgpuTextureViewRelease(OwnView);
		wgpuTextureRelease(OwnTexture);
		wgpuBindGroupRelease(SharedBindGroup);
		wgpuTextureViewRelease(SharedView);
		wgpuBufferRelease(Params);
		Shared.Reset();
		RHICmdList.SubmitAndBlockUntilGPUIdle();
	});

	if (!TestTrue(FString::Printf(TEXT("Benchmark ran: %s"), *Error), bOk))
	{
		return false;
	}

	const FString Summary = FString::Printf(TEXT("1080p RGBA16F WebGPU compute -> UE: zero-copy latency %.2f ms, zero-copy %.3f ms/frame (%d frames), CPU-staged %.2f ms/frame (%d frames: dispatch+map wait %.2f, memcpy %.2f, UE upload+consume+sync %.2f)"),
		ZeroCopyLatencyMs, ZeroCopyFrameMs, ZeroCopyFrames, StagedFrameMs, StagedFrames,
		StagedMapSeconds * 1000.0 / StagedFrames, StagedCopySeconds * 1000.0 / StagedFrames, StagedUploadSeconds * 1000.0 / StagedFrames);
	AddInfo(Summary);
	UE_LOG(LogWebGPU, Display, TEXT("%s"), *Summary);
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
