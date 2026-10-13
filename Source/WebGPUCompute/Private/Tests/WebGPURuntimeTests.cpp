// Copyright 2025-current Getnamo. All Rights Reserved.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "WebGPUBlueprintLibrary.h"
#include "WebGPUBuffer.h"
#include "WebGPUCompute.h"
#include "WebGPUContext.h"
#include "WebGPUKernel.h"
#include "WebGPUSubsystem.h"
#include "HAL/PlatformProcess.h"
#include "Tests/AutomationCommon.h"

namespace WebGPURuntimeTests
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

	/** Result slot shared with worker-thread callbacks, so a timed-out test never leaves a dangling capture. */
	struct FReadResult
	{
		std::atomic<bool> bDone = false;
		std::atomic<bool> bSuccess = false;
		TArray<uint8> Data;
		double CompletedAt = 0.0;
	};

	TSharedRef<FReadResult> Read(FWebGPUContext& Context, const TSharedRef<FWebGPUBuffer>& Buffer, uint64 Offset, uint64 Size)
	{
		TSharedRef<FReadResult> Result = MakeShared<FReadResult>();
		Context.ReadBuffer(Buffer, Offset, Size, [Result](bool bSuccess, TArray<uint8>&& Data)
		{
			Result->Data = MoveTemp(Data);
			Result->bSuccess = bSuccess;
			Result->CompletedAt = FPlatformTime::Seconds();
			Result->bDone = true;
		}, EWebGPUCallbackThread::WorkerThread);
		return Result;
	}

	struct FCompileResult
	{
		std::atomic<int32> Count = 0;
		std::atomic<bool> bLastSuccess = false;
		FCriticalSection Lock;
		TArray<FWebGPUCompileMessage> LastMessages;
	};

	FWebGPUCompileCallback CompileListener(const TSharedRef<FCompileResult>& Result)
	{
		return [Result](bool bSuccess, const TArray<FWebGPUCompileMessage>& Messages)
		{
			{
				FScopeLock Lock(&Result->Lock);
				Result->LastMessages = Messages;
			}
			Result->bLastSuccess = bSuccess;
			Result->Count++;
		};
	}

	// y = a * x + y over n elements
	const TCHAR* SaxpySource = TEXT(R"(
struct Params { a: f32, n: u32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	if (id.x >= params.n) { return; }
	y[id.x] = params.a * x[id.x] + y[id.x];
}
)");

	struct FSaxpy
	{
		TSharedPtr<FWebGPUBuffer> Params;
		TSharedPtr<FWebGPUBuffer> X;
		TSharedPtr<FWebGPUBuffer> Y;
		TSharedPtr<FWebGPUKernel> Kernel;
		TSharedRef<FCompileResult> Compiled = MakeShared<FCompileResult>();
		uint32 Count = 0;

		void Create(FWebGPUContext& Context, uint32 InCount, const FString& Source = SaxpySource)
		{
			Count = InCount;
			Params = Context.CreateBuffer(16, EWebGPUBufferUsage::Uniform, TEXT("saxpy.params"));
			X = Context.CreateBuffer(Count * sizeof(float), EWebGPUBufferUsage::Storage, TEXT("saxpy.x"));
			Y = Context.CreateBuffer(Count * sizeof(float), EWebGPUBufferUsage::Storage, TEXT("saxpy.y"));
			Kernel = Context.CreateKernel(Source, TEXT("main"), TEXT("saxpy"), CompileListener(Compiled), EWebGPUCallbackThread::WorkerThread);
			Context.SetBuffer(Kernel.ToSharedRef(), 0, 0, Params);
			Context.SetBuffer(Kernel.ToSharedRef(), 0, 1, X);
			Context.SetBuffer(Kernel.ToSharedRef(), 0, 2, Y);
		}

		void Upload(FWebGPUContext& Context, float A)
		{
			uint32 ParamData[4] = { 0, Count, 0, 0 };
			FMemory::Memcpy(&ParamData[0], &A, sizeof(float));
			Context.WriteBuffer(Params.ToSharedRef(), 0, ParamData, sizeof(ParamData));

			TArray<float> XData;
			XData.SetNumUninitialized(Count);
			for (uint32 Index = 0; Index < Count; Index++)
			{
				XData[Index] = (float)(Index % 100);
			}
			Context.WriteBuffer(X.ToSharedRef(), 0, XData.GetData(), Count * sizeof(float));

			TArray<uint8> Zeros;
			Zeros.SetNumZeroed(Count * sizeof(float));
			Context.WriteBuffer(Y.ToSharedRef(), 0, MoveTemp(Zeros));
		}

		void Dispatch(FWebGPUContext& Context)
		{
			Context.Dispatch(Kernel.ToSharedRef(), FMath::DivideAndRoundUp(Count, 64u));
		}

		/** Counts elements where y != Expected * x. */
		int32 Mismatches(const TArray<uint8>& Data, float Expected) const
		{
			const float* Values = reinterpret_cast<const float*>(Data.GetData());
			int32 Bad = 0;
			for (uint32 Index = 0; Index < Count; Index++)
			{
				Bad += FMath::IsNearlyEqual(Values[Index], Expected * (float)(Index % 100), 1e-3f) ? 0 : 1;
			}
			return Bad;
		}
	};
}

using namespace WebGPURuntimeTests;

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUBufferRoundTripTest, "WebGPU.Runtime.BufferRoundTrip", WebGPURuntimeTests::Flags)
bool FWebGPUBufferRoundTripTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}

	const int32 Count = 1 << 18;
	TArray<float> Data;
	Data.SetNumUninitialized(Count);
	for (int32 Index = 0; Index < Count; Index++)
	{
		Data[Index] = Index * 0.5f;
	}

	TSharedRef<FWebGPUBuffer> Buffer = Context->CreateBuffer(Count * sizeof(float), EWebGPUBufferUsage::Storage, TEXT("RoundTrip"));
	Context->WriteBuffer(Buffer, 0, Data.GetData(), Count * sizeof(float));

	TSharedRef<FReadResult> Whole = Read(*Context, Buffer, 0, Count * sizeof(float));
	TSharedRef<FReadResult> Part = Read(*Context, Buffer, 1024, 64);
	if (!TestTrue(TEXT("Reads completed"), WaitUntil([&] { return Whole->bDone && Part->bDone; })))
	{
		return false;
	}

	TestTrue(TEXT("Whole read succeeded"), Whole->bSuccess.load());
	TestEqual(TEXT("Whole read size"), Whole->Data.Num(), Count * (int32)sizeof(float));
	TestTrue(TEXT("Whole read matches"), Whole->Data.Num() == Count * (int32)sizeof(float) && FMemory::Memcmp(Whole->Data.GetData(), Data.GetData(), Whole->Data.Num()) == 0);
	TestTrue(TEXT("Partial read matches"), Part->bSuccess && Part->Data.Num() == 64 && FMemory::Memcmp(Part->Data.GetData(), reinterpret_cast<const uint8*>(Data.GetData()) + 1024, 64) == 0);

	// Misaligned access is rejected up front and reported as a failed read
	AddExpectedMessage(TEXT("must be"), ELogVerbosity::Error, EAutomationExpectedMessageFlags::Contains, 2, false);
	const uint8 Odd[3] = { 1, 2, 3 };
	Context->WriteBuffer(Buffer, 2, Odd, 3);
	TSharedRef<FReadResult> Bad = Read(*Context, Buffer, 2, 8);
	TestTrue(TEXT("Misaligned read fails"), WaitUntil([&] { return Bad->bDone.load(); }) && !Bad->bSuccess);

	const FWebGPUStats Stats = Context->GetStats();
	AddInfo(FString::Printf(TEXT("Stats: %lld submits, %lld readbacks, %lld bytes written, %lld bytes read"),
		Stats.Submits, Stats.Readbacks, Stats.BytesWritten, Stats.BytesRead));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUKernelDispatchTest, "WebGPU.Runtime.KernelDispatch", WebGPURuntimeTests::Flags)
bool FWebGPUKernelDispatchTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}

	FSaxpy Saxpy;
	Saxpy.Create(*Context, 1 << 20);
	Saxpy.Upload(*Context, 2.0f);
	if (!TestTrue(TEXT("Kernel compiled"), WaitUntil([&] { return Saxpy.Kernel->IsReady(); })))
	{
		AddError(FWebGPUDevice::FormatCompileMessages(Saxpy.Kernel->GetMessages()));
		return false;
	}
	// The listener runs just after the kernel turns ready
	TestTrue(TEXT("Listener fired once"), WaitUntil([&] { return Saxpy.Compiled->Count == 1; }) && Saxpy.Compiled->bLastSuccess);

	// Ten dispatches enqueued back to back accumulate y += 2x; they batch into few submits
	const FWebGPUStats Before = Context->GetStats();
	const double EnqueueStart = FPlatformTime::Seconds();
	for (int32 Pass = 0; Pass < 10; Pass++)
	{
		Saxpy.Dispatch(*Context);
	}
	TSharedRef<FReadResult> Result = Read(*Context, Saxpy.Y.ToSharedRef(), 0, Saxpy.Count * sizeof(float));
	const double EnqueueMs = (FPlatformTime::Seconds() - EnqueueStart) * 1000.0;

	if (!TestTrue(TEXT("Read completed"), WaitUntil([&] { return Result->bDone.load(); })) || !TestTrue(TEXT("Read succeeded"), Result->bSuccess.load()))
	{
		return false;
	}
	TestEqual(TEXT("y == 20 * x"), Saxpy.Mismatches(Result->Data, 20.0f), 0);

	const FWebGPUStats After = Context->GetStats();
	TestEqual(TEXT("Dispatches counted"), After.Dispatches - Before.Dispatches, (int64)10);
	TestEqual(TEXT("No failed submits"), After.FailedSubmits - Before.FailedSubmits, (int64)0);
	AddInfo(FString::Printf(TEXT("10 dispatches of 1M-element saxpy + 4 MB readback: %lld submits, enqueue %.3f ms on the calling thread, %.2f ms until the data was back"),
		After.Submits - Before.Submits, EnqueueMs, (Result->CompletedAt - EnqueueStart) * 1000.0));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUCompileAsyncTest, "WebGPU.Runtime.CompileAndRecompile", WebGPURuntimeTests::Flags)
bool FWebGPUCompileAsyncTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}

	const FString Broken = FString(SaxpySource).Replace(TEXT("params.a * x[id.x]"), TEXT("params.a * missing[id.x]"));
	const int32 BrokenLine = 10;

	// A kernel whose first compile fails reports it and skips dispatches
	FSaxpy Saxpy;
	Saxpy.Create(*Context, 4096, Broken);
	if (!TestTrue(TEXT("Listener fired"), WaitUntil([&] { return Saxpy.Compiled->Count >= 1; })))
	{
		return false;
	}
	TestFalse(TEXT("Broken source fails"), Saxpy.Compiled->bLastSuccess.load());
	TestFalse(TEXT("Not ready"), Saxpy.Kernel->IsReady());
	{
		FScopeLock Lock(&Saxpy.Compiled->Lock);
		const FWebGPUCompileMessage* Error = Saxpy.Compiled->LastMessages.FindByPredicate([](const FWebGPUCompileMessage& M) { return M.bIsError; });
		if (TestNotNull(TEXT("Error message"), Error))
		{
			TestEqual(TEXT("Error line"), Error->Line, (uint64)BrokenLine);
			AddInfo(FString::Printf(TEXT("Async compile error: %s"), *FWebGPUDevice::FormatCompileMessages(Saxpy.Compiled->LastMessages)));
		}
	}

	AddExpectedMessage(TEXT("skipped: no compiled pipeline"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
	const int64 SkippedBefore = Context->GetStats().SkippedDispatches;
	Saxpy.Upload(*Context, 1.0f);
	Saxpy.Dispatch(*Context);
	Context->Flush();
	TestEqual(TEXT("Dispatch skipped while not compiled"), Context->GetStats().SkippedDispatches - SkippedBefore, (int64)1);

	// Fixing the source makes it ready
	TSharedRef<std::atomic<bool>> bRecompiled = MakeShared<std::atomic<bool>>(false);
	Context->RecompileKernel(Saxpy.Kernel.ToSharedRef(), SaxpySource, [bRecompiled](bool bSuccess, const TArray<FWebGPUCompileMessage>&) { *bRecompiled = bSuccess; }, EWebGPUCallbackThread::WorkerThread);
	if (!TestTrue(TEXT("Recompiled"), WaitUntil([&] { return Saxpy.Kernel->IsReady(); })))
	{
		return false;
	}
	TestTrue(TEXT("Per-call callback fired"), WaitUntil([&] { return bRecompiled->load(); }));
	TestTrue(TEXT("Listener saw the fix"), WaitUntil([&] { return Saxpy.Compiled->Count >= 2; }) && Saxpy.Compiled->bLastSuccess);

	// Breaking it again keeps the last good pipeline running (the hot reload contract)
	Context->RecompileKernel(Saxpy.Kernel.ToSharedRef(), Broken);
	TestTrue(TEXT("Second failure reported"), WaitUntil([&] { return Saxpy.Compiled->Count >= 3; }) && !Saxpy.Compiled->bLastSuccess);
	TestTrue(TEXT("Still ready"), Saxpy.Kernel->IsReady());

	Saxpy.Upload(*Context, 3.0f);
	Saxpy.Dispatch(*Context);
	TSharedRef<FReadResult> Result = Read(*Context, Saxpy.Y.ToSharedRef(), 0, Saxpy.Count * sizeof(float));
	TestTrue(TEXT("Read completed"), WaitUntil([&] { return Result->bDone.load(); }) && Result->bSuccess);
	TestEqual(TEXT("Last good pipeline computed y == 3x"), Saxpy.Mismatches(Result->Data, 3.0f), 0);

	// A bad entry point fails at pipeline creation, also asynchronously
	TSharedRef<FCompileResult> EntryResult = MakeShared<FCompileResult>();
	TSharedRef<FWebGPUKernel> NoEntry = Context->CreateKernel(SaxpySource, TEXT("not_main"), TEXT("no-entry"), CompileListener(EntryResult), EWebGPUCallbackThread::WorkerThread);
	TestTrue(TEXT("Bad entry point reported"), WaitUntil([&] { return EntryResult->Count >= 1; }) && !EntryResult->bLastSuccess);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUDeviceLossTest, "WebGPU.Runtime.DeviceLossRecovery", WebGPURuntimeTests::Flags)
bool FWebGPUDeviceLossTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}

	FSaxpy Saxpy;
	Saxpy.Create(*Context, 1 << 16);
	Saxpy.Upload(*Context, 2.0f);
	if (!TestTrue(TEXT("Kernel compiled"), WaitUntil([&] { return Saxpy.Kernel->IsReady(); })))
	{
		return false;
	}
	Saxpy.Dispatch(*Context);
	TSharedRef<FReadResult> First = Read(*Context, Saxpy.Y.ToSharedRef(), 0, Saxpy.Count * sizeof(float));
	TestTrue(TEXT("First run"), WaitUntil([&] { return First->bDone.load(); }) && First->bSuccess && Saxpy.Mismatches(First->Data, 2.0f) == 0);

	// Lose the device; the worker recreates it, its buffers (zeroed) and recompiles the kernel
	AddExpectedMessage(TEXT("WebGPU device lost"), ELogVerbosity::Error, EAutomationExpectedMessageFlags::Contains, 0, false);
	AddExpectedMessage(TEXT("WebGPU device lost"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 0, false);
	const uint32 GenerationBefore = Context->GetDeviceGeneration();
	const int32 CompilesBefore = Saxpy.Compiled->Count;
	const double LossStart = FPlatformTime::Seconds();
	Context->ForceDeviceLoss();

	if (!TestTrue(TEXT("Device recreated"), WaitUntil([&] { return Context->GetDeviceGeneration() > GenerationBefore; })))
	{
		return false;
	}
	TestTrue(TEXT("Kernel recompiled after recovery"), WaitUntil([&] { return Saxpy.Kernel->IsReady() && Saxpy.Compiled->Count > CompilesBefore; }));
	const double RecoveryMs = (FPlatformTime::Seconds() - LossStart) * 1000.0;

	// Contents were lost: y reads back zero until re-uploaded
	TSharedRef<FReadResult> Zeroed = Read(*Context, Saxpy.Y.ToSharedRef(), 0, Saxpy.Count * sizeof(float));
	TestTrue(TEXT("Buffers recreated zeroed"), WaitUntil([&] { return Zeroed->bDone.load(); }) && Zeroed->bSuccess && Saxpy.Mismatches(Zeroed->Data, 0.0f) == 0);

	Saxpy.Upload(*Context, 5.0f);
	Saxpy.Dispatch(*Context);
	TSharedRef<FReadResult> After = Read(*Context, Saxpy.Y.ToSharedRef(), 0, Saxpy.Count * sizeof(float));
	TestTrue(TEXT("Read after recovery"), WaitUntil([&] { return After->bDone.load(); }) && After->bSuccess);
	TestEqual(TEXT("y == 5x after recovery"), Saxpy.Mismatches(After->Data, 5.0f), 0);

	AddInfo(FString::Printf(TEXT("Device generation %u -> %u, recovered with the kernel recompiled in %.1f ms"), GenerationBefore, Context->GetDeviceGeneration(), RecoveryMs));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUTimestampTest, "WebGPU.Runtime.Timestamps", WebGPURuntimeTests::Flags)
bool FWebGPUTimestampTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}
	TSharedPtr<FWebGPUDevice> Device = Context->GetDevice();
	if (!Device || !Device->HasFeature(WGPUFeatureName_TimestampQuery))
	{
		AddInfo(TEXT("Skipped: no TimestampQuery feature"));
		return true;
	}

	// Enough arithmetic to take a measurable amount of time
	const FString Heavy = TEXT(R"(
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	var v = data[id.x];
	for (var i = 0u; i < 2048u; i = i + 1u) { v = v * 1.0001 + 0.0001; }
	data[id.x] = v;
}
)");
	const uint32 Count = 1 << 20;
	TSharedRef<FWebGPUBuffer> Data = Context->CreateBuffer(Count * sizeof(float), EWebGPUBufferUsage::Storage, TEXT("timestamps.data"));
	TSharedRef<FWebGPUKernel> Kernel = Context->CreateKernel(Heavy, TEXT("main"), TEXT("heavy"));
	Context->SetBuffer(Kernel, 0, 0, Data);
	Context->SetProfiling(Kernel, true);
	if (!TestTrue(TEXT("Kernel compiled"), WaitUntil([&] { return Kernel->IsReady(); })))
	{
		return false;
	}

	for (int32 Pass = 0; Pass < 5; Pass++)
	{
		Context->Dispatch(Kernel, Count / 64);
	}
	TestTrue(TEXT("Flush"), Context->Flush());
	TestTrue(TEXT("GPU time measured"), WaitUntil([&] { return Kernel->GetLastGpuTimeMs() > 0.0; }));
	AddInfo(FString::Printf(TEXT("1M invocations x 2048 FMAs: last %.3f ms, average %.3f ms on the GPU"), Kernel->GetLastGpuTimeMs(), Kernel->GetAverageGpuTimeMs()));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPULatencyTest, "WebGPU.Runtime.RoundTripLatency", WebGPURuntimeTests::Flags)
bool FWebGPULatencyTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}

	// A per-frame pattern: upload params, dispatch over 64k agents, read back 64k results
	FSaxpy Saxpy;
	Saxpy.Create(*Context, 1 << 16);
	Saxpy.Upload(*Context, 1.0f);
	if (!TestTrue(TEXT("Kernel compiled"), WaitUntil([&] { return Saxpy.Kernel->IsReady(); })))
	{
		return false;
	}

	const int32 Frames = 200;
	TArray<double> Latencies;
	double EnqueueTotal = 0.0;
	for (int32 Frame = 0; Frame < Frames; Frame++)
	{
		const double Start = FPlatformTime::Seconds();
		uint32 ParamData[4] = { 0, Saxpy.Count, 0, 0 };
		const float A = 1.0f;
		FMemory::Memcpy(&ParamData[0], &A, sizeof(float));
		Context->WriteBuffer(Saxpy.Params.ToSharedRef(), 0, ParamData, sizeof(ParamData));
		Saxpy.Dispatch(*Context);
		TSharedRef<FReadResult> Result = Read(*Context, Saxpy.Y.ToSharedRef(), 0, Saxpy.Count * sizeof(float));
		EnqueueTotal += FPlatformTime::Seconds() - Start;
		if (!WaitUntil([&] { return Result->bDone.load(); }) || !Result->bSuccess)
		{
			AddError(TEXT("Read timed out"));
			return false;
		}
		Latencies.Add((Result->CompletedAt - Start) * 1000.0);
	}

	Latencies.Sort();
	AddInfo(FString::Printf(TEXT("Write + dispatch (64k) + 256 KB readback round trip over %d frames: p50 %.3f ms, p90 %.3f ms, max %.3f ms; enqueue %.1f us/frame"),
		Frames, Latencies[Frames / 2], Latencies[Frames * 9 / 10], Latencies.Last(), EnqueueTotal * 1.0e6 / Frames));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUBlueprintApiTest, "WebGPU.Runtime.BlueprintAPI", WebGPURuntimeTests::Flags)
bool FWebGPUBlueprintApiTest::RunTest(const FString& Parameters)
{
	// The UObject layer with game-thread callbacks, as Blueprint and unreal.js use it
	const int32 Count = 4096;
	UWebGPUBuffer* X = UWebGPUBlueprintLibrary::CreateWebGPUBuffer(Count * sizeof(float), (int32)EWebGPUBufferUsage::Storage, TEXT("bp.x"));
	UWebGPUBuffer* Y = UWebGPUBlueprintLibrary::CreateWebGPUBuffer(Count * sizeof(float), (int32)EWebGPUBufferUsage::Storage, TEXT("bp.y"));
	UWebGPUKernel* Kernel = UWebGPUBlueprintLibrary::CreateWebGPUKernel(TEXT(R"(
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	if (id.x < arrayLength(&x)) { y[id.x] = x[id.x] * 2.0; }
}
)"), TEXT("main"), TEXT("bp.double"));
	if (!TestNotNull(TEXT("Buffer x"), X) || !TestNotNull(TEXT("Buffer y"), Y) || !TestNotNull(TEXT("Kernel"), Kernel))
	{
		return false;
	}
	X->AddToRoot();
	Y->AddToRoot();
	Kernel->AddToRoot();

	TArray<float> Input;
	for (int32 Index = 0; Index < Count; Index++)
	{
		Input.Add((float)Index);
	}
	X->WriteFloats(Input);
	Kernel->SetBuffer(0, X);
	Kernel->SetBuffer(1, Y);

	struct FState
	{
		bool bDispatched = false;
		bool bDone = false;
		bool bSuccess = false;
		TArray<uint8> Data;
		double Deadline = 0.0;
	};
	TSharedRef<FState> State = MakeShared<FState>();
	State->Deadline = FPlatformTime::Seconds() + 10.0;

	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([this, X, Y, Kernel, State, Count]() -> bool
	{
		const bool bTimedOut = FPlatformTime::Seconds() > State->Deadline;
		if (!bTimedOut && !State->bDispatched)
		{
			if (!Kernel->IsReady())
			{
				return false;
			}
			Kernel->Dispatch(UWebGPUBlueprintLibrary::WorkgroupsForCount(Count, 64));
			Y->Read(0, -1, [State](bool bSuccess, TArray<uint8>&& Data)
			{
				check(IsInGameThread());
				State->bSuccess = bSuccess;
				State->Data = MoveTemp(Data);
				State->bDone = true;
			});
			State->bDispatched = true;
			return false;
		}
		if (!bTimedOut && !State->bDone)
		{
			return false;
		}

		TestTrue(TEXT("Completed in time"), State->bDone && State->bSuccess);
		const TArray<float> Output = UWebGPUBlueprintLibrary::WebGPUBytesToFloats(State->Data);
		int32 Mismatches = Output.Num() == Count ? 0 : Count;
		for (int32 Index = 0; Index < Output.Num(); Index++)
		{
			Mismatches += Output[Index] == Index * 2.0f ? 0 : 1;
		}
		TestEqual(TEXT("y == 2x through the UObject API"), Mismatches, 0);
		TestTrue(TEXT("Compile messages empty"), Kernel->GetCompileMessages().IsEmpty());

		X->RemoveFromRoot();
		Y->RemoveFromRoot();
		Kernel->RemoveFromRoot();
		return true;
	}));
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
