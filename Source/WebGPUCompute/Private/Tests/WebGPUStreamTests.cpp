// Copyright 2025-current Getnamo. All Rights Reserved.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "WebGPUBlueprintLibrary.h"
#include "WebGPUBuffer.h"
#include "WebGPUCompute.h"
#include "WebGPUContext.h"
#include "WebGPUKernel.h"
#include "WebGPUReadbackStream.h"
#include "WebGPUStreams.h"
#include "WebGPUStructMirror.h"
#include "WebGPUSubsystem.h"
#include "WebGPUTestStructs.h"
#include "HAL/Event.h"
#include "HAL/PlatformProcess.h"
#include "Math/RandomStream.h"
#include "Misc/ScopeExit.h"
#include "Tests/AutomationCommon.h"

namespace WebGPUStreamTests
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

	/** Holds the worker inside a command so that everything enqueued meanwhile runs back to back. */
	struct FWorkerGate
	{
		FEvent* Event = FPlatformProcess::GetSynchEventFromPool(true);
		~FWorkerGate() { FPlatformProcess::ReturnSynchEventToPool(Event); }
		void Open() { Event->Trigger(); }
	};

	TSharedRef<FWorkerGate> HoldWorker(FWebGPUContext& Context)
	{
		TSharedRef<FWorkerGate> Gate = MakeShared<FWorkerGate>();
		Context.Enqueue([Gate]() { Gate->Event->Wait(5000); });
		return Gate;
	}

	struct FReadResult
	{
		std::atomic<bool> bDone = false;
		std::atomic<bool> bSuccess = false;
		TArray<uint8> Data;
	};

	bool ReadNow(FWebGPUContext& Context, const TSharedRef<FWebGPUBuffer>& Buffer, uint64 Offset, uint64 Size, TArray<uint8>& OutData)
	{
		TSharedRef<FReadResult> Result = MakeShared<FReadResult>();
		Context.ReadBuffer(Buffer, Offset, Size, [Result](bool bSuccess, TArray<uint8>&& Data)
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

	/** data[i] = value * 100003 + i: every frame's content is unique and checkable. */
	const TCHAR* FillSource = TEXT(R"(
struct Params { value: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	if (id.x < arrayLength(&data)) { data[id.x] = params.value * 100003u + id.x; }
}
)");

	struct FFill
	{
		TSharedPtr<FWebGPUBuffer> Params;
		TSharedPtr<FWebGPUBuffer> Data;
		TSharedPtr<FWebGPUKernel> Kernel;
		uint32 Count = 0;

		bool Create(FWebGPUContext& Context, uint32 InCount)
		{
			Count = InCount;
			Params = Context.CreateBuffer(16, EWebGPUBufferUsage::Uniform, TEXT("fill.params"));
			Data = Context.CreateBuffer(Count * sizeof(uint32), EWebGPUBufferUsage::Storage, TEXT("fill.data"));
			Kernel = Context.CreateKernel(FillSource, TEXT("main"), TEXT("fill"));
			Context.SetBufferByName(Kernel.ToSharedRef(), TEXT("params"), Params);
			Context.SetBufferByName(Kernel.ToSharedRef(), TEXT("data"), Data);
			return WaitUntil([this] { return Kernel->IsReady(); });
		}

		void Run(FWebGPUContext& Context, uint32 Value)
		{
			const uint32 ParamData[4] = { Value, 0, 0, 0 };
			Context.WriteBuffer(Params.ToSharedRef(), 0, ParamData, sizeof(ParamData));
			Context.DispatchForCount(Kernel.ToSharedRef(), Count);
		}

		static bool Matches(const uint8* Bytes, uint64 Size, uint32 Value)
		{
			if (!Bytes)
			{
				return false;
			}
			const uint32* Words = reinterpret_cast<const uint32*>(Bytes);
			for (uint64 Index = 0; Index < Size / 4; Index++)
			{
				if (Words[Index] != Value * 100003u + (uint32)Index)
				{
					return false;
				}
			}
			return true;
		}
	};

	bool SameAgent(const FWebGPUTestAgent& A, const FWebGPUTestAgent& B)
	{
		return A.Position == B.Position && A.Speed == B.Speed && A.Target == B.Target && A.Flags == B.Flags && A.Heading == B.Heading;
	}

	bool SameMixed(const FWebGPUTestMixed& A, const FWebGPUTestMixed& B)
	{
		// Location goes through f32 on the GPU
		return A.Location.Equals(B.Location, 1e-2) && A.bActive == B.bActive && A.Team == B.Team && A.Velocity == B.Velocity
			&& A.Color == B.Color && A.Values[0] == B.Values[0] && A.Values[1] == B.Values[1] && A.Values[2] == B.Values[2]
			&& A.Inner.A == B.Inner.A && A.Inner.B == B.Inner.B && A.Small == B.Small;
	}

	FWebGPUTestAgent MakeAgent(int32 Index, int32 Salt = 0)
	{
		FWebGPUTestAgent Agent;
		Agent.Position = FVector3f(Index, Index * 2.f + Salt, Index * 3.f);
		Agent.Speed = 0.5f + (Index % 7) + Salt;
		Agent.Target = Index + Salt;
		Agent.Flags = (uint32)(Index * 2 + Salt);
		Agent.Heading = FVector2f(1.f, -1.f * Salt);
		return Agent;
	}

	FWebGPUTestMixed MakeMixed(int32 Index, int32 Salt = 0)
	{
		FWebGPUTestMixed Mixed;
		Mixed.Location = FVector(Index * 1.5, -Index * 0.25 + Salt, 1000.0 + Index);
		Mixed.bActive = (Index + Salt) % 2 == 0;
		Mixed.Team = (uint8)((Index + Salt) % 200);
		Mixed.Velocity = FVector3f(1.f, 2.f, 3.f + Salt);
		Mixed.Color = FLinearColor(Index * 0.001f, 0.2f, 0.3f, 1.f);
		Mixed.Values[0] = (float)Index;
		Mixed.Values[1] = Index * 0.5f + Salt;
		Mixed.Values[2] = -1.f;
		Mixed.Inner.A = 7.f + Salt;
		Mixed.Inner.B = FVector2f(Index, 1.f);
		Mixed.Small = (int16)(Index - 500 + Salt);
		return Mixed;
	}

	/** Reads a mirror's GPU buffer back and counts elements that differ from its CPU copy. */
	template <typename T>
	int32 MirrorMismatches(FWebGPUContext& Context, TWebGPUStructMirror<T>& Mirror, TFunctionRef<bool(const T&, const T&)> Same)
	{
		TArray<uint8> Bytes;
		if (!ReadNow(Context, Mirror.GetBuffer(), 0, Mirror.GetBuffer()->GetSize(), Bytes))
		{
			return -1;
		}
		TArray<T> Gpu;
		Gpu.SetNum(Mirror.Num());
		FWebGPUStructLayout::Get(T::StaticStruct())->Unpack(Bytes.GetData(), Mirror.Num(), Gpu.GetData());
		int32 Bad = 0;
		for (int32 Index = 0; Index < Mirror.Num(); Index++)
		{
			Bad += Same(Gpu[Index], Mirror[Index]) ? 0 : 1;
		}
		return Bad;
	}

	/** Spins until Seconds after Start: a stand-in for the rest of a game frame. */
	void PaceUntil(double Start, double Seconds)
	{
		while (FPlatformTime::Seconds() < Start + Seconds)
		{
			FPlatformProcess::SleepNoStats(0.0f);
		}
	}

	double Percentile(TArray<double> Values, double Fraction)
	{
		if (Values.Num() == 0)
		{
			return 0.0;
		}
		Values.Sort();
		return Values[FMath::Clamp((int32)(Values.Num() * Fraction), 0, Values.Num() - 1)];
	}
}

using namespace WebGPUStreamTests;

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUBatchedCommandsTest, "WebGPU.Streams.BatchedCommands", WebGPUStreamTests::Flags)
bool FWebGPUBatchedCommandsTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}

	const uint32 Count = 4096;
	TSharedRef<FWebGPUBuffer> Params = Context->CreateBuffer(16, EWebGPUBufferUsage::Uniform, TEXT("batch.params"));
	TSharedRef<FWebGPUBuffer> Acc = Context->CreateBuffer(Count * sizeof(float), EWebGPUBufferUsage::Storage, TEXT("batch.acc"));
	TSharedRef<FWebGPUKernel> Kernel = Context->CreateKernel(TEXT(R"(
struct Params { a: f32, n: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> acc: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	if (id.x < params.n) { acc[id.x] = acc[id.x] + params.a * f32(id.x); }
}
)"), TEXT("main"), TEXT("accumulate"));
	Context->SetBufferByName(Kernel, TEXT("params"), Params);
	Context->SetBufferByName(Kernel, TEXT("acc"), Acc);
	if (!TestTrue(TEXT("Compiled"), WaitUntil([&] { return Kernel->IsReady(); })))
	{
		return false;
	}

	auto WriteParams = [&](float A)
	{
		uint32 Data[4] = { 0, Count, 0, 0 };
		FMemory::Memcpy(&Data[0], &A, sizeof(float));
		Context->WriteBuffer(Params, 0, Data, sizeof(Data));
	};

	// Writes between dispatches must land between them: acc = (1 + 2 + 4) * i, not 12 * i
	const FWebGPUStats Before = Context->GetStats();
	TSharedRef<FReadResult> Result = MakeShared<FReadResult>();
	{
		TSharedRef<FWorkerGate> Gate = HoldWorker(*Context);
		ON_SCOPE_EXIT { Gate->Open(); };
		TArray<uint8> Zeros;
		Zeros.SetNumZeroed(Count * sizeof(float));
		Context->WriteBuffer(Acc, 0, MoveTemp(Zeros));
		for (const float A : { 1.f, 2.f, 4.f })
		{
			WriteParams(A);
			Context->DispatchForCount(Kernel, Count);
		}
		Context->ReadBuffer(Acc, 0, Count * sizeof(float), [Result](bool bSuccess, TArray<uint8>&& Data)
		{
			Result->Data = MoveTemp(Data);
			Result->bSuccess = bSuccess;
			Result->bDone = true;
		}, EWebGPUCallbackThread::WorkerThread);
	}
	if (!TestTrue(TEXT("Read back"), WaitUntil([&] { return Result->bDone.load(); }) && Result->bSuccess))
	{
		return false;
	}

	const float* Values = reinterpret_cast<const float*>(Result->Data.GetData());
	int32 Bad = 0;
	for (uint32 Index = 0; Index < Count; Index++)
	{
		Bad += Values[Index] == 7.f * Index ? 0 : 1;
	}
	TestEqual(TEXT("Writes ordered between dispatches"), Bad, 0);
	const FWebGPUStats After = Context->GetStats();
	TestEqual(TEXT("Writes, dispatches and the read in one submit"), After.Submits - Before.Submits, (int64)1);
	TestEqual(TEXT("No failed submits"), After.FailedSubmits - Before.FailedSubmits, (int64)0);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUReadbackLatestTest, "WebGPU.Streams.ReadbackLatest", WebGPUStreamTests::Flags)
bool FWebGPUReadbackLatestTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	FFill Fill;
	if (!TestTrue(TEXT("Context"), Context.IsValid()) || !TestTrue(TEXT("Fill kernel"), Fill.Create(*Context, 16384)))
	{
		return false;
	}
	const TSharedRef<FWebGPUBuffer> Data = Fill.Data.ToSharedRef();
	TSharedRef<FWebGPUReadbackStream> Stream = Context->CreateReadbackStream(Data->GetSize(), 3, EWebGPUStreamPolicy::Latest, TEXT("latest"));

	auto CaptureAndWaitReady = [&](uint32 Value, int32 ReadyAfter)
	{
		Fill.Run(*Context, Value);
		const bool bCaptured = Stream->Capture(Data);
		return bCaptured && WaitUntil([&] { return Stream->GetReadyCount() == ReadyAfter; });
	};

	// Three unread frames fill the ring; a fourth capture recycles the oldest instead of dropping
	TestTrue(TEXT("Frame 1"), CaptureAndWaitReady(1, 1));
	TestTrue(TEXT("Frame 2"), CaptureAndWaitReady(2, 2));
	TestTrue(TEXT("Frame 3"), CaptureAndWaitReady(3, 3));
	TestTrue(TEXT("Frame 4 recycles frame 1"), CaptureAndWaitReady(4, 3));
	TestEqual(TEXT("One frame recycled"), Stream->GetStats().Skipped, (int64)1);

	FWebGPUReadbackFrame Frame;
	if (!TestTrue(TEXT("Acquire"), Stream->Acquire(Frame)))
	{
		return false;
	}
	TestEqual(TEXT("Newest frame"), Frame.GetSequence(), (uint64)4);
	TestTrue(TEXT("Frame 4 data"), FFill::Matches(Frame.GetData(), Frame.GetSize(), 4));
	TestEqual(TEXT("Frames 2 and 3 skipped"), Stream->GetStats().Skipped, (int64)3);
	TestEqual(TEXT("Nothing else ready"), Stream->GetReadyCount(), 0);

	// A held frame is never recycled: its memory stays intact while newer frames come and go
	TestTrue(TEXT("Frame 5"), CaptureAndWaitReady(5, 1));
	TestTrue(TEXT("Frame 6"), CaptureAndWaitReady(6, 2));
	TestTrue(TEXT("Frame 7 recycles frame 5, not the held one"), CaptureAndWaitReady(7, 2));
	TestTrue(TEXT("Held frame unchanged"), FFill::Matches(Frame.GetData(), Frame.GetSize(), 4));
	Frame.Release();
	TestTrue(TEXT("Acquire newest"), Stream->Acquire(Frame) && Frame.GetSequence() == 7 && FFill::Matches(Frame.GetData(), Frame.GetSize(), 7));
	Frame.Release();

	// With every slot on the GPU there is nothing to recycle: the capture is dropped
	const int64 DroppedBefore = Stream->GetStats().Dropped;
	{
		TSharedRef<FWorkerGate> Gate = HoldWorker(*Context);
		ON_SCOPE_EXIT { Gate->Open(); };
		Fill.Run(*Context, 8);
		TestTrue(TEXT("Capture 8"), Stream->Capture(Data));
		TestTrue(TEXT("Capture 9"), Stream->Capture(Data));
		TestTrue(TEXT("Capture 10"), Stream->Capture(Data));
		TestFalse(TEXT("Capture 11 dropped"), Stream->Capture(Data));
	}
	TestEqual(TEXT("Drop counted"), Stream->GetStats().Dropped - DroppedBefore, (int64)1);
	TestTrue(TEXT("Captures 8-10 arrive"), Stream->WaitForFrame(5.0) && WaitUntil([&] { return Stream->GetReadyCount() == 3; }));
	TestTrue(TEXT("Acquire after drop"), Stream->Acquire(Frame) && Frame.GetSequence() == 10 && FFill::Matches(Frame.GetData(), Frame.GetSize(), 8));
	Frame.Release();

	// Steady state: every frame dispatches, captures, and reads whatever is newest
	TMap<uint64, uint32> ValueOfSequence;
	uint64 LastSequence = 10;
	int32 Corrupt = 0;
	int32 Read = 0;
	int32 OutOfOrder = 0;
	uint64 PreviousRead = 0;
	for (uint32 Value = 100; Value < 400; Value++)
	{
		const double FrameStart = FPlatformTime::Seconds();
		ON_SCOPE_EXIT { PaceUntil(FrameStart, 0.001); };
		Fill.Run(*Context, Value);
		if (Stream->Capture(Data))
		{
			ValueOfSequence.Add(++LastSequence, Value);
		}
		if (Stream->Acquire(Frame))
		{
			const uint32* Expected = ValueOfSequence.Find(Frame.GetSequence());
			Corrupt += Expected && FFill::Matches(Frame.GetData(), Frame.GetSize(), *Expected) ? 0 : 1;
			OutOfOrder += Frame.GetSequence() > PreviousRead ? 0 : 1;
			PreviousRead = Frame.GetSequence();
			Read++;
			Frame.Release();
		}
	}
	TestEqual(TEXT("Every frame read matches its capture"), Corrupt, 0);
	TestEqual(TEXT("Frames never go backwards"), OutOfOrder, 0);
	const FWebGPUStreamStats Stats = Stream->GetStats();
	AddInfo(FString::Printf(TEXT("Latest, 3 x 64 KB slots, 300 frames at 1 ms: %d read, %lld skipped, %lld dropped, capture-to-CPU %.3f ms"),
		Read, Stats.Skipped, Stats.Dropped, Stats.LatencyMs));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUReadbackQueueTest, "WebGPU.Streams.ReadbackQueue", WebGPUStreamTests::Flags)
bool FWebGPUReadbackQueueTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	FFill Fill;
	if (!TestTrue(TEXT("Context"), Context.IsValid()) || !TestTrue(TEXT("Fill kernel"), Fill.Create(*Context, 4096)))
	{
		return false;
	}
	const TSharedRef<FWebGPUBuffer> Data = Fill.Data.ToSharedRef();
	TSharedRef<FWebGPUReadbackStream> Stream = Context->CreateReadbackStream(Data->GetSize(), 3, EWebGPUStreamPolicy::Queue, TEXT("queue"));

	// Queue never recycles unread frames: with every slot taken a capture is dropped
	{
		TSharedRef<FWorkerGate> Gate = HoldWorker(*Context);
		ON_SCOPE_EXIT { Gate->Open(); };
		for (uint32 Value = 1; Value <= 3; Value++)
		{
			Fill.Run(*Context, Value);
			TestTrue(*FString::Printf(TEXT("Capture %u"), Value), Stream->Capture(Data));
		}
		TestFalse(TEXT("Capture 4 dropped"), Stream->Capture(Data));
	}
	TestEqual(TEXT("Drop counted"), Stream->GetStats().Dropped, (int64)1);

	FWebGPUReadbackFrame Frame;
	for (uint32 Value = 1; Value <= 3; Value++)
	{
		const bool bRead = Stream->WaitForFrame(5.0) && Stream->Acquire(Frame);
		TestTrue(*FString::Printf(TEXT("Frame %u in order"), Value), bRead && Frame.GetSequence() == Value && FFill::Matches(Frame.GetData(), Frame.GetSize(), Value));
		Frame.Release();
	}

	// Steady state: every captured frame is delivered, in order
	TMap<uint64, uint32> ValueOfSequence;
	uint64 LastCaptured = 3;
	uint64 LastRead = 3;
	int32 Bad = 0;
	int32 Captured = 0;
	auto Drain = [&]()
	{
		while (Stream->Acquire(Frame))
		{
			const uint32* Expected = ValueOfSequence.Find(Frame.GetSequence());
			Bad += Frame.GetSequence() == LastRead + 1 && Expected && FFill::Matches(Frame.GetData(), Frame.GetSize(), *Expected) ? 0 : 1;
			LastRead = Frame.GetSequence();
			Frame.Release();
		}
	};
	for (uint32 Value = 10; Value < 210; Value++)
	{
		const double FrameStart = FPlatformTime::Seconds();
		ON_SCOPE_EXIT { PaceUntil(FrameStart, 0.001); };
		Fill.Run(*Context, Value);
		if (Stream->Capture(Data))
		{
			ValueOfSequence.Add(++LastCaptured, Value);
			Captured++;
		}
		Drain();
	}
	TestTrue(TEXT("Remaining frames arrive"), WaitUntil([&] { Drain(); return LastRead == LastCaptured; }));
	TestEqual(TEXT("Every captured frame delivered in order with its data"), Bad, 0);
	const FWebGPUStreamStats Stats = Stream->GetStats();
	TestEqual(TEXT("Delivered == captured"), Stats.Delivered, (int64)(Captured + 3));
	AddInfo(FString::Printf(TEXT("Queue, 3 x 16 KB slots, 200 frames at 1 ms: %d captured and delivered, %lld dropped for lack of a slot"), Captured, Stats.Dropped - 1));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUReadbackLockstepTest, "WebGPU.Streams.Lockstep", WebGPUStreamTests::Flags)
bool FWebGPUReadbackLockstepTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	FFill Fill;
	if (!TestTrue(TEXT("Context"), Context.IsValid()) || !TestTrue(TEXT("Fill kernel"), Fill.Create(*Context, 16384)))
	{
		return false;
	}
	const TSharedRef<FWebGPUBuffer> Data = Fill.Data.ToSharedRef();
	TSharedRef<FWebGPUReadbackStream> Stream = Context->CreateReadbackStream(Data->GetSize(), 2, EWebGPUStreamPolicy::Lockstep, TEXT("lockstep"));

	// Capture returns once the frame is on the CPU, so the same tick reads this tick's result
	TArray<double> StepMs;
	int32 Bad = 0;
	for (uint32 Value = 1; Value <= 200; Value++)
	{
		const double Start = FPlatformTime::Seconds();
		Fill.Run(*Context, Value);
		const bool bCaptured = Stream->Capture(Data);
		FWebGPUReadbackFrame Frame;
		const bool bRead = bCaptured && Stream->Acquire(Frame);
		StepMs.Add((FPlatformTime::Seconds() - Start) * 1000.0);
		Bad += bRead && Frame.GetSequence() == Value && FFill::Matches(Frame.GetData(), Frame.GetSize(), Value) ? 0 : 1;
	}
	TestEqual(TEXT("Every step returns its own result"), Bad, 0);
	AddInfo(FString::Printf(TEXT("Lockstep (write + dispatch 16k + capture 64 KB, blocking): p50 %.3f ms, p90 %.3f ms, max %.3f ms per step"),
		Percentile(StepMs, 0.5), Percentile(StepMs, 0.9), Percentile(StepMs, 1.0)));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUUploadStreamTest, "WebGPU.Streams.UploadStream", WebGPUStreamTests::Flags)
bool FWebGPUUploadStreamTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}

	const uint64 SlotBytes = 64 * 1024;
	TSharedRef<FWebGPUUploadStream> Upload = Context->CreateUploadStream(SlotBytes, 3, TEXT("upload"));
	TSharedRef<FWebGPUBuffer> Target = Context->CreateBuffer(6 * SlotBytes, EWebGPUBufferUsage::Storage, TEXT("upload.target"));

	auto Pattern = [](uint32 Chunk, uint64 Words)
	{
		TArray<uint32> Values;
		Values.SetNumUninitialized((int32)Words);
		for (int32 Index = 0; Index < Values.Num(); Index++)
		{
			Values[Index] = Chunk * 0x01000000u + (uint32)Index;
		}
		return Values;
	};

	// Slots are created mapped on the worker
	const int32 First = Upload->AcquireSlot(5.0);
	if (!TestTrue(TEXT("Slot available"), First != INDEX_NONE))
	{
		return false;
	}
	TArray<uint32> Chunk0 = Pattern(0, SlotBytes / 4);
	FMemory::Memcpy(Upload->GetSlotData(First), Chunk0.GetData(), SlotBytes);
	Upload->Commit(First, Target, 0, SlotBytes);

	// Write() uses a slot when one is free and copies otherwise; either way the data lands in order
	for (uint32 Chunk = 1; Chunk < 4; Chunk++)
	{
		TArray<uint32> Values = Pattern(Chunk, SlotBytes / 4);
		Upload->Write(Target, Chunk * SlotBytes, Values.GetData(), SlotBytes);
	}
	TArray<uint32> Big = Pattern(4, SlotBytes / 2);
	Upload->Write(Target, 4 * SlotBytes, Big.GetData(), SlotBytes * 2);

	// Several regions of one slot
	const int32 Regions = Upload->AcquireSlot(5.0);
	if (!TestTrue(TEXT("Slot for regions"), Regions != INDEX_NONE))
	{
		return false;
	}
	uint32* Words = reinterpret_cast<uint32*>(Upload->GetSlotData(Regions));
	for (uint32 Index = 0; Index < 64; Index++)
	{
		Words[Index] = 0xABC00000u + Index;
	}
	TArray<FWebGPUCopyRegion> Copies = { { 0, 100 * 4, 16 * 4 }, { 16 * 4, 2000 * 4, 48 * 4 } };
	Upload->Commit(Regions, Target, MoveTemp(Copies));

	// A cancelled slot comes back
	const int32 Cancelled = Upload->AcquireSlot(5.0);
	TestTrue(TEXT("Slot to cancel"), Cancelled != INDEX_NONE);
	Upload->Cancel(Cancelled);
	TestTrue(TEXT("Slot free again after cancel"), WaitUntil([&] { const int32 Slot = Upload->AcquireSlot(0.0); if (Slot != INDEX_NONE) { Upload->Cancel(Slot); } return Slot != INDEX_NONE; }));

	TArray<uint8> Bytes;
	if (!TestTrue(TEXT("Read back"), ReadNow(*Context, Target, 0, Target->GetSize(), Bytes)))
	{
		return false;
	}
	const uint32* Gpu = reinterpret_cast<const uint32*>(Bytes.GetData());
	int32 Bad = 0;
	for (uint32 Chunk = 0; Chunk < 4; Chunk++)
	{
		for (uint64 Index = 0; Index < SlotBytes / 4; Index++)
		{
			const uint64 Word = Chunk * SlotBytes / 4 + Index;
			uint32 Expected = Chunk * 0x01000000u + (uint32)Index;
			if (Chunk == 0 && Index >= 100 && Index < 116)
			{
				Expected = 0xABC00000u + (uint32)(Index - 100);
			}
			if (Chunk == 0 && Index >= 2000 && Index < 2048)
			{
				Expected = 0xABC00000u + 16 + (uint32)(Index - 2000);
			}
			Bad += Gpu[Word] == Expected ? 0 : 1;
		}
	}
	for (uint64 Index = 0; Index < SlotBytes / 2; Index++)
	{
		Bad += Gpu[4 * SlotBytes / 4 + Index] == 4 * 0x01000000u + (uint32)Index ? 0 : 1;
	}
	TestEqual(TEXT("Uploaded data in place"), Bad, 0);

	const FWebGPUStreamStats Stats = Upload->GetStats();
	TestTrue(TEXT("The oversized write took the copying path"), Stats.Fallbacks >= 1);
	TestEqual(TEXT("Every write accounted for"), Stats.Committed + Stats.Fallbacks, (int64)6);
	AddInfo(FString::Printf(TEXT("Upload stream: %lld slot commits, %lld fallbacks, slot reusable %.3f ms after commit"), Stats.Committed, Stats.Fallbacks, Stats.LatencyMs));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUStructMirrorTest, "WebGPU.Streams.StructMirror", WebGPUStreamTests::Flags)
bool FWebGPUStructMirrorTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}

	const int32 Count = 10000;
	TWebGPUStructMirror<FWebGPUTestAgent> Agents = TWebGPUStructMirror<FWebGPUTestAgent>::Create(Context.ToSharedRef(), Count, EWebGPUBufferUsage::Storage, TEXT("mirror.agents"));
	if (!TestTrue(TEXT("Mirror"), Agents.IsValid()))
	{
		return false;
	}
	auto Same = [](const FWebGPUTestAgent& A, const FWebGPUTestAgent& B) { return SameAgent(A, B); };

	for (int32 Index = 0; Index < Count; Index++)
	{
		Agents.Edit(Index) = MakeAgent(Index);
	}
	TestEqual(TEXT("First sync uploads everything"), Agents.Sync(), (uint64)Count * 32);
	TestEqual(TEXT("GPU copy matches"), MirrorMismatches<FWebGPUTestAgent>(*Context, Agents, Same), 0);
	TestEqual(TEXT("Nothing dirty: nothing uploaded"), Agents.Sync(), (uint64)0);

	// Scattered single edits: one copy each
	for (const int32 Index : { 100, 5000, 9000 })
	{
		Agents.Edit(Index) = MakeAgent(Index, 1);
	}
	const uint64 ThreeBytes = Agents.Sync();
	const FWebGPUMirrorSyncStats Three = Agents.Get()->GetLastSync();
	TestTrue(TEXT("Three elements, three copies"), ThreeBytes == 96 && Three.Copies == 3 && Three.Scatters == 0);

	// Edits a small gap apart merge into one copy
	for (int32 Index = 1000; Index < 2000; Index += 2)
	{
		Agents.Edit(Index) = MakeAgent(Index, 2);
	}
	const uint64 MergedBytes = Agents.Sync();
	TestTrue(TEXT("Every other element: one merged copy"), MergedBytes == 999 * 32 && Agents.Get()->GetLastSync().Copies == 1);
	TestEqual(TEXT("GPU copy matches after partial syncs"), MirrorMismatches<FWebGPUTestAgent>(*Context, Agents, Same), 0);

	// Too scattered for copies: the dirty elements and their indices are scattered by a kernel
	if (!TestTrue(TEXT("Scatter kernel compiled"), WaitUntil([&] { return Agents.Get()->IsScatterReady(); })))
	{
		return false;
	}
	int32 Scattered = 0;
	for (int32 Index = 3; Index < Count; Index += 37)
	{
		Agents.Edit(Index) = MakeAgent(Index, 3);
		Scattered++;
	}
	const uint64 ScatterBytes = Agents.Sync();
	const FWebGPUMirrorSyncStats Scatter = Agents.Get()->GetLastSync();
	TestTrue(TEXT("Scatter mode"), Scatter.Scatters == 1 && Scatter.Copies == 0 && Scatter.Elements == Scattered);
	TestEqual(TEXT("Scatter uploads the elements and their indices only"), ScatterBytes, (uint64)(8 + Scattered * (32 + 4)));
	TestEqual(TEXT("GPU copy matches after a scatter"), MirrorMismatches<FWebGPUTestAgent>(*Context, Agents, Same), 0);

	// A converted layout (doubles, bools, small ints, nested) through both paths
	TWebGPUStructMirror<FWebGPUTestMixed> Mixed = TWebGPUStructMirror<FWebGPUTestMixed>::Create(Context.ToSharedRef(), 1000, EWebGPUBufferUsage::Storage, TEXT("mirror.mixed"));
	auto SameM = [](const FWebGPUTestMixed& A, const FWebGPUTestMixed& B) { return SameMixed(A, B); };
	for (int32 Index = 0; Index < 1000; Index++)
	{
		Mixed.Edit(Index) = MakeMixed(Index);
	}
	Mixed.Sync();
	Mixed.Get()->MaxCopyRegions = 16;
	for (int32 Index = 0; Index < 1000; Index += 50)
	{
		Mixed.Edit(Index) = MakeMixed(Index, 5);
	}
	TestTrue(TEXT("Mixed scatter kernel compiled"), WaitUntil([&] { return Mixed.Get()->IsScatterReady(); }));
	Mixed.Sync();
	TestTrue(TEXT("Mixed scattered"), Mixed.Get()->GetLastSync().Scatters == 1);
	TestEqual(TEXT("Mixed GPU copy matches"), MirrorMismatches<FWebGPUTestMixed>(*Context, Mixed, SameM), 0);

	// After a device loss the buffer comes back zeroed, and the next Sync uploads everything again
	AddExpectedMessage(TEXT("WebGPU device lost"), ELogVerbosity::Error, EAutomationExpectedMessageFlags::Contains, 0, false);
	AddExpectedMessage(TEXT("WebGPU device lost"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 0, false);
	const uint32 Generation = Context->GetDeviceGeneration();
	Context->ForceDeviceLoss();
	if (!TestTrue(TEXT("Device recovered"), WaitUntil([&] { return Context->GetDeviceGeneration() > Generation; })))
	{
		return false;
	}
	TestEqual(TEXT("Full re-upload after recovery"), Agents.Sync(), (uint64)Count * 32);
	TestEqual(TEXT("GPU copy restored"), MirrorMismatches<FWebGPUTestAgent>(*Context, Agents, Same), 0);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUHeldMemoryDeviceLossTest, "WebGPU.Streams.HeldMemoryAcrossDeviceLoss", WebGPUStreamTests::Flags)
bool FWebGPUHeldMemoryDeviceLossTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	FFill Fill;
	if (!TestTrue(TEXT("Context"), Context.IsValid()) || !TestTrue(TEXT("Fill kernel"), Fill.Create(*Context, 16384)))
	{
		return false;
	}
	const TSharedRef<FWebGPUBuffer> Data = Fill.Data.ToSharedRef();
	TSharedRef<FWebGPUReadbackStream> Stream = Context->CreateReadbackStream(Data->GetSize(), 2, EWebGPUStreamPolicy::Latest, TEXT("held"));
	TSharedRef<FWebGPUUploadStream> Upload = Context->CreateUploadStream(Data->GetSize(), 2, TEXT("held-upload"));

	Fill.Run(*Context, 42);
	FWebGPUReadbackFrame Frame;
	if (!TestTrue(TEXT("Captured"), Stream->Capture(Data) && Stream->WaitForFrame(5.0) && Stream->Acquire(Frame)))
	{
		return false;
	}
	const int32 Slot = Upload->AcquireSlot(5.0);
	if (!TestTrue(TEXT("Upload slot"), Slot != INDEX_NONE))
	{
		return false;
	}

	// The device goes away while the game reads a frame and writes a slot: both memories must stay valid
	AddExpectedMessage(TEXT("WebGPU device lost"), ELogVerbosity::Error, EAutomationExpectedMessageFlags::Contains, 0, false);
	AddExpectedMessage(TEXT("WebGPU device lost"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 0, false);
	const uint32 Generation = Context->GetDeviceGeneration();
	Context->ForceDeviceLoss();
	if (!TestTrue(TEXT("Device recovered"), WaitUntil([&] { return Context->GetDeviceGeneration() > Generation; })))
	{
		return false;
	}
	TestTrue(TEXT("Lost device kept while its memory is held"), Context->HasRetiredDevices());
	TestTrue(TEXT("Held frame still readable"), FFill::Matches(Frame.GetData(), Frame.GetSize(), 42));
	FMemory::Memset(Upload->GetSlotData(Slot), 0x5A, Upload->GetSlotBytes());
	Upload->Commit(Slot, Data, 0, Data->GetSize());
	Frame.Release();
	TestTrue(TEXT("Lost device released once nothing holds its memory"), WaitUntil([&] { return !Context->HasRetiredDevices(); }));
	TestTrue(TEXT("The stale upload was dropped"), WaitUntil([&] { return Upload->GetStats().Dropped == 1; }));

	// Both streams carry on with the new device
	TestTrue(TEXT("Kernel recompiled"), WaitUntil([&] { return Fill.Kernel->IsReady(); }));
	Fill.Run(*Context, 43);
	TestTrue(TEXT("Readback after recovery"), Stream->Capture(Data) && Stream->WaitForFrame(5.0) && Stream->Acquire(Frame)
		&& FFill::Matches(Frame.GetData(), Frame.GetSize(), 43));
	Frame.Release();

	TArray<uint32> Values;
	Values.Init(0xC0FFEEu, 16384);
	TestTrue(TEXT("Upload slot after recovery"), WaitUntil([&] { const int32 Next = Upload->AcquireSlot(0.0); if (Next != INDEX_NONE) { Upload->Cancel(Next); } return Next != INDEX_NONE; }));
	Upload->Write(Data, 0, Values.GetData(), Values.Num() * sizeof(uint32));
	TArray<uint8> Bytes;
	TestTrue(TEXT("Upload after recovery"), ReadNow(*Context, Data, 0, Data->GetSize(), Bytes) && FMemory::Memcmp(Bytes.GetData(), Values.GetData(), Bytes.Num()) == 0);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUStreamBlueprintTest, "WebGPU.Streams.BlueprintAPI", WebGPUStreamTests::Flags)
bool FWebGPUStreamBlueprintTest::RunTest(const FString& Parameters)
{
	// The UObject layer as Blueprint uses it: a mirror in, a kernel, a readback stream out, polled per tick
	const int32 Count = 512;
	UWebGPUStructMirror* Mirror = UWebGPUBlueprintLibrary::CreateWebGPUStructMirror(FWebGPUTestAgent::StaticStruct(), Count, (int32)EWebGPUBufferUsage::Storage, TEXT("bp.mirror"));
	UWebGPUBuffer* Out = UWebGPUBlueprintLibrary::CreateWebGPUStructBuffer(FWebGPUTestAgent::StaticStruct(), Count, (int32)EWebGPUBufferUsage::Storage, TEXT("bp.out"));
	UWebGPUKernel* Kernel = UWebGPUBlueprintLibrary::CreateWebGPUKernelWithStructs(TEXT(R"(
@group(0) @binding(0) var<storage, read> agents: array<WebGPUTestAgent>;
@group(0) @binding(1) var<storage, read_write> moved: array<WebGPUTestAgent>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let i = id.x;
	if (i < arrayLength(&agents)) {
		var a = agents[i];
		a.Position = a.Position + vec3<f32>(a.Heading, 0.0) * a.Speed;
		moved[i] = a;
	}
}
)"), { FWebGPUTestAgent::StaticStruct() }, TEXT("main"), TEXT("bp.move"));
	UWebGPUReadbackStream* Stream = UWebGPUBlueprintLibrary::CreateWebGPUReadbackStreamForBuffer(Out, 3, EWebGPUStreamPolicy::Latest);
	if (!TestNotNull(TEXT("Mirror"), Mirror) || !TestNotNull(TEXT("Out"), Out) || !TestNotNull(TEXT("Kernel"), Kernel) || !TestNotNull(TEXT("Stream"), Stream))
	{
		return false;
	}
	TestEqual(TEXT("Mirror size"), Mirror->Num(), Count);
	TestTrue(TEXT("Mirror struct type"), Mirror->GetStructType() == FWebGPUTestAgent::StaticStruct());
	for (UObject* Object : TArray<UObject*>{ Mirror, Out, Kernel, Stream })
	{
		Object->AddToRoot();
	}

	// Set Elements (wildcard array), Set Element (wildcard struct), Get Element
	FWebGPUTestArrays Arrays;
	for (int32 Index = 0; Index < Count; Index++)
	{
		Arrays.Agents.Add(MakeAgent(Index));
	}
	const FArrayProperty* AgentsProperty = FindFProperty<FArrayProperty>(FWebGPUTestArrays::StaticStruct(), TEXT("Agents"));
	TestTrue(TEXT("SetElements"), Mirror->GenericSetElements(AgentsProperty, AgentsProperty->ContainerPtrToValuePtr<void>(&Arrays), 0));
	const FWebGPUTestAgent Special = MakeAgent(7, 9);
	TestTrue(TEXT("SetElement"), Mirror->GenericSetElement(7, FWebGPUTestAgent::StaticStruct(), &Special));
	FWebGPUTestAgent Got;
	TestTrue(TEXT("GetElement"), Mirror->GenericGetElement(7, FWebGPUTestAgent::StaticStruct(), &Got) && SameAgent(Got, Special));
	AddExpectedMessage(TEXT("the mirror holds"), ELogVerbosity::Error, EAutomationExpectedMessageFlags::Contains, 1, false);
	FWebGPUTestMixed Wrong;
	TestFalse(TEXT("Wrong struct type rejected"), Mirror->GenericSetElement(0, FWebGPUTestMixed::StaticStruct(), &Wrong));
	Arrays.Agents[7] = Special;

	Kernel->SetBufferByName(TEXT("agents"), Mirror->GetBuffer());
	Kernel->SetBufferByName(TEXT("moved"), Out);

	struct FState
	{
		int32 Phase = 0;
		double Deadline = 0.0;
		int64 Uploaded = 0;
		TArray<FWebGPUTestAgent> Result;
	};
	TSharedRef<FState> State = MakeShared<FState>();
	State->Deadline = FPlatformTime::Seconds() + 10.0;

	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([this, Mirror, Out, Kernel, Stream, State, Arrays, Count]() -> bool
	{
		const bool bTimedOut = FPlatformTime::Seconds() > State->Deadline;
		if (!bTimedOut && State->Phase == 0)
		{
			if (!Kernel->IsReady())
			{
				return false;
			}
			// One tick: sync what changed, dispatch, capture
			State->Uploaded = Mirror->Sync();
			Kernel->DispatchForCount(Count);
			Stream->Capture(Out);
			State->Phase = 1;
			return false;
		}
		if (!bTimedOut && State->Phase == 1)
		{
			// Later ticks: poll for the frame
			const FArrayProperty* ResultProperty = FindFProperty<FArrayProperty>(FWebGPUTestArrays::StaticStruct(), TEXT("Agents"));
			FWebGPUTestArrays Holder;
			if (!Stream->GenericReadStructs(ResultProperty, ResultProperty->ContainerPtrToValuePtr<void>(&Holder)))
			{
				return false;
			}
			State->Result = Holder.Agents;
			State->Phase = 2;
		}

		TestEqual(TEXT("Completed in time"), State->Phase, 2);
		TestEqual(TEXT("First sync uploaded the mirror"), State->Uploaded, (int64)Count * 32);
		int32 Bad = State->Result.Num() == Count ? 0 : Count;
		for (int32 Index = 0; Index < State->Result.Num(); Index++)
		{
			FWebGPUTestAgent Expected = Arrays.Agents[Index];
			Expected.Position += FVector3f(Expected.Heading.X, Expected.Heading.Y, 0.f) * Expected.Speed;
			Bad += State->Result[Index].Position.Equals(Expected.Position, 1e-4f) && State->Result[Index].Target == Expected.Target ? 0 : 1;
		}
		TestEqual(TEXT("Moved agents read back through the stream"), Bad, 0);
		TestEqual(TEXT("Frame info"), Stream->GetLastFrameInfo().Sequence, (int64)1);
		TestEqual(TEXT("Nothing more to read"), Stream->GetReadyCount(), 0);

		for (UObject* Object : TArray<UObject*>{ Mirror, Out, Kernel, Stream })
		{
			Object->RemoveFromRoot();
		}
		return true;
	}));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUArmyBenchmarkTest, "WebGPU.Streams.ArmyBenchmark", WebGPUStreamTests::Flags)
bool FWebGPUArmyBenchmarkTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}

	// 262k agents: goals and speeds owned by the CPU (mirror), positions owned by the GPU, a cell id per agent read back
	const int32 Count = 1 << 18;
	TWebGPUStructMirror<FWebGPUTestAgent> Agents = TWebGPUStructMirror<FWebGPUTestAgent>::Create(Context.ToSharedRef(), Count, EWebGPUBufferUsage::Storage, TEXT("army.agents"));
	TSharedRef<FWebGPUBuffer> Positions = Context->CreateBuffer(Count * 16, EWebGPUBufferUsage::Storage, TEXT("army.positions"));
	TSharedRef<FWebGPUBuffer> Cells = Context->CreateBuffer(Count * 4, EWebGPUBufferUsage::Storage, TEXT("army.cells"));
	TSharedRef<FWebGPUBuffer> Params = Context->CreateBuffer(16, EWebGPUBufferUsage::Uniform, TEXT("army.params"));
	TSharedRef<FWebGPUReadbackStream> Results = Context->CreateReadbackStream(Cells->GetSize(), 3, EWebGPUStreamPolicy::Latest, TEXT("army.results"));

	FWebGPUKernelDesc Desc;
	Desc.Label = TEXT("army.step");
	Desc.Structs = { FWebGPUTestAgent::StaticStruct() };
	Desc.Source = TEXT(R"(
struct Params { dt: f32, grid: f32, count: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> agents: array<WebGPUTestAgent>;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cells: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let i = id.x;
	if (i >= params.count) { return; }
	let a = agents[i];
	var p = positions[i];
	p = vec4<f32>(p.xy + a.Heading * a.Speed * params.dt, p.z, p.w);
	positions[i] = p;
	let cell = vec2<i32>(floor(p.xy / params.grid));
	cells[i] = (u32(cell.x) & 0xffffu) | ((u32(cell.y) & 0xffffu) << 16u);
}
)");
	TSharedRef<FWebGPUKernel> Step = Context->CreateKernel(Desc);
	Context->SetBufferByName(Step, TEXT("agents"), Agents.GetBuffer());
	Context->SetBufferByName(Step, TEXT("positions"), Positions);
	Context->SetBufferByName(Step, TEXT("cells"), Cells);
	Context->SetBufferByName(Step, TEXT("params"), Params);

	FRandomStream Random(1234);
	for (int32 Index = 0; Index < Count; Index++)
	{
		FWebGPUTestAgent& Agent = Agents.Edit(Index);
		Agent.Heading = FVector2f(Random.FRandRange(-1.f, 1.f), Random.FRandRange(-1.f, 1.f));
		Agent.Speed = Random.FRandRange(1.f, 5.f);
		Agent.Target = Index;
	}
	Agents.Sync();
	if (!TestTrue(TEXT("Compiled"), WaitUntil([&] { return Step->IsReady() && Agents.Get()->IsScatterReady(); })))
	{
		return false;
	}
	Context->Flush();

	// 300 frames paced at 4 ms: 1% of agents get new goals, upload what changed, step, capture, read the newest result
	const int32 Frames = 300;
	const int32 EditsPerFrame = Count / 100;
	TArray<double> EditMs, SyncMs, DispatchMs, ConsumeMs, FrameMs;
	uint64 UploadedBytes = 0;
	int32 ScatterFrames = 0;
	int32 Read = 0;
	int64 Lag = 0;
	uint64 LastCaptured = 0;
	uint64 Checksum = 0;
	for (int32 Frame = 0; Frame < Frames; Frame++)
	{
		const double T0 = FPlatformTime::Seconds();
		for (int32 Edit = 0; Edit < EditsPerFrame; Edit++)
		{
			FWebGPUTestAgent& Agent = Agents.Edit(Random.RandHelper(Count));
			Agent.Heading = FVector2f(Random.FRandRange(-1.f, 1.f), Random.FRandRange(-1.f, 1.f));
			Agent.Speed = Random.FRandRange(1.f, 5.f);
		}
		const double T1 = FPlatformTime::Seconds();
		UploadedBytes += Agents.Sync();
		ScatterFrames += Agents.Get()->GetLastSync().Scatters > 0 ? 1 : 0;
		const double T2 = FPlatformTime::Seconds();

		const struct { float Dt; float Grid; uint32 Count; uint32 Pad; } ParamData = { 1.f / 60.f, 10.f, (uint32)Count, 0 };
		Context->WriteBuffer(Params, 0, &ParamData, sizeof(ParamData));
		Context->DispatchForCount(Step, Count);
		if (Results->Capture(Cells))
		{
			LastCaptured++;
		}
		const double T3 = FPlatformTime::Seconds();

		FWebGPUReadbackFrame Result;
		if (Results->Acquire(Result))
		{
			for (const uint32 Cell : Result.View<uint32>())
			{
				Checksum += Cell;
			}
			Lag += (int64)(LastCaptured - Result.GetSequence());
			Read++;
		}
		Result.Release();
		const double T4 = FPlatformTime::Seconds();

		EditMs.Add((T1 - T0) * 1000.0);
		SyncMs.Add((T2 - T1) * 1000.0);
		DispatchMs.Add((T3 - T2) * 1000.0);
		ConsumeMs.Add((T4 - T3) * 1000.0);
		FrameMs.Add((T4 - T0) * 1000.0);
		PaceUntil(T0, 0.004);
	}

	// The GPU's copy of the goals equals the CPU's after all those scattered syncs
	auto Same = [](const FWebGPUTestAgent& A, const FWebGPUTestAgent& B) { return SameAgent(A, B); };
	TestEqual(TEXT("Mirror intact under load"), MirrorMismatches<FWebGPUTestAgent>(*Context, Agents, Same), 0);
	TestTrue(TEXT("Results read"), Read > Frames / 2);
	TestTrue(TEXT("Sparse edits scattered"), ScatterFrames > Frames / 2);

	const FWebGPUStreamStats Stats = Results->GetStats();
	AddInfo(FString::Printf(TEXT("Army: %d agents (8 MB of goals, 4 MB of positions), %d edits/frame, 1 MB of results/frame, 300 frames at 4 ms"), Count, EditsPerFrame));
	AddInfo(FString::Printf(TEXT("Game thread per frame: p50 %.3f ms (edits %.3f, sync %.3f, write + dispatch + capture %.3f, read newest 1 MB %.3f), p90 %.3f ms"),
		Percentile(FrameMs, 0.5), Percentile(EditMs, 0.5), Percentile(SyncMs, 0.5), Percentile(DispatchMs, 0.5), Percentile(ConsumeMs, 0.5), Percentile(FrameMs, 0.9)));
	AddInfo(FString::Printf(TEXT("Uploaded %.1f KB/frame (%d of %d frames scattered); results: %d read, %.2f frames behind the latest capture, capture-to-CPU %.3f ms, %lld skipped, %lld dropped (checksum %llu)"),
		UploadedBytes / 1024.0 / Frames, ScatterFrames, Frames, Read, Read ? (double)Lag / Read : 0.0, Stats.LatencyMs, Stats.Skipped, Stats.Dropped, Checksum));
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
