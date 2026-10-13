// Copyright 2025-current Getnamo. All Rights Reserved.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "WebGPUCompute.h"
#include "WebGPUContext.h"
#include "WebGPUShaderFiles.h"
#include "WebGPUSubsystem.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformProcess.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Tests/AutomationCommon.h"

namespace WebGPUHotReloadTests
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

	bool ReadAll(FWebGPUContext& Context, const TSharedRef<FWebGPUBuffer>& Buffer, TArray<uint8>& OutData)
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

	TArray<float> ReadFloats(FWebGPUContext& Context, const TSharedRef<FWebGPUBuffer>& Buffer)
	{
		TArray<uint8> Bytes;
		TArray<float> Floats;
		if (ReadAll(Context, Buffer, Bytes))
		{
			Floats.SetNumUninitialized(Bytes.Num() / 4);
			FMemory::Memcpy(Floats.GetData(), Bytes.GetData(), Floats.Num() * 4);
		}
		return Floats;
	}

	/** Counts a kernel's compiles (listener on the worker thread, so tests can wait without ticking). */
	struct FCompileLog
	{
		std::atomic<int32> Count = 0;
		std::atomic<bool> bLastSuccess = false;
		FCriticalSection Lock;
		TArray<FWebGPUCompileMessage> LastMessages;

		TArray<FWebGPUCompileMessage> GetMessages()
		{
			FScopeLock ScopeLock(&Lock);
			return LastMessages;
		}
	};

	FWebGPUCompileCallback Listen(const TSharedRef<FCompileLog>& Log)
	{
		return [Log](bool bSuccess, const TArray<FWebGPUCompileMessage>& Messages)
		{
			{
				FScopeLock ScopeLock(&Log->Lock);
				Log->LastMessages = Messages;
			}
			Log->bLastSuccess = bSuccess;
			Log->Count++;
		};
	}

	bool WaitCompiles(const TSharedRef<FCompileLog>& Log, int32 Count)
	{
		return WaitUntil([&] { return Log->Count.load() >= Count; });
	}

	FString TestDir(const TCHAR* Name)
	{
		const FString Dir = FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir() / TEXT("WebGPUTests/HotReload") / Name);
		IFileManager::Get().DeleteDirectory(*Dir, false, true);
		IFileManager::Get().MakeDirectory(*Dir, true);
		return Dir;
	}

	/** Writes a shader file with a timestamp later than any before, so every write counts as a change. */
	void WriteShader(const FString& File, const FString& Text)
	{
		static int32 Stamp = 0;
		FFileHelper::SaveStringToFile(Text, *File);
		IFileManager::Get().SetTimeStamp(*File, FDateTime::UtcNow() + FTimespan::FromSeconds(++Stamp));
	}

	const FWebGPUCompileMessage* FindError(const TArray<FWebGPUCompileMessage>& Messages)
	{
		return Messages.FindByPredicate([](const FWebGPUCompileMessage& Message) { return Message.bIsError; });
	}

	bool SameFile(const FString& A, const FString& B)
	{
		return FPaths::IsSamePath(A, B);
	}
}

using namespace WebGPUHotReloadTests;

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUShaderIncludesTest, "WebGPU.HotReload.Includes", WebGPUHotReloadTests::Flags)
bool FWebGPUShaderIncludesTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!Context)
	{
		AddError(TEXT("No WebGPU context"));
		return false;
	}

	const FString Dir = TestDir(TEXT("Includes"));
	const FString Common = Dir / TEXT("common.wgsl");
	const FString Helpers = Dir / TEXT("sub/helpers.wgsl");
	const FString Main = Dir / TEXT("main.wgsl");
	WriteShader(Common, TEXT("// shared\nconst SCALE: f32 = 3.0;\nfn scaled(x: f32) -> f32 { return x * SCALE; }\n"));
	WriteShader(Helpers, TEXT("#include \"../common.wgsl\"\nfn twice(x: f32) -> f32 { return scaled(x) * 2.0; }\n"));
	WriteShader(Main, TEXT(
		"#include \"common.wgsl\"\n"
		"#include \"sub/helpers.wgsl\"\n"
		"  #include \"common.wgsl\"\n"
		"@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n"
		"@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {\n"
		"\tif (id.x < arrayLength(&data)) { data[id.x] = twice(f32(id.x)); }\n"
		"}\n"));

	// Expansion: each file once, in order, with a line map back to the files
	FWebGPUShaderSource Source = FWebGPUShaderFiles::Load(Main);
	TestTrue(TEXT("Loads without errors"), Source.IsValid());
	TestTrue(TEXT("Has a line map"), Source.Map.IsValid());
	if (!Source.IsValid() || !Source.Map)
	{
		AddError(FWebGPUDevice::FormatCompileMessages(Source.Errors));
		return false;
	}
	TestEqual(TEXT("Files read"), Source.Map->Files.Num(), 3);
	TestTrue(TEXT("Main file first"), Source.Map->Files.Num() == 3 && SameFile(Source.Map->Files[0], Main) && SameFile(Source.Map->Files[1], Common) && SameFile(Source.Map->Files[2], Helpers));
	TestEqual(TEXT("A file included twice is pasted once"), Source.Source.Find(TEXT("const SCALE")), Source.Source.Find(TEXT("const SCALE"), ESearchCase::CaseSensitive, ESearchDir::FromEnd));
	TestEqual(TEXT("Timestamps per file"), Source.FileTimes.Num(), 3);

	TArray<FString> Lines;
	Source.Source.ParseIntoArrayLines(Lines, false);
	const int32 TwiceLine = Lines.IndexOfByPredicate([](const FString& Line) { return Line.Contains(TEXT("fn twice")); }) + 1;
	FString File;
	uint64 Line = 0;
	TestTrue(TEXT("Line map resolves"), Source.Map->Resolve((uint64)TwiceLine, File, Line));
	TestTrue(TEXT("fn twice comes from helpers.wgsl line 2"), SameFile(File, Helpers) && Line == 2);

	// The expanded source runs
	FWebGPUKernelDesc Desc;
	Desc.SourceFile = Main;
	Desc.Label = TEXT("IncludesTest");
	TSharedRef<FCompileLog> Log = MakeShared<FCompileLog>();
	TSharedRef<FWebGPUKernel> Kernel = Context->CreateKernel(Desc, Listen(Log), EWebGPUCallbackThread::WorkerThread);
	TestTrue(TEXT("Compiled"), WaitCompiles(Log, 1) && Log->bLastSuccess);
	TSharedRef<FWebGPUBuffer> Data = Context->CreateBuffer(64 * sizeof(float), EWebGPUBufferUsage::Storage, TEXT("IncludesData"));
	Context->SetBufferByName(Kernel, TEXT("data"), Data);
	Context->DispatchForCount(Kernel, 64);
	TArray<float> Values = ReadFloats(*Context, Data);
	TestTrue(TEXT("data[5] = 5 * 3 * 2"), Values.Num() == 64 && Values[5] == 30.f);
	TestEqual(TEXT("Kernel remembers its file"), Kernel->GetSourceFile(), Main);

	// An error inside an included file is reported in that file, at its line
	const FString Broken = Dir / TEXT("broken_part.wgsl");
	WriteShader(Broken, TEXT("// one\n// two\nfn broken() -> f32 { return undefined_name; }\n"));
	const FString BrokenMain = Dir / TEXT("broken_main.wgsl");
	WriteShader(BrokenMain, TEXT("#include \"broken_part.wgsl\"\n@compute @workgroup_size(1) fn main() { _ = broken(); }\n"));
	Desc.SourceFile = BrokenMain;
	TSharedRef<FCompileLog> BrokenLog = MakeShared<FCompileLog>();
	TSharedRef<FWebGPUKernel> BrokenKernel = Context->CreateKernel(Desc, Listen(BrokenLog), EWebGPUCallbackThread::WorkerThread);
	TestTrue(TEXT("Broken include fails"), WaitCompiles(BrokenLog, 1) && !BrokenLog->bLastSuccess);
	{
		const TArray<FWebGPUCompileMessage> Messages = BrokenLog->GetMessages();
		const FWebGPUCompileMessage* Error = FindError(Messages);
		AddInfo(FWebGPUDevice::FormatCompileMessages(Messages));
		TestTrue(TEXT("Error names the included file"), Error && SameFile(Error->File, Broken));
		TestTrue(TEXT("Error is on its line 3"), Error && Error->Line == 3);
		TestTrue(TEXT("Formatted as file:line:column"), FWebGPUDevice::FormatCompileMessages(Messages).Contains(TEXT("broken_part.wgsl:3:")));
	}

	// A missing include fails at the #include line; so does a cycle
	const FString Missing = Dir / TEXT("missing.wgsl");
	WriteShader(Missing, TEXT("// first\n#include \"nope.wgsl\"\n"));
	FWebGPUShaderSource MissingSource = FWebGPUShaderFiles::Load(Missing);
	const FWebGPUCompileMessage* MissingError = FindError(MissingSource.Errors);
	TestTrue(TEXT("Missing include reported at line 2"), MissingError && SameFile(MissingError->File, Missing) && MissingError->Line == 2 && MissingError->Message.Contains(TEXT("nope.wgsl")));
	TestTrue(TEXT("Missing include is still watched"), MissingSource.Map && MissingSource.Map->Files.ContainsByPredicate([](const FString& Entry) { return Entry.EndsWith(TEXT("nope.wgsl")); }));

	WriteShader(Dir / TEXT("a.wgsl"), TEXT("#include \"b.wgsl\"\n"));
	WriteShader(Dir / TEXT("b.wgsl"), TEXT("\n#include \"a.wgsl\"\n"));
	FWebGPUShaderSource Cycle = FWebGPUShaderFiles::Load(Dir / TEXT("a.wgsl"));
	const FWebGPUCompileMessage* CycleError = FindError(Cycle.Errors);
	TestTrue(TEXT("Include cycle reported"), CycleError && CycleError->Message.Contains(TEXT("include cycle: a.wgsl -> b.wgsl -> a.wgsl")) && CycleError->Line == 2);

	FWebGPUShaderSource NoFile = FWebGPUShaderFiles::Load(Dir / TEXT("does_not_exist.wgsl"));
	TestTrue(TEXT("Missing file reported"), !NoFile.IsValid() && NoFile.Errors[0].Message.Contains(TEXT("not found")));

	// Content-style paths: /PluginName/... is the plugin's Content folder
	const FString Example = FWebGPUShaderFiles::ResolvePath(TEXT("/WebGPUCompute/Examples/Army/army_common.wgsl"));
	TestTrue(FString::Printf(TEXT("Content path resolves to an existing file (%s)"), *Example), FPaths::FileExists(Example));
	TestTrue(TEXT("Relative paths are relative to the project"), SameFile(FWebGPUShaderFiles::ResolvePath(TEXT("Saved/x.wgsl")), FPaths::ProjectDir() / TEXT("Saved/x.wgsl")));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUHotReloadWatchTest, "WebGPU.HotReload.Watch", WebGPUHotReloadTests::Flags)
bool FWebGPUHotReloadWatchTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!Context)
	{
		AddError(TEXT("No WebGPU context"));
		return false;
	}

	const FString Dir = TestDir(TEXT("Watch"));
	const FString Value = Dir / TEXT("value.wgsl");
	const FString Main = Dir / TEXT("kernel.wgsl");
	auto KernelSource = [](const TCHAR* Expression)
	{
		return FString::Printf(TEXT(
			"#include \"value.wgsl\"\n"
			"@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n"
			"@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {\n"
			"\tif (id.x < arrayLength(&data)) { data[id.x] = %s; }\n"
			"}\n"), Expression);
	};
	WriteShader(Value, TEXT("const VALUE: f32 = 1.0;\n"));
	WriteShader(Main, KernelSource(TEXT("VALUE")));

	FWebGPUKernelDesc Desc;
	Desc.SourceFile = Main;
	Desc.bHotReload = true;
	Desc.Label = TEXT("WatchTest");
	TSharedRef<FCompileLog> Log = MakeShared<FCompileLog>();
	TSharedRef<FWebGPUKernel> Kernel = Context->CreateKernel(Desc, Listen(Log), EWebGPUCallbackThread::WorkerThread);
	TSharedRef<FWebGPUBuffer> Data = Context->CreateBuffer(64 * sizeof(float), EWebGPUBufferUsage::Storage, TEXT("WatchData"));
	Context->SetBufferByName(Kernel, TEXT("data"), Data);
	TestTrue(TEXT("Watched"), Context->IsHotReloading(Kernel));

	int32 Compiles = 1;
	auto Run = [&]() -> float
	{
		Context->DispatchForCount(Kernel, 64);
		TArray<float> Values = ReadFloats(*Context, Data);
		return Values.Num() == 64 ? Values[63] : -1.f;
	};
	// A save is picked up once it's been stable for a poll: the first poll sees the change, the second reloads
	auto PollTwice = [&]() -> int32
	{
		const int32 First = Context->PollShaderFiles();
		return First + Context->PollShaderFiles();
	};

	TestTrue(TEXT("First compile"), WaitCompiles(Log, Compiles) && Log->bLastSuccess);
	TestEqual(TEXT("VALUE 1"), Run(), 1.f);
	TestEqual(TEXT("Nothing changed, nothing reloads"), PollTwice(), 0);

	// Editing an included file reloads the kernel
	WriteShader(Value, TEXT("const VALUE: f32 = 2.0;\n"));
	TestEqual(TEXT("First poll only notices the change"), Context->PollShaderFiles(), 0);
	TestEqual(TEXT("Second poll reloads"), Context->PollShaderFiles(), 1);
	TestTrue(TEXT("Recompiled after the include changed"), WaitCompiles(Log, ++Compiles) && Log->bLastSuccess);
	TestEqual(TEXT("VALUE 2"), Run(), 2.f);

	// File times have one second precision: a second save within the same second still reloads (content hash)
	{
		const FDateTime Time = IFileManager::Get().GetTimeStamp(*Value);
		FFileHelper::SaveStringToFile(TEXT("const VALUE: f32 = 7.0;\n"), *Value);
		IFileManager::Get().SetTimeStamp(*Value, Time);
		TestEqual(TEXT("A save that keeps the file's time reloads"), PollTwice(), 1);
		TestTrue(TEXT("Same-time save compiles"), WaitCompiles(Log, ++Compiles) && Log->bLastSuccess);
		TestEqual(TEXT("VALUE 7"), Run(), 7.f);
		WriteShader(Value, TEXT("const VALUE: f32 = 2.0;\n"));
		TestEqual(TEXT("Back to 2"), PollTwice(), 1);
		TestTrue(TEXT("VALUE 2 compiles"), WaitCompiles(Log, ++Compiles) && Log->bLastSuccess);
	}

	// A broken edit: reported in its file and line, the last good pipeline keeps running
	WriteShader(Main, KernelSource(TEXT("VALUE +")));
	TestEqual(TEXT("Broken edit reloads"), PollTwice(), 1);
	TestTrue(TEXT("Broken edit fails"), WaitCompiles(Log, ++Compiles) && !Log->bLastSuccess);
	{
		const TArray<FWebGPUCompileMessage> Messages = Log->GetMessages();
		const FWebGPUCompileMessage* Error = FindError(Messages);
		TestTrue(TEXT("Error in kernel.wgsl line 4"), Error && SameFile(Error->File, Main) && Error->Line == 4);
	}
	TestEqual(TEXT("Last good version still runs"), Run(), 2.f);

	WriteShader(Main, KernelSource(TEXT("VALUE * 3.0")));
	TestEqual(TEXT("Fix reloads"), PollTwice(), 1);
	TestTrue(TEXT("Fix compiles"), WaitCompiles(Log, ++Compiles) && Log->bLastSuccess);
	TestEqual(TEXT("VALUE * 3"), Run(), 6.f);

	// A deleted include fails the reload; bringing it back recovers
	IFileManager::Get().Delete(*Value);
	TestEqual(TEXT("Deleted include reloads"), PollTwice(), 1);
	TestTrue(TEXT("Deleted include fails"), WaitCompiles(Log, ++Compiles) && !Log->bLastSuccess);
	{
		const TArray<FWebGPUCompileMessage> Messages = Log->GetMessages();
		const FWebGPUCompileMessage* Error = FindError(Messages);
		TestTrue(TEXT("Reported at the #include line"), Error && SameFile(Error->File, Main) && Error->Line == 1 && Error->Message.Contains(TEXT("value.wgsl")));
	}
	TestEqual(TEXT("Still the last good version"), Run(), 6.f);
	WriteShader(Value, TEXT("const VALUE: f32 = 4.0;\n"));
	TestEqual(TEXT("Restored include reloads"), PollTwice(), 1);
	TestTrue(TEXT("Restored include compiles"), WaitCompiles(Log, ++Compiles) && Log->bLastSuccess);
	TestEqual(TEXT("VALUE 4 * 3"), Run(), 12.f);

	// Unwatched: edits are ignored
	Context->SetHotReload(Kernel, false);
	TestFalse(TEXT("No longer watched"), Context->IsHotReloading(Kernel));
	WriteShader(Value, TEXT("const VALUE: f32 = 5.0;\n"));
	TestEqual(TEXT("No reload when unwatched"), PollTwice(), 0);
	Context->SetHotReload(Kernel, true);
	TestTrue(TEXT("Watched again"), Context->IsHotReloading(Kernel));
	TestEqual(TEXT("Watching again starts from the files as they are"), PollTwice(), 0);

	// The reload reports run on the game thread: let them land inside the test
	AddExpectedMessage(TEXT("doesn't compile; the last good version keeps running"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 2, false);
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(0.2f));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUBindingSetTest, "WebGPU.HotReload.BindingSets", WebGPUHotReloadTests::Flags)
bool FWebGPUBindingSetTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!Context)
	{
		AddError(TEXT("No WebGPU context"));
		return false;
	}

	auto Source = [](const TCHAR* ParamsStruct)
	{
		return FString::Printf(TEXT(R"(
%s
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;
@group(0) @binding(2) var<storage, read_write> extra: array<f32>;
@group(0) @binding(3) var<uniform> gain: f32;

@compute @workgroup_size(64) fn write(@builtin(global_invocation_id) id: vec3u) {
	if (id.x >= params.count) { return; }
	data[id.x] = f32(id.x) * params.scale + params.offset.y + f32(params.bias);
}

@compute @workgroup_size(64) fn copy(@builtin(global_invocation_id) id: vec3u) {
	if (id.x < arrayLength(&extra)) { extra[id.x] = data[id.x] * gain; }
}
)"), ParamsStruct);
	};
	const FString Original = Source(TEXT("struct Params { scale: f32, count: u32, offset: vec2f, bias: i32 }"));

	TSharedRef<FWebGPUBindingSet> Set = Context->CreateBindingSet(TEXT("SetTest"));
	const int32 Count = 16;
	TSharedRef<FWebGPUBuffer> Data = Context->CreateBuffer(Count * sizeof(float), EWebGPUBufferUsage::Storage, TEXT("SetData"));
	TSharedRef<FWebGPUBuffer> Extra = Context->CreateBuffer(Count * sizeof(float), EWebGPUBufferUsage::Storage, TEXT("SetExtra"));
	TSharedRef<FWebGPUBuffer> Own = Context->CreateBuffer(Count * sizeof(float), EWebGPUBufferUsage::Storage, TEXT("KernelsOwnExtra"));
	Context->SetSharedBuffer(Set, TEXT("data"), Data);
	Context->SetSharedBuffer(Set, TEXT("extra"), Extra);
	Context->SetUniformValue(Set, TEXT("params.scale"), FWebGPUUniformValue::Float(2.f));
	Context->SetUniformValue(Set, TEXT("params.count"), FWebGPUUniformValue::Int(8));
	Context->SetUniformValue(Set, TEXT("params.offset"), FWebGPUUniformValue::Vector(FVector4f(0.f, 10.f, 0.f, 0.f)));
	Context->SetUniformValue(Set, TEXT("params.bias"), FWebGPUUniformValue::Float(-3.7f));
	Context->SetUniformValue(Set, TEXT("gain"), FWebGPUUniformValue::Float(2.f));

	FWebGPUKernelDesc Desc;
	Desc.Source = Original;
	Desc.Bindings = Set;
	Desc.EntryPoint = TEXT("write");
	TSharedRef<FCompileLog> WriteLog = MakeShared<FCompileLog>();
	TSharedRef<FWebGPUKernel> Write = Context->CreateKernel(Desc, Listen(WriteLog), EWebGPUCallbackThread::WorkerThread);
	Desc.EntryPoint = TEXT("copy");
	TSharedRef<FCompileLog> CopyLog = MakeShared<FCompileLog>();
	TSharedRef<FWebGPUKernel> Copy = Context->CreateKernel(Desc, Listen(CopyLog), EWebGPUCallbackThread::WorkerThread);
	TestTrue(TEXT("Both compile"), WaitCompiles(WriteLog, 1) && WaitCompiles(CopyLog, 1) && WriteLog->bLastSuccess && CopyLog->bLastSuccess);

	// Shared buffers and uniform members by name; a value change between two dispatches applies between them
	Context->DispatchForCount(Write, Count);
	Context->DispatchForCount(Copy, Count);
	Context->SetUniformValue(Set, TEXT("params.scale"), FWebGPUUniformValue::Float(5.f));
	Context->DispatchForCount(Write, Count);
	TArray<float> A = ReadFloats(*Context, Data);
	TArray<float> B = ReadFloats(*Context, Extra);
	bool bWrite = A.Num() == Count && B.Num() == Count;
	bool bCopy = bWrite;
	for (int32 Index = 0; bWrite && Index < Count; Index++)
	{
		// bias -3.7 truncates to -3 in the i32 member; only the first params.count elements are written
		bWrite &= A[Index] == (Index < 8 ? Index * 5.f + 10.f - 3.f : 0.f);
		bCopy &= B[Index] == (Index < 8 ? (Index * 2.f + 7.f) * 2.f : 0.f);
	}
	TestTrue(TEXT("write saw scale 5 (set after the first dispatch), offset.y 10, bias -3, count 8"), bWrite);
	TestTrue(TEXT("copy saw the first write and gain 2 (a scalar uniform)"), bCopy);

	// A kernel's own binding wins over the set's
	Context->SetBufferByName(Copy, TEXT("extra"), Own);
	Context->DispatchForCount(Copy, Count);
	TArray<float> Mine = ReadFloats(*Context, Own);
	TArray<float> Shared = ReadFloats(*Context, Extra);
	TestTrue(TEXT("Own binding written"), Mine.Num() == Count && Mine[1] == (5.f + 7.f) * 2.f);
	TestTrue(TEXT("Shared one untouched"), Shared.Num() == Count && Shared[1] == (2.f + 7.f) * 2.f);

	// Hot reload moves and grows the struct: the values follow their names
	Context->RecompileKernel(Write, Source(TEXT("struct Params { bias: i32, added: vec4f, offset: vec2f, count: u32, scale: f32 }")));
	TestTrue(TEXT("Reordered struct compiles"), WaitCompiles(WriteLog, 2) && WriteLog->bLastSuccess);
	Context->SetUniformValue(Set, TEXT("params.scale"), FWebGPUUniformValue::Float(1.f));
	Context->DispatchForCount(Write, Count);
	Context->DispatchForCount(Copy, Count);
	A = ReadFloats(*Context, Data);
	Mine = ReadFloats(*Context, Own);
	TestTrue(TEXT("Values follow the members"), A.Num() == Count && A[3] == 3.f + 7.f && A[9] == 0.f);
	TestTrue(TEXT("The other kernel still works"), Mine.Num() == Count && Mine[3] == (3.f + 7.f) * 2.f);

	// A value for a member that doesn't exist warns once and the rest still works
	AddExpectedMessage(TEXT("has no member 'missing'"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
	Context->SetUniformValue(Set, TEXT("params.missing"), FWebGPUUniformValue::Float(1.f));
	Context->SetUniformValue(Set, TEXT("params.scale"), FWebGPUUniformValue::Float(4.f));
	Context->DispatchForCount(Write, Count);
	A = ReadFloats(*Context, Data);
	TestTrue(TEXT("Other values still applied"), A.Num() == Count && A[2] == 8.f + 7.f);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUArmyShadersTest, "WebGPU.Examples.ArmyShaders", WebGPUHotReloadTests::Flags)
bool FWebGPUArmyShadersTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!Context)
	{
		AddError(TEXT("No WebGPU context"));
		return false;
	}

	// The example's constants (army_common.wgsl)
	constexpr int32 Grid = 256;
	constexpr int32 Display = 1024;
	constexpr int32 Teams = 2;
	constexpr float CaptureHeight = 2000.f;
	constexpr uint32 Unreachable = 0x3fffffffu;
	constexpr int32 AgentCount = 8192;
	constexpr int32 AgentStride = 32;

	// What the example's Blueprint sets up, without the shared render targets: a wall at x 60..63 from y 0 to 179
	TSharedRef<FWebGPUBindingSet> Set = Context->CreateBindingSet(TEXT("ArmyTest"));
	TSharedRef<FWebGPUBuffer> Agents = Context->CreateBuffer(AgentCount * AgentStride, EWebGPUBufferUsage::Storage, TEXT("agents"));
	TSharedRef<FWebGPUBuffer> Field = Context->CreateBuffer(Teams * Grid * Grid * 4, EWebGPUBufferUsage::Storage, TEXT("field"));
	TSharedRef<FWebGPUBuffer> Density = Context->CreateBuffer(Teams * Grid * Grid * 4, EWebGPUBufferUsage::Storage, TEXT("density"));
	TSharedRef<FWebGPUTexture> Obstacles = Context->CreateTexture(Grid, Grid, EWebGPUTextureFormat::R32Float, EWebGPUTextureUsage::Sampled, TEXT("obstacles"));
	TSharedRef<FWebGPUTexture> DisplayTexture = Context->CreateTexture(Display, Display, EWebGPUTextureFormat::RGBA16Float, EWebGPUTextureUsage::Storage, TEXT("display"));
	auto IsWall = [](int32 X, int32 Y) { return X >= 60 && X < 64 && Y < 180; };
	TArray<float> Depth;
	Depth.SetNumUninitialized(Grid * Grid);
	for (int32 Y = 0; Y < Grid; Y++)
	{
		for (int32 X = 0; X < Grid; X++)
		{
			Depth[Y * Grid + X] = IsWall(X, Y) ? CaptureHeight - 500.f : CaptureHeight;
		}
	}
	Context->WriteTexture(Obstacles, Depth.GetData(), Depth.Num() * sizeof(float));
	Context->SetSharedBuffer(Set, TEXT("agents"), Agents);
	Context->SetSharedBuffer(Set, TEXT("field"), Field);
	Context->SetSharedBuffer(Set, TEXT("density"), Density);
	Context->SetSharedTexture(Set, TEXT("obstacles"), Obstacles);
	Context->SetSharedTexture(Set, TEXT("display"), DisplayTexture);
	Context->SetUniformValue(Set, TEXT("params.dt"), FWebGPUUniformValue::Float(1.f / 60.f));

	TMap<FString, TSharedRef<FWebGPUKernel>> Kernels;
	TArray<TSharedRef<FCompileLog>> Logs;
	auto Make = [&](const TCHAR* File, const TCHAR* Entry)
	{
		FWebGPUKernelDesc Desc;
		Desc.SourceFile = FString::Printf(TEXT("/WebGPUCompute/Examples/Army/%s"), File);
		Desc.EntryPoint = Entry;
		Desc.Bindings = Set;
		TSharedRef<FCompileLog> Log = Logs.Add_GetRef(MakeShared<FCompileLog>());
		Kernels.Add(Entry, Context->CreateKernel(Desc, Listen(Log), EWebGPUCallbackThread::WorkerThread));
	};
	Make(TEXT("army_field.wgsl"), TEXT("prepare"));
	Make(TEXT("army_field.wgsl"), TEXT("count"));
	Make(TEXT("army_field.wgsl"), TEXT("relax"));
	Make(TEXT("army_behavior.wgsl"), TEXT("simulate"));
	Make(TEXT("army_behavior.wgsl"), TEXT("respawn"));
	for (const TSharedRef<FCompileLog>& Log : Logs)
	{
		if (!WaitCompiles(Log, 1) || !Log->bLastSuccess)
		{
			AddError(FString::Printf(TEXT("Example shader doesn't compile:\n%s"), *FWebGPUDevice::FormatCompileMessages(Log->GetMessages())));
			return false;
		}
	}

	// What the Blueprint does every frame, for 4 seconds of 60 Hz frames
	const double Start = FPlatformTime::Seconds();
	for (int32 Frame = 0; Frame < 240; Frame++)
	{
		Context->SetUniformValue(Set, TEXT("params.time"), FWebGPUUniformValue::Float(Frame / 60.f));
		Context->DispatchForCount(Kernels[TEXT("prepare")], Display, Display);
		Context->DispatchForCount(Kernels[TEXT("count")], AgentCount);
		for (int32 Pass = 0; Pass < 24; Pass++)
		{
			Context->DispatchForCount(Kernels[TEXT("relax")], Grid, Grid, Teams);
		}
		Context->DispatchForCount(Kernels[TEXT("simulate")], AgentCount);
	}
	TArray<uint8> AgentBytes;
	TArray<uint8> FieldBytes;
	TestTrue(TEXT("Read agents"), ReadAll(*Context, Agents, AgentBytes));
	TestTrue(TEXT("Read field"), ReadAll(*Context, Field, FieldBytes));
	AddInfo(FString::Printf(TEXT("240 frames (%d agents, 27 dispatches each) in %.0f ms"), AgentCount, (FPlatformTime::Seconds() - Start) * 1000.0));
	if (AgentBytes.Num() != AgentCount * AgentStride || FieldBytes.Num() != Teams * Grid * Grid * 4)
	{
		return false;
	}

	struct FAgentView
	{
		FVector2f Pos;
		FVector2f Vel;
		uint32 Team;
		uint32 Goal;
		uint32 Seed;
		uint32 State;
	};
	static_assert(sizeof(FAgentView) == AgentStride);
	const FAgentView* View = reinterpret_cast<const FAgentView*>(AgentBytes.GetData());
	int32 Unspawned = 0;
	int32 InWall = 0;
	int32 Outside = 0;
	double SumX[Teams] = {};
	for (int32 Index = 0; Index < AgentCount; Index++)
	{
		const FAgentView& Agent = View[Index];
		Unspawned += Agent.State == 0 ? 1 : 0;
		Outside += (Agent.Pos.X < 0.5f || Agent.Pos.Y < 0.5f || Agent.Pos.X > Grid - 0.5f || Agent.Pos.Y > Grid - 0.5f) ? 1 : 0;
		InWall += IsWall((int32)Agent.Pos.X, (int32)Agent.Pos.Y) ? 1 : 0;
		SumX[Agent.Team % Teams] += Agent.Pos.X;
	}
	const double MeanWest = SumX[0] / (AgentCount / 2);
	const double MeanEast = SumX[1] / (AgentCount / 2);
	AddInfo(FString::Printf(TEXT("Mean x after 4 s: team 0 %.1f (spawned 2..30), team 1 %.1f (spawned 226..254)"), MeanWest, MeanEast));
	TestEqual(TEXT("Every agent spawned"), Unspawned, 0);
	TestEqual(TEXT("No agent outside the grid"), Outside, 0);
	TestEqual(TEXT("No agent inside the wall"), InWall, 0);
	TestTrue(TEXT("Team 0 marched east"), MeanWest > 40.0);
	TestTrue(TEXT("Team 1 marched west"), MeanEast < Grid - 40.0);

	// The flow field: walls unreachable, and the way around the wall is longer than the way through the gap
	const uint32* Distances = reinterpret_cast<const uint32*>(FieldBytes.GetData());
	auto DistanceEast = [&](int32 X, int32 Y) { return Distances[Grid * Grid + Y * Grid + X]; };
	TestEqual(TEXT("Wall cells unreachable"), DistanceEast(61, 50), Unreachable);
	TestEqual(TEXT("Goal cells are 0"), DistanceEast(Grid - 1, 100), 0u);
	const uint32 Behind = DistanceEast(50, 100);
	const uint32 Open = DistanceEast(50, 220);
	AddInfo(FString::Printf(TEXT("Steps to the east goal from (50,100), behind the wall: %.1f; from (50,220), in line with the gap: %.1f"), Behind / 16.f, Open / 16.f));
	TestTrue(TEXT("Behind the wall the path detours"), Behind < Unreachable && Behind > Open + 16 * 40);

	// Respawn puts everyone back at the start
	Context->DispatchForCount(Kernels[TEXT("respawn")], AgentCount);
	TestTrue(TEXT("Read agents after respawn"), ReadAll(*Context, Agents, AgentBytes));
	View = reinterpret_cast<const FAgentView*>(AgentBytes.GetData());
	int32 AtStart = 0;
	for (int32 Index = 0; Index < AgentCount; Index++)
	{
		const float X = View[Index].Pos.X;
		AtStart += (View[Index].Team == 0 ? X <= 30.5f : X >= Grid - 30.5f) ? 1 : 0;
	}
	TestEqual(TEXT("Everyone back at the start"), AtStart, AgentCount);
	return true;
}

#endif
