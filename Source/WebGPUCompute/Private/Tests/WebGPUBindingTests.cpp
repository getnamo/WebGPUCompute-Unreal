// Copyright 2025-current Getnamo. All Rights Reserved.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "WebGPUBlueprintLibrary.h"
#include "WebGPUBuffer.h"
#include "WebGPUCompute.h"
#include "WebGPUContext.h"
#include "WebGPUReflection.h"
#include "WebGPUStructLayout.h"
#include "WebGPUSubsystem.h"
#include "WebGPUTestStructs.h"
#include "HAL/PlatformProcess.h"

namespace WebGPUBindingTests
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

	/** Reads a whole buffer on the worker thread and waits for it. */
	bool ReadAll(FWebGPUContext& Context, const TSharedRef<FWebGPUBuffer>& Buffer, TArray<uint8>& OutData)
	{
		TSharedRef<FReadResult> Result = MakeShared<FReadResult>();
		Context.ReadBuffer(Buffer, 0, Align(Buffer->GetSize(), 4), [Result](bool bSuccess, TArray<uint8>&& Data)
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

	bool WaitReady(const TSharedPtr<FWebGPUKernel>& Kernel)
	{
		return WaitUntil([&] { return Kernel->IsReady() || (Kernel->GetMessages().Num() > 0 && !Kernel->IsReady()); }) && Kernel->IsReady();
	}

	FWebGPUTestAgent MakeAgent(int32 Index)
	{
		FWebGPUTestAgent Agent;
		Agent.Position = FVector3f(Index, Index * 2.f, Index * 3.f);
		Agent.Speed = 0.5f + (Index % 7);
		Agent.Target = Index;
		Agent.Flags = (uint32)(Index * 2);
		Agent.Heading = FVector2f(1.f, -1.f);
		return Agent;
	}

	FWebGPUTestMixed MakeMixed(int32 Index)
	{
		FWebGPUTestMixed Mixed;
		Mixed.Location = FVector(Index * 1.5, -Index * 0.25, 1000.0 + Index);
		Mixed.bActive = Index % 2 == 0;
		Mixed.Team = (uint8)(Index % 200);
		Mixed.Velocity = FVector3f(1.f, 2.f, 3.f);
		Mixed.Color = FLinearColor(Index * 0.001f, 0.2f, 0.3f, 1.f);
		Mixed.Values[0] = (float)Index;
		Mixed.Values[1] = Index * 0.5f;
		Mixed.Values[2] = -1.f;
		Mixed.Inner.A = 7.f;
		Mixed.Inner.B = FVector2f(Index, 1.f);
		Mixed.Small = (int16)(Index - 500);
		return Mixed;
	}

	FString ShaderUsingStruct(const FWebGPUStructLayout& Layout, bool bUniform)
	{
		const FString Declaration = bUniform
			? FString::Printf(TEXT("@group(0) @binding(0) var<uniform> item: %s;\n@compute @workgroup_size(1) fn main() { _ = item; }\n"), *Layout.GetWgslName())
			: FString::Printf(TEXT("@group(0) @binding(0) var<storage, read_write> items: array<%s>;\n@compute @workgroup_size(1) fn main() { _ = items[0]; }\n"), *Layout.GetWgslName());
		return Layout.GenerateWGSL() + Declaration;
	}
}

using namespace WebGPUBindingTests;

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUReflectionTest, "WebGPU.Binding.Reflection", WebGPUBindingTests::Flags)
bool FWebGPUReflectionTest::RunTest(const FString& Parameters)
{
	const FString Source = TEXT(R"(
struct Params { count: u32, scale: f32, offset: vec3<f32> };
struct Agent { pos: vec3<f32>, speed: f32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> agents: array<Agent>;
@group(1) @binding(0) var<storage, read> weights: array<f32>;
@group(1) @binding(3) var<storage, read_write> unused: array<u32>;
override block_size: u32 = 8u;

@compute @workgroup_size(8, 4, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let i = id.x;
	if (i >= params.count) { return; }
	agents[i].pos = agents[i].pos * params.scale + params.offset * weights[i % arrayLength(&weights)];
	agents[i].speed = f32(block_size);
}

@compute @workgroup_size(64)
fn other() { }
)");

	FWebGPUShaderReflection Reflection;
	TArray<FWebGPUCompileMessage> Messages;
	if (!TestTrue(TEXT("Reflect"), FWebGPUShaderReflection::Reflect(Source, Reflection, Messages)))
	{
		AddError(FWebGPUDevice::FormatCompileMessages(Messages));
		return false;
	}

	TestEqual(TEXT("Entry points"), Reflection.EntryPoints.Num(), 2);
	const FWebGPUEntryPointInfo* Main = Reflection.FindEntryPoint(TEXT("main"));
	if (!TestNotNull(TEXT("main"), Main))
	{
		return false;
	}
	TestTrue(TEXT("main is compute"), Main->bCompute);
	TestEqual(TEXT("main workgroup size"), Main->WorkgroupSize, FIntVector(8, 4, 1));
	TestTrue(TEXT("main uses override block_size"), Main->Overrides.Contains(TEXT("block_size")));
	TestEqual(TEXT("other workgroup size"), Reflection.FindEntryPoint(TEXT("other"))->WorkgroupSize, FIntVector(64, 1, 1));

	TestEqual(TEXT("main uses 3 bindings (not 'unused')"), Main->Bindings.Num(), 3);
	TestNull(TEXT("'unused' not listed"), Main->FindBinding(TEXT("unused")));

	const FWebGPUBindingInfo* Params = Main->FindBinding(TEXT("params"));
	const FWebGPUBindingInfo* Agents = Main->FindBinding(TEXT("agents"));
	const FWebGPUBindingInfo* Weights = Main->FindBinding(TEXT("weights"));
	if (!TestNotNull(TEXT("params"), Params) || !TestNotNull(TEXT("agents"), Agents) || !TestNotNull(TEXT("weights"), Weights))
	{
		return false;
	}
	TestTrue(TEXT("params type"), Params->Type == EWebGPUBindingType::UniformBuffer);
	TestEqual(TEXT("params group/binding"), FIntPoint(Params->Group, Params->Binding), FIntPoint(0, 0));
	TestEqual(TEXT("params min size"), Params->MinSize, (int64)32);
	TestTrue(TEXT("agents type"), Agents->Type == EWebGPUBindingType::StorageBuffer);
	TestEqual(TEXT("agents min size (one element)"), Agents->MinSize, (int64)16);
	TestTrue(TEXT("weights type"), Weights->Type == EWebGPUBindingType::ReadOnlyStorageBuffer);
	TestEqual(TEXT("weights group/binding"), FIntPoint(Weights->Group, Weights->Binding), FIntPoint(1, 0));

	TSharedPtr<const FWebGPUTypeLayout> AgentElement = Agents->GetRuntimeArrayElement();
	if (TestTrue(TEXT("agents element is Agent"), AgentElement.IsValid() && AgentElement->Name == TEXT("Agent")))
	{
		TestEqual(TEXT("Agent size"), AgentElement->Size, 16u);
		TestEqual(TEXT("Agent.speed offset"), AgentElement->FindField(TEXT("speed"))->Offset, 12u);
	}
	const TSharedPtr<const FWebGPUTypeLayout>* ParamsStruct = Reflection.Structs.Find(TEXT("Params"));
	if (TestNotNull(TEXT("Params struct"), ParamsStruct))
	{
		TestEqual(TEXT("Params.offset at 16 (vec3 aligns to 16)"), (*ParamsStruct)->FindField(TEXT("offset"))->Offset, 16u);
		TestEqual(TEXT("Params size"), (*ParamsStruct)->Size, 32u);
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUStructLayoutTest, "WebGPU.Binding.StructLayoutMatchesTint", WebGPUBindingTests::Flags)
bool FWebGPUStructLayoutTest::RunTest(const FString& Parameters)
{
	// Our layout math vs Tint's, for generated declarations
	struct FCase
	{
		UScriptStruct* Struct;
		EWebGPUAddressSpace Space;
		uint32 ExpectedSize;
		bool bExpectMemcpy;
	};
	const FCase Cases[] = {
		{ FWebGPUTestAgent::StaticStruct(), EWebGPUAddressSpace::Storage, 32, true },
		{ FWebGPUTestAgent::StaticStruct(), EWebGPUAddressSpace::Uniform, 32, true },
		{ FWebGPUTestInner::StaticStruct(), EWebGPUAddressSpace::Storage, 16, false },
		{ FWebGPUTestMixed::StaticStruct(), EWebGPUAddressSpace::Storage, 112, false },
	};

	for (const FCase& Case : Cases)
	{
		const FString CaseName = FString::Printf(TEXT("%s (%s)"), *Case.Struct->GetName(), Case.Space == EWebGPUAddressSpace::Uniform ? TEXT("uniform") : TEXT("storage"));
		FString Error;
		TSharedPtr<const FWebGPUStructLayout> Layout = FWebGPUStructLayout::Get(Case.Struct, Case.Space, Error);
		if (!TestTrue(FString::Printf(TEXT("%s layout: %s"), *CaseName, *Error), Layout.IsValid()))
		{
			continue;
		}
		TestEqual(FString::Printf(TEXT("%s size"), *CaseName), Layout->GetSize(), Case.ExpectedSize);
		TestEqual(FString::Printf(TEXT("%s memcpy-compatible"), *CaseName), Layout->IsMemcpyCompatible(), Case.bExpectMemcpy);

		const FString Shader = ShaderUsingStruct(*Layout, Case.Space == EWebGPUAddressSpace::Uniform);
		FWebGPUShaderReflection Reflection;
		TArray<FWebGPUCompileMessage> Messages;
		if (!TestTrue(FString::Printf(TEXT("%s generated WGSL compiles"), *CaseName), FWebGPUShaderReflection::Reflect(Shader, Reflection, Messages)))
		{
			AddError(FWebGPUDevice::FormatCompileMessages(Messages) + Shader);
			continue;
		}
		const TSharedPtr<const FWebGPUTypeLayout>* TintStruct = Reflection.Structs.Find(Layout->GetWgslName());
		FString Differences;
		TestTrue(FString::Printf(TEXT("%s matches Tint"), *CaseName), TintStruct && Layout->Matches(**TintStruct, Differences));
		if (!Differences.IsEmpty())
		{
			AddError(Differences);
		}
	}

	// Uniform rules reject 4-byte array elements
	FString UniformError;
	TestFalse(TEXT("Mixed has no uniform layout"), FWebGPUStructLayout::Get(FWebGPUTestMixed::StaticStruct(), EWebGPUAddressSpace::Uniform, UniformError).IsValid());
	TestTrue(TEXT("Uniform error names the array"), UniformError.Contains(TEXT("Values")));

	AddInfo(FWebGPUStructLayout::Get(FWebGPUTestMixed::StaticStruct())->GenerateWGSL());
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUStructPackTest, "WebGPU.Binding.StructPackRoundTrip", WebGPUBindingTests::Flags)
bool FWebGPUStructPackTest::RunTest(const FString& Parameters)
{
	TArray<FWebGPUTestMixed> Source;
	for (int32 Index = 0; Index < 5; Index++)
	{
		Source.Add(MakeMixed(Index));
	}
	TSharedPtr<const FWebGPUStructLayout> Layout = FWebGPUStructLayout::Get(FWebGPUTestMixed::StaticStruct());
	if (!TestNotNull(TEXT("Layout"), Layout.Get()))
	{
		return false;
	}

	const TArray<uint8> Bytes = Layout->Pack(Source.GetData(), Source.Num());
	TestEqual(TEXT("Packed size"), Bytes.Num(), Source.Num() * 112);
	// Spot check the GPU bytes: element 1's Team (u32 at 16) and Velocity.y (f32 at 36)
	TestEqual(TEXT("Team as u32"), *reinterpret_cast<const uint32*>(&Bytes[112 + 16]), 1u);
	TestEqual(TEXT("Velocity.y at offset 36"), *reinterpret_cast<const float*>(&Bytes[112 + 36]), 2.f);

	TArray<FWebGPUTestMixed> Unpacked;
	Unpacked.SetNum(Source.Num());
	Layout->Unpack(Bytes.GetData(), Source.Num(), Unpacked.GetData());
	for (int32 Index = 0; Index < Source.Num(); Index++)
	{
		const FWebGPUTestMixed& A = Source[Index];
		const FWebGPUTestMixed& B = Unpacked[Index];
		const bool bSame = A.Location.Equals(B.Location, 1e-3) && A.bActive == B.bActive && A.Team == B.Team && A.Velocity == B.Velocity
			&& A.Color == B.Color && A.Values[0] == B.Values[0] && A.Values[2] == B.Values[2] && A.Inner.A == B.Inner.A
			&& A.Inner.B == B.Inner.B && A.Small == B.Small;
		TestTrue(FString::Printf(TEXT("Element %d round trips"), Index), bSame);
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUStructKernelTest, "WebGPU.Binding.StructKernel", WebGPUBindingTests::Flags)
bool FWebGPUStructKernelTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}

	// The shader uses the UStructs by name through the generated preamble
	FWebGPUKernelDesc Desc;
	Desc.Label = TEXT("struct-kernel");
	Desc.Structs = { FWebGPUTestAgent::StaticStruct(), FWebGPUTestMixed::StaticStruct() };
	Desc.Source = TEXT(R"(
@group(0) @binding(0) var<storage, read_write> agents: array<WebGPUTestAgent>;
@group(0) @binding(1) var<storage, read_write> mixed: array<WebGPUTestMixed>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let i = id.x;
	if (i < arrayLength(&agents)) {
		var a = agents[i];
		a.Position = a.Position + vec3<f32>(a.Heading, 0.0) * a.Speed;
		a.Target = a.Target + 1;
		a.Flags = a.Flags | 1u;
		agents[i] = a;
	}
	if (i < arrayLength(&mixed)) {
		var m = mixed[i];
		m.Location = m.Location + m.Velocity;
		m.bActive = 1u - m.bActive;
		m.Team = m.Team + 1u;
		for (var k = 0; k < 3; k = k + 1) { m.Values[k] = m.Values[k] * 2.0; }
		m.Inner.A = m.Color.r;
		m.Inner.B = m.Inner.B * 2.0;
		m.Small = -m.Small;
		mixed[i] = m;
	}
}
)");
	TSharedRef<FWebGPUKernel> Kernel = Context->CreateKernel(Desc, nullptr, EWebGPUCallbackThread::WorkerThread);

	const int32 Count = 1000;
	TArray<FWebGPUTestAgent> Agents;
	TArray<FWebGPUTestMixed> Mixed;
	for (int32 Index = 0; Index < Count; Index++)
	{
		Agents.Add(MakeAgent(Index));
		Mixed.Add(MakeMixed(Index));
	}

	TSharedPtr<FWebGPUBuffer> AgentBuffer = Context->CreateStructBuffer(FWebGPUTestAgent::StaticStruct(), Count);
	TSharedPtr<FWebGPUBuffer> MixedBuffer = Context->CreateStructBuffer(FWebGPUTestMixed::StaticStruct(), Count);
	if (!TestTrue(TEXT("Struct buffers"), AgentBuffer.IsValid() && MixedBuffer.IsValid()))
	{
		return false;
	}
	TestEqual(TEXT("Agent buffer element count"), AgentBuffer->GetElementCount(), Count);
	TestTrue(TEXT("Write agents"), Context->WriteStructArray(AgentBuffer.ToSharedRef(), Agents));
	TestTrue(TEXT("Write mixed"), Context->WriteStructArray(MixedBuffer.ToSharedRef(), Mixed));

	Context->SetBufferByName(Kernel, TEXT("agents"), AgentBuffer);
	Context->SetBufferByName(Kernel, TEXT("mixed"), MixedBuffer);
	if (!TestTrue(TEXT("Compiled"), WaitReady(Kernel)))
	{
		AddError(FWebGPUDevice::FormatCompileMessages(Kernel->GetMessages()));
		AddInfo(Kernel->GetCompiledSource());
		return false;
	}

	FWebGPUEntryPointInfo Info;
	TestTrue(TEXT("Entry point reflected"), Kernel->GetEntryPointInfo(Info) && Info.WorkgroupSize == FIntVector(64, 1, 1));

	const FWebGPUStats Before = Context->GetStats();
	Context->DispatchForCount(Kernel, Count);

	TArray<uint8> AgentBytes;
	TArray<uint8> MixedBytes;
	if (!TestTrue(TEXT("Read agents"), ReadAll(*Context, AgentBuffer.ToSharedRef(), AgentBytes)) || !TestTrue(TEXT("Read mixed"), ReadAll(*Context, MixedBuffer.ToSharedRef(), MixedBytes)))
	{
		return false;
	}
	const FWebGPUStats After = Context->GetStats();
	TestEqual(TEXT("One dispatch"), After.Dispatches - Before.Dispatches, (int64)1);
	TestEqual(TEXT("Nothing skipped"), After.SkippedDispatches - Before.SkippedDispatches, (int64)0);

	TArray<FWebGPUTestAgent> AgentsOut;
	AgentsOut.SetNum(Count);
	FWebGPUStructLayout::Get(FWebGPUTestAgent::StaticStruct())->Unpack(AgentBytes.GetData(), Count, AgentsOut.GetData());
	TArray<FWebGPUTestMixed> MixedOut;
	MixedOut.SetNum(Count);
	FWebGPUStructLayout::Get(FWebGPUTestMixed::StaticStruct())->Unpack(MixedBytes.GetData(), Count, MixedOut.GetData());

	int32 AgentErrors = 0;
	int32 MixedErrors = 0;
	for (int32 Index = 0; Index < Count; Index++)
	{
		const FWebGPUTestAgent& In = Agents[Index];
		const FWebGPUTestAgent& Out = AgentsOut[Index];
		const FVector3f ExpectedPosition = In.Position + FVector3f(In.Heading.X, In.Heading.Y, 0.f) * In.Speed;
		AgentErrors += (Out.Position.Equals(ExpectedPosition, 1e-4f) && Out.Target == In.Target + 1 && Out.Flags == (In.Flags | 1u) && Out.Speed == In.Speed) ? 0 : 1;

		const FWebGPUTestMixed& M = Mixed[Index];
		const FWebGPUTestMixed& R = MixedOut[Index];
		const bool bOk = R.Location.Equals(M.Location + FVector(M.Velocity), 1e-2) && R.bActive == !M.bActive && R.Team == M.Team + 1
			&& R.Values[0] == M.Values[0] * 2.f && R.Values[1] == M.Values[1] * 2.f && R.Values[2] == M.Values[2] * 2.f
			&& R.Inner.A == M.Color.R && R.Inner.B == M.Inner.B * 2.f && R.Small == -M.Small && R.Color == M.Color && R.Velocity == M.Velocity;
		MixedErrors += bOk ? 0 : 1;
	}
	TestEqual(TEXT("Agents updated on the GPU (memcpy layout)"), AgentErrors, 0);
	TestEqual(TEXT("Mixed updated on the GPU (converted layout)"), MixedErrors, 0);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUBindingValidationTest, "WebGPU.Binding.Validation", WebGPUBindingTests::Flags)
bool FWebGPUBindingValidationTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
	if (!TestTrue(TEXT("Context"), Context.IsValid()))
	{
		return false;
	}

	const FString Source = TEXT(R"(
struct Params { scale: f32, count: u32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	if (id.x < params.count) { data[id.x] = data[id.x] * params.scale; }
}
)");
	TSharedRef<FWebGPUKernel> Kernel = Context->CreateKernel(Source, TEXT("main"), TEXT("validated"));
	TSharedRef<FWebGPUBuffer> Params = Context->CreateBuffer(16, EWebGPUBufferUsage::Uniform, TEXT("params"));
	TSharedRef<FWebGPUBuffer> StorageParams = Context->CreateBuffer(16, EWebGPUBufferUsage::Storage, TEXT("storage-params"));
	TSharedRef<FWebGPUBuffer> Data = Context->CreateBuffer(256 * sizeof(float), EWebGPUBufferUsage::Storage, TEXT("data"));
	TSharedRef<FWebGPUBuffer> Tiny = Context->CreateBuffer(4, EWebGPUBufferUsage::Uniform, TEXT("tiny"));
	if (!TestTrue(TEXT("Compiled"), WaitReady(Kernel)))
	{
		return false;
	}

	const struct { float Scale; uint32 Count; uint32 Pad[2]; } ParamData = { 3.f, 256, { 0, 0 } };
	Context->WriteBuffer(Params, 0, &ParamData, sizeof(ParamData));
	TArray<float> Ones;
	Ones.Init(1.f, 256);
	Context->WriteBuffer(Data, 0, Ones.GetData(), Ones.Num() * sizeof(float));

	auto Expect = [this](const TCHAR* Pattern) { AddExpectedMessage(Pattern, ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false); };
	auto Skipped = [&Context]() { return Context->GetStats().SkippedDispatches; };
	auto Failed = [&Context]() { return Context->GetStats().FailedSubmits; };

	// 1. Missing binding: skipped with a warning instead of rejecting the batch
	Expect(TEXT("'data' (@group(0) @binding(1)) is not bound"));
	Context->SetBufferByName(Kernel, TEXT("params"), Params);
	int64 SkippedBefore = Skipped();
	const int64 FailedBefore = Failed();
	Context->DispatchForCount(Kernel, 256);
	Context->Flush();
	TestEqual(TEXT("Missing binding skips the dispatch"), Skipped() - SkippedBefore, (int64)1);

	// 2. A name the shader doesn't use only warns; extra bindings are ignored
	Expect(TEXT("uses no binding named 'nonexistent'"));
	Context->SetBufferByName(Kernel, TEXT("data"), Data);
	Context->SetBufferByName(Kernel, TEXT("nonexistent"), Data);
	Context->SetBuffer(Kernel, 3, 7, Data);
	SkippedBefore = Skipped();
	Context->DispatchForCount(Kernel, 256);
	TArray<uint8> Result;
	TestTrue(TEXT("Read"), ReadAll(*Context, Data, Result));
	TestEqual(TEXT("Dispatched despite extras"), Skipped() - SkippedBefore, (int64)0);
	TestEqual(TEXT("data == 3"), reinterpret_cast<const float*>(Result.GetData())[255], 3.f);

	// 3. Too small, and 4. wrong usage
	Expect(TEXT("'params' needs at least 8 bytes, buffer 'tiny' provides 4"));
	Context->SetBufferByName(Kernel, TEXT("params"), Tiny);
	SkippedBefore = Skipped();
	Context->DispatchForCount(Kernel, 256);
	Context->Flush();
	TestEqual(TEXT("Too-small buffer skips"), Skipped() - SkippedBefore, (int64)1);

	Expect(TEXT("was not created with the Uniform usage"));
	Context->SetBufferByName(Kernel, TEXT("params"), StorageParams);
	SkippedBefore = Skipped();
	Context->DispatchForCount(Kernel, 256);
	Context->Flush();
	TestEqual(TEXT("Wrong usage skips"), Skipped() - SkippedBefore, (int64)1);
	TestEqual(TEXT("No batch was rejected by WebGPU"), Failed() - FailedBefore, (int64)0);

	// 5. A struct buffer whose UStruct differs from the shader's struct warns
	Expect(TEXT("holds WebGPUTestAgent but the shader reads Agent"));
	TSharedRef<FWebGPUKernel> AgentKernel = Context->CreateKernel(TEXT(R"(
struct Agent { pos: vec3<f32>, speed: f32 };
@group(0) @binding(0) var<storage, read_write> agents: array<Agent>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	if (id.x < arrayLength(&agents)) { agents[id.x].speed = 1.0; }
}
)"), TEXT("main"), TEXT("mismatch"));
	TSharedPtr<FWebGPUBuffer> Agents = Context->CreateStructBuffer(FWebGPUTestAgent::StaticStruct(), 64);
	Context->SetBufferByName(AgentKernel, TEXT("agents"), Agents);
	TestTrue(TEXT("Mismatch kernel compiled"), WaitReady(AgentKernel));
	Context->DispatchForCount(AgentKernel, 64);
	Context->Flush();

	// 6. Errors in the source report source lines even with a struct preamble
	FWebGPUKernelDesc Desc;
	Desc.Structs = { FWebGPUTestMixed::StaticStruct() };
	Desc.Source = TEXT("@group(0) @binding(0) var<storage, read_write> m: array<WebGPUTestMixed>;\n@compute @workgroup_size(1)\nfn main() {\n\tm[0].Team = undefined_thing;\n}\n");
	TSharedRef<FWebGPUKernel> Broken = Context->CreateKernel(Desc, nullptr, EWebGPUCallbackThread::WorkerThread);
	TestTrue(TEXT("Broken kernel reported"), WaitUntil([&] { return Broken->GetMessages().Num() > 0; }));
	const TArray<FWebGPUCompileMessage> BrokenMessages = Broken->GetMessages();
	const FWebGPUCompileMessage* Error = BrokenMessages.FindByPredicate([](const FWebGPUCompileMessage& M) { return M.bIsError; });
	TestTrue(TEXT("Error on source line 4"), Error && Error->Line == 4);
	TestTrue(TEXT("Preamble kept separate"), Broken->GetPreamble().Contains(TEXT("struct WebGPUTestMixed")));

	// 7. Unknown entry point names the ones that exist
	TSharedRef<FWebGPUKernel> NoEntry = Context->CreateKernel(Source, TEXT("missing_entry"), TEXT("no-entry"));
	TestTrue(TEXT("Missing entry point reported"), WaitUntil([&] { return NoEntry->GetMessages().Num() > 0; }));
	const TArray<FWebGPUCompileMessage> EntryMessages = NoEntry->GetMessages();
	TestTrue(TEXT("Message lists main"), EntryMessages.Num() > 0 && EntryMessages.Last().Message.Contains(TEXT("compute entry points: main")));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUBlueprintStructNodesTest, "WebGPU.Binding.BlueprintStructNodes", WebGPUBindingTests::Flags)
bool FWebGPUBlueprintStructNodesTest::RunTest(const FString& Parameters)
{
	// The wildcard nodes' implementations, driven through a real TArray property
	const FArrayProperty* MixedProperty = FindFProperty<FArrayProperty>(FWebGPUTestArrays::StaticStruct(), TEXT("Mixed"));
	if (!TestNotNull(TEXT("Array property"), MixedProperty))
	{
		return false;
	}

	UWebGPUBuffer* Buffer = UWebGPUBlueprintLibrary::CreateWebGPUStructBuffer(FWebGPUTestMixed::StaticStruct(), 16);
	if (!TestNotNull(TEXT("Struct buffer"), Buffer))
	{
		return false;
	}
	TestEqual(TEXT("Element count"), Buffer->GetElementCount(), 16);
	TestEqual(TEXT("Element stride"), Buffer->GetElementStride(), 112);

	FWebGPUTestArrays Input;
	for (int32 Index = 0; Index < 16; Index++)
	{
		Input.Mixed.Add(MakeMixed(Index));
	}
	TestTrue(TEXT("Write struct array"), UWebGPUBlueprintLibrary::GenericWriteStructArray(Buffer, MixedProperty, &Input.Mixed, 0));

	TArray<uint8> Bytes;
	if (!TestTrue(TEXT("Read"), ReadAll(*UWebGPUSubsystem::GetSharedContext(), Buffer->GetNative().ToSharedRef(), Bytes)))
	{
		return false;
	}
	FWebGPUTestArrays Output;
	TestTrue(TEXT("Bytes to struct array"), UWebGPUBlueprintLibrary::GenericBytesToStructArray(Bytes, MixedProperty, &Output.Mixed));
	TestEqual(TEXT("Count"), Output.Mixed.Num(), 16);
	int32 Errors = 0;
	for (int32 Index = 0; Index < FMath::Min(Output.Mixed.Num(), 16); Index++)
	{
		Errors += (Output.Mixed[Index].Team == Input.Mixed[Index].Team && Output.Mixed[Index].Small == Input.Mixed[Index].Small
			&& Output.Mixed[Index].Location.Equals(Input.Mixed[Index].Location, 1e-3)) ? 0 : 1;
	}
	TestEqual(TEXT("Round trip through the GPU"), Errors, 0);
	TestTrue(TEXT("WGSL for struct"), UWebGPUBlueprintLibrary::GetWGSLForStruct(FWebGPUTestAgent::StaticStruct()).Contains(TEXT("struct WebGPUTestAgent")));
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
