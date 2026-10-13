// Copyright 2025-current Getnamo. All Rights Reserved.

// WebGPU from JavaScript: webgpu.js (Content/Scripts) in a real UnrealJs context, driven by the engine tick.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS && WITH_WEBGPU_UNREALJS

#include "JavascriptContext.h"
#include "JavascriptIsolate.h"
#include "WebGPUContext.h"
#include "WebGPUSubsystem.h"
#include "Tests/AutomationCommon.h"

namespace WebGPUJsTests
{
	constexpr EAutomationTestFlags Flags = EAutomationTestFlags::EditorContext | EAutomationTestFlags::ClientContext | EAutomationTestFlags::ProductFilter;

	/** An isolate and context for one test, rooted until Teardown. */
	struct FJsEnv
	{
		UJavascriptIsolate* Isolate = nullptr;
		UJavascriptContext* Context = nullptr;

		bool Init()
		{
			Isolate = NewObject<UJavascriptIsolate>(GetTransientPackage());
			Isolate->AddToRoot();
			TMap<FString, FString> Features = UJavascriptIsolate::DefaultIsolateFeatures();
			Isolate->Init(GIsEditor, Features);
			Context = Isolate->CreateContext();
			return Context != nullptr;
		}

		void Teardown()
		{
			Context = nullptr;
			if (Isolate)
			{
				// Dispose the isolate now: full isolates left for a later GC add up (V8 shares one pointer cage)
				Isolate->RemoveFromRoot();
				Isolate = nullptr;
				CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			}
		}

		FString Run(const FString& Script)
		{
			return Context ? Context->RunScript(Script, false) : FString();
		}
	};

	/** Runs Script (which sets result.done) and, once done or after Timeout seconds, hands the env to Check. */
	void RunAsync(FAutomationTestBase& Test, const FString& Script, TFunction<void(FJsEnv&)> Check, double TimeoutSeconds = 20.0)
	{
		TSharedRef<FJsEnv> Env = MakeShared<FJsEnv>();
		if (!Test.TestTrue(TEXT("Script context"), Env->Init()))
		{
			Env->Teardown();
			return;
		}
		Env->Run(TEXT("globalThis.result = { done: false };"));
		Env->Run(Script);
		const double Deadline = FPlatformTime::Seconds() + TimeoutSeconds;
		ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([&Test, Env, Check, Deadline]() -> bool
		{
			if (Env->Run(TEXT("result.done")) != TEXT("true") && FPlatformTime::Seconds() < Deadline)
			{
				return false;
			}
			const FString Error = Env->Run(TEXT("result.error || ''"));
			if (!Error.IsEmpty())
			{
				Test.AddError(TEXT("Script failed: ") + Error);
			}
			Test.TestEqual(TEXT("Script finished"), Env->Run(TEXT("result.done")), FString(TEXT("true")));
			Test.AddInfo(TEXT("result: ") + Env->Run(TEXT("JSON.stringify(result)")));
			Check(*Env);
			Env->Teardown();
			return true;
		}));
	}
}

using namespace WebGPUJsTests;

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUJsComputeTest, "WebGPU.Js.Compute", WebGPUJsTests::Flags)
bool FWebGPUJsComputeTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FWebGPUContext> GPU = UWebGPUSubsystem::GetSharedContext();
	if (!GPU)
	{
		AddError(TEXT("No WebGPU context"));
		return false;
	}
	GPU->Flush();
	const FWebGPUStats Before = GPU->GetStats();

	const FString Script = TEXT(R"js(
globalThis.gpu = require('webgpu');
(async () => {
	result.available = gpu.available();
	const scope = gpu.scope('jstest');
	const bindings = scope.bindings();
	const data = bindings.buffer('data', 256, 4);
	const kernel = scope.kernel({ entry: 'main', bindings, source: `
struct Params { scale: f32, offset: f32 }
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
	if (id.x < arrayLength(&data)) { data[id.x] = data[id.x] * params.scale + params.offset + f32(id.x); }
}` });
	const fileKernel = scope.kernel({ file: '/WebGPUCompute/Examples/Army/army_behavior.wgsl', entry: 'respawn', bindings });

	await kernel.ready;
	result.ready = kernel.isReady;
	await fileKernel.ready;
	result.file = fileKernel.native.GetSourceFile() + ' ' + fileKernel.native.IsHotReloading();

	// Script memory in, uniforms by name, results back as an ArrayBuffer
	gpu.write(data, new Float32Array(256).fill(2));
	bindings.set('params.scale', 3).set('params.offset', 0.5);
	kernel.dispatch(256);
	let out = new Float32Array(await gpu.read(data));
	result.compute = [out[0], out[1], out[255]].join(',');

	// A view into a bigger array, written at byte 16
	gpu.write(data, new Float32Array([1, 2, 30, 40]).subarray(2), 16);
	out = new Float32Array(await gpu.read(data, 16, 8));
	result.subarray = [out[0], out[1]].join(',');

	// Typed arrays into Blueprint functions (TArray<float>)
	data.WriteFloats(new Float32Array([9, 8]), 0);
	out = new Float32Array(await gpu.read(data, 0, 8));
	result.writeFloats = [out[0], out[1]].join(',');

	// A stream frame viewed in place, detached when the reader returns
	const stream = scope.stream(data, { slots: 2, policy: 'Lockstep' });
	stream.capture();
	let kept;
	result.frame = stream.latest((bytes, info) => {
		kept = new Float32Array(bytes);
		return [kept[0], kept[255], info.Bytes, info.Sequence].join(',');
	});
	result.detached = kept.length;

	// A frame held across calls
	stream.capture();
	const held = stream.acquire();
	result.held = held ? new Float32Array(held.bytes)[1] : -1;
	held.release();
	result.heldDetached = held.bytes.byteLength;

	// Releasing the scope frees it all
	scope.release();
	result.released = [kernel.native.IsReady(), fileKernel.native.IsReady(), data.IsBufferValid(), stream.native.GetReadyCount()].join(',');
	scope.release();
	result.done = true;
})().catch((e) => { result.error = String(e.stack || e); result.done = true; });
)js");

	RunAsync(*this, Script, [this, GPU, Before](FJsEnv& Env)
	{
		// Everything the script made is gone once the commands that used it have run
		GPU->Flush();
		const FWebGPUStats After = GPU->GetStats();
		AddInfo(FString::Printf(TEXT("Live before / after: %lld / %lld buffers (%lld / %lld bytes), %lld / %lld kernels"),
			Before.LiveBuffers, After.LiveBuffers, Before.LiveBufferBytes, After.LiveBufferBytes, Before.LiveKernels, After.LiveKernels));
		TestTrue(TEXT("No buffers left behind"), After.LiveBuffers <= Before.LiveBuffers && After.LiveBufferBytes <= Before.LiveBufferBytes);
		TestTrue(TEXT("No kernels left behind"), After.LiveKernels <= Before.LiveKernels);

		TestEqual(TEXT("WebGPUJs available"), Env.Run(TEXT("result.available")), FString(TEXT("true")));
		TestEqual(TEXT("Kernel ready"), Env.Run(TEXT("result.ready")), FString(TEXT("true")));
		TestEqual(TEXT("Kernel from a hot reloading file"), Env.Run(TEXT("result.file")), FString(TEXT("/WebGPUCompute/Examples/Army/army_behavior.wgsl true")));
		TestEqual(TEXT("write, uniforms by name, dispatch, read"), Env.Run(TEXT("result.compute")), FString(TEXT("6.5,7.5,261.5")));
		TestEqual(TEXT("A subarray written at an offset"), Env.Run(TEXT("result.subarray")), FString(TEXT("30,40")));
		TestEqual(TEXT("Float32Array into WriteFloats"), Env.Run(TEXT("result.writeFloats")), FString(TEXT("9,8")));
		TestEqual(TEXT("Stream frame in place"), Env.Run(TEXT("result.frame")), FString(TEXT("9,261.5,1024,1")));
		TestEqual(TEXT("Frame view detached after the reader"), Env.Run(TEXT("result.detached")), FString(TEXT("0")));
		TestEqual(TEXT("Held frame"), Env.Run(TEXT("result.held")), FString(TEXT("8")));
		TestEqual(TEXT("Held frame detached on release"), Env.Run(TEXT("result.heldDetached")), FString(TEXT("0")));
		TestEqual(TEXT("Scope release frees kernels, buffers, streams"), Env.Run(TEXT("result.released")), FString(TEXT("false,false,false,0")));
	});
	return true;
}

#endif
