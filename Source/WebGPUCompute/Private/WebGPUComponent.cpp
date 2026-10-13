#include "WebGPUComponent.h"
#include "WebGPUCompute.h"
#include "WebGPUDevice.h"
#include "WebGPUSubsystem.h"
#include "FlopBenchmark.h"

namespace
{
	const TCHAR* CollatzShaderSource = TEXT(R"(
@group(0)
@binding(0)
var<storage, read_write> v_indices: array<u32>; // this is used as both input and output for convenience

// The Collatz Conjecture states that for any integer n:
// If n is even, n = n/2
// If n is odd, n = 3n+1
// And repeat this process for each new n, you will always eventually reach 1.
// Though the conjecture has not been proven, no counterexample has ever been found.
// This function returns how many times this recurrence needs to be applied to reach 1.
fn collatz_iterations(n_base: u32) -> u32 {
	var n : u32 = n_base;
	var i : u32 = 0u;
	loop{
		if (n <= 1u) {
			break;
		}
		if (n % 2u == 0u) {
			n = n / 2u;
		}
		else {
			// Overflow? (i.e. 3*n + 1 > 0xffffffffu?)
			if (n >= 1431655765u) {   // 0x55555555u
				return 4294967295u;   // 0xffffffffu
			}

			n = 3u * n + 1u;
		}
		i = i + 1u;
	}
	return i;
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	v_indices[global_id.x] = collatz_iterations(v_indices[global_id.x]);
}
)");
}

UWebGPUComponent::UWebGPUComponent(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
}

void UWebGPUComponent::Test()
{
	TArray<int32> Data = { 1, 2, 3, 4 };
	TArray<int32> Result;
	RunShader(CollatzShaderSource, Data, Result);

	FString Values;
	for (int32 Value : Result)
	{
		Values += FString::Printf(TEXT("%d,"), Value);
	}
	UE_LOG(LogWebGPU, Log, TEXT("Collatz test output: [%s]"), *Values);
}

void UWebGPUComponent::RunShader(const FString& ShaderSource, const TArray<int32>& InData, TArray<int32>& OutData)
{
	TSharedPtr<FWebGPUDevice> Device = UWebGPUSubsystem::GetSharedDevice();
	if (!Device)
	{
		UE_LOG(LogWebGPU, Error, TEXT("RunShader: WebGPU device unavailable"));
		return;
	}

	// NB: shader uses u32; int32 bits pass through unchanged
	TArray<uint32> In(reinterpret_cast<const uint32*>(InData.GetData()), InData.Num());
	TArray<uint32> Out;
	FString Error;
	if (!Device->RunBufferKernel(ShaderSource, In, Out, Error))
	{
		UE_LOG(LogWebGPU, Error, TEXT("RunShader failed:\n%s"), *Error);
		return;
	}
	OutData = TArray<int32>(reinterpret_cast<const int32*>(Out.GetData()), Out.Num());
}

void UWebGPUComponent::PrintCPUInfo()
{
	FFlopBenchmark Bench;

	Bench.PrintCPUIDInfo();
}

void UWebGPUComponent::BenchmarkFlops(int32 Threads, int64 Iterations, bool bTestAVX, bool bAVX512)
{
	FFlopBenchmark Bench;

	if (bTestAVX)
	{
		if (bAVX512)
		{
			Bench.BenchmarkAVX512(Iterations);
		}
		else
		{
			Bench.BenchmarkAVX2(Iterations);
		}
	}
	else
	{
		Bench.BenchmarkCPUScalar(Threads, Iterations);
	}
}
