#pragma once

#include "CoreMinimal.h"

/**
* General utility class to identify speed (for now, bandwidth later)
* of hardware currently running, this can give information about
* what kind of additional tuning we want to do.
*/
class FFlopBenchmark
{
public:
	void PrintCPUIDInfo();
	float GetCPUFrequencyMHz();
	/** True when both the CPU and the OS support AVX2 + FMA (x86 only). */
	bool SupportsAVX2();
	/** True when both the CPU and the OS support AVX-512F (x86 only). */
	bool SupportsAVX512();
	void BenchmarkCPUScalar(int32 ThreadCount, uint64 IterationsPerThread);
	/** Logs a warning and returns when SupportsAVX2() is false. */
	void BenchmarkAVX2(uint64 Iterations);
	/** Logs a warning and returns when SupportsAVX512() is false. */
	void BenchmarkAVX512(uint64 Iterations);
};
