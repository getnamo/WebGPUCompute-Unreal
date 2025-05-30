#pragma once

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
	bool SupportsAVX2();
	bool SupportsAVX512();
	void BenchmarkCPUScalar(int32 ThreadCount, uint64 IterationsPerThread);
	void BenchmarkAVX2(uint64 Iterations);
	void BenchmarkAVX512(uint64 Iterations);
};