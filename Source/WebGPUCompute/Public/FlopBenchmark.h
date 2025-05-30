#pragma once

class FFlopBenchmark
{
public:
	float GetCPUFrequencyMHz();
	bool SupportsAVX2();
	bool SupportsAVX512();
	void BenchmarkCPUScalar(int32 ThreadCount, uint64 IterationsPerThread);
	void BenchmarkAVX2(uint64 Iterations);
};