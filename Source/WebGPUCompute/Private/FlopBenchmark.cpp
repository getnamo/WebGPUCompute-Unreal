#include "FlopBenchmark.h"

// gpu_flops_benchmark.cpp
#include <chrono>
#include <vector>
#include <cmath>
#include <immintrin.h>
#include <thread>
#include <future>

#include "HAL/PlatformMisc.h"
#include "GenericPlatform/GenericPlatformMisc.h"
#include "Misc/OutputDeviceDebug.h"
#include "Misc/Paths.h"
#include "Logging/LogMacros.h"
#include "Logging/LogVerbosity.h"

DEFINE_LOG_CATEGORY_STATIC(LogFlopBenchmark, Log, All);

#if PLATFORM_WINDOWS
#include <windows.h>
#endif

float FFlopBenchmark::GetCPUFrequencyMHz()
{
#if PLATFORM_WINDOWS
	HKEY hKey;
	DWORD data;
	DWORD dataSize = sizeof(data);

	if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
		"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
		0, KEY_READ, &hKey) == ERROR_SUCCESS)
	{
		if (RegQueryValueExA(hKey, "~MHz", nullptr, nullptr, (LPBYTE)&data, &dataSize) == ERROR_SUCCESS)
		{
			RegCloseKey(hKey);
			return static_cast<float>(data);
		}
		RegCloseKey(hKey);
	}
#endif
	return 0.0f;
}

bool FFlopBenchmark::SupportsAVX2() 
{
#if PLATFORM_ENABLE_VECTORINTRINSICS
	return true;
#else
	return false;
#endif
}

bool FFlopBenchmark::SupportsAVX512() 
{
#if PLATFORM_ENABLE_VECTORINTRINSICS
	return true;
#else
	return false;
#endif
}

void FFlopBenchmark::BenchmarkCPUScalar(int32 ThreadCount, uint64 IterationsPerThread)
{
	float FrequencyMHz = GetCPUFrequencyMHz();
	UE_LOG(LogTemp, Log, TEXT("CPU Frequency: %.2f MHz"), FrequencyMHz);

	auto Start = std::chrono::high_resolution_clock::now();

	float a1 = 1.0f, b1 = 2.0f, c1 = 3.0f;
	float a2 = 4.0f, b2 = 5.0f, c2 = 6.0f;
	float a3 = 7.0f, b3 = 8.0f, c3 = 9.0f;
	float a4 = 10.0f, b4 = 11.0f, c4 = 12.0f;

	for (uint64 j = 0; j < IterationsPerThread; ++j) {
		a1 = a1 * b1 + c1;
		b1 = b1 * c1 + a1;
		c1 = c1 * a1 + b1;

		a2 = a2 * b2 + c2;
		b2 = b2 * c2 + a2;
		c2 = c2 * a2 + b2;

		a3 = a3 * b3 + c3;
		b3 = b3 * c3 + a3;
		c3 = c3 * a3 + b3;

		a4 = a4 * b4 + c4;
		b4 = b4 * c4 + a4;
		c4 = c4 * a4 + b4;
	}

	auto End = std::chrono::high_resolution_clock::now();
	double Seconds = std::chrono::duration<double>(End - Start).count();

	volatile float sink = a1 + b1 + c1 + a2 + b2 + c2 + a3 + b3 + c3 + a4 + b4 + c4;

	double Flops = 24.0 * IterationsPerThread;
	UE_LOG(LogFlopBenchmark, Log, TEXT("[CPU Scalar Optimized] Elapsed Time: %.6fs"), Seconds);
	UE_LOG(LogFlopBenchmark, Log, TEXT("[CPU Scalar Optimized] Total FLOPs: %.0f"), Flops);
	UE_LOG(LogFlopBenchmark, Log, TEXT("[CPU Scalar Optimized] FLOPs/s: %.2f GFLOPs"), (Flops / Seconds) / 1e9);
}

void FFlopBenchmark::BenchmarkAVX2(uint64 Iterations)
{
	if (!SupportsAVX2()) return;

	__m256 a = _mm256_set_ps(1, 2, 3, 4, 5, 6, 7, 8);
	__m256 b = _mm256_add_ps(a, _mm256_set1_ps(1.0f));
	__m256 c = _mm256_add_ps(b, _mm256_set1_ps(1.0f));

	auto Start = std::chrono::high_resolution_clock::now();

	for (uint64 i = 0; i < Iterations; ++i)
	{
		a = _mm256_fmadd_ps(a, b, c);
		b = _mm256_fmadd_ps(b, c, a);
		c = _mm256_fmadd_ps(c, a, b);
	}

	auto End = std::chrono::high_resolution_clock::now();

	// Prevent optimization: use the result
	__m256 result = _mm256_add_ps(a, _mm256_add_ps(b, c));
	float resultArray[8];
	_mm256_storeu_ps(resultArray, result);
	volatile float sink = 0.0f;
	for (int i = 0; i < 8; ++i)
		sink += resultArray[i];

	double Seconds = std::chrono::duration<double>(End - Start).count();
	double Flops = 6.0 * 8 * Iterations;

	UE_LOG(LogFlopBenchmark, Log, TEXT("[CPU AVX2] Elapsed Time: %.6fs"), Seconds);
	UE_LOG(LogFlopBenchmark, Log, TEXT("[CPU AVX2] Total FLOPs: %.0f"), Flops);
	UE_LOG(LogFlopBenchmark, Log, TEXT("[CPU AVX2] FLOPs/s: %.2f GFLOPs"), (Flops / Seconds) / 1e9);
	UE_LOG(LogFlopBenchmark, Verbose, TEXT("[CPU AVX2] Result checksum: %f"), sink);
}