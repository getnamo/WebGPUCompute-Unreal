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

void FFlopBenchmark::PrintCPUIDInfo()
{
	//from https://gist.github.com/boxmein/7d8e5fae7febafc5851e
#if PLATFORM_WINDOWS
	int cpuinfo[4];
	UE_LOG(LogFlopBenchmark, Log, TEXT("CPU identification thing"));
	UE_LOG(LogFlopBenchmark, Log, TEXT("prints out various data the cpuid instruction returns"));
	UE_LOG(LogFlopBenchmark, Log, TEXT("instructions marked with an asterisk (*) are Intel-only"));
	UE_LOG(LogFlopBenchmark, Log, TEXT("approximated from http://msdn.microsoft.com/en-us/library/hskdteyh(v=vs.90).aspx\n"));

	__cpuid(cpuinfo, 0);
	UE_LOG(LogFlopBenchmark, Log, TEXT("- __cpuid(0) -"));

	char ident[13] = {
		static_cast<char>(cpuinfo[1] & 0xFF), static_cast<char>((cpuinfo[1] >> 8) & 0xFF), static_cast<char>((cpuinfo[1] >> 16) & 0xFF), static_cast<char>((cpuinfo[1] >> 24) & 0xFF),
		static_cast<char>(cpuinfo[3] & 0xFF), static_cast<char>((cpuinfo[3] >> 8) & 0xFF), static_cast<char>((cpuinfo[3] >> 16) & 0xFF), static_cast<char>((cpuinfo[3] >> 24) & 0xFF),
		static_cast<char>(cpuinfo[2] & 0xFF), static_cast<char>((cpuinfo[2] >> 8) & 0xFF), static_cast<char>((cpuinfo[2] >> 16) & 0xFF), static_cast<char>((cpuinfo[2] >> 24) & 0xFF),
		0
	};

	UE_LOG(LogFlopBenchmark, Log, TEXT("  ident string: %hs"), ident);

	__cpuid(cpuinfo, 1);
	UE_LOG(LogFlopBenchmark, Log, TEXT("\n- __cpuid(1) eax -"));
	UE_LOG(LogFlopBenchmark, Log, TEXT("  stepping id: %d"), cpuinfo[0] & 0xf);
	UE_LOG(LogFlopBenchmark, Log, TEXT("  model: %d"), (cpuinfo[0] >> 4) & 0xf);
	UE_LOG(LogFlopBenchmark, Log, TEXT("  family: %d"), (cpuinfo[0] >> 8) & 0xf);
	UE_LOG(LogFlopBenchmark, Log, TEXT("* processor type: %d"), (cpuinfo[0] >> 12) & 0x3);
	UE_LOG(LogFlopBenchmark, Log, TEXT("  reserved bits 14-15: %d"), (cpuinfo[0] >> 14) & 0x3);
	UE_LOG(LogFlopBenchmark, Log, TEXT("  extended model: %d"), (cpuinfo[0] >> 16) & 0xf);
	UE_LOG(LogFlopBenchmark, Log, TEXT("  extended family: %d"), (cpuinfo[0] >> 20) & 0xff);
	UE_LOG(LogFlopBenchmark, Log, TEXT("  reserved bits 28-31: %d"), (cpuinfo[0] >> 28) & 0xf);

	UE_LOG(LogFlopBenchmark, Log, TEXT("\n- __cpuid(1) ebx -"));
	UE_LOG(LogFlopBenchmark, Log, TEXT("  brand index: %d"), cpuinfo[1] & 0xff);
	UE_LOG(LogFlopBenchmark, Log, TEXT("  clflush cache line size: %d (quadwords)"), (cpuinfo[1] >> 8) & 0xff);
	UE_LOG(LogFlopBenchmark, Log, TEXT("  logical processors: %d"), (cpuinfo[1] >> 16) & 0xff);
	UE_LOG(LogFlopBenchmark, Log, TEXT("  initial apic id: %d"), (cpuinfo[1] >> 24) & 0xff);

	// You can continue similarly with ecx and edx if needed,
	// but this demonstrates the format translation into UE_LOG macros.
#endif
}

void FFlopBenchmark::BenchmarkCPUScalar(int32 ThreadCount, uint64 IterationsPerThread)
{
	auto Start = std::chrono::high_resolution_clock::now();

	std::vector<std::future<double>> futures;
	for (int32 t = 0; t < ThreadCount; ++t) 
	{
		futures.push_back(std::async(std::launch::async, [IterationsPerThread]() 
		{
			float a1 = 1.0f, b1 = 2.0f, c1 = 3.0f;
			float a2 = 4.0f, b2 = 5.0f, c2 = 6.0f;
			float a3 = 7.0f, b3 = 8.0f, c3 = 9.0f;
			float a4 = 10.0f, b4 = 11.0f, c4 = 12.0f;

			for (uint64 j = 0; j < IterationsPerThread; ++j)
			{
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

			volatile float sink = a1 + b1 + c1 + a2 + b2 + c2 + a3 + b3 + c3 + a4 + b4 + c4;
			return static_cast<double>(sink); // placeholder to prevent optimization
		}));
	}

	for (auto& f : futures) f.get();

	auto End = std::chrono::high_resolution_clock::now();
	double Seconds = std::chrono::duration<double>(End - Start).count();
	double Flops = 24.0 * IterationsPerThread * ThreadCount;
	UE_LOG(LogFlopBenchmark, Log, TEXT("[CPU Scalar MT] Elapsed Time: %.6fs"), Seconds);
	UE_LOG(LogFlopBenchmark, Log, TEXT("[CPU Scalar MT] Total FLOPs: %.0f"), Flops);
	UE_LOG(LogFlopBenchmark, Log, TEXT("[CPU Scalar MT] FLOPs/s: %.2f GFLOPs"), (Flops / Seconds) / 1e9);
}

void FFlopBenchmark::BenchmarkAVX2(uint64 Iterations)
{
	if (!SupportsAVX2()) return;

	auto Start = std::chrono::high_resolution_clock::now();

	int32 ThreadCount = std::thread::hardware_concurrency();
	std::vector<std::future<float>> futures;

	for (int32 t = 0; t < ThreadCount; ++t)
	{
		futures.push_back(std::async(std::launch::async, [Iterations]() -> float
			{
				__m256 a = _mm256_set_ps(1, 2, 3, 4, 5, 6, 7, 8);
				__m256 b = _mm256_add_ps(a, _mm256_set1_ps(1.0f));
				__m256 c = _mm256_add_ps(b, _mm256_set1_ps(1.0f));

				for (uint64 i = 0; i < Iterations; ++i)
				{
					a = _mm256_fmadd_ps(a, b, c);
					b = _mm256_fmadd_ps(b, c, a);
					c = _mm256_fmadd_ps(c, a, b);
				}

				__m256 result = _mm256_add_ps(a, _mm256_add_ps(b, c));
				float resultArray[8];
				_mm256_storeu_ps(resultArray, result);
				volatile float sink = 0.0f;
				for (int i = 0; i < 8; ++i)
				{
					sink += resultArray[i];
				}
				return sink;
			}));
	}

	float totalSink = 0.0f;
	for (auto& f : futures)
	{
		totalSink += f.get();
	}

	auto End = std::chrono::high_resolution_clock::now();
	double Seconds = std::chrono::duration<double>(End - Start).count();
	double Flops = 6.0 * 8 * Iterations * ThreadCount;

	UE_LOG(LogFlopBenchmark, Log, TEXT("[CPU AVX2 MT] Elapsed Time: %.6fs"), Seconds);
	UE_LOG(LogFlopBenchmark, Log, TEXT("[CPU AVX2 MT] Total FLOPs: %.0f"), Flops);
	UE_LOG(LogFlopBenchmark, Log, TEXT("[CPU AVX2 MT] FLOPs/s: %.2f GFLOPs"), (Flops / Seconds) / 1e9);
	UE_LOG(LogFlopBenchmark, Verbose, TEXT("[CPU AVX2 MT] Result checksum: %f"), totalSink);
}

void FFlopBenchmark::BenchmarkAVX512(uint64 Iterations)
{
	if (!SupportsAVX512()) return;

	auto Start = std::chrono::high_resolution_clock::now();

	int32 ThreadCount = std::thread::hardware_concurrency();
	std::vector<std::future<float>> futures;

	for (int32 t = 0; t < ThreadCount; ++t)
	{
		futures.push_back(std::async(std::launch::async, [Iterations]() -> float
			{
				__m512 a = _mm512_set_ps(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
				__m512 b = _mm512_add_ps(a, _mm512_set1_ps(1.0f));
				__m512 c = _mm512_add_ps(b, _mm512_set1_ps(1.0f));

				for (uint64 i = 0; i < Iterations; ++i)
				{
					a = _mm512_fmadd_ps(a, b, c);
					b = _mm512_fmadd_ps(b, c, a);
					c = _mm512_fmadd_ps(c, a, b);
				}

				__m512 result = _mm512_add_ps(a, _mm512_add_ps(b, c));
				alignas(64) float resultArray[16];
				_mm512_storeu_ps(resultArray, result);
				volatile float sink = 0.0f;
				for (int i = 0; i < 16; ++i)
				{
					sink += resultArray[i];
				}
				return sink;
			}));
	}

	float totalSink = 0.0f;
	for (auto& f : futures)
	{
		totalSink += f.get();
	}

	auto End = std::chrono::high_resolution_clock::now();
	double Seconds = std::chrono::duration<double>(End - Start).count();
	double Flops = 6.0 * 16 * Iterations * ThreadCount; // 3 FMAs × 2 FLOPs × 16-wide vector

	UE_LOG(LogFlopBenchmark, Log, TEXT("[CPU AVX-512 MT] Elapsed Time: %.6fs"), Seconds);
	UE_LOG(LogFlopBenchmark, Log, TEXT("[CPU AVX-512 MT] Total FLOPs: %.0f"), Flops);
	UE_LOG(LogFlopBenchmark, Log, TEXT("[CPU AVX-512 MT] FLOPs/s: %.2f GFLOPs"), (Flops / Seconds) / 1e9);
	UE_LOG(LogFlopBenchmark, Verbose, TEXT("[CPU AVX-512 MT] Result checksum: %f"), totalSink);
}