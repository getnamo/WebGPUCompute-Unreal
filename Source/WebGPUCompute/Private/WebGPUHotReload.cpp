// Copyright 2025-current Getnamo. All Rights Reserved.

// Hot reload: kernels compiled from .wgsl files are recompiled when a file they read changes

#include "WebGPUContext.h"
#include "WebGPUCompute.h"
#include "HAL/IConsoleManager.h"
#include "Engine/Engine.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

namespace
{
	TAutoConsoleVariable<float> CVarHotReloadInterval(
		TEXT("webgpu.HotReloadInterval"), 0.25f,
		TEXT("Seconds between checks of .wgsl files watched for hot reload (0 stops checking)."));

	TAutoConsoleVariable<bool> CVarHotReloadMessages(
		TEXT("webgpu.HotReloadMessages"), true,
		TEXT("Show the result of each shader hot reload on screen (development builds)."));

	/** Within this long of a file's time, saves can share that time: compare the text too. */
	constexpr double SameTimeWindowSeconds = 3.0;

	/** Game thread: the outcome of a reload in the log and, in development builds, on screen. */
	void ReportReload(uint64 Key, const FString& Label, const FString& File, bool bSuccess, const TArray<FWebGPUCompileMessage>& Messages)
	{
		const FString Name = FPaths::GetCleanFilename(File);
		FString Text;
		if (bSuccess)
		{
			Text = FString::Printf(TEXT("WebGPU: reloaded %s (%s)"), *Name, *Label);
			UE_LOG(LogWebGPU, Display, TEXT("%s"), *Text);
		}
		else
		{
			Text = FString::Printf(TEXT("WebGPU: %s (%s) doesn't compile; the last good version keeps running"), *Name, *Label);
			UE_LOG(LogWebGPU, Warning, TEXT("%s:\n%s"), *Text, *FWebGPUDevice::FormatCompileMessages(Messages));
			int32 Shown = 0;
			for (const FWebGPUCompileMessage& Message : Messages)
			{
				if (Message.bIsError && Shown++ < 3)
				{
					Text += FString::Printf(TEXT("\n  %s:%llu: %s"), Message.File.IsEmpty() ? *Name : *FPaths::GetCleanFilename(Message.File), Message.Line, *Message.Message);
				}
			}
		}
#if !UE_BUILD_SHIPPING
		if (GEngine && CVarHotReloadMessages.GetValueOnGameThread())
		{
			// One line per kernel: a fix replaces its error
			GEngine->AddOnScreenDebugMessage(Key, bSuccess ? 4.f : 30.f, bSuccess ? FColor::Green : FColor(255, 90, 70), Text);
		}
#endif
	}
}

void FWebGPUContext::WatchKernelFile(const TSharedRef<FWebGPUKernel>& Kernel, const FWebGPUShaderSource& Source)
{
	FScopeLock Lock(&WatchLock);
	FWatchedKernel* Watch = WatchedKernels.FindByPredicate([&Kernel](const FWatchedKernel& Entry) { return Entry.Kernel.HasSameObject(&Kernel.Get()); });
	if (!Watch)
	{
		Watch = &WatchedKernels.AddDefaulted_GetRef();
		Watch->Kernel = Kernel;
	}
	Watch->Files = Source.Map ? Source.Map->Files : TArray<FString>();
	Watch->Read.Reset();
	for (int32 Index = 0; Index < Watch->Files.Num(); Index++)
	{
		FFileSignature& Signature = Watch->Read.AddDefaulted_GetRef();
		Signature.Time = Source.FileTimes.IsValidIndex(Index) ? Source.FileTimes[Index] : FDateTime::MinValue();
		Signature.Hash = Source.FileHashes.IsValidIndex(Index) ? Source.FileHashes[Index] : 0;
		Signature.bHashed = true;
	}
	Watch->Pending.Reset();

	if (!WatchTicker.IsValid())
	{
		TWeakPtr<FWebGPUContext> WeakContext = AsShared();
		WatchTicker = FTSTicker::GetCoreTicker().AddTicker(TEXT("WebGPUHotReload"), 0.f, [WeakContext, NextPoll = 0.0](float) mutable
		{
			const float Interval = CVarHotReloadInterval.GetValueOnGameThread();
			const double Now = FPlatformTime::Seconds();
			if (Interval > 0.f && Now >= NextPoll)
			{
				NextPoll = Now + Interval;
				if (TSharedPtr<FWebGPUContext> Context = WeakContext.Pin())
				{
					Context->PollShaderFiles();
				}
			}
			return true;
		});
	}
}

int32 FWebGPUContext::PollShaderFiles()
{
	// Each file is looked at once per poll, however many kernels include it
	const FDateTime NowUtc = FDateTime::UtcNow();
	TMap<FString, FFileSignature> Seen;
	auto Look = [&Seen, &NowUtc](const FString& File) -> FFileSignature
	{
		if (const FFileSignature* Known = Seen.Find(File))
		{
			return *Known;
		}
		FFileSignature Signature;
		Signature.Time = FWebGPUShaderFiles::GetTimestamp(File);
		if (Signature.Time != FDateTime::MinValue() && (NowUtc - Signature.Time).GetTotalSeconds() < SameTimeWindowSeconds)
		{
			FString Text;
			if (FFileHelper::LoadFileToString(Text, *File))
			{
				Signature.Hash = FCrc::StrCrc32(*Text);
				Signature.bHashed = true;
			}
		}
		return Seen.Add(File, Signature);
	};
	auto AllSame = [](const TArray<FFileSignature>& A, const TArray<FFileSignature>& B)
	{
		if (A.Num() != B.Num())
		{
			return false;
		}
		for (int32 Index = 0; Index < A.Num(); Index++)
		{
			if (!A[Index].SameAs(B[Index]))
			{
				return false;
			}
		}
		return true;
	};

	TArray<TSharedRef<FWebGPUKernel>> Changed;
	{
		FScopeLock Lock(&WatchLock);
		for (int32 Index = WatchedKernels.Num() - 1; Index >= 0; Index--)
		{
			FWatchedKernel& Watch = WatchedKernels[Index];
			TSharedPtr<FWebGPUKernel> Kernel = Watch.Kernel.Pin();
			if (!Kernel)
			{
				WatchedKernels.RemoveAtSwap(Index);
				continue;
			}
			TArray<FFileSignature> Now;
			for (const FString& File : Watch.Files)
			{
				Now.Add(Look(File));
			}
			if (AllSame(Now, Watch.Read))
			{
				Watch.Pending.Reset();
			}
			else if (AllSame(Now, Watch.Pending))
			{
				// Unchanged since the last poll: the save is complete
				Changed.Add(Kernel.ToSharedRef());
				Watch.Pending.Reset();
			}
			else
			{
				Watch.Pending = MoveTemp(Now);
			}
		}
	}

	for (const TSharedRef<FWebGPUKernel>& Kernel : Changed)
	{
		FWebGPUShaderSource Source = FWebGPUShaderFiles::Load(Kernel->GetSourceFile());
		WatchKernelFile(Kernel, Source);
		const uint64 Key = (uint64)(UPTRINT)&Kernel.Get();
		RecompileKernel(Kernel, Source, [Key, Label = Kernel->GetLabel(), File = FWebGPUShaderFiles::ResolvePath(Kernel->GetSourceFile())](bool bSuccess, const TArray<FWebGPUCompileMessage>& Messages)
		{
			ReportReload(Key, Label, File, bSuccess, Messages);
		}, EWebGPUCallbackThread::GameThread);
	}
	return Changed.Num();
}

void FWebGPUContext::SetHotReload(const TSharedRef<FWebGPUKernel>& Kernel, bool bEnabled)
{
	if (!bEnabled)
	{
		FScopeLock Lock(&WatchLock);
		WatchedKernels.RemoveAll([&Kernel](const FWatchedKernel& Entry) { return Entry.Kernel.HasSameObject(&Kernel.Get()); });
		return;
	}
	if (Kernel->GetSourceFile().IsEmpty())
	{
		UE_LOG(LogWebGPU, Warning, TEXT("Kernel '%s' wasn't created from a file; there is nothing to watch"), *Kernel->GetLabel());
		return;
	}
	if (!IsHotReloading(Kernel))
	{
		// Watch from what's on disk now; a later change reloads
		WatchKernelFile(Kernel, FWebGPUShaderFiles::Load(Kernel->GetSourceFile()));
	}
}

bool FWebGPUContext::IsHotReloading(const TSharedRef<FWebGPUKernel>& Kernel) const
{
	FScopeLock Lock(&WatchLock);
	return WatchedKernels.ContainsByPredicate([&Kernel](const FWatchedKernel& Entry) { return Entry.Kernel.HasSameObject(&Kernel.Get()); });
}
