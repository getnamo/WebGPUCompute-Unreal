// Copyright 2025-current Getnamo. All Rights Reserved.

// Plays the JavaScript army example (/WebGPUCompute/Examples/Army/ArmyExampleJs) in PIE:
// 1. examples/webgpu/army.js runs the army and shows per-team numbers it reads back from the GPU every frame.
// 2. Saving army.js (a new title) reloads the script: its scope frees the old run's GPU objects, the new run
//    makes its own, and the numbers keep coming. The file is restored however the test ends.
// Needs UnrealJs and a real RHI (-RenderOffscreen); screenshots land in Saved/Screenshots.

#include "CoreMinimal.h"
#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS && WITH_EDITOR && WITH_WEBGPU_UNREALJS

#include "WebGPUContext.h"
#include "WebGPUSubsystem.h"
#include "Components/TextRenderComponent.h"
#include "DynamicRHI.h"
#include "Editor.h"
#include "EngineUtils.h"
#include "Engine/World.h"
#include "IV8.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Tests/AutomationCommon.h"
#include "Tests/AutomationEditorCommon.h"
#include "UnrealClient.h"

namespace WebGPUJsArmyMapTests
{
	const TCHAR* TitleLine = TEXT("const TITLE = 'WebGPU army, driven by JavaScript (examples/webgpu/army.js)';");
	const TCHAR* NewTitle = TEXT("WebGPU army, reloaded from a saved script");

	enum class EPhase : uint8 { FindArmy, Running, Reloading, Restoring, Done };

	struct FState
	{
		double Start = 0.0;
		double PhaseStart = 0.0;
		double Since = 0.0;
		EPhase Phase = EPhase::FindArmy;
		TWeakObjectPtr<AActor> Army;
		FString ScriptFile;
		FString Original;
		bool bModified = false;
		FWebGPUStats Running;

		void Restore()
		{
			if (bModified)
			{
				FFileHelper::SaveStringToFile(Original, *ScriptFile);
				bModified = false;
			}
		}

		~FState()
		{
			Restore();
		}
	};

	FString TextOf(AActor* Actor, const TCHAR* Name)
	{
		for (UActorComponent* Component : Actor->GetComponents())
		{
			if (UTextRenderComponent* Text = Cast<UTextRenderComponent>(Component); Text && Text->GetName() == Name)
			{
				return Text->Text.ToString();
			}
		}
		return FString();
	}

	int32 ReloadCount(AActor* Actor)
	{
		const FIntProperty* Property = FindFProperty<FIntProperty>(Actor->GetClass(), TEXT("ReloadCount"));
		return Property ? Property->GetPropertyValue_InContainer(Actor) : -1;
	}

	/** "orange: ... , 11.6 cells/s, ..." -> the speeds of both teams */
	TArray<float> Speeds(const FString& Stats)
	{
		TArray<float> Result;
		int32 Search = 0;
		for (;;)
		{
			const int32 At = Stats.Find(TEXT(" cells/s"), ESearchCase::CaseSensitive, ESearchDir::FromStart, Search);
			if (At == INDEX_NONE)
			{
				return Result;
			}
			int32 Begin = At;
			while (Begin > 0 && (FChar::IsDigit(Stats[Begin - 1]) || Stats[Begin - 1] == TEXT('.')))
			{
				Begin--;
			}
			Result.Add(FCString::Atof(*Stats.Mid(Begin, At - Begin)));
			Search = At + 1;
		}
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUJsArmyMapTest, "WebGPU.Js.ArmyMap", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FWebGPUJsArmyMapTest::RunTest(const FString&)
{
	using namespace WebGPUJsArmyMapTests;
	if (!FApp::CanEverRender() || !GDynamicRHI || FCString::Strcmp(GDynamicRHI->GetName(), TEXT("Null")) == 0)
	{
		AddWarning(TEXT("needs a real RHI (run with -RenderOffscreen instead of -nullrhi): skipped"));
		return true;
	}

	TSharedRef<FState> S = MakeShared<FState>();
	for (const FString& Dir : IV8::Get().GetGlobalScriptSearchPaths())
	{
		const FString Candidate = FPaths::ConvertRelativePathToFull(Dir / TEXT("examples/webgpu/army.js"));
		if (FPaths::FileExists(Candidate))
		{
			S->ScriptFile = Candidate;
		}
	}
	if (S->ScriptFile.IsEmpty() || !FFileHelper::LoadFileToString(S->Original, *S->ScriptFile) || !S->Original.Contains(TitleLine))
	{
		AddError(FString::Printf(TEXT("examples/webgpu/army.js not found in the script paths, or without '%s'"), TitleLine));
		return false;
	}

	ADD_LATENT_AUTOMATION_COMMAND(FEditorLoadMap(TEXT("/WebGPUCompute/Examples/Army/ArmyExampleJs")));
	ADD_LATENT_AUTOMATION_COMMAND(FStartPIECommand(false));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([this, S]() -> bool
	{
		const double Now = FPlatformTime::Seconds();
		if (S->Start == 0.0)
		{
			S->Start = Now;
			S->PhaseStart = Now;
		}
		auto Next = [S, Now](EPhase Phase) { S->Phase = Phase; S->PhaseStart = Now; S->Since = 0.0; };
		if (Now - S->PhaseStart > 40.0)
		{
			AddError(FString::Printf(TEXT("stuck in phase %d"), (int32)S->Phase));
			S->Restore();
			return true;
		}
		UWorld* World = GEditor ? GEditor->PlayWorld.Get() : nullptr;
		if (!World)
		{
			return false;
		}
		if (!S->Army.IsValid())
		{
			for (TActorIterator<AActor> It(World); It; ++It)
			{
				if (It->GetClass()->GetName().StartsWith(TEXT("BP_WebGPUArmyJs")))
				{
					S->Army = *It;
				}
			}
			return false;
		}
		AActor* Army = S->Army.Get();
		const FString Title = TextOf(Army, TEXT("TitleText"));
		const FString Stats = TextOf(Army, TEXT("StatsText"));
		TSharedPtr<FWebGPUContext> GPU = UWebGPUSubsystem::GetSharedContext();

		switch (S->Phase)
		{
		case EPhase::FindArmy:
			// The script runs and its numbers come back from the GPU
			if (!Stats.Contains(TEXT("marching east")))
			{
				return false;
			}
			Next(EPhase::Running);
			return false;

		case EPhase::Running:
		{
			// Four seconds of marching
			if (Now - S->PhaseStart < 4.0)
			{
				return false;
			}
			const TArray<float> TeamSpeeds = Speeds(Stats);
			AddInfo(TEXT("Stats: ") + Stats.Replace(TEXT("<br>"), TEXT(" | ")));
			TestEqual(TEXT("The script's title"), Title, FString(TEXT("WebGPU army, driven by JavaScript (examples/webgpu/army.js)")));
			TestTrue(TEXT("Both armies marching (stats read back by the script)"), TeamSpeeds.Num() == 2 && TeamSpeeds[0] > 6.f && TeamSpeeds[1] > 6.f);
			FScreenshotRequest::RequestScreenshot(TEXT("WebGPU_ArmyExampleJs.png"), false, false);
			S->Running = GPU ? GPU->GetStats() : FWebGPUStats();

			// Save the script with a new title: the file watcher reloads it
			S->bModified = true;
			FFileHelper::SaveStringToFile(S->Original.Replace(TitleLine, *FString::Printf(TEXT("const TITLE = '%s';"), NewTitle)), *S->ScriptFile);
			Next(EPhase::Reloading);
			return false;
		}

		case EPhase::Reloading:
			// The new run is up and its numbers are back
			if (Title != NewTitle || !Stats.Contains(TEXT("script reload #1")))
			{
				S->Since = Now;
				return false;
			}
			if (Now - S->Since < 1.0)
			{
				return false;
			}
			{
				const FWebGPUStats Reloaded = GPU ? GPU->GetStats() : FWebGPUStats();
				AddInfo(FString::Printf(TEXT("Live while running / after the reload: %lld / %lld buffers (%.1f / %.1f MB), %lld / %lld kernels, %lld / %lld textures"),
					S->Running.LiveBuffers, Reloaded.LiveBuffers, S->Running.LiveBufferBytes / 1048576.0, Reloaded.LiveBufferBytes / 1048576.0,
					S->Running.LiveKernels, Reloaded.LiveKernels, S->Running.LiveTextures, Reloaded.LiveTextures));
				TestEqual(TEXT("Reloaded once"), ReloadCount(Army), 1);
				TestTrue(TEXT("The old run's GPU objects were freed (no growth across the reload)"),
					Reloaded.LiveBuffers <= S->Running.LiveBuffers && Reloaded.LiveKernels <= S->Running.LiveKernels && Reloaded.LiveTextures <= S->Running.LiveTextures);
				const TArray<float> TeamSpeeds = Speeds(Stats);
				TestTrue(TEXT("The reloaded script runs the army"), TeamSpeeds.Num() == 2 && TeamSpeeds[0] > 3.f && TeamSpeeds[1] > 3.f);
			}
			FScreenshotRequest::RequestScreenshot(TEXT("WebGPU_ArmyExampleJs_Reloaded.png"), false, false);
			S->Restore();
			Next(EPhase::Restoring);
			return false;

		case EPhase::Restoring:
			if (!Title.Contains(TEXT("driven by JavaScript")) || !Stats.Contains(TEXT("marching east")))
			{
				return false;
			}
			TestEqual(TEXT("Reloaded again after the restore"), ReloadCount(Army), 2);
			Next(EPhase::Done);
			return false;

		default:
			AddInfo(FString::Printf(TEXT("done after %.1f s"), Now - S->Start));
			return true;
		}
	}));
	ADD_LATENT_AUTOMATION_COMMAND(FEndPlayMapCommand());
	return true;
}

#endif
