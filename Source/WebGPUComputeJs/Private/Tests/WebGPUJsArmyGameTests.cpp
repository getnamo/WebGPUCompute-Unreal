// Copyright 2025-current Getnamo. All Rights Reserved.

// The JavaScript army example in a game (a packaged build, or the editor with -game): opens
// /WebGPUCompute/Examples/Army/ArmyExampleJs and checks that army.js (staged with the plugin) runs the army and shows
// the numbers it reads back from the GPU. Needs UnrealJs and a real RHI; a screenshot lands in Saved/Screenshots.

#include "CoreMinimal.h"
#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS && WITH_WEBGPU_UNREALJS

#include "Components/TextRenderComponent.h"
#include "DynamicRHI.h"
#include "EngineUtils.h"
#include "Engine/World.h"
#include "Tests/AutomationCommon.h"
#include "UnrealClient.h"

namespace WebGPUJsArmyGameTests
{
	struct FState
	{
		double Start = 0.0;
		double Running = 0.0;
		double Done = 0.0;
		TWeakObjectPtr<AActor> Army;
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

	/** "... 11.6 cells/s ..." -> the speeds of both teams */
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

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUJsArmyGameTest, "WebGPU.Js.ArmyGame", EAutomationTestFlags::ClientContext | EAutomationTestFlags::ProductFilter)
bool FWebGPUJsArmyGameTest::RunTest(const FString&)
{
	using namespace WebGPUJsArmyGameTests;
	if (!FApp::CanEverRender() || !GDynamicRHI || FCString::Strcmp(GDynamicRHI->GetName(), TEXT("Null")) == 0)
	{
		AddWarning(TEXT("needs a real RHI (not -nullrhi): skipped"));
		return true;
	}

	AutomationOpenMap(TEXT("/WebGPUCompute/Examples/Army/ArmyExampleJs"));
	TSharedRef<FState> S = MakeShared<FState>();
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([this, S]() -> bool
	{
		const double Now = FPlatformTime::Seconds();
		if (S->Start == 0.0)
		{
			S->Start = Now;
		}
		UWorld* World = AutomationCommon::GetAnyGameWorld();
		if (World && !S->Army.IsValid())
		{
			for (TActorIterator<AActor> It(World); It; ++It)
			{
				if (It->GetClass()->GetName().StartsWith(TEXT("BP_WebGPUArmyJs")))
				{
					S->Army = *It;
				}
			}
		}
		const FString Stats = S->Army.IsValid() ? TextOf(S->Army.Get(), TEXT("StatsText")) : FString();
		if (Now - S->Start > 60.0)
		{
			AddError(FString::Printf(TEXT("the script's numbers didn't show within 60 s (stats: '%s')"), *Stats));
			return true;
		}
		if (!Stats.Contains(TEXT("marching east")))
		{
			return false;
		}
		if (S->Running == 0.0)
		{
			S->Running = Now;
		}
		// Four seconds of marching
		if (Now - S->Running < 4.0)
		{
			return false;
		}
		if (S->Done == 0.0)
		{
			const TArray<float> TeamSpeeds = Speeds(Stats);
			AddInfo(TEXT("Stats: ") + Stats.Replace(TEXT("<br>"), TEXT(" | ")));
			TestTrue(TEXT("Both armies marching (numbers read back by army.js)"), TeamSpeeds.Num() == 2 && TeamSpeeds[0] > 6.f && TeamSpeeds[1] > 6.f);
			FScreenshotRequest::RequestScreenshot(TEXT("WebGPU_ArmyGameJs.png"), false, false);
			S->Done = Now;
		}
		return Now - S->Done > 0.5;
	}));
	return true;
}

#endif
