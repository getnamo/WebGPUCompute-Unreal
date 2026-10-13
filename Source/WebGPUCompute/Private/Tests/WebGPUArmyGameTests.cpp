// Copyright 2025-current Getnamo. All Rights Reserved.

// The army example in a game (a packaged build, or the editor with -game): opens /WebGPUCompute/Examples/Army/ArmyExample
// and checks that every agent spawned and the armies march, i.e. the cooked Blueprint, the staged .wgsl files and
// the shared render targets work outside the editor. Needs a real RHI; a screenshot lands in Saved/Screenshots.

#include "CoreMinimal.h"
#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "WebGPUBindings.h"
#include "WebGPUBuffer.h"
#include "WebGPUKernel.h"
#include "DynamicRHI.h"
#include "EngineUtils.h"
#include "Engine/World.h"
#include "Tests/AutomationCommon.h"
#include "UnrealClient.h"

namespace WebGPUArmyGameTests
{
	struct FAgentView
	{
		FVector2f Pos;
		FVector2f Vel;
		uint32 Team;
		uint32 Goal;
		uint32 Seed;
		uint32 State;
	};
	static_assert(sizeof(FAgentView) == 32);

	struct FState
	{
		double Start = 0.0;
		double Found = 0.0;
		double Done = 0.0;
		TWeakObjectPtr<AActor> Army;
		bool bReading = false;
		bool bRead = false;
		TArray<uint8> Agents;
	};

	template <typename T>
	T* GetMember(AActor* Actor, const TCHAR* Name)
	{
		const FObjectProperty* Property = FindFProperty<FObjectProperty>(Actor->GetClass(), Name);
		return Property ? Cast<T>(Property->GetObjectPropertyValue_InContainer(Actor)) : nullptr;
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUArmyGameTest, "WebGPU.Examples.ArmyGame", EAutomationTestFlags::ClientContext | EAutomationTestFlags::ProductFilter)
bool FWebGPUArmyGameTest::RunTest(const FString&)
{
	using namespace WebGPUArmyGameTests;
	if (!FApp::CanEverRender() || !GDynamicRHI || FCString::Strcmp(GDynamicRHI->GetName(), TEXT("Null")) == 0)
	{
		AddWarning(TEXT("needs a real RHI (not -nullrhi): skipped"));
		return true;
	}

	AutomationOpenMap(TEXT("/WebGPUCompute/Examples/Army/ArmyExample"));
	TSharedRef<FState> S = MakeShared<FState>();
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([this, S]() -> bool
	{
		const double Now = FPlatformTime::Seconds();
		if (S->Start == 0.0)
		{
			S->Start = Now;
		}
		if (Now - S->Start > 60.0)
		{
			AddError(TEXT("the army didn't come up within 60 s"));
			return true;
		}
		UWorld* World = AutomationCommon::GetAnyGameWorld();
		if (!World)
		{
			return false;
		}
		if (!S->Army.IsValid())
		{
			for (TActorIterator<AActor> It(World); It; ++It)
			{
				const FString ClassName = It->GetClass()->GetName();
				if (ClassName.StartsWith(TEXT("BP_WebGPUArmy")) && !ClassName.StartsWith(TEXT("BP_WebGPUArmyJs")))
				{
					S->Army = *It;
					S->Found = Now;
				}
			}
			return false;
		}

		// Four seconds of marching, then one read of the agents
		if (Now - S->Found < 4.0)
		{
			return false;
		}
		if (!S->bRead)
		{
			UWebGPUBindings* Bindings = GetMember<UWebGPUBindings>(S->Army.Get(), TEXT("Bindings"));
			UWebGPUBuffer* Agents = Bindings ? Bindings->GetBuffer(TEXT("agents")) : nullptr;
			if (!Agents)
			{
				AddError(TEXT("the army has no agents buffer"));
				return true;
			}
			if (!S->bReading)
			{
				S->bReading = true;
				Agents->Read(0, -1, [S](bool bSuccess, TArray<uint8>&& Data)
				{
					S->Agents = MoveTemp(Data);
					S->bRead = true;
				});
			}
			return false;
		}

		if (S->Done == 0.0)
		{
			const FAgentView* Agents = reinterpret_cast<const FAgentView*>(S->Agents.GetData());
			const int32 Count = S->Agents.Num() / (int32)sizeof(FAgentView);
			int32 Spawned = 0;
			double Speed = 0.0;
			for (int32 Index = 0; Index < Count; Index++)
			{
				Spawned += Agents[Index].State != 0 ? 1 : 0;
				Speed += Agents[Index].Vel.Size();
			}
			Speed /= FMath::Max(Count, 1);
			AddInfo(FString::Printf(TEXT("%d agents, %d spawned, marching at %.1f cells/s"), Count, Spawned, Speed));
			TestEqual(TEXT("131072 agents"), Count, 131072);
			TestEqual(TEXT("All spawned"), Spawned, Count);
			TestTrue(TEXT("Marching"), Speed > 6.0);
			FScreenshotRequest::RequestScreenshot(TEXT("WebGPU_ArmyGame.png"), false, false);
			S->Done = Now;
		}
		// A moment for the screenshot
		return Now - S->Done > 0.5;
	}));
	return true;
}

#endif
