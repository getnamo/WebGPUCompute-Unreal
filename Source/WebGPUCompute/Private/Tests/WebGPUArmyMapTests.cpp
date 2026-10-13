// Copyright 2025-current Getnamo. All Rights Reserved.

// Plays the army example map (/WebGPUCompute/Examples/Army/ArmyExample) in PIE:
// 1. The army marches: every agent spawned, none inside an obstacle the scene capture sees.
// 2. Hot reload: army_behavior.wgsl is edited on disk (SPEED 0) and the running agents slow down; a broken edit
//    keeps the last good version running and shows the error on screen; restoring the file brings the speed back.
// The file is restored however the test ends. Needs a real RHI (-RenderOffscreen) for the shared textures;
// screenshots land in Saved/Screenshots.

#include "CoreMinimal.h"
#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS && WITH_EDITOR

#include "WebGPUBindings.h"
#include "WebGPUBuffer.h"
#include "WebGPUContext.h"
#include "WebGPUKernel.h"
#include "WebGPUShaderFiles.h"
#include "WebGPUTexture.h"
#include "DynamicRHI.h"
#include "Editor.h"
#include "EngineUtils.h"
#include "Engine/World.h"
#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Tests/AutomationCommon.h"
#include "Tests/AutomationEditorCommon.h"
#include "UnrealClient.h"

namespace WebGPUArmyMapTests
{
	constexpr int32 Grid = 256;
	constexpr int32 AgentStride = 32;
	constexpr float CaptureHeight = 2000.f;
	constexpr float BlockHeight = 40.f;
	const TCHAR* BehaviorPath = TEXT("/WebGPUCompute/Examples/Army/army_behavior.wgsl");
	const TCHAR* SpeedLine = TEXT("const SPEED: f32 = 14.0;");

	enum class EPhase : uint8 { FindArmy, Marching, ReadObstacles, Stopping, Broken, Restoring, Done };

	struct FState
	{
		double Start = 0.0;
		double PhaseStart = 0.0;
		/** When the phase's condition was first met (or a request was made). */
		double Now0 = 0.0;
		EPhase Phase = EPhase::FindArmy;
		TWeakObjectPtr<AActor> Army;
		FString OriginalBehavior;
		FString BehaviorFile;
		bool bModified = false;

		// The latest agents readback
		bool bReading = false;
		TArray<uint8> Agents;
		double AgentsTime = 0.0;
		TArray<float> Depth;

		float MarchingSpeed = 0.f;
		float StoppedSpeed = 0.f;
		float RestoredSpeed = 0.f;

		void Restore()
		{
			if (bModified)
			{
				FFileHelper::SaveStringToFile(OriginalBehavior, *BehaviorFile);
				bModified = false;
			}
		}

		~FState()
		{
			Restore();
		}
	};

	template <typename T>
	T* GetMember(AActor* Actor, const TCHAR* Name)
	{
		const FObjectProperty* Property = FindFProperty<FObjectProperty>(Actor->GetClass(), Name);
		return Property ? Cast<T>(Property->GetObjectPropertyValue_InContainer(Actor)) : nullptr;
	}

	struct FAgentView
	{
		FVector2f Pos;
		FVector2f Vel;
		uint32 Team;
		uint32 Goal;
		uint32 Seed;
		uint32 State;
	};
	static_assert(sizeof(FAgentView) == AgentStride);

	TArrayView<const FAgentView> View(const TArray<uint8>& Bytes)
	{
		return TArrayView<const FAgentView>(reinterpret_cast<const FAgentView*>(Bytes.GetData()), Bytes.Num() / AgentStride);
	}

	float MeanSpeed(const TArray<uint8>& Bytes)
	{
		double Sum = 0.0;
		TArrayView<const FAgentView> Agents = View(Bytes);
		for (const FAgentView& Agent : Agents)
		{
			Sum += Agent.Vel.Size();
		}
		return Agents.Num() ? (float)(Sum / Agents.Num()) : 0.f;
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FWebGPUArmyMapTest, "WebGPU.Examples.ArmyMap", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FWebGPUArmyMapTest::RunTest(const FString&)
{
	using namespace WebGPUArmyMapTests;
	if (!FApp::CanEverRender() || !GDynamicRHI || FCString::Strcmp(GDynamicRHI->GetName(), TEXT("Null")) == 0)
	{
		AddWarning(TEXT("needs a real RHI (run with -RenderOffscreen instead of -nullrhi): skipped"));
		return true;
	}

	TSharedRef<FState> S = MakeShared<FState>();
	S->BehaviorFile = FWebGPUShaderFiles::ResolvePath(BehaviorPath);
	if (!FFileHelper::LoadFileToString(S->OriginalBehavior, *S->BehaviorFile) || !S->OriginalBehavior.Contains(SpeedLine))
	{
		AddError(FString::Printf(TEXT("%s should contain '%s'"), *S->BehaviorFile, SpeedLine));
		return false;
	}

	// The broken edit, once per kernel of army_behavior.wgsl (simulate and respawn)
	AddExpectedMessage(TEXT("doesn't compile; the last good version keeps running"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 0, false);
	AddExpectedMessage(TEXT("failed to compile"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 0, false);
	ADD_LATENT_AUTOMATION_COMMAND(FEditorLoadMap(TEXT("/WebGPUCompute/Examples/Army/ArmyExample")));
	ADD_LATENT_AUTOMATION_COMMAND(FStartPIECommand(false));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([this, S]() -> bool
	{
		const double Now = FPlatformTime::Seconds();
		if (S->Start == 0.0)
		{
			S->Start = Now;
			S->PhaseStart = Now;
		}
		auto Next = [S, Now](EPhase Phase) { S->Phase = Phase; S->PhaseStart = Now; S->Now0 = 0.0; };
		auto Fail = [this, S](const FString& Message) { AddError(Message); S->Restore(); return true; };
		if (Now - S->PhaseStart > 30.0)
		{
			return Fail(FString::Printf(TEXT("stuck in phase %d"), (int32)S->Phase));
		}

		UWorld* World = GEditor ? GEditor->PlayWorld.Get() : nullptr;
		if (!World)
		{
			return false;
		}
		AActor* Army = S->Army.Get();
		UWebGPUBindings* Bindings = Army ? GetMember<UWebGPUBindings>(Army, TEXT("Bindings")) : nullptr;
		UWebGPUBuffer* Agents = Bindings ? Bindings->GetBuffer(TEXT("agents")) : nullptr;
		UWebGPUKernel* Simulate = Army ? GetMember<UWebGPUKernel>(Army, TEXT("SimulateKernel")) : nullptr;
		if (S->Phase != EPhase::FindArmy && (!Simulate || !Agents))
		{
			return Fail(TEXT("the army actor or its resources are gone"));
		}

		// Keeps a fresh copy of the agents coming
		auto ReadAgents = [S, Agents]()
		{
			if (Agents && !S->bReading)
			{
				S->bReading = true;
				Agents->Read(0, -1, [S](bool bSuccess, TArray<uint8>&& Data)
				{
					S->bReading = false;
					if (bSuccess)
					{
						S->Agents = MoveTemp(Data);
						S->AgentsTime = FPlatformTime::Seconds();
					}
				});
			}
		};
		auto WriteBehavior = [S](const FString& Text)
		{
			S->bModified = true;
			FFileHelper::SaveStringToFile(Text, *S->BehaviorFile);
		};

		switch (S->Phase)
		{
		case EPhase::FindArmy:
			for (TActorIterator<AActor> It(World); It; ++It)
			{
				if (It->GetClass()->GetName().StartsWith(TEXT("BP_WebGPUArmy")))
				{
					S->Army = *It;
				}
			}
			if (S->Army.IsValid() && Simulate && Simulate->IsReady() && Agents)
			{
				Next(EPhase::Marching);
			}
			return false;

		case EPhase::Marching:
			// Four seconds in: the armies are on their way
			if (Now - S->PhaseStart < 4.0)
			{
				return false;
			}
			if (S->AgentsTime < S->PhaseStart + 4.0)
			{
				ReadAgents();
				return false;
			}
			FScreenshotRequest::RequestScreenshot(TEXT("WebGPU_ArmyExample.png"), false, false);
			S->MarchingSpeed = MeanSpeed(S->Agents);
			Next(EPhase::ReadObstacles);
			return false;

		case EPhase::ReadObstacles:
			if (S->Depth.Num() == 0)
			{
				if (S->Now0 == 0.0)
				{
					UWebGPUTexture* Obstacles = GetMember<UWebGPUTexture>(Army, TEXT("ObstacleTexture"));
					TSharedPtr<FWebGPUTexture> Native = Obstacles ? Obstacles->GetNative() : nullptr;
					TSharedPtr<FWebGPUContext> Context = Native ? Native->GetContext() : nullptr;
					if (!Context)
					{
						return Fail(TEXT("no obstacle texture"));
					}
					S->Now0 = Now;
					Context->BeginTextureAccess(Native.ToSharedRef());
					Context->ReadTexture(Native.ToSharedRef(), [S](bool bSuccess, TArray<uint8>&& Data)
					{
						if (bSuccess && Data.Num() == Grid * Grid * 4)
						{
							S->Depth.SetNumUninitialized(Grid * Grid);
							FMemory::Memcpy(S->Depth.GetData(), Data.GetData(), Data.Num());
						}
					});
					Context->EndTextureAccess(Native.ToSharedRef());
				}
				return false;
			}
			{
				auto Blocked = [S](int32 X, int32 Y)
				{
					const float Depth = S->Depth[Y * Grid + X];
					return Depth > 0.f && Depth < CaptureHeight - BlockHeight;
				};
				// World (X, Y) -> cell: the capture looks down with +X up and +Y right
				auto CellOf = [](float WorldX, float WorldY) { return FIntPoint((int32)((WorldY / 10000.f + 0.5f) * Grid), (int32)((0.5f - WorldX / 10000.f) * Grid)); };
				int32 BlockedCells = 0;
				float MinDepth = MAX_flt;
				float MaxDepth = -MAX_flt;
				for (int32 Index = 0; Index < Grid * Grid; Index++)
				{
					BlockedCells += Blocked(Index % Grid, Index / Grid) ? 1 : 0;
					MinDepth = FMath::Min(MinDepth, S->Depth[Index]);
					MaxDepth = FMath::Max(MaxDepth, S->Depth[Index]);
				}
				AddInfo(FString::Printf(TEXT("Obstacle capture: depth %.1f..%.1f, %d of %d cells blocked"), MinDepth, MaxDepth, BlockedCells, Grid * Grid));
				const FIntPoint WallNear = CellOf(-3400.f, 0.f);
				const FIntPoint Gap = CellOf(-1350.f, 0.f);
				const FIntPoint PillarFar = CellOf(3300.f, 2600.f);
				const FIntPoint Open = CellOf(0.f, -1500.f);
				TestTrue(TEXT("Floor seen at the capture height"), FMath::IsNearlyEqual(MaxDepth, CaptureHeight, 5.f));
				TestTrue(TEXT("The near wall blocks its cells"), Blocked(WallNear.X, WallNear.Y));
				TestTrue(TEXT("The pillar at (3300, 2600) blocks its cells"), Blocked(PillarFar.X, PillarFar.Y));
				TestFalse(TEXT("The gap in the wall is open"), Blocked(Gap.X, Gap.Y));
				TestFalse(TEXT("Open floor is open"), Blocked(Open.X, Open.Y));

				TArrayView<const FAgentView> Marching = View(S->Agents);
				int32 Unspawned = 0;
				int32 Inside = 0;
				for (const FAgentView& Agent : Marching)
				{
					Unspawned += Agent.State == 0 ? 1 : 0;
					const int32 X = FMath::Clamp((int32)Agent.Pos.X, 0, Grid - 1);
					const int32 Y = FMath::Clamp((int32)Agent.Pos.Y, 0, Grid - 1);
					Inside += Blocked(X, Y) ? 1 : 0;
				}
				AddInfo(FString::Printf(TEXT("%d agents marching at %.1f cells/s on average"), Marching.Num(), S->MarchingSpeed));
				TestEqual(TEXT("131072 agents"), Marching.Num(), 131072);
				TestEqual(TEXT("All spawned"), Unspawned, 0);
				TestTrue(FString::Printf(TEXT("Agents inside obstacles: %d"), Inside), Inside < Marching.Num() / 1000);
				TestTrue(TEXT("Marching"), S->MarchingSpeed > 6.f);
			}
			// Hot reload: stop the army by editing the behavior file on disk
			WriteBehavior(S->OriginalBehavior.Replace(SpeedLine, TEXT("const SPEED: f32 = 0.0;")));
			Next(EPhase::Stopping);
			return false;

		case EPhase::Stopping:
			// Reloaded (polled every 0.25 s), then a second for the agents to slow down
			if (!Simulate->GetCompiledSource().Contains(TEXT("const SPEED: f32 = 0.0;")))
			{
				S->Now0 = Now;
				return false;
			}
			if (Now - S->Now0 < 1.5 || S->AgentsTime < S->Now0 + 1.5)
			{
				ReadAgents();
				return false;
			}
			S->StoppedSpeed = MeanSpeed(S->Agents);
			AddInfo(FString::Printf(TEXT("After SPEED = 0 was saved: %.2f cells/s (reloaded %.2f s after the save)"), S->StoppedSpeed, S->Now0 - S->PhaseStart));
			TestTrue(TEXT("The edit slowed the army down"), S->StoppedSpeed < S->MarchingSpeed * 0.3f);
			WriteBehavior(S->OriginalBehavior.Replace(SpeedLine, TEXT("const SPEED: f32 = 14.0 this is not wgsl;")));
			Next(EPhase::Broken);
			return false;

		case EPhase::Broken:
			if (!Simulate->GetCompileMessages().Contains(TEXT("error")))
			{
				S->Now0 = Now;
				return false;
			}
			if (Now - S->Now0 < 0.5)
			{
				return false;
			}
			// With the UI: the error is on screen
			FScreenshotRequest::RequestScreenshot(TEXT("WebGPU_ArmyExample_CompileError.png"), true, false);
			AddInfo(TEXT("Broken edit: ") + Simulate->GetCompileMessages().TrimEnd());
			TestTrue(TEXT("The error names the file"), Simulate->GetCompileMessages().Contains(TEXT("army_behavior.wgsl:")));
			TestTrue(TEXT("The last good kernel stays ready"), Simulate->IsReady());
			S->Restore();
			Next(EPhase::Restoring);
			return false;

		case EPhase::Restoring:
			if (!Simulate->GetCompiledSource().Contains(SpeedLine) || Simulate->GetCompileMessages().Contains(TEXT("error")))
			{
				S->Now0 = Now;
				return false;
			}
			if (Now - S->Now0 < 2.0 || S->AgentsTime < S->Now0 + 2.0)
			{
				ReadAgents();
				return false;
			}
			S->RestoredSpeed = MeanSpeed(S->Agents);
			AddInfo(FString::Printf(TEXT("Restored file: %.1f cells/s"), S->RestoredSpeed));
			TestTrue(TEXT("Restoring the file brought the speed back"), S->RestoredSpeed > S->MarchingSpeed * 0.6f);
			FScreenshotRequest::RequestScreenshot(TEXT("WebGPU_ArmyExample_Restored.png"), false, false);
			Next(EPhase::Done);
			return false;

		default:
			// A little longer so the last screenshot is taken
			if (Now - S->PhaseStart < 0.5)
			{
				return false;
			}
			AddInfo(FString::Printf(TEXT("done after %.1f s"), Now - S->Start));
			return true;
		}
	}));
	ADD_LATENT_AUTOMATION_COMMAND(FEndPlayMapCommand());
	return true;
}

#endif
