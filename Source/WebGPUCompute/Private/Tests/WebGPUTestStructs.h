// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "WebGPUTestStructs.generated.h"

/** Test struct whose UE layout equals its WGSL layout (packing is a memcpy): 32 bytes. */
USTRUCT()
struct FWebGPUTestAgent
{
	GENERATED_BODY()

	UPROPERTY()
	FVector3f Position = FVector3f::ZeroVector;

	UPROPERTY()
	float Speed = 0.f;

	UPROPERTY()
	int32 Target = 0;

	UPROPERTY()
	uint32 Flags = 0;

	UPROPERTY()
	FVector2f Heading = FVector2f::ZeroVector;
};

USTRUCT()
struct FWebGPUTestInner
{
	GENERATED_BODY()

	UPROPERTY()
	float A = 0.f;

	UPROPERTY()
	FVector2f B = FVector2f::ZeroVector;
};

/** Test struct that needs conversion: doubles, bool, small ints, a C array and a nested struct. */
USTRUCT()
struct FWebGPUTestMixed
{
	GENERATED_BODY()

	UPROPERTY()
	FVector Location = FVector::ZeroVector;

	UPROPERTY()
	bool bActive = false;

	UPROPERTY()
	uint8 Team = 0;

	UPROPERTY()
	FVector3f Velocity = FVector3f::ZeroVector;

	UPROPERTY()
	FLinearColor Color = FLinearColor::Black;

	UPROPERTY()
	float Values[3] = { 0.f, 0.f, 0.f };

	UPROPERTY()
	FWebGPUTestInner Inner;

	UPROPERTY()
	int16 Small = 0;
};

/** Holds arrays so tests can reach FArrayProperty for the wildcard Blueprint nodes. */
USTRUCT()
struct FWebGPUTestArrays
{
	GENERATED_BODY()

	UPROPERTY()
	TArray<FWebGPUTestAgent> Agents;

	UPROPERTY()
	TArray<FWebGPUTestMixed> Mixed;
};
