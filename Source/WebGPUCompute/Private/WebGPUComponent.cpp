#include "WebGPUComponent.h"

UWebGPUComponent::UWebGPUComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
}

void UWebGPUComponent::BeginPlay()
{
	Super::BeginPlay();

	// Initialization logic
}

void UWebGPUComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// Per-frame logic
}


void UWebGPUComponent::Test()
{
	UE_LOG(LogTemp, Log, TEXT("Test start.");

	//moar logic here

	UE_LOG(LogTemp, Log, TEXT("Test end.");
}