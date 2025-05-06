#include "WebGPUComponent.h"
#include "webgpu/webgpu.h"

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
	UE_LOG(LogTemp, Log, TEXT("Test start."));

	//From: https://eliemichel.github.io/LearnWebGPU/getting-started/hello-webgpu.html#lit-6

	// We create a descriptor
	WGPUInstanceDescriptor desc = {};
	desc.nextInChain = nullptr;

	// We create the instance using this descriptor
	WGPUInstance instance = wgpuCreateInstance(&desc);

	// We can check whether there is actually an instance created
	if (!instance) {
		UE_LOG(LogTemp, Error, TEXT("Could not initialize WebGPU!"));
		return;
	}

	// Display the object (WGPUInstance is a simple pointer, it may be
	// copied around without worrying about its size).
	UE_LOG(LogTemp, Log, TEXT("WGPU instance: %d"), instance);

	// We clean up the WebGPU instance
	wgpuInstanceRelease(instance);

	UE_LOG(LogTemp, Log, TEXT("Test end."));
}