#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "WebGPUComponent.generated.h"

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class WEBGPUCOMPUTE_API UWebGPUComponent : public UActorComponent
{
	GENERATED_BODY()

public:	
	UWebGPUComponent();

	//Debug function for first pass test
	UFUNCTION(BlueprintCallable, Category = "Utility")
	void Test();

	UFUNCTION(BlueprintCallable, Category = "Utility")
	void RunShader(const FString& ShaderSource);

protected:
	virtual void BeginPlay() override;

public:	
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;
};
