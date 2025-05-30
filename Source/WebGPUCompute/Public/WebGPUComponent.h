#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "WebGPUComponent.generated.h"

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class WEBGPUCOMPUTE_API UWebGPUComponent : public UActorComponent
{
	GENERATED_BODY()

public:	
	UWebGPUComponent(const FObjectInitializer& ObjectInitializer);
	~UWebGPUComponent();

	//Debug function for first pass test
	UFUNCTION(BlueprintCallable, Category = "Utility")
	void Test();

	UFUNCTION(BlueprintCallable, Category = "Utility")
	void PrintCPUInfo();

	UFUNCTION(BlueprintCallable, Category = "Utility")
	void BenchmarkFlops(int32 Threads = 1, int64 Iterations = 1000000, bool bTestAVX = false, bool bAVX512 = false);

	//Example shader with int array data in/out bind
	//Todo: generalize data binding (auto generate binds)
	UFUNCTION(BlueprintCallable, Category = "Utility")
	void RunShader(const FString& ShaderSource, const TArray<int32>& InData, TArray<int32>& OutData);

protected:
	virtual void BeginPlay() override;

public:	
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

protected:

	class FWebGPUInternal* Internal = nullptr;
};
