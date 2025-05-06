#include "WebGPUComponent.h"
#define WEBGPU_CPP_IMPLEMENTATION
#include "webgpu/webgpu.hpp"

class FWebGPUInternal
{
public:
	WGPUAdapter RequestAdapterSync(WGPUInstance instance, WGPURequestAdapterOptions const* options) 
	{
		// A simple structure holding the local information shared with the
		// onAdapterRequestEnded callback.
		struct UserData {
			WGPUAdapter adapter = nullptr;
			bool requestEnded = false;
		};
		UserData userData;

		// Callback called by wgpuInstanceRequestAdapter when the request returns
		// This is a C++ lambda function, but could be any function defined in the
		// global scope. It must be non-capturing (the brackets [] are empty) so
		// that it behaves like a regular C function pointer, which is what
		// wgpuInstanceRequestAdapter expects (WebGPU being a C API). The workaround
		// is to convey what we want to capture through the pUserData pointer,
		// provided as the last argument of wgpuInstanceRequestAdapter and received
		// by the callback as its last argument.
		auto onAdapterRequestEnded = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* pUserData, void* pUserData2) {
			UserData& userData = *reinterpret_cast<UserData*>(pUserData);
			if (status == WGPURequestAdapterStatus_Success) 
			{
				userData.adapter = adapter;
			}
			else 
			{
				UE_LOG(LogTemp, Error, TEXT("Could not get WebGPU adapter: %hs"), message.data);
			}
			userData.requestEnded = true;
		};

		WGPURequestAdapterCallbackInfo callbackInfo;
		callbackInfo.nextInChain = nullptr;
		callbackInfo.mode = WGPUCallbackMode_WaitAnyOnly;
		callbackInfo.callback = onAdapterRequestEnded;
		callbackInfo.userdata1 = &userData;


		// Call to the WebGPU request adapter procedure
		wgpuInstanceRequestAdapter(
			instance /* equivalent of navigator.gpu */,
			options,
			callbackInfo
		);

		FPlatformProcess::ConditionalSleep([&userData] { return userData.requestEnded; }, 0.01f);

		return userData.adapter;
	}

	void InspectAdapter(WGPUAdapter adapter) 
	{
		WGPULimits limits = {};
		limits.nextInChain = nullptr;

		WGPUStatus status = wgpuAdapterGetLimits(adapter, &limits);

		if (status == WGPUStatus_Success) {
			UE_LOG(LogTemp, Log, TEXT("Adapter limits:"));
			UE_LOG(LogTemp, Log, TEXT(" - maxTextureDimension1D: %u"), limits.maxTextureDimension1D);
			UE_LOG(LogTemp, Log, TEXT(" - maxTextureDimension2D: %u"), limits.maxTextureDimension2D);
			UE_LOG(LogTemp, Log, TEXT(" - maxTextureDimension3D: %u"), limits.maxTextureDimension3D);
			UE_LOG(LogTemp, Log, TEXT(" - maxTextureArrayLayers: %u"), limits.maxTextureArrayLayers);
			UE_LOG(LogTemp, Log, TEXT(" - maxBufferSize: %u"), limits.maxBufferSize);
			UE_LOG(LogTemp, Log, TEXT(" - maxComputeWorkgroupSizeX: %u"), limits.maxComputeWorkgroupSizeX);
			UE_LOG(LogTemp, Log, TEXT(" - maxComputeWorkgroupSizeY: %u"), limits.maxComputeWorkgroupSizeY);
			UE_LOG(LogTemp, Log, TEXT(" - maxComputeWorkgroupSizeZ: %u"), limits.maxComputeWorkgroupSizeZ);
		}
	}

	void Startup() 
	{
		// We create a descriptor
		WGPUInstanceDescriptor desc = {};
		desc.nextInChain = nullptr;

		// We create the instance using this descriptor
		Instance = wgpuCreateInstance(&desc);

		// We can check whether there is actually an instance created
		if (!Instance) 
		{
			UE_LOG(LogTemp, Error, TEXT("Could not initialize WebGPU!"));
			return;
		}

		WGPURequestAdapterOptions adapterOpts = {};
		adapterOpts.nextInChain = nullptr;

		Adapter = RequestAdapterSync(Instance, &adapterOpts);
	}
	void Shutdown()
	{
		// We clean up the WebGPU instance
		wgpuInstanceRelease(Instance);


		wgpuAdapterRelease(Adapter);
	}

	WGPUInstance Instance = nullptr;
	WGPUAdapter Adapter = nullptr;
};

UWebGPUComponent::UWebGPUComponent(const FObjectInitializer& ObjectInitializer)
{
	PrimaryComponentTick.bCanEverTick = true;
	Internal = new FWebGPUInternal();
}

UWebGPUComponent::~UWebGPUComponent()
{
	delete Internal;
	Internal = nullptr;
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
	UE_LOG(LogTemp, Log, TEXT("## Test start. ##"));

	//From: https://eliemichel.github.io/LearnWebGPU/getting-started/hello-webgpu.html#lit-6

	
	Internal->Startup();

	// Display the object (WGPUInstance is a simple pointer, it may be
	// copied around without worrying about its size).
	UE_LOG(LogTemp, Log, TEXT("WGPU instance: %p"), Internal->Instance);


	Internal->InspectAdapter(Internal->Adapter);

	Internal->Shutdown();

	UE_LOG(LogTemp, Log, TEXT("## Test end. ##"));
}

void UWebGPUComponent::RunShader(const FString& ShaderSource)
{
}
