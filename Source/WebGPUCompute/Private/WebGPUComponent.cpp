#include "WebGPUComponent.h"
#define WEBGPU_CPP_IMPLEMENTATION
#include "webgpu/webgpu.hpp"

class FWebGPUInternal
{
public:
	WGPUAdapter requestAdapterSync(WGPUInstance instance, WGPURequestAdapterOptions const* options) {
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

	void inspectAdapter(WGPUAdapter adapter) {
		WGPULimits limits = {};
		limits.nextInChain = nullptr;

		WGPUStatus status = wgpuAdapterGetLimits(adapter, &limits);

		if (status == WGPUStatus_Success) {
			UE_LOG(LogTemp, Log, TEXT("Adapter limits:"));
			UE_LOG(LogTemp, Log, TEXT(" - maxTextureDimension1D: %u"), limits.maxTextureDimension1D);
			UE_LOG(LogTemp, Log, TEXT(" - maxTextureDimension2D: %u"), limits.maxTextureDimension2D);
			UE_LOG(LogTemp, Log, TEXT(" - maxTextureDimension3D: %u"), limits.maxTextureDimension3D);
			UE_LOG(LogTemp, Log, TEXT(" - maxTextureArrayLayers: %u"), limits.maxTextureArrayLayers);
		}

		std::vector<WGPUFeatureName> features;
		
		/*size_t featureCount = wgpuAdapterEnumerateFeatures(adapter, nullptr);
		features.resize(featureCount);
		wgpuAdapterEnumerateFeatures(adapter, features.data());

		UE_LOG(LogTemp, Log, TEXT("Adapter features:"));
		for (auto f : features) {
			UE_LOG(LogTemp, Log, TEXT(" - 0x%X"), static_cast<uint32>(f));
		}

		WGPUAdapterProperties properties = {};
		properties.nextInChain = nullptr;
		wgpuAdapterGetProperties(adapter, &properties);

		UE_LOG(LogTemp, Log, TEXT("Adapter properties:"));
		UE_LOG(LogTemp, Log, TEXT(" - vendorID: %u"), properties.vendorID);
		if (properties.vendorName) {
			UE_LOG(LogTemp, Log, TEXT(" - vendorName: %s"), ANSI_TO_TCHAR(properties.vendorName));
		}
		if (properties.architecture) {
			UE_LOG(LogTemp, Log, TEXT(" - architecture: %s"), ANSI_TO_TCHAR(properties.architecture));
		}
		UE_LOG(LogTemp, Log, TEXT(" - deviceID: %u"), properties.deviceID);
		if (properties.name) {
			UE_LOG(LogTemp, Log, TEXT(" - name: %s"), ANSI_TO_TCHAR(properties.name));
		}
		if (properties.driverDescription) {
			UE_LOG(LogTemp, Log, TEXT(" - driverDescription: %s"), ANSI_TO_TCHAR(properties.driverDescription));
		}
		UE_LOG(LogTemp, Log, TEXT(" - adapterType: 0x%X"), static_cast<uint32>(properties.adapterType));
		UE_LOG(LogTemp, Log, TEXT(" - backendType: 0x%X"), static_cast<uint32>(properties.backendType));*/
	}
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
	UE_LOG(LogTemp, Log, TEXT("WGPU instance: %p"), instance);

	WGPURequestAdapterOptions adapterOpts = {};
	adapterOpts.nextInChain = nullptr;

	WGPUAdapter adapter = Internal->requestAdapterSync(instance, &adapterOpts);

	Internal->inspectAdapter(adapter);

	// We clean up the WebGPU instance
	wgpuInstanceRelease(instance);

	UE_LOG(LogTemp, Log, TEXT("Test end."));
}

void UWebGPUComponent::RunShader(const FString& ShaderSource)
{
}
