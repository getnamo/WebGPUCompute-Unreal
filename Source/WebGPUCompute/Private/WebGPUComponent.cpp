#include "WebGPUComponent.h"
#define WEBGPU_CPP_IMPLEMENTATION
#include "webgpu/webgpu.hpp"

class FWebGPUInternal
{
public:

	struct ErrorUserData 
	{
		bool bDidError = false;
	};
	ErrorUserData AnyErrorUserData;

	//We use synchronous variants because it's intended to run on a background/non-blocking thread
	//in production (not yet implemented)
	WGPUAdapter RequestAdapterSync(WGPUInstance instance, WGPURequestAdapterOptions const* options) 
	{
		// A simple structure holding the local information shared with the
		// onAdapterRequestEnded callback.
		struct UserData 
		{
			WGPUAdapter Adapter = nullptr;
			bool bRequestEnded = false;
		};
		UserData AdapterUserData;

		// Callback called by wgpuInstanceRequestAdapter when the request returns
		// This is a C++ lambda function, but could be any function defined in the
		// global scope. It must be non-capturing (the brackets [] are empty) so
		// that it behaves like a regular C function pointer, which is what
		// wgpuInstanceRequestAdapter expects (WebGPU being a C API). The workaround
		// is to convey what we want to capture through the pUserData pointer,
		// provided as the last argument of wgpuInstanceRequestAdapter and received
		// by the callback as its last argument.
		auto onAdapterRequestEnded = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* pUserData, void* pUserData2) 
		{
			UserData& AdapterUserData = *reinterpret_cast<UserData*>(pUserData);
			if (status == WGPURequestAdapterStatus_Success)
			{
				AdapterUserData.Adapter = adapter;
			}
			else
			{
				UE_LOG(LogTemp, Error, TEXT("Could not get WebGPU adapter: %hs"), message.data);
			}
			AdapterUserData.bRequestEnded = true;
		};

		WGPURequestAdapterCallbackInfo callbackInfo;
		callbackInfo.nextInChain = nullptr;
		callbackInfo.mode = WGPUCallbackMode_WaitAnyOnly;
		callbackInfo.callback = onAdapterRequestEnded;
		callbackInfo.userdata1 = &AdapterUserData;


		// Call to the WebGPU request adapter procedure
		wgpuInstanceRequestAdapter(
			instance /* equivalent of navigator.gpu */,
			options,
			callbackInfo
		);

		FPlatformProcess::ConditionalSleep([&AdapterUserData]
		{ 
			return AdapterUserData.bRequestEnded;
		}, 0.01f);

		return AdapterUserData.Adapter;
	}

	WGPUDevice RequestDeviceSync(WGPUAdapter InAdapter, WGPUDeviceDescriptor const* descriptor) {
		WGPUDevice TempDevice = NULL;

		WGPURequestDeviceCallbackInfo CallbackInfo = {};
		CallbackInfo.userdata1 = &TempDevice;
		CallbackInfo.callback = [](
			WGPURequestDeviceStatus Status, WGPUDevice InDevice, WGPUStringView Message,
		void* UserData1, void* UserData2)
		{
			*(WGPUDevice*)UserData1 = InDevice;
		};

		//This is needed to receive errors instead of panics
		WGPUUncapturedErrorCallbackInfo UncapturedErrorCallbackInfo = {};
		UncapturedErrorCallbackInfo.nextInChain = nullptr;
		UncapturedErrorCallbackInfo.userdata1 = &AnyErrorUserData;

		UncapturedErrorCallbackInfo.callback = [](
			WGPUDevice const* device,
			WGPUErrorType type, WGPUStringView message,
			void* userdata1, void* userdata2)
		{
			ErrorUserData& AnyErrorUserData = *reinterpret_cast<ErrorUserData*>(userdata1);
			AnyErrorUserData.bDidError = true;
			UE_LOG(LogTemp, Error, TEXT("Uncaught error: %s"), UTF8_TO_TCHAR(message.data));
		};

		WGPUDeviceDescriptor DeviceDescriptor = {};
		DeviceDescriptor.requiredLimits = nullptr;
		//DeviceDescriptor.deviceLostCallbackInfo =	//we don't handle this case gracefully yet
		DeviceDescriptor.uncapturedErrorCallbackInfo = UncapturedErrorCallbackInfo;

		wgpuAdapterRequestDevice(InAdapter, &DeviceDescriptor, CallbackInfo);
		assert(TempDevice);

		return TempDevice;
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
		Device = RequestDeviceSync(Adapter, nullptr);

		Queue = wgpuDeviceGetQueue(Device);
		assert(Queue);

		wgpuSetLogCallback([](WGPULogLevel level, WGPUStringView message,
			void* userdata)
		{
			if (level == WGPULogLevel_Error)
			{
				UE_LOG(LogTemp, Error, TEXT("%hs"), message.data);
			}
			else if (level == WGPULogLevel_Warn)
			{
				UE_LOG(LogTemp, Warning, TEXT("%hs"), message.data);
			}
			else
			{ 
				UE_LOG(LogTemp, Log, TEXT("%hs"), message.data);
			}
		}, nullptr);
	}

	//Array In/out data bind shader, e.g. collatz count
	//largely from: https://github.com/gfx-rs/wgpu-native/blob/trunk/examples/compute/main.c
	void RunExampleShader(const FString& Source, const TArray<int32>& InData, TArray<int32>& OutData)
	{

		//Human readable error handling
		ErrorUserData ErrorScopeUserData;

		WGPUPopErrorScopeCallbackInfo CallbackInfo;
		CallbackInfo.userdata1 = &ErrorScopeUserData; //todo pass through callback info for errors
		CallbackInfo.callback = [](WGPUPopErrorScopeStatus Status,
			WGPUErrorType ErrorType,
			WGPUStringView Message,
			void* UserData1,
			void* UserData2)
		{
			ErrorUserData& ErrorScopeUserData = *reinterpret_cast<ErrorUserData*>(UserData1);
			ErrorScopeUserData.bDidError = ErrorType != WGPUErrorType_NoError;

			if (Message.data && ErrorScopeUserData.bDidError)
			{
				UE_LOG(LogTemp, Error, TEXT("%s"), UTF8_TO_TCHAR(Message.data));
			}
		};

		//Proper way of converting FString to char*
		FTCHARToUTF8 Converter(*Source);
		const char* SourceBuffer = Converter.Get();

		//Enabling validation catching, makes it caught here instead of uncaught on device
		wgpuDevicePushErrorScope(Device, WGPUErrorFilter_Validation);
		//wgpuDevicePushErrorScope(Device, WGPUErrorFilter_OutOfMemory);
		//wgpuDevicePushErrorScope(Device, WGPUErrorFilter_Internal);
		

		WGPUShaderSourceWGSL SourceDesc = {};
		SourceDesc.chain.next = nullptr;
		SourceDesc.chain.sType = WGPUSType_ShaderSourceWGSL;


		//NB: shader technically uses uint32_t, but this is compatible for early tests
		const TArray<int32>& Numbers = InData; //{ 1, 2, 3, 4 }; //fixed data example
		int32 NumbersSize = Numbers.Num() * sizeof(int32);
		int32 NumbersLength = Numbers.Num();

		SourceDesc.code = { SourceBuffer, WGPU_STRLEN};


		//Top level
		WGPUShaderModuleDescriptor ShaderDesc = {};
		ShaderDesc.label = { "shader.wgsl", WGPU_STRLEN };
		ShaderDesc.nextInChain = reinterpret_cast<const WGPUChainedStruct*>(&SourceDesc);

		// --- Create shader module (this is the compilation call) ---
		WGPUShaderModule ShaderModule = wgpuDeviceCreateShaderModule(Device, &ShaderDesc);

		//This callback for catching compilation errors, but unfortunate doesn't work in our context
		//WGPUCompilationInfoCallbackInfo CompilationCallbackInfo = {};
		//CompilationCallbackInfo.nextInChain = nullptr;
		//CompilationCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
		//CompilationCallbackInfo.userdata1 = &CompilationUserData;
		//CompilationCallbackInfo.callback = [](WGPUCompilationInfoRequestStatus status,
		//	struct WGPUCompilationInfo const* compilationInfo,
		//	void* userdata1, void* userdata2)
		//{
		//	UserData& CompilationUserData = *reinterpret_cast<UserData*>(userdata1);
		//	CompilationUserData.bDidError = status != WGPUCompilationInfoRequestStatus_Success;

		//	//for (int32 i = 0; i < compilationInfo->messageCount; i++)
		//	//{
		//	//	WGPUCompilationMessage message = compilationInfo->messages[i];
		//	//	UE_LOG(LogTemp, Log, TEXT("%hs at %d"), message.message.data, message.lineNum);
		//	//}
		//};

		//This call creates a panic, it looks like we need to capture this via error scopes
		//wgpuShaderModuleGetCompilationInfo(ShaderModule, CompilationCallbackInfo);

		//pop it here to find out if we errored on compilation
		wgpuDevicePopErrorScope(Device, CallbackInfo);

		//AnyErrorUserData or ErrorScopeUserData depending on whether error scope is set
		if (!ShaderModule || AnyErrorUserData.bDidError || ErrorScopeUserData.bDidError)
		{
			UE_LOG(LogTemp, Log, TEXT("ShaderModule Failed to compile"));

			if (ShaderModule)
			{
				wgpuShaderModuleRelease(ShaderModule);
			}

			//Reset the error trigger for future compiles
			AnyErrorUserData.bDidError = false;
			return;
		}

		//wgpuDevicePopErrorScope(Device, CallbackInfo);
		assert(ShaderModule);

		// --- Create staging buffer ---
		WGPUBufferDescriptor StagingDesc = {};
		StagingDesc.label = { "staging_buffer", WGPU_STRLEN };
		StagingDesc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
		StagingDesc.size = NumbersSize;
		StagingDesc.mappedAtCreation = false;

		WGPUBuffer StagingBuffer = wgpuDeviceCreateBuffer(Device, &StagingDesc);
		assert(StagingBuffer);

		// --- Create storage buffer ---
		WGPUBufferDescriptor StorageDesc = {};
		StorageDesc.label = { "storage_buffer", WGPU_STRLEN };
		StorageDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
		StorageDesc.size = NumbersSize;
		StorageDesc.mappedAtCreation = false;

		WGPUBuffer StorageBuffer = wgpuDeviceCreateBuffer(Device, &StorageDesc);
		assert(StorageBuffer);

		// --- Create compute pipeline ---
		WGPUProgrammableStageDescriptor StageDesc = {};
		StageDesc.module = ShaderModule;
		StageDesc.entryPoint = { "main", WGPU_STRLEN };

		WGPUComputePipelineDescriptor PipelineDesc = {};
		PipelineDesc.label = { "compute_pipeline", WGPU_STRLEN };
		PipelineDesc.compute = StageDesc;

		WGPUComputePipeline ComputePipeline = wgpuDeviceCreateComputePipeline(Device, &PipelineDesc);
		assert(ComputePipeline);

		// --- Create bind group layout ---
		WGPUBindGroupLayout BindGroupLayout = wgpuComputePipelineGetBindGroupLayout(ComputePipeline, 0);
		assert(BindGroupLayout);

		// --- Create bind group ---
		WGPUBindGroupEntry BindEntry = {};
		BindEntry.binding = 0;
		BindEntry.buffer = StorageBuffer;
		BindEntry.offset = 0;
		BindEntry.size = NumbersSize;

		WGPUBindGroupDescriptor BindGroupDesc = {};
		BindGroupDesc.label = { "bind_group", WGPU_STRLEN };
		BindGroupDesc.layout = BindGroupLayout;
		BindGroupDesc.entryCount = 1;
		BindGroupDesc.entries = &BindEntry;

		WGPUBindGroup BindGroup = wgpuDeviceCreateBindGroup(Device, &BindGroupDesc);
		assert(BindGroup);

		// --- Create command encoder ---
		WGPUCommandEncoderDescriptor EncoderDesc = {};
		EncoderDesc.label = { "command_encoder", WGPU_STRLEN };

		WGPUCommandEncoder CommandEncoder = wgpuDeviceCreateCommandEncoder(Device, &EncoderDesc);
		assert(CommandEncoder);

		// --- Begin compute pass ---
		WGPUComputePassDescriptor computePassDesc = {};
		computePassDesc.label = { "compute_pass", WGPU_STRLEN };

		WGPUComputePassEncoder ComputePassEncoder = wgpuCommandEncoderBeginComputePass(CommandEncoder, &computePassDesc);
		assert(ComputePassEncoder);

		// --- Dispatch compute ---
		wgpuComputePassEncoderSetPipeline(ComputePassEncoder, ComputePipeline);
		wgpuComputePassEncoderSetBindGroup(ComputePassEncoder, 0, BindGroup, 0, nullptr);
		wgpuComputePassEncoderDispatchWorkgroups(ComputePassEncoder, NumbersLength, 1, 1);
		wgpuComputePassEncoderEnd(ComputePassEncoder);
		wgpuComputePassEncoderRelease(ComputePassEncoder);

		// --- Copy buffer ---
		wgpuCommandEncoderCopyBufferToBuffer(CommandEncoder, StorageBuffer, 0, StagingBuffer, 0, NumbersSize);

		// --- Finish command buffer ---
		WGPUCommandBufferDescriptor CmdBufDesc = {};
		CmdBufDesc.label = { "command_buffer", WGPU_STRLEN };

		WGPUCommandBuffer CommandBuffer = wgpuCommandEncoderFinish(CommandEncoder, &CmdBufDesc);
		assert(CommandBuffer);

		// --- Write data to buffer ---
		wgpuQueueWriteBuffer(Queue, StorageBuffer, 0, Numbers.GetData(), NumbersSize);

		// --- Submit commands ---
		wgpuQueueSubmit(Queue, 1, &CommandBuffer);

		// --- Map staging buffer ---
		WGPUBufferMapCallbackInfo mapInfo = {};
		mapInfo.callback = [](WGPUMapAsyncStatus status,
		WGPUStringView message,
		void* userdata1, void* userdata2)
		{
			UE_LOG(LogTemp, Log, TEXT(" buffer_map status=%#.8x"), status)
		};

		wgpuBufferMapAsync(StagingBuffer, WGPUMapMode_Read, 0, NumbersSize, mapInfo);

		// --- Poll for map completion ---
		wgpuDevicePoll(Device, true, nullptr);

		// --- Access mapped buffer ---
		uint32_t* buf = static_cast<uint32_t*>(wgpuBufferGetMappedRange(StagingBuffer, 0, NumbersSize));
		assert(buf);

		//Set the out data
		int32 NumElements = NumbersSize / sizeof(uint32_t);
		OutData.Append(reinterpret_cast<int32*>(buf), NumElements);

		FString Times;
		for (auto& Element : OutData)
		{
			Times += FString::Printf(TEXT("%d,"), Element);
		}

		UE_LOG(LogTemp, Log, TEXT("Output: [%s]"), *Times);

		//wgpuDevicePopErrorScope(Device, CallbackInfo);
		//wgpuDevicePopErrorScope(Device, CallbackInfo);
		//wgpuDevicePopErrorScope(Device, CallbackInfo);

		wgpuBufferUnmap(StagingBuffer);
		wgpuCommandBufferRelease(CommandBuffer);
		wgpuCommandEncoderRelease(CommandEncoder);
		wgpuBindGroupRelease(BindGroup);
		wgpuBindGroupLayoutRelease(BindGroupLayout);
		wgpuComputePipelineRelease(ComputePipeline);
		wgpuBufferRelease(StorageBuffer);
		wgpuBufferRelease(StagingBuffer);
		wgpuShaderModuleRelease(ShaderModule);
	}

	//release all memories used
	void Shutdown()
	{
		if (Queue)
		{
			wgpuQueueRelease(Queue);
			Queue = nullptr;
		}
		if (Device)
		{
			wgpuDeviceRelease(Device);
			Device = nullptr;
		}
		if (Adapter)
		{
			wgpuAdapterRelease(Adapter);
			Adapter = nullptr;
		}
		if (Instance)
		{
			wgpuInstanceRelease(Instance);
			Instance = nullptr;
		}
	}

	bool HasStarted() 
	{
		return Instance != nullptr;
	}

	WGPUInstance Instance = nullptr;
	WGPUAdapter Adapter = nullptr;
	WGPUDevice Device = nullptr;
	WGPUQueue Queue = nullptr;
	
};

UWebGPUComponent::UWebGPUComponent(const FObjectInitializer& ObjectInitializer)
{
	PrimaryComponentTick.bCanEverTick = true;
	Internal = new FWebGPUInternal();
}

UWebGPUComponent::~UWebGPUComponent()
{
	Internal->Shutdown();

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

	if (!Internal->HasStarted())
	{
		Internal->Startup();

		// Display the object (WGPUInstance is a simple pointer, it may be
		// copied around without worrying about its size).
		UE_LOG(LogTemp, Log, TEXT("WGPU instance: %p"), Internal->Instance);

		Internal->InspectAdapter(Internal->Adapter);
	}

	const char* RawSource = (R"(
@group(0)
@binding(0)
var<storage, read_write> v_indices: array<u32>; // this is used as both input and output for convenience

// The Collatz Conjecture states that for any integer n:
// If n is even, n = n/2
// If n is odd, n = 3n+1
// And repeat this process for each new n, you will always eventually reach 1.
// Though the conjecture has not been proven, no counterexample has ever been found.
// This function returns how many times this recurrence needs to be applied to reach 1.
fn collatz_iterations(n_base: u32) -> u32 {
	var n : u32 = n_base;
	var i : u32 = 0u;
	loop{
		if (n <= 1u) {
			break;
		}
		if (n % 2u == 0u) {
			n = n / 2u;
		}
		else {
			// Overflow? (i.e. 3*n + 1 > 0xffffffffu?)
			if (n >= 1431655765u) {   // 0x55555555u
				return 4294967295u;   // 0xffffffffu
			}

			n = 3u * n + 1u;
		}
		i = i + 1u;
	}
	return i;
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	v_indices[global_id.x] = collatz_iterations(v_indices[global_id.x]);
}
		)");

	FString Source = FString(RawSource);

	TArray<int32> Data = { 1,2,3,4 };
	Internal->RunExampleShader(Source, Data, Data);

	UE_LOG(LogTemp, Log, TEXT("## Test end. ##"));
}

void UWebGPUComponent::RunShader(const FString& ShaderSource, const TArray<int32>& InData, TArray<int32>& OutData)
{
	if (!Internal->HasStarted())
	{
		Internal->Startup();

		// Display the object (WGPUInstance is a simple pointer, it may be
		// copied around without worrying about its size).
		UE_LOG(LogTemp, Log, TEXT("WGPU instance: %p"), Internal->Instance);

		Internal->InspectAdapter(Internal->Adapter);
	}

	Internal->RunExampleShader(ShaderSource, InData, OutData);
}
