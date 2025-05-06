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

		wgpuAdapterRequestDevice(InAdapter, NULL, CallbackInfo);
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
	}

	//todo: export source and auto-bind
	void RunShader()
	{
		WGPUShaderSourceWGSL sourceDesc = {};
		sourceDesc.chain.sType = WGPUSType_ShaderSourceWGSL;

		const char* Source = (R"(
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

		uint32_t numbers[] = { 1, 2, 3, 4 };
		uint32_t numbers_size = sizeof(numbers);
		uint32_t numbers_length = numbers_size / sizeof(uint32_t);

		sourceDesc.code = { Source, WGPU_STRLEN };

		WGPUShaderModuleDescriptor shaderDesc = {};
		shaderDesc.label = { "shader.wgsl", WGPU_STRLEN };
		shaderDesc.nextInChain = reinterpret_cast<const WGPUChainedStruct*>(&sourceDesc);

		WGPUShaderModule shader_module = wgpuDeviceCreateShaderModule(Device, &shaderDesc);
		assert(shader_module);

		// --- Create staging buffer ---
		WGPUBufferDescriptor stagingDesc = {};
		stagingDesc.label = { "staging_buffer", WGPU_STRLEN };
		stagingDesc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
		stagingDesc.size = numbers_size;
		stagingDesc.mappedAtCreation = false;

		WGPUBuffer staging_buffer = wgpuDeviceCreateBuffer(Device, &stagingDesc);
		assert(staging_buffer);

		// --- Create storage buffer ---
		WGPUBufferDescriptor storageDesc = {};
		storageDesc.label = { "storage_buffer", WGPU_STRLEN };
		storageDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
		storageDesc.size = numbers_size;
		storageDesc.mappedAtCreation = false;

		WGPUBuffer storage_buffer = wgpuDeviceCreateBuffer(Device, &storageDesc);
		assert(storage_buffer);

		// --- Create compute pipeline ---
		WGPUProgrammableStageDescriptor stageDesc = {};
		stageDesc.module = shader_module;
		stageDesc.entryPoint = { "main", WGPU_STRLEN };

		WGPUComputePipelineDescriptor pipelineDesc = {};
		pipelineDesc.label = { "compute_pipeline", WGPU_STRLEN };
		pipelineDesc.compute = stageDesc;

		WGPUComputePipeline compute_pipeline = wgpuDeviceCreateComputePipeline(Device, &pipelineDesc);
		assert(compute_pipeline);

		// --- Create bind group layout ---
		WGPUBindGroupLayout bind_group_layout = wgpuComputePipelineGetBindGroupLayout(compute_pipeline, 0);
		assert(bind_group_layout);

		// --- Create bind group ---
		WGPUBindGroupEntry bindEntry = {};
		bindEntry.binding = 0;
		bindEntry.buffer = storage_buffer;
		bindEntry.offset = 0;
		bindEntry.size = numbers_size;

		WGPUBindGroupDescriptor bindGroupDesc = {};
		bindGroupDesc.label = { "bind_group", WGPU_STRLEN };
		bindGroupDesc.layout = bind_group_layout;
		bindGroupDesc.entryCount = 1;
		bindGroupDesc.entries = &bindEntry;

		WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(Device, &bindGroupDesc);
		assert(bind_group);

		// --- Create command encoder ---
		WGPUCommandEncoderDescriptor encoderDesc = {};
		encoderDesc.label = { "command_encoder", WGPU_STRLEN };

		WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(Device, &encoderDesc);
		assert(command_encoder);

		// --- Begin compute pass ---
		WGPUComputePassDescriptor computePassDesc = {};
		computePassDesc.label = { "compute_pass", WGPU_STRLEN };

		WGPUComputePassEncoder compute_pass_encoder = wgpuCommandEncoderBeginComputePass(command_encoder, &computePassDesc);
		assert(compute_pass_encoder);

		// --- Dispatch compute ---
		wgpuComputePassEncoderSetPipeline(compute_pass_encoder, compute_pipeline);
		wgpuComputePassEncoderSetBindGroup(compute_pass_encoder, 0, bind_group, 0, nullptr);
		wgpuComputePassEncoderDispatchWorkgroups(compute_pass_encoder, numbers_length, 1, 1);
		wgpuComputePassEncoderEnd(compute_pass_encoder);
		wgpuComputePassEncoderRelease(compute_pass_encoder);

		// --- Copy buffer ---
		wgpuCommandEncoderCopyBufferToBuffer(command_encoder, storage_buffer, 0, staging_buffer, 0, numbers_size);

		// --- Finish command buffer ---
		WGPUCommandBufferDescriptor cmdBufDesc = {};
		cmdBufDesc.label = { "command_buffer", WGPU_STRLEN };

		WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(command_encoder, &cmdBufDesc);
		assert(command_buffer);

		// --- Write data to buffer ---
		wgpuQueueWriteBuffer(Queue, storage_buffer, 0, &numbers, numbers_size);

		// --- Submit commands ---
		wgpuQueueSubmit(Queue, 1, &command_buffer);

		// --- Map staging buffer ---
		WGPUBufferMapCallbackInfo mapInfo = {};
		mapInfo.callback = [](WGPUMapAsyncStatus status,
			WGPUStringView message,
			void* userdata1, void* userdata2)
			{
				UE_LOG(LogTemp, Log, TEXT(" buffer_map status=%#.8x"), status)
			};

		wgpuBufferMapAsync(staging_buffer, WGPUMapMode_Read, 0, numbers_size, mapInfo);

		// --- Poll for map completion ---
		wgpuDevicePoll(Device, true, nullptr);

		// --- Access mapped buffer ---
		uint32_t* buf = static_cast<uint32_t*>(wgpuBufferGetMappedRange(staging_buffer, 0, numbers_size));
		assert(buf);

		UE_LOG(LogTemp, Log, TEXT("collatz times: [%d, %d, %d, %d]"), buf[0], buf[1], buf[2], buf[3]);

		wgpuBufferUnmap(staging_buffer);
		wgpuCommandBufferRelease(command_buffer);
		wgpuCommandEncoderRelease(command_encoder);
		wgpuBindGroupRelease(bind_group);
		wgpuBindGroupLayoutRelease(bind_group_layout);
		wgpuComputePipelineRelease(compute_pipeline);
		wgpuBufferRelease(storage_buffer);
		wgpuBufferRelease(staging_buffer);
		wgpuShaderModuleRelease(shader_module);
	}

	//release all memories used
	void Shutdown()
	{
		wgpuQueueRelease(Queue);
		wgpuDeviceRelease(Device);
		wgpuAdapterRelease(Adapter);
		wgpuInstanceRelease(Instance);
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

	Internal->RunShader();

	Internal->Shutdown();

	UE_LOG(LogTemp, Log, TEXT("## Test end. ##"));
}

void UWebGPUComponent::RunShader(const FString& ShaderSource)
{
}
