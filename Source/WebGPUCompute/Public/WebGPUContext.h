// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "HAL/Runnable.h"
#include "Containers/Queue.h"
#include "webgpu/webgpu.h"
#include "WebGPUDevice.h"
#include "WebGPUTypes.h"
#include <atomic>

class FWebGPUContext;

using FWebGPUReadCallback = TUniqueFunction<void(bool bSuccess, TArray<uint8>&& Data)>;
using FWebGPUCompileCallback = TFunction<void(bool bSuccess, const TArray<FWebGPUCompileMessage>& Messages)>;

/**
 * A GPU buffer owned by an FWebGPUContext. Create it with FWebGPUContext::CreateBuffer.
 * Its WebGPU handle is only touched on the context's worker thread; it is recreated (zeroed) after a device loss.
 */
class WEBGPUCOMPUTE_API FWebGPUBuffer
{
public:
	~FWebGPUBuffer();

	uint64 GetSize() const { return Size; }
	EWebGPUBufferUsage GetUsage() const { return Usage; }
	const FString& GetLabel() const { return Label; }
	TSharedPtr<FWebGPUContext> GetContext() const;

	/** Worker thread only (FWebGPUContext::Enqueue). */
	WGPUBuffer GetHandle() const { return Handle; }

private:
	friend class FWebGPUContext;

	TWeakPtr<FWebGPUContext> Context;
	uint64 Size = 0;
	EWebGPUBufferUsage Usage = EWebGPUBufferUsage::Storage;
	FString Label;

	// Worker thread only
	WGPUBuffer Handle = nullptr;
};

/**
 * A compute pipeline plus its bindings. Create it with FWebGPUContext::CreateKernel.
 * Compiles asynchronously; a recompile keeps the last good pipeline until the new one succeeds.
 */
class WEBGPUCOMPUTE_API FWebGPUKernel
{
public:
	~FWebGPUKernel();

	const FString& GetLabel() const { return Label; }
	TSharedPtr<FWebGPUContext> GetContext() const;

	/** True once a pipeline compiled successfully (stays true while a recompile is pending or failed). */
	bool IsReady() const { return bReady; }

	/** Errors and warnings from the most recent compile. */
	TArray<FWebGPUCompileMessage> GetMessages() const;

	/** GPU time of the last profiled dispatch and its running average (see FWebGPUContext::SetProfiling). */
	double GetLastGpuTimeMs() const { return LastGpuTimeMs; }
	double GetAverageGpuTimeMs() const { return AverageGpuTimeMs; }

private:
	friend class FWebGPUContext;

	struct FBinding
	{
		TSharedPtr<FWebGPUBuffer> Buffer;
		uint64 Offset = 0;
		uint64 Size = WGPU_WHOLE_SIZE;
	};

	TWeakPtr<FWebGPUContext> Context;
	FString Label;
	std::atomic<bool> bReady = false;
	std::atomic<double> LastGpuTimeMs = 0.0;
	std::atomic<double> AverageGpuTimeMs = 0.0;

	mutable FCriticalSection MessagesLock;
	TArray<FWebGPUCompileMessage> Messages;

	// Worker thread only
	FString Source;
	FString EntryPoint;
	WGPUComputePipeline Pipeline = nullptr;
	uint32 CompileSerial = 0;
	uint32 AppliedSerial = 0;
	bool bProfile = false;
	bool bWarnedNotReady = false;
	FWebGPUCompileCallback Listener;
	EWebGPUCallbackThread ListenerThread = EWebGPUCallbackThread::GameThread;
	TSortedMap<uint32, TSortedMap<uint32, FBinding>> Bindings;
	TSortedMap<uint32, WGPUBindGroup> BindGroups;
	bool bBindGroupsDirty = true;

	void SetMessages(const TArray<FWebGPUCompileMessage>& InMessages);
	void ReleaseBindGroups();
	void ReleaseGPU();
};

/**
 * The WebGPU runtime: one device plus a worker thread that executes every command in order.
 *
 * Calls return immediately from any thread. Dispatches enqueued back to back are encoded into one
 * command buffer and submitted when the queue runs dry; writes and reads keep their order relative
 * to dispatches. Results come back through callbacks on the game thread (default) or the worker.
 * If the device is lost (TDR, driver reset) the worker recreates it, recreates buffers (zeroed) and
 * recompiles kernels, then broadcasts OnDeviceRecovered so owners can re-upload their data.
 */
class WEBGPUCOMPUTE_API FWebGPUContext : public TSharedFromThis<FWebGPUContext>, private FRunnable
{
public:
	/** Creates the device and starts the worker. Null if no WebGPU device is available. */
	static TSharedPtr<FWebGPUContext> Create(const FWebGPUDeviceOptions& Options = FWebGPUDeviceOptions());

	virtual ~FWebGPUContext();

	/** Stops the worker and releases every GPU object. Pending callbacks fail. Not from the worker thread. */
	void Shutdown();

	// Resources (any thread)

	TSharedRef<FWebGPUBuffer> CreateBuffer(uint64 Size, EWebGPUBufferUsage Usage = EWebGPUBufferUsage::Storage, const FString& Label = FString());

	/**
	 * Compiles asynchronously. Dispatches are skipped (and counted) until the first compile succeeds.
	 * OnCompiled is kept as the kernel's listener: it also fires after RecompileKernel and after device recovery.
	 */
	TSharedRef<FWebGPUKernel> CreateKernel(const FString& Source, const FString& EntryPoint = TEXT("main"), const FString& Label = FString(),
		FWebGPUCompileCallback OnCompiled = nullptr, EWebGPUCallbackThread CallbackThread = EWebGPUCallbackThread::GameThread);

	/** Compiles new source. The previous pipeline keeps running until the new one succeeds. OnCompiled fires once (the listener fires too). */
	void RecompileKernel(const TSharedRef<FWebGPUKernel>& Kernel, const FString& Source,
		FWebGPUCompileCallback OnCompiled = nullptr, EWebGPUCallbackThread CallbackThread = EWebGPUCallbackThread::GameThread);

	// Commands (any thread, executed in order)

	/** Offset and size must be multiples of 4. The data is copied. */
	void WriteBuffer(const TSharedRef<FWebGPUBuffer>& Buffer, uint64 Offset, const void* Data, uint64 Size);
	void WriteBuffer(const TSharedRef<FWebGPUBuffer>& Buffer, uint64 Offset, TArray<uint8>&& Data);

	/** Binds a buffer range to @group(Group) @binding(Binding). Null unbinds. */
	void SetBuffer(const TSharedRef<FWebGPUKernel>& Kernel, uint32 Group, uint32 Binding, const TSharedPtr<FWebGPUBuffer>& Buffer,
		uint64 Offset = 0, uint64 Size = WGPU_WHOLE_SIZE);

	void Dispatch(const TSharedRef<FWebGPUKernel>& Kernel, uint32 X, uint32 Y = 1, uint32 Z = 1);

	/** Measures each dispatch of this kernel with GPU timestamps (needs the TimestampQuery feature). */
	void SetProfiling(const TSharedRef<FWebGPUKernel>& Kernel, bool bEnabled);

	/** Copies a buffer range to the CPU after all earlier commands. Offset and size must be multiples of 4. */
	void ReadBuffer(const TSharedRef<FWebGPUBuffer>& Buffer, uint64 Offset, uint64 Size, FWebGPUReadCallback&& Callback,
		EWebGPUCallbackThread CallbackThread = EWebGPUCallbackThread::GameThread);

	/** Blocks until every command enqueued so far has finished on the GPU. Never call from the worker thread. */
	bool Flush(double TimeoutSeconds = 10.0);

	/** Runs code on the worker in order with other commands, for raw Dawn access through GetDevice(). */
	void Enqueue(TUniqueFunction<void()>&& Command);

	/** Simulates a device loss (like a TDR) to exercise recovery. */
	void ForceDeviceLoss();

	// State

	/** The current device; replaced after a device loss. Thread-safe Dawn access for interop. */
	TSharedPtr<FWebGPUDevice> GetDevice() const;
	uint32 GetDeviceGeneration() const { return DeviceGeneration; }
	FWebGPUStats GetStats() const;
	bool IsInWorkerThread() const;

	/** Broadcast on the game thread after the device was recreated. Buffer contents are lost; re-upload them. */
	TMulticastDelegate<void(uint32 /*Generation*/)> OnDeviceRecovered;

private:
	explicit FWebGPUContext(const FWebGPUDeviceOptions& InOptions);
	bool Start();

	// FRunnable
	virtual uint32 Run() override;

	// Worker thread
	void CreateBufferHandle(FWebGPUBuffer& Buffer);
	void CompileKernel(const TSharedRef<FWebGPUKernel>& Kernel, FWebGPUCompileCallback OnCompiled, EWebGPUCallbackThread CallbackThread);
	void FinishCompile(FWebGPUKernel& Kernel, bool bSuccess, const TArray<FWebGPUCompileMessage>& Messages, FWebGPUCompileCallback&& OnCompiled, EWebGPUCallbackThread CallbackThread);
	void EncodeDispatch(const TSharedRef<FWebGPUKernel>& Kernel, uint32 X, uint32 Y, uint32 Z);
	bool UpdateBindGroups(FWebGPUKernel& Kernel);
	void ReadBufferOnWorker(FWebGPUBuffer& Buffer, uint64 Offset, uint64 Size, FWebGPUReadCallback&& Callback, EWebGPUCallbackThread CallbackThread);
	WGPUCommandEncoder GetEncoder();
	void SubmitBatch();
	WGPUBuffer AcquireStaging(uint64 Size, uint64& OutCapacity);
	void ReleaseStaging(WGPUBuffer Staging, uint64 Capacity, uint32 Generation);
	void CreateDeviceObjects();
	void ReleaseDeviceObjects();
	void Recover();
	void ReleaseAllResources();
	bool HasDevice() const { return WorkerDevice.IsValid() && WorkerDevice->IsValid(); }
	void Wake();

	static void Deliver(EWebGPUCallbackThread Thread, TUniqueFunction<void()>&& Function);

	// Async callback trampolines (run on the worker inside wgpuInstanceProcessEvents)
	struct FReadRequest;
	struct FCompileRequest;
	struct FBatchScopeRequest;
	struct FTimestampRequest;
	struct FFlushRequest;
	static void OnReadMapped(WGPUMapAsyncStatus Status, WGPUStringView Message, void* UserData1, void* UserData2);
	static void OnPipelineCreated(WGPUCreatePipelineAsyncStatus Status, WGPUComputePipeline Pipeline, WGPUStringView Message, void* UserData1, void* UserData2);
	static void OnBatchScopePopped(WGPUPopErrorScopeStatus Status, WGPUErrorType Type, WGPUStringView Message, void* UserData1, void* UserData2);
	static void OnTimestampsMapped(WGPUMapAsyncStatus Status, WGPUStringView Message, void* UserData1, void* UserData2);
	static void OnFlushDone(WGPUQueueWorkDoneStatus Status, WGPUStringView Message, void* UserData1, void* UserData2);

	FWebGPUDeviceOptions Options;

	// Current device, readable from any thread
	mutable FCriticalSection DeviceLock;
	TSharedPtr<FWebGPUDevice> Device;
	std::atomic<uint32> DeviceGeneration = 0;

	FRunnableThread* Thread = nullptr;
	std::atomic<uint32> WorkerThreadId = 0;
	std::atomic<bool> bStopping = false;
	std::atomic<bool> bDeviceLostSignal = false;
	FEvent* WakeEvent = nullptr;
	TQueue<TUniqueFunction<void()>, EQueueMode::Mpsc> Commands;

	// Worker thread only
	TSharedPtr<FWebGPUDevice> WorkerDevice;
	WGPUCommandEncoder Encoder = nullptr;
	TArray<FString> BatchLabels;
	int32 PendingCallbacks = 0;
	bool bReleasingDevice = false;
	double BusyPollUntil = 0.0;
	double NextRecoveryTime = 0.0;
	TMap<uint64, TArray<WGPUBuffer>> StagingPool;
	TArray<TWeakPtr<FWebGPUBuffer>> LiveBuffers;
	TArray<TWeakPtr<FWebGPUKernel>> LiveKernels;

	static constexpr uint32 MaxTimestampQueries = 128;
	WGPUQuerySet TimestampQueries = nullptr;
	WGPUBuffer TimestampResolve = nullptr;
	uint32 BatchQueryCount = 0;
	TArray<TWeakPtr<FWebGPUKernel>> BatchProfiled;

	// Stats
	std::atomic<int64> StatSubmits = 0;
	std::atomic<int64> StatDispatches = 0;
	std::atomic<int64> StatSkippedDispatches = 0;
	std::atomic<int64> StatFailedSubmits = 0;
	std::atomic<int64> StatBytesWritten = 0;
	std::atomic<int64> StatBytesRead = 0;
	std::atomic<int64> StatReadbacks = 0;
};
