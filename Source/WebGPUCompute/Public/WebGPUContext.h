// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "HAL/Runnable.h"
#include "Containers/Queue.h"
#include "Containers/Ticker.h"
#include "webgpu/webgpu.h"
#include "WebGPUDevice.h"
#include "WebGPUReflection.h"
#include "WebGPUShaderFiles.h"
#include "WebGPUStructLayout.h"
#include "WebGPUTypes.h"
#include <atomic>

class FWebGPUContext;
class FRHITexture;
class FWebGPUReadbackStream;
struct FWebGPUSharedTextureState;
class FWebGPUUploadStream;
enum class EWebGPUStreamPolicy : uint8;

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

	/** Struct buffers (FWebGPUContext::CreateStructBuffer): the element layout; null for raw buffers. */
	const TSharedPtr<const FWebGPUStructLayout>& GetElementLayout() const { return ElementLayout; }
	int32 GetElementCount() const { return ElementLayout ? (int32)(Size / ElementLayout->GetStride()) : 0; }

	/** Worker thread only (FWebGPUContext::Enqueue). */
	WGPUBuffer GetHandle() const { return Handle; }

private:
	friend class FWebGPUContext;

	TWeakPtr<FWebGPUContext> Context;
	uint64 Size = 0;
	EWebGPUBufferUsage Usage = EWebGPUBufferUsage::Storage;
	FString Label;
	TSharedPtr<const FWebGPUStructLayout> ElementLayout;
	/** Counted in the context's live stats. */
	bool bCounted = false;

	// Worker thread only
	WGPUBuffer Handle = nullptr;
};

/**
 * A 2D texture owned by an FWebGPUContext (CreateTexture), or a UE RHI texture shared without copies
 * (ImportTexture, D3D12). Its WebGPU handle is only touched on the worker; after a device loss it is
 * recreated (zeroed) or imported again.
 */
class WEBGPUCOMPUTE_API FWebGPUTexture
{
public:
	~FWebGPUTexture();

	uint32 GetWidth() const { return Width; }
	uint32 GetHeight() const { return Height; }
	EWebGPUTextureFormat GetFormat() const { return Format; }
	EWebGPUTextureUsage GetUsage() const { return Usage; }
	const FString& GetLabel() const { return Label; }
	TSharedPtr<FWebGPUContext> GetContext() const;

	/** A shared UE texture: use it between FWebGPUContext::BeginTextureAccess and EndTextureAccess. */
	bool IsShared() const { return Shared.IsValid(); }

	uint32 GetBytesPerPixel() const { return BytesPerPixel(Format); }
	/** Bytes per row of texture <-> buffer copies (rows are 256-byte aligned there). */
	uint32 GetCopyBytesPerRow() const { return Align(Width * GetBytesPerPixel(), 256u); }

	static uint32 BytesPerPixel(EWebGPUTextureFormat Format);
	/** The WGSL storage format spelling, e.g. "rgba16float". */
	static const TCHAR* WgslFormat(EWebGPUTextureFormat Format);
	static WGPUTextureFormat ToWGPU(EWebGPUTextureFormat Format);

	/** Worker thread only. */
	WGPUTexture GetHandle() const { return Handle; }
	WGPUTextureView GetView() const { return View; }

private:
	friend class FWebGPUContext;

	TWeakPtr<FWebGPUContext> Context;
	uint32 Width = 0;
	uint32 Height = 0;
	EWebGPUTextureFormat Format = EWebGPUTextureFormat::RGBA8Unorm;
	EWebGPUTextureUsage Usage = EWebGPUTextureUsage::Storage | EWebGPUTextureUsage::Sampled;
	FString Label;
	TSharedPtr<FWebGPUSharedTextureState> Shared;
	/** Counted in the context's live stats. */
	bool bCounted = false;

	// Worker thread only
	WGPUTexture Handle = nullptr;
	WGPUTextureView View = nullptr;
	bool bAccessing = false;
};

/** One resource bound to a kernel: a buffer range, a texture, or a built-in sampler. */
struct FWebGPUResourceBinding
{
	TSharedPtr<FWebGPUBuffer> Buffer;
	uint64 Offset = 0;
	uint64 Size = WGPU_WHOLE_SIZE;
	TSharedPtr<FWebGPUTexture> Texture;
	TOptional<EWebGPUSampler> Sampler;
};

/** A value for a uniform member set by name on a binding set (FWebGPUContext::SetUniformValue). */
struct FWebGPUUniformValue
{
	enum class EType : uint8
	{
		Float,
		Int,
		Vector,
	};

	EType Type = EType::Float;
	/** Float and Int values are in X. Vectors fill as many components as the member has. */
	FVector4f Floats = FVector4f(0.f, 0.f, 0.f, 0.f);
	FIntVector4 Ints = FIntVector4(0, 0, 0, 0);

	static FWebGPUUniformValue Float(float Value) { FWebGPUUniformValue Result; Result.Floats.X = Value; return Result; }
	static FWebGPUUniformValue Int(int32 Value) { FWebGPUUniformValue Result; Result.Type = EType::Int; Result.Ints.X = Value; return Result; }
	static FWebGPUUniformValue Vector(const FVector4f& Value) { FWebGPUUniformValue Result; Result.Type = EType::Vector; Result.Floats = Value; return Result; }
};

/**
 * Named resources shared by kernels (FWebGPUContext::CreateBindingSet). A kernel that uses a binding set
 * takes every binding it doesn't set itself from the set, by WGSL variable name. The set can also own
 * uniform buffers whose members are set by name ("params.dt"); they are packed with the layout the
 * kernel's shader declares, so editing that struct (hot reload) keeps each value in its member.
 */
class WEBGPUCOMPUTE_API FWebGPUBindingSet
{
public:
	const FString& GetLabel() const { return Label; }
	TSharedPtr<FWebGPUContext> GetContext() const { return Context.Pin(); }

private:
	friend class FWebGPUContext;
	friend class FWebGPUKernel;

	/** A uniform buffer owned by the set, written from its values whenever they or the layout change. */
	struct FUniformBlock
	{
		TMap<FString, FWebGPUUniformValue> Values;
		FWebGPUResourceBinding Binding;
		TArray<uint8> Uploaded;
		uint32 UploadedGeneration = 0;
	};

	TWeakPtr<FWebGPUContext> Context;
	FString Label;

	// Worker thread only
	TMap<FString, FWebGPUResourceBinding> Named;
	TMap<FString, TSharedPtr<FUniformBlock>> Uniforms;
	/** Bumped when a resource changes, so kernels rebuild their bind groups. */
	uint32 Version = 1;
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

	/** Errors and warnings from the most recent compile (lines are relative to the source, after the preamble; File is set for .wgsl files). */
	TArray<FWebGPUCompileMessage> GetMessages() const;

	/** The .wgsl file the kernel was created from (FWebGPUKernelDesc::SourceFile), as given; empty for source strings. */
	const FString& GetSourceFile() const { return SourceFile; }

	/** Bindings, workgroup size and structs of the last successful compile; null before one. */
	TSharedPtr<const FWebGPUShaderReflection> GetReflection() const;

	/** The kernel's entry point in GetReflection(). False before a successful compile. */
	bool GetEntryPointInfo(FWebGPUEntryPointInfo& OutInfo) const;

	/** WGSL generated from UStructs and prepended to the source (see FWebGPUKernelDesc::Structs). */
	const FString& GetPreamble() const { return Preamble; }

	/** Preamble plus the source of the most recent compile: what Dawn actually compiled. */
	FString GetCompiledSource() const;

	/** GPU time of the last profiled dispatch and its running average (see FWebGPUContext::SetProfiling). */
	double GetLastGpuTimeMs() const { return LastGpuTimeMs; }
	double GetAverageGpuTimeMs() const { return AverageGpuTimeMs; }

private:
	friend class FWebGPUContext;

	using FBinding = FWebGPUResourceBinding;

	TWeakPtr<FWebGPUContext> Context;
	FString Label;
	FString EntryPoint;
	FString SourceFile;
	/** Counted in the context's live stats. */
	bool bCounted = false;
	FString Preamble;
	int32 PreambleLines = 0;
	std::atomic<bool> bReady = false;
	std::atomic<double> LastGpuTimeMs = 0.0;
	std::atomic<double> AverageGpuTimeMs = 0.0;

	mutable FCriticalSection StateLock;
	TArray<FWebGPUCompileMessage> Messages;
	TSharedPtr<const FWebGPUShaderReflection> Reflection;
	FString CompiledSource;

	// Worker thread only
	FString Source;
	/** Lines of Source -> .wgsl files and lines (null for source strings). */
	TSharedPtr<const FWebGPUSourceMap> SourceMap;
	TSharedPtr<FWebGPUBindingSet> BindingSet;
	uint32 BindingSetVersion = 0;
	/** Uniform bindings the binding set provides for the current bind groups: written before each dispatch when their values change. */
	TArray<TPair<const FWebGPUBindingInfo*, FWebGPUBindingSet::FUniformBlock*>> SetUniforms;
	TSharedPtr<const FWebGPUShaderReflection> WorkerReflection;
	const FWebGPUEntryPointInfo* EntryInfo = nullptr;
	WGPUComputePipeline Pipeline = nullptr;
	uint32 CompileSerial = 0;
	uint32 AppliedSerial = 0;
	bool bProfile = false;
	bool bWarnedNotReady = false;
	/** The latest compile failed (vs. still compiling): skipped dispatches warn. */
	bool bCompileFailed = false;
	FWebGPUCompileCallback Listener;
	EWebGPUCallbackThread ListenerThread = EWebGPUCallbackThread::GameThread;
	TSortedMap<uint32, TSortedMap<uint32, FBinding>> Bindings;
	TMap<FString, FBinding> NamedBindings;
	TSortedMap<uint32, WGPUBindGroup> BindGroups;
	bool bBindGroupsDirty = true;
	bool bBindingsValid = false;
	/** Shared textures in the current bind groups: each dispatch checks they are being accessed. */
	TArray<TWeakPtr<FWebGPUTexture>> SharedTextures;
	TSet<FString> Warned;

	void SetMessages(const TArray<FWebGPUCompileMessage>& InMessages);
	void ReleaseBindGroups();
	void ReleaseGPU();
};

/** How to build a kernel. */
struct FWebGPUKernelDesc
{
	/** WGSL source. */
	FString Source;
	FString EntryPoint = TEXT("main");
	FString Label;

	/** UStructs declared as WGSL structs (storage layout) in a preamble before the source, so shaders can use them by name. */
	TArray<const UScriptStruct*> Structs;

	/**
	 * A .wgsl file to compile instead of Source, with #include support (see FWebGPUShaderFiles for path
	 * forms). Compile messages name the file and line.
	 */
	FString SourceFile;

	/**
	 * SourceFile: recompile when the file, or a file it includes, changes on disk. The last good pipeline
	 * keeps running while an edit doesn't compile. Files are polled on the game thread
	 * (webgpu.HotReloadInterval seconds), or call FWebGPUContext::PollShaderFiles.
	 */
	bool bHotReload = false;

	/** Named resources and uniform values shared with other kernels (FWebGPUContext::CreateBindingSet). */
	TSharedPtr<FWebGPUBindingSet> Bindings;
};

/**
 * The WebGPU runtime: one device plus a worker thread that executes every command in order.
 *
 * Calls return immediately from any thread. Commands enqueued back to back (writes, dispatches, reads) are
 * encoded in order into one command buffer, submitted when the queue runs dry. Results come back through
 * callbacks on the game thread (default) or the worker, or through streams (WebGPUStreams.h).
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

	/** Same, from a description. Struct layouts are read from UObject reflection: call from the game thread when Structs is set. */
	TSharedRef<FWebGPUKernel> CreateKernel(const FWebGPUKernelDesc& Desc, FWebGPUCompileCallback OnCompiled = nullptr,
		EWebGPUCallbackThread CallbackThread = EWebGPUCallbackThread::GameThread);

	/** Compiles new source (the preamble is kept). The previous pipeline keeps running until the new one succeeds. OnCompiled fires once (the listener fires too). */
	void RecompileKernel(const TSharedRef<FWebGPUKernel>& Kernel, const FString& Source,
		FWebGPUCompileCallback OnCompiled = nullptr, EWebGPUCallbackThread CallbackThread = EWebGPUCallbackThread::GameThread);

	/** Same, from a loaded .wgsl file. A source with Errors (a missing include, ...) fails the compile with them and changes nothing. */
	void RecompileKernel(const TSharedRef<FWebGPUKernel>& Kernel, const FWebGPUShaderSource& Source,
		FWebGPUCompileCallback OnCompiled = nullptr, EWebGPUCallbackThread CallbackThread = EWebGPUCallbackThread::GameThread);

	/**
	 * Hot reload: recompiles every watched kernel (FWebGPUKernelDesc::bHotReload) whose file, or a file it
	 * includes, changed and has stayed unchanged since the previous poll (so half-written saves are skipped).
	 * Runs on the game thread every webgpu.HotReloadInterval seconds; call it directly to poll now. Returns
	 * the number of kernels it started recompiling.
	 */
	int32 PollShaderFiles();

	/** Starts or stops watching a kernel created from a file. */
	void SetHotReload(const TSharedRef<FWebGPUKernel>& Kernel, bool bEnabled);
	bool IsHotReloading(const TSharedRef<FWebGPUKernel>& Kernel) const;

	// Binding sets (any thread)

	/** Named resources and uniform values shared by kernels (see FWebGPUBindingSet). */
	TSharedRef<FWebGPUBindingSet> CreateBindingSet(const FString& Label = FString());

	/** Makes the kernel take unset bindings from the set (null detaches it). */
	void SetKernelBindingSet(const TSharedRef<FWebGPUKernel>& Kernel, const TSharedPtr<FWebGPUBindingSet>& Set);

	/** A buffer for every kernel of the set that uses a binding of this name. Null removes it. */
	void SetSharedBuffer(const TSharedRef<FWebGPUBindingSet>& Set, const FString& Name, const TSharedPtr<FWebGPUBuffer>& Buffer,
		uint64 Offset = 0, uint64 Size = WGPU_WHOLE_SIZE);
	void SetSharedTexture(const TSharedRef<FWebGPUBindingSet>& Set, const FString& Name, const TSharedPtr<FWebGPUTexture>& Texture);
	void SetSharedSampler(const TSharedRef<FWebGPUBindingSet>& Set, const FString& Name, EWebGPUSampler Sampler);

	/**
	 * A value for a member of a var<uniform> struct, by path: "params.dt" is member dt of the uniform binding
	 * named params ("speed" alone is a scalar uniform named speed). The set owns that uniform buffer and
	 * writes it, packed with the reflected layout, before dispatches whenever a value or the layout changed.
	 * Floats convert to i32/u32 members and back; a vector fills as many components as the member has.
	 */
	void SetUniformValue(const TSharedRef<FWebGPUBindingSet>& Set, const FString& Path, const FWebGPUUniformValue& Value);

	/** A 2D texture (zeroed). Storage use needs a WGSL storage format (see EWebGPUTextureFormat). */
	TSharedRef<FWebGPUTexture> CreateTexture(uint32 Width, uint32 Height, EWebGPUTextureFormat Format,
		EWebGPUTextureUsage Usage = EWebGPUTextureUsage::Storage | EWebGPUTextureUsage::Sampled, const FString& Label = FString());

	/**
	 * Shares a UE RHI texture with WebGPU without copies: D3D12 RHI, texture created with
	 * ETextureCreateFlags::Shared (plus UAV for storage use; bGPUSharedFlag / bCanCreateUAV on render targets).
	 * Game thread. Null, with OutError, when it can't be shared. Re-imported automatically after a device loss.
	 */
	TSharedPtr<FWebGPUTexture> ImportTexture(FRHITexture* Texture, FString& OutError, const FString& Label = FString());

	// Commands (any thread, executed in order)

	/** Offset and size must be multiples of 4. The data is copied. */
	void WriteBuffer(const TSharedRef<FWebGPUBuffer>& Buffer, uint64 Offset, const void* Data, uint64 Size);
	void WriteBuffer(const TSharedRef<FWebGPUBuffer>& Buffer, uint64 Offset, TArray<uint8>&& Data);

	/** Binds a buffer range to @group(Group) @binding(Binding). Null unbinds. */
	void SetBuffer(const TSharedRef<FWebGPUKernel>& Kernel, uint32 Group, uint32 Binding, const TSharedPtr<FWebGPUBuffer>& Buffer,
		uint64 Offset = 0, uint64 Size = WGPU_WHOLE_SIZE);

	/**
	 * Binds by WGSL variable name, resolved against the kernel's reflection at dispatch time, so it follows
	 * bindings that move when the shader is edited. Null unbinds.
	 */
	void SetBufferByName(const TSharedRef<FWebGPUKernel>& Kernel, const FString& Name, const TSharedPtr<FWebGPUBuffer>& Buffer,
		uint64 Offset = 0, uint64 Size = WGPU_WHOLE_SIZE);

	/**
	 * Dispatches X*Y*Z workgroups. Before encoding, the kernel's bindings are checked against its reflection:
	 * a missing, too small or wrongly-used buffer skips this dispatch with a warning instead of failing the batch.
	 */
	void Dispatch(const TSharedRef<FWebGPUKernel>& Kernel, uint32 X, uint32 Y = 1, uint32 Z = 1);

	/** Dispatches enough workgroups to cover X*Y*Z invocations, using the entry point's @workgroup_size. */
	void DispatchForCount(const TSharedRef<FWebGPUKernel>& Kernel, uint32 X, uint32 Y = 1, uint32 Z = 1);

	/** Binds a texture (texture_2d or texture_storage_2d) to @group(Group) @binding(Binding). Null unbinds. */
	void SetTexture(const TSharedRef<FWebGPUKernel>& Kernel, uint32 Group, uint32 Binding, const TSharedPtr<FWebGPUTexture>& Texture);

	/** Binds a texture by WGSL variable name. Null unbinds. */
	void SetTextureByName(const TSharedRef<FWebGPUKernel>& Kernel, const FString& Name, const TSharedPtr<FWebGPUTexture>& Texture);

	/** Picks the built-in sampler for a sampler binding. Sampler bindings left unset use LinearClamp. */
	void SetSamplerByName(const TSharedRef<FWebGPUKernel>& Kernel, const FString& Name, EWebGPUSampler Sampler);

	/** Rows tightly packed (Width * bytes per pixel each). The data is copied. */
	void WriteTexture(const TSharedRef<FWebGPUTexture>& Texture, const void* Pixels, uint64 Size);
	void WriteTexture(const TSharedRef<FWebGPUTexture>& Texture, TArray<uint8>&& Pixels);

	/** Reads the whole texture back after all earlier commands, rows tightly packed. */
	void ReadTexture(const TSharedRef<FWebGPUTexture>& Texture, FWebGPUReadCallback&& Callback,
		EWebGPUCallbackThread CallbackThread = EWebGPUCallbackThread::GameThread);

	/**
	 * Shared textures (no-op for others), game thread. Commands enqueued between Begin and End may use the
	 * texture: WebGPU sees UE rendering enqueued before Begin, and UE rendering enqueued after End sees
	 * WebGPU's results. The synchronization is GPU-side fences; neither the game nor the render thread waits.
	 */
	void BeginTextureAccess(const TSharedRef<FWebGPUTexture>& Texture);
	void EndTextureAccess(const TSharedRef<FWebGPUTexture>& Texture);

	/** Measures each dispatch of this kernel with GPU timestamps (needs the TimestampQuery feature). */
	void SetProfiling(const TSharedRef<FWebGPUKernel>& Kernel, bool bEnabled);

	/** Copies a buffer range to the CPU after all earlier commands. Offset and size must be multiples of 4. */
	void ReadBuffer(const TSharedRef<FWebGPUBuffer>& Buffer, uint64 Offset, uint64 Size, FWebGPUReadCallback&& Callback,
		EWebGPUCallbackThread CallbackThread = EWebGPUCallbackThread::GameThread);

	// Struct arrays. Layouts come from UObject reflection: call these from the game thread.

	/** A buffer of Count elements of Struct in its WGSL storage layout (see FWebGPUStructLayout). */
	TSharedPtr<FWebGPUBuffer> CreateStructBuffer(const UScriptStruct* Struct, int32 Count, EWebGPUBufferUsage Usage = EWebGPUBufferUsage::Storage, const FString& Label = FString());

	/** Packs Count UE structs (a memcpy when the layouts match) and writes them from element FirstElement on. */
	bool WriteStructs(const TSharedRef<FWebGPUBuffer>& Buffer, const UScriptStruct* Struct, const void* Elements, int32 Count, int32 FirstElement = 0);

	template <typename T>
	bool WriteStructArray(const TSharedRef<FWebGPUBuffer>& Buffer, const TArray<T>& Items, int32 FirstElement = 0)
	{
		return WriteStructs(Buffer, T::StaticStruct(), Items.GetData(), Items.Num(), FirstElement);
	}

	/** Reads Count elements from FirstElement on and unpacks them into UE structs on the game thread. */
	template <typename T>
	void ReadStructArray(const TSharedRef<FWebGPUBuffer>& Buffer, int32 FirstElement, int32 Count, TUniqueFunction<void(bool bSuccess, TArray<T>&& Items)>&& Callback)
	{
		const EWebGPUAddressSpace Space = Buffer->GetElementLayout() ? Buffer->GetElementLayout()->GetAddressSpace() : EWebGPUAddressSpace::Storage;
		TSharedPtr<const FWebGPUStructLayout> Layout = FWebGPUStructLayout::Get(T::StaticStruct(), Space);
		if (!Layout || Count <= 0)
		{
			Callback(false, TArray<T>());
			return;
		}
		ReadBuffer(Buffer, (uint64)FirstElement * Layout->GetStride(), (uint64)Count * Layout->GetStride(),
			[Layout, Count, Callback = MoveTemp(Callback)](bool bSuccess, TArray<uint8>&& Bytes) mutable
			{
				TArray<T> Items;
				if (bSuccess)
				{
					Items.SetNum(Count);
					Layout->Unpack(Bytes.GetData(), Count, Items.GetData());
				}
				Callback(bSuccess, MoveTemp(Items));
			}, EWebGPUCallbackThread::GameThread);
	}

	// Streams (WebGPUStreams.h)

	/** A ring of Slots readback slots of SlotBytes each, for per-frame results (see FWebGPUReadbackStream). */
	TSharedRef<FWebGPUReadbackStream> CreateReadbackStream(uint64 SlotBytes, int32 Slots, EWebGPUStreamPolicy Policy, const FString& Label = FString());

	/** A ring of Slots upload slots of SlotBytes each that stay mapped for writing (see FWebGPUUploadStream). */
	TSharedRef<FWebGPUUploadStream> CreateUploadStream(uint64 SlotBytes, int32 Slots = 3, const FString& Label = FString());

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

	/** True while a lost device is kept alive because the game still holds memory it mapped (see the streams). */
	bool HasRetiredDevices() const { return bHasRetiredDevices; }
	FWebGPUStats GetStats() const;
	bool IsInWorkerThread() const;

	/** Broadcast on the game thread after the device was recreated. Buffer contents are lost; re-upload them. */
	TMulticastDelegate<void(uint32 /*Generation*/)> OnDeviceRecovered;

private:
	friend class FWebGPUBuffer;
	friend class FWebGPUTexture;
	friend class FWebGPUReadbackStream;
	friend class FWebGPUUploadStream;

	explicit FWebGPUContext(const FWebGPUDeviceOptions& InOptions);
	bool Start();

	// FRunnable
	virtual uint32 Run() override;

	// Worker thread
	void CreateBufferHandle(FWebGPUBuffer& Buffer);
	void CreateTextureHandle(FWebGPUTexture& Texture);
	void ReleaseTextureHandle(FWebGPUTexture& Texture);
	bool CanUseTexture(FWebGPUTexture& Texture, const TCHAR* What);
	void WriteTextureOnWorker(FWebGPUTexture& Texture, const uint8* Pixels, uint64 Size);
	void ReadTextureOnWorker(FWebGPUTexture& Texture, FWebGPUReadCallback&& Callback, EWebGPUCallbackThread CallbackThread);
	void CompileKernel(const TSharedRef<FWebGPUKernel>& Kernel, FWebGPUCompileCallback OnCompiled, EWebGPUCallbackThread CallbackThread);
	void FinishCompile(FWebGPUKernel& Kernel, bool bSuccess, const TArray<FWebGPUCompileMessage>& Messages, FWebGPUCompileCallback&& OnCompiled, EWebGPUCallbackThread CallbackThread);
	void EncodeDispatch(const TSharedRef<FWebGPUKernel>& Kernel, uint32 X, uint32 Y, uint32 Z);
	bool UpdateBindGroups(FWebGPUKernel& Kernel);
	/** Writes the set-owned uniform buffers this kernel uses when their packed bytes changed. */
	void UploadSetUniforms(FWebGPUKernel& Kernel);
	FWebGPUBindingSet::FUniformBlock* GetUniformBlock(FWebGPUKernel& Kernel, FWebGPUBindingSet& Set, const FWebGPUBindingInfo& Info);
	void WatchKernelFile(const TSharedRef<FWebGPUKernel>& Kernel, const FWebGPUShaderSource& Source);
	void WarnOnce(FWebGPUKernel& Kernel, const FString& Message);
	void ReadBufferOnWorker(FWebGPUBuffer& Buffer, uint64 Offset, uint64 Size, FWebGPUReadCallback&& Callback, EWebGPUCallbackThread CallbackThread);
	void UploadOnWorker(FWebGPUBuffer& Buffer, uint64 Offset, const void* Data, uint64 Size);
	void DestroyBufferOnWorker(WGPUBuffer Buffer);
	WGPUCommandEncoder GetEncoder();
	/** Runs after the current batch is submitted (true), or when it is abandoned on device loss (false). */
	void AfterSubmit(TUniqueFunction<void(bool bSubmitted)>&& Action);
	void SubmitBatch();
	void MapAsync(WGPUBuffer Buffer, WGPUMapMode Mode, uint64 Offset, uint64 Size, TUniqueFunction<void(bool bSuccess)>&& Handler);
	void CallbackDone(uint32 Generation);
	WGPUBuffer CreateMappableBuffer(WGPUBufferUsage Usage, uint64 Size, bool bMappedAtCreation, const char* Label);
	WGPUBuffer AcquireStaging(uint64 Size, uint64& OutCapacity);
	void ReleaseStaging(WGPUBuffer Staging, uint64 Capacity, uint32 Generation);
	/** Upload memory for writes: mapped for writing, unmapped for the submit that copies from it, then remapped. */
	struct FUploadPage
	{
		WGPUBuffer Buffer = nullptr;
		uint64 Capacity = 0;
		uint64 Used = 0;
		uint8* Mapped = nullptr;
		uint32 Generation = 0;
	};
	FUploadPage* AllocateUpload(uint64 Size, uint64& OutOffset, uint64 Alignment = 4);
	void RecycleUploadPage(FUploadPage* Page);
	void DeleteUploadPage(FUploadPage* Page);
	void CreateDeviceObjects();
	void ReleaseDeviceObjects();
	void Recover();
	void ShutdownRetiredDevices(bool bForce);
	void ReleaseAllResources();
	bool HasDevice() const { return WorkerDevice.IsValid() && WorkerDevice->IsValid(); }
	void Wake();

	static void Deliver(EWebGPUCallbackThread Thread, TUniqueFunction<void()>&& Function);

	// Async callback trampolines (run on the worker inside wgpuInstanceProcessEvents)
	struct FMapRequest;
	struct FCompileRequest;
	struct FBatchScopeRequest;
	struct FFlushRequest;
	static void OnBufferMapped(WGPUMapAsyncStatus Status, WGPUStringView Message, void* UserData1, void* UserData2);
	static void OnPipelineCreated(WGPUCreatePipelineAsyncStatus Status, WGPUComputePipeline Pipeline, WGPUStringView Message, void* UserData1, void* UserData2);
	static void OnBatchScopePopped(WGPUPopErrorScopeStatus Status, WGPUErrorType Type, WGPUStringView Message, void* UserData1, void* UserData2);
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
	std::atomic<bool> bHasRetiredDevices = false;
	FEvent* WakeEvent = nullptr;
	TQueue<TUniqueFunction<void()>, EQueueMode::Mpsc> Commands;

	// Worker thread only
	TSharedPtr<FWebGPUDevice> WorkerDevice;
	WGPUCommandEncoder Encoder = nullptr;
	TArray<FString> BatchLabels;
	TArray<TUniqueFunction<void(bool)>> BatchAfterSubmit;
	TArray<FUploadPage*> BatchUploadPages;
	FUploadPage* CurrentUploadPage = nullptr;
	TMap<uint64, TArray<FUploadPage*>> UploadPagePool;
	int32 BatchCommands = 0;
	int32 PendingCallbacks = 0;
	bool bReleasingDevice = false;
	double BusyPollUntil = 0.0;
	double NextRecoveryTime = 0.0;
	TMap<uint64, TArray<WGPUBuffer>> StagingPool;
	TArray<TWeakPtr<FWebGPUBuffer>> LiveBuffers;
	TArray<TWeakPtr<FWebGPUKernel>> LiveKernels;
	TArray<TWeakPtr<FWebGPUTexture>> LiveTextures;
	WGPUSampler Samplers[4] = {};
	TArray<TWeakPtr<FWebGPUReadbackStream>> LiveReadbackStreams;
	TArray<TWeakPtr<FWebGPUUploadStream>> LiveUploadStreams;
	/** Lost devices kept alive (not destroyed) while the game still holds memory they mapped. */
	TArray<TSharedPtr<FWebGPUDevice>> RetiredDevices;

	static constexpr uint32 MaxTimestampQueries = 128;
	WGPUQuerySet TimestampQueries = nullptr;
	WGPUBuffer TimestampResolve = nullptr;
	uint32 BatchQueryCount = 0;
	TArray<TWeakPtr<FWebGPUKernel>> BatchProfiled;

	/** Kernels compiled from files and watched for changes (hot reload). */
	/** A file as last seen: its time, plus a hash of its text while the time is too recent to tell saves apart (one second precision). */
	struct FFileSignature
	{
		FDateTime Time;
		uint32 Hash = 0;
		bool bHashed = false;

		bool SameAs(const FFileSignature& Other) const { return Time == Other.Time && (!bHashed || !Other.bHashed || Hash == Other.Hash); }
	};
	struct FWatchedKernel
	{
		TWeakPtr<FWebGPUKernel> Kernel;
		TArray<FString> Files;
		/** The files as the kernel's source was read from them. */
		TArray<FFileSignature> Read;
		/** As seen changed on the last poll: reloaded once they stay the same for a poll. */
		TArray<FFileSignature> Pending;
	};
	mutable FCriticalSection WatchLock;
	TArray<FWatchedKernel> WatchedKernels;
	FTSTicker::FDelegateHandle WatchTicker;

	// Stats
	std::atomic<int64> StatSubmits = 0;
	std::atomic<int64> StatDispatches = 0;
	std::atomic<int64> StatSkippedDispatches = 0;
	std::atomic<int64> StatFailedSubmits = 0;
	std::atomic<int64> StatBytesWritten = 0;
	std::atomic<int64> StatBytesRead = 0;
	std::atomic<int64> StatReadbacks = 0;
	std::atomic<int64> StatLiveBuffers = 0;
	std::atomic<int64> StatLiveBufferBytes = 0;
	std::atomic<int64> StatLiveKernels = 0;
	std::atomic<int64> StatLiveTextures = 0;

public:
	/** Live-object bookkeeping, for the objects' constructors and destructors. */
	void CountBuffer(FWebGPUBuffer& Buffer, bool bAlive);
	void CountKernel(FWebGPUKernel& Kernel, bool bAlive);
	void CountTexture(FWebGPUTexture& Texture, bool bAlive);
};
