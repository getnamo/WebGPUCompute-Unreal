# WebGPU Compute Unreal

Leverages [Dawn](https://dawn.googlesource.com/dawn) (Google's WebGPU implementation) to run WGSL shaders in unreal for compute purposes. Experimental plugin meant to explore more dynamic runtime compute shader code. YMMV.

On the D3D12 RHI, render targets can be shared with WebGPU without copies, synchronized with GPU fences (see Textures).

## Setup

Dawn is not checked in. Build the pinned version from source before building the plugin (Win64 only for now):

```
powershell -ExecutionPolicy Bypass -File Tools/BuildDawn.ps1
```

This needs Visual Studio with the MSVC 14.44 toolset (the oldest UE 5.8 accepts, so the static lib links under either 14.44 or 14.50), Git for Windows and Python 3. It clones Dawn, fetches its dependencies, builds it with the D3D12 backend and DXC, and installs headers, `webgpu_dawn.lib` and `dxcompiler.dll` into `ThirdParty/Dawn`. The source and build trees go to `ThirdParty/.dawn-build` (several GB); pass `-WorkDir` to put them elsewhere.

## Usage

All GPU work goes through one runtime (`UWebGPUSubsystem`): a device plus a worker thread that runs commands in order. Calls return immediately; dispatches enqueued together are submitted as one command buffer, and results come back on the game thread.

Blueprint: **Create WebGPU Buffer**, **Create WebGPU Kernel** (WGSL source; compiles asynchronously, see `OnCompiled` / `IsReady`), **Set Buffer**, **Dispatch**, and the async **Read WebGPU Buffer** node. `Recompile` keeps the last good pipeline running if the new source fails, and compile errors carry line and column.

C++:

```cpp
TSharedPtr<FWebGPUContext> Context = UWebGPUSubsystem::GetSharedContext();
TSharedRef<FWebGPUBuffer> Data = Context->CreateBuffer(Count * sizeof(float));
TSharedRef<FWebGPUKernel> Kernel = Context->CreateKernel(WgslSource);   // entry point "main"
Context->WriteBuffer(Data, 0, Values.GetData(), Count * sizeof(float));
Context->SetBuffer(Kernel, /*Group*/ 0, /*Binding*/ 0, Data);
Context->Dispatch(Kernel, FMath::DivideAndRoundUp(Count, 64u));
Context->ReadBuffer(Data, 0, Count * sizeof(float), [](bool bSuccess, TArray<uint8>&& Bytes) { /* game thread */ });
```

If the GPU device is lost (driver reset, TDR), the runtime recreates it along with every buffer (zeroed) and kernel, then fires `OnDeviceRecovered` so you can re-upload data. Per-kernel GPU timing is available with `SetProfiling`.

### Bindings and structs

Kernels are reflected with Tint (Dawn's WGSL compiler), so buffers can be bound by WGSL variable name (`SetBufferByName`) and `DispatchForCount` sizes the dispatch from `@workgroup_size`. Before each dispatch the bindings are checked against the shader: a missing, too small or wrongly-created buffer skips that dispatch with a warning that says what's wrong, and bindings the shader doesn't use are ignored.

Arrays of UStructs map onto WGSL structs. The layout follows WGSL's alignment rules (a `FVector3f` takes 16 bytes unless a 4-byte member follows it); when it matches the UE memory layout, packing is a single memcpy, otherwise members are converted (`double` vectors to `f32`, `bool` to `u32`, small ints and enums widened). Pass the structs when creating a kernel and they are declared in WGSL ahead of your source:

```cpp
FWebGPUKernelDesc Desc;
Desc.Source = TEXT("@group(0) @binding(0) var<storage, read_write> agents: array<AgentState>; ...");
Desc.Structs = { FAgentState::StaticStruct() };
TSharedRef<FWebGPUKernel> Kernel = Context->CreateKernel(Desc);
TSharedPtr<FWebGPUBuffer> Agents = Context->CreateStructBuffer(FAgentState::StaticStruct(), Count);
Context->WriteStructArray(Agents.ToSharedRef(), AgentArray);
Context->SetBufferByName(Kernel, TEXT("agents"), Agents);
Context->DispatchForCount(Kernel, Count);
Context->ReadStructArray<FAgentState>(Agents.ToSharedRef(), 0, Count, [](bool bSuccess, TArray<FAgentState>&& Result) { });
```

Blueprint has the same through **Create WebGPU Struct Buffer**, **Create WebGPU Kernel With Structs**, **Write WebGPU Struct Array** and **WebGPU Bytes To Struct Array** (wildcard struct arrays), **Get WGSL For Struct**, and the kernel's **Get Bindings** / **Get Workgroup Size**. If a struct buffer's UStruct and the shader's struct disagree, a warning lists the differing fields.

### Streams

Writes and reads join the command buffer of the dispatches around them, so write → dispatch → write → dispatch → read is a single submit. For work that repeats every frame, `WebGPUStreams.h` adds three pieces:

- **Readback streams** (`CreateReadbackStream`): a ring of readback slots. `Capture` copies a buffer range after the dispatches before it; the consumer `Acquire`s a frame and reads the mapped memory in place until it releases it. Policies, as in PythonCLI:
  - **Latest**: the newest frame; stale ones are skipped, and nothing waits.
  - **Queue**: every frame in order; a capture with no free slot is dropped and counted.
  - **Lockstep**: `Capture` blocks until the data is back, for same-tick results.
- **Upload streams** (`CreateUploadStream`): slots that stay mapped for writing. Pack data straight into one and `Commit` it to a buffer; when no slot is free, the write takes the copying path instead of being dropped.
- **Struct mirrors** (`TWebGPUStructMirror<T>`): a struct array kept on the CPU and the GPU. Edit elements and `Sync` uploads only what changed. Nearby edits merge into a few copies; widely scattered edits upload just the changed elements and their indices, and a small kernel scatters them into place. After a device loss, the next Sync re-uploads everything. Mirrors are for data the CPU owns (grids, goals, parameters); keep state that kernels write in its own buffer.

```cpp
TWebGPUStructMirror<FAgentGoal> Goals = TWebGPUStructMirror<FAgentGoal>::Create(Context.ToSharedRef(), Count);
TSharedRef<FWebGPUReadbackStream> Results = Context->CreateReadbackStream(Count * sizeof(uint32), 3, EWebGPUStreamPolicy::Latest);

// Every tick
Goals.Edit(Index).Target = NewTarget;        // marks the element dirty
Goals.Sync();                                // uploads only the changed goals
Context->DispatchForCount(Step, Count);
Results->Capture(Cells);
FWebGPUReadbackFrame Frame;
if (Results->Acquire(Frame))                 // usually the previous tick's result
{
	for (uint32 Cell : Frame.View<uint32>()) { /* ... */ }
}
```

Blueprint: **Create WebGPU Readback Stream (For Buffer)**, **Capture**, **Read Bytes / Floats / Ints / Structs** (wildcard array), **Get Last Frame Info**; **Create WebGPU Struct Mirror**, **Set Element / Get Element** (wildcard struct), **Set Elements**, **Sync**, **Get Buffer**.

### Textures

`CreateTexture(Width, Height, Format, Usage)` makes a 2D texture. Bind it by WGSL name with `SetTextureByName` (`texture_2d` or `texture_storage_2d`) and choose built-in samplers with `SetSamplerByName`; sampler bindings left unset use linear clamp. `WriteTexture` and `ReadTexture` take tightly packed rows, and readback streams can `CaptureTexture`. Texture bindings are checked against the shader (2D only, storage format, usage). Anything WebGPU itself rejects, such as sampling an unfilterable format, skips that kernel with WebGPU's message rather than failing the batch.

On the D3D12 RHI, UE render targets can be shared without copies: `ImportTexture(FRHITexture*)` in C++, or **Create WebGPU Texture From Render Target** in Blueprint. The render target needs **Shared** enabled, plus **Can Create UAV** for kernels that write it. Use a shared texture between `BeginTextureAccess` and `EndTextureAccess`, called on the game thread:

```cpp
Context->BeginTextureAccess(Camera);   // WebGPU sees UE's rendering enqueued so far
Context->DispatchForCount(Detect, Width, Height);
Context->EndTextureAccess(Camera);     // UE rendering enqueued after this sees WebGPU's results
```

Synchronization uses GPU fences, so neither the game thread nor the render thread waits. Shared textures are imported again after a device loss. (`FWebGPUSharedTexture` is the lower-level, render-thread version of the same sharing, for direct Dawn use.)

Blueprint: **Create WebGPU Texture**, **Create WebGPU Texture From Render Target**, the texture's **Begin Access / End Access / Write Bytes / Write Floats**, the kernel's **Set Texture By Name / Set Sampler By Name**, and **Create WebGPU Readback Stream For Texture** + **Capture Texture**.

See https://github.com/getnamo/WebGPUCompute-Unreal/issues/1 for tracking on initial usable build.

[Discord Server](https://discord.gg/qfJUyxaW4s)
