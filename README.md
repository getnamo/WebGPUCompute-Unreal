# WebGPU Compute Unreal

Leverages [Dawn](https://dawn.googlesource.com/dawn) (Google's WebGPU implementation) to run WGSL shaders in unreal for compute purposes. Experimental plugin meant to explore more dynamic runtime compute shader code. YMMV.

On the D3D12 RHI, render targets can be shared with WebGPU without copies (`FWebGPUSharedTexture`), synchronized with GPU fences.

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

See https://github.com/getnamo/WebGPUCompute-Unreal/issues/1 for tracking on initial usable build.

[Discord Server](https://discord.gg/qfJUyxaW4s)
