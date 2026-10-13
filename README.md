# WebGPU Compute Unreal

Leverages [Dawn](https://dawn.googlesource.com/dawn) (Google's WebGPU implementation) to run WGSL shaders in unreal for compute purposes. Experimental plugin meant to explore more dynamic runtime compute shader code. YMMV.

On the D3D12 RHI, render targets can be shared with WebGPU without copies (`FWebGPUSharedTexture`), synchronized with GPU fences.

## Setup

Dawn is not checked in. Build the pinned version from source before building the plugin (Win64 only for now):

```
powershell -ExecutionPolicy Bypass -File Tools/BuildDawn.ps1
```

This needs Visual Studio with the MSVC 14.44 toolset (the oldest UE 5.8 accepts, so the static lib links under either 14.44 or 14.50), Git for Windows and Python 3. It clones Dawn, fetches its dependencies, builds it with the D3D12 backend and DXC, and installs headers, `webgpu_dawn.lib` and `dxcompiler.dll` into `ThirdParty/Dawn`. The source and build trees go to `ThirdParty/.dawn-build` (several GB); pass `-WorkDir` to put them elsewhere.

See https://github.com/getnamo/WebGPUCompute-Unreal/issues/1 for tracking on initial usable build.

[Discord Server](https://discord.gg/qfJUyxaW4s)
