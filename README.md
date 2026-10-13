# WebGPU Compute Unreal

Leverages [Dawn](https://dawn.googlesource.com/dawn) (Google's WebGPU implementation) to run WGSL shaders in unreal for compute purposes. Experimental plugin meant to explore more dynamic runtime compute shader code. YMMV.

On the D3D12 RHI, render targets can be shared with WebGPU without copies (`FWebGPUSharedTexture`), synchronized with GPU fences.

## Setup

Dawn is not checked in. Fetch the pinned prebuilt before building:

```
powershell -ExecutionPolicy Bypass -File Tools/FetchDawn.ps1
```

This downloads a pinned Dawn release from [google/dawn releases](https://github.com/google/dawn/releases) into `ThirdParty/Dawn` (Win64 only for now).

See https://github.com/getnamo/WebGPUCompute-Unreal/issues/1 for tracking on initial usable build.

[Discord Server](https://discord.gg/qfJUyxaW4s)
