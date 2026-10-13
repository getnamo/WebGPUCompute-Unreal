# Builds Dawn (WebGPU) from source and installs it into ThirdParty/Dawn.
#
# Why from source: Dawn's nightly prebuilts use the newest MSVC STL, which references helpers that
# older toolchains' msvcprt.lib lack. Building with the oldest MSVC that UE accepts (14.44 for UE 5.8,
# see Engine/Config/Windows/Windows_SDK.json) gives a static lib that links under any toolchain UE picks.
#
# Requires Visual Studio with the MSVC toolset given by -MsvcVersion (CMake and Ninja come with it),
# Git for Windows and Python 3. Dependencies are fetched by Dawn's own tools/fetch_dawn_dependencies.py.
#
# Usage: powershell -ExecutionPolicy Bypass -File Tools/BuildDawn.ps1 [-WorkDir <dir>] [-MsvcVersion 14.44] [-Clean]

param(
	# Source checkout and build trees (several GB). Defaults to ThirdParty/.dawn-build (gitignored).
	[string]$WorkDir = '',
	[string]$MsvcVersion = '14.44',
	# Dawn's D3D12 backend needs SM 6.8/6.9 declarations (Windows SDK 10.0.26100+). Only Dawn is built
	# with it; the plugin itself still builds with whatever SDK UE selects.
	[string]$WindowsSdkVersion = '10.0.26100.0',
	[switch]$Clean
)

$ErrorActionPreference = 'Stop'

$DawnCommit = 'c943ea32b614ff53e4fe410fc8f4968b6b8fe46d'
$DawnRepo   = 'https://github.com/google/dawn.git'

$PluginDir = Split-Path -Parent $PSScriptRoot
$DawnDir   = Join-Path $PluginDir 'ThirdParty\Dawn'
if (-not $WorkDir) { $WorkDir = Join-Path $PluginDir 'ThirdParty\.dawn-build' }
$WorkDir    = [System.IO.Path]::GetFullPath($WorkDir)
$SourceDir  = Join-Path $WorkDir 'dawn'
$BuildDir   = Join-Path $WorkDir "out\win64-msvc$MsvcVersion"
$InstallDir = Join-Path $WorkDir "install\win64-msvc$MsvcVersion"

# Git for Windows (depot_tools' git.bat wrappers may shadow it on PATH)
$Git = (Get-Command git.exe -ErrorAction SilentlyContinue | Select-Object -First 1).Source
if (-not $Git) { throw 'git.exe not found on PATH' }

# A Visual Studio install that has the requested MSVC toolset with ATL (DXC includes atlbase.h)
$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$VcVars = $null
foreach ($Install in (& $VsWhere -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath))
{
	$Toolsets = Get-ChildItem (Join-Path $Install 'VC\Tools\MSVC') -Directory -Filter "$MsvcVersion*" -ErrorAction SilentlyContinue
	if ($Toolsets | Where-Object { Test-Path (Join-Path $_.FullName 'atlmfc\include\atlbase.h') })
	{
		$VcVars = Join-Path $Install 'VC\Auxiliary\Build\vcvarsall.bat'
		break
	}
}
if (-not $VcVars) { throw "No Visual Studio install has MSVC $MsvcVersion with ATL (add the 'C++ ATL for v14.x build tools' component)" }

$SdkD3D12Header = "${env:ProgramFiles(x86)}\Windows Kits\10\Include\$WindowsSdkVersion\um\d3d12.h"
if (-not ((Test-Path $SdkD3D12Header) -and (Select-String -Path $SdkD3D12Header -Pattern 'D3D_SHADER_MODEL_6_9' -Quiet)))
{
	throw "Windows SDK $WindowsSdkVersion is missing or too old for Dawn (needs D3D_SHADER_MODEL_6_9, SDK 10.0.26100+)"
}
Write-Host "Using $VcVars (MSVC $MsvcVersion, Windows SDK $WindowsSdkVersion)"

# Source at the pinned commit
if (-not (Test-Path (Join-Path $SourceDir '.git')))
{
	New-Item -ItemType Directory -Force $SourceDir | Out-Null
	& $Git -C $SourceDir init -q
	& $Git -C $SourceDir remote add origin $DawnRepo
}
$Head = & $Git -C $SourceDir rev-parse HEAD 2>$null
if ($Head -ne $DawnCommit)
{
	Write-Host "Fetching Dawn $DawnCommit"
	& $Git -C $SourceDir fetch --depth 1 origin $DawnCommit
	if ($LASTEXITCODE -ne 0) { throw 'git fetch failed' }
	& $Git -C $SourceDir checkout -q FETCH_HEAD
	if ($LASTEXITCODE -ne 0) { throw 'git checkout failed' }
}

if ($Clean)
{
	foreach ($Dir in $BuildDir, $InstallDir) { if (Test-Path $Dir) { Remove-Item $Dir -Recurse -Force } }
}

# D3D12 only on Windows (UE's RHI there), with DXC built from source so f16/subgroups work and
# dxcompiler.dll can ship next to the plugin.
$CMakeArgs = @(
	'-DCMAKE_BUILD_TYPE=Release',
	'-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL',
	"-DCMAKE_SYSTEM_VERSION=$WindowsSdkVersion",
	'-DDAWN_FETCH_DEPENDENCIES=ON',
	'-DDAWN_BUILD_MONOLITHIC_LIBRARY=STATIC',
	'-DDAWN_ENABLE_INSTALL=ON',
	'-DTINT_ENABLE_INSTALL=OFF',
	'-DDAWN_ENABLE_D3D12=ON',
	'-DDAWN_ENABLE_D3D11=OFF',
	'-DDAWN_ENABLE_VULKAN=OFF',
	'-DDAWN_ENABLE_DESKTOP_GL=OFF',
	'-DDAWN_ENABLE_OPENGLES=OFF',
	'-DDAWN_ENABLE_NULL=ON',
	'-DDAWN_ENABLE_WEBGPU_ON_WEBGPU=OFF',
	'-DDAWN_USE_BUILT_DXC=ON',
	'-DDAWN_DXC_ENABLE_ASSERTS_IN_NDEBUG=OFF',
	'-DDAWN_USE_GLFW=OFF',
	'-DDAWN_USE_WINDOWS_UI=OFF',
	'-DDAWN_BUILD_SAMPLES=OFF',
	'-DDAWN_BUILD_TESTS=OFF',
	'-DDAWN_BUILD_BENCHMARKS=OFF',
	'-DDAWN_BUILD_PROTOBUF=OFF',
	'-DTINT_BUILD_TESTS=OFF',
	'-DTINT_BUILD_CMD_TOOLS=OFF',
	'-DTINT_BUILD_IR_BINARY=OFF',
	'-DTINT_BUILD_GLSL_WRITER=OFF',
	'-DTINT_BUILD_GLSL_VALIDATOR=OFF',
	'-DTINT_BUILD_SPV_READER=OFF',
	'-DTINT_BUILD_SPV_WRITER=OFF',
	'-DTINT_BUILD_MSL_WRITER=OFF'
)

New-Item -ItemType Directory -Force $WorkDir | Out-Null
$GitDir = Split-Path -Parent $Git
function Invoke-Msvc([string]$Name, [string[]]$Commands)
{
	$Script = Join-Path $WorkDir $Name
	Set-Content -Path $Script -Encoding ASCII -Value (@(
		'@echo off',
		# vcvarsall runs vswhere, which isn't on every PATH
		'set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"',
		"call `"$VcVars`" x64 $WindowsSdkVersion -vcvars_ver=$MsvcVersion || exit /b 1",
		"set `"PATH=$GitDir;%PATH%`""
	) + $Commands)
	& cmd.exe /c $Script
	if ($LASTEXITCODE -ne 0) { throw "$Name failed ($LASTEXITCODE)" }
}

# Configure (this also fetches the dependencies)
Invoke-Msvc 'configure-dawn.cmd' @(
	'cl 2>&1 | findstr /C:"Version"',
	"cmake -S `"$SourceDir`" -B `"$BuildDir`" -G Ninja $($CMakeArgs -join ' ') || exit /b 1"
)

# Dawn and V8 (UnrealJs) both carry Abseil configured without an inline namespace, so a monolithic game
# that links both plugins gets duplicate absl:: symbols. Dawn's copy gets its own: absl::dawn::...
$AbslOptions = Join-Path $SourceDir 'third_party\abseil-cpp\absl\base\options.h'
$Options = [IO.File]::ReadAllText($AbslOptions)
$Namespaced = $Options.Replace('#define ABSL_OPTION_USE_INLINE_NAMESPACE 0', '#define ABSL_OPTION_USE_INLINE_NAMESPACE 1').Replace('#define ABSL_OPTION_INLINE_NAMESPACE_NAME head', '#define ABSL_OPTION_INLINE_NAMESPACE_NAME dawn')
if (-not $Namespaced.Contains('#define ABSL_OPTION_INLINE_NAMESPACE_NAME dawn')) { throw "Unexpected ${AbslOptions}: can't give Abseil an inline namespace" }
if ($Namespaced -ne $Options) { [IO.File]::WriteAllText($AbslOptions, $Namespaced) }

Invoke-Msvc 'build-dawn.cmd' @(
	"cmake --build `"$BuildDir`" || exit /b 1",
	"cmake --install `"$BuildDir`" --prefix `"$InstallDir`" || exit /b 1"
)

# Stage into ThirdParty/Dawn: headers, the static lib, DXC runtime DLLs, licenses
if (Test-Path $DawnDir) { Remove-Item $DawnDir -Recurse -Force }
New-Item -ItemType Directory -Force (Join-Path $DawnDir 'lib\Win64'), (Join-Path $DawnDir 'bin\Win64') | Out-Null

Copy-Item (Join-Path $InstallDir 'include') $DawnDir -Recurse

# Tint headers (for the WGSL inspector, which is compiled into webgpu_dawn.lib). Tint's own install
# rules expect its command-line tools, so stage the headers from source as include/src/{tint,utils}.
foreach ($Sub in 'src\tint', 'src\utils')
{
	$Root = Join-Path $SourceDir $Sub
	Get-ChildItem $Root -Recurse -File -Filter '*.h' | ForEach-Object {
		$Target = Join-Path (Join-Path $DawnDir 'include') ($_.FullName.Substring($SourceDir.Length + 1))
		New-Item -ItemType Directory -Force (Split-Path -Parent $Target) | Out-Null
		Copy-Item $_.FullName $Target
	}
}
Copy-Item (Join-Path $InstallDir 'lib\webgpu_dawn.lib') (Join-Path $DawnDir 'lib\Win64')

$DxcDlls = Get-ChildItem $BuildDir -Recurse -File -Include 'dxcompiler.dll', 'dxil.dll'
if (-not ($DxcDlls | Where-Object Name -eq 'dxcompiler.dll')) { throw 'dxcompiler.dll not found in the build tree' }
$DxcDlls | Copy-Item -Destination (Join-Path $DawnDir 'bin\Win64')

Copy-Item (Join-Path $SourceDir 'LICENSE') (Join-Path $DawnDir 'LICENSE')
$DxcLicense = Get-Item (Join-Path $SourceDir 'third_party\directx-shader-compiler\src\LICENSE.TXT') -ErrorAction SilentlyContinue
if ($DxcLicense) { Copy-Item $DxcLicense.FullName (Join-Path $DawnDir 'LICENSE-DXC.txt') }

$ClVersion = (Get-ChildItem (Join-Path (Split-Path -Parent (Split-Path -Parent $VcVars)) '..\Tools\MSVC') -Directory -Filter "$MsvcVersion*" | Select-Object -First 1).Name
Set-Content -Path (Join-Path $DawnDir 'VERSION.txt') -Value "Dawn $DawnCommit (source build, MSVC $ClVersion, Windows SDK $WindowsSdkVersion, D3D12 + built DXC)"

Write-Host "Dawn $DawnCommit installed to $DawnDir"
