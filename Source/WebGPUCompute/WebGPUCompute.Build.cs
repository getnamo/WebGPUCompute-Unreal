// Copyright 2025-current Getnamo. All Rights Reserved.

using System.IO;
using UnrealBuildTool;

public class WebGPUCompute : ModuleRules
{
	private string WebGpuThirdPartyPath
	{
		get { return Path.GetFullPath(Path.Combine(ModuleDirectory, "../../ThirdParty/WebGPU")); }
	}

	private string WebGpuIncludePath
	{
		get { return Path.GetFullPath(Path.Combine(WebGpuThirdPartyPath,  "Include")); }
	}

	private string WebGpuLibPath
	{
		get { return Path.GetFullPath(Path.Combine(WebGpuThirdPartyPath, "Lib")); }
	}

	private string WebGpuBinariesPath
	{
		get { return Path.GetFullPath(Path.Combine(WebGpuThirdPartyPath, "Binaries")); }
	}

	public WebGPUCompute(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicIncludePaths.AddRange(
			new string[] {
				// ... add public include paths required here ...
			}
			);


		PrivateIncludePaths.AddRange(
			new string[] {
				// ... add other private include paths required here ...
			}
			);


		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				// ... add other public dependencies that you statically link with here ...
			}
			);


		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"Slate",
				"SlateCore",
				// ... add private dependencies that you statically link with here ...	
			}
			);


		DynamicallyLoadedModuleNames.AddRange(
			new string[]
			{
				// ... add any modules that your module loads dynamically here ...
			}
			);

		PublicIncludePaths.Add(WebGpuIncludePath);

		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			string Win64LibPath = Path.Combine(WebGpuLibPath, "Win64");
			string Win64BinariesPath = Path.Combine(WebGpuBinariesPath, "Win64");

			PublicAdditionalLibraries.Add(Path.Combine(Win64LibPath, "wgpu_native.lib"));
			RuntimeDependencies.Add("$(BinaryOutputDir)/wgpu_native.dll", Path.Combine(Win64BinariesPath, "wgpu_native.dll"));
		}
	}
}
