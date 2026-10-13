// Copyright 2025-current Getnamo. All Rights Reserved.

using System.IO;
using UnrealBuildTool;

public class WebGPUCompute : ModuleRules
{
	private string DawnPath
	{
		get { return Path.GetFullPath(Path.Combine(ModuleDirectory, "../../ThirdParty/Dawn")); }
	}

	public WebGPUCompute(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"CoreUObject",
				"Engine",
				"AudioMixer",
			}
			);

		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"RHI",
				"RenderCore",
			}
			);

		// Dawn (WebGPU), built from source by Tools/BuildDawn.ps1
		PublicIncludePaths.Add(Path.Combine(DawnPath, "include"));

		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			string DawnLib = Path.Combine(DawnPath, "lib", "Win64", "webgpu_dawn.lib");
			if (!File.Exists(DawnLib))
			{
				throw new BuildException("Dawn not found at " + DawnLib + ". Run Plugins/WebGPUCompute-Unreal/Tools/BuildDawn.ps1 first.");
			}

			// Static monolithic Dawn (/MD, _ITERATOR_DEBUG_LEVEL=0, matches UE)
			PublicAdditionalLibraries.Add(DawnLib);
			PublicSystemLibraries.AddRange(new string[] { "user32.lib", "dxguid.lib", "onecore_apiset.lib" });

			// DXC, loaded by Dawn at runtime from the directory of the module that contains it
			string DawnBinaries = Path.Combine(DawnPath, "bin", "Win64");
			foreach (string DxcDll in new string[] { "dxcompiler.dll", "dxil.dll" })
			{
				string DxcPath = Path.Combine(DawnBinaries, DxcDll);
				if (File.Exists(DxcPath))
				{
					RuntimeDependencies.Add("$(BinaryOutputDir)/" + DxcDll, DxcPath);
				}
			}

			// D3D12 interop: shared textures/fences between UE's D3D12 RHI and Dawn's D3D12 device
			PrivateDependencyModuleNames.Add("D3D12RHI");
			AddEngineThirdPartyPrivateStaticDependencies(Target, "DX12");
			PublicDefinitions.Add("WITH_WEBGPU_D3D12_INTEROP=1");
		}
		else
		{
			PublicDefinitions.Add("WITH_WEBGPU_D3D12_INTEROP=0");
		}
	}
}
