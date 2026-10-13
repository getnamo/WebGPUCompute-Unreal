// Copyright 2025-current Getnamo. All Rights Reserved.

using System;
using UnrealBuildTool;

/** Script access to WebGPU (UnrealJs): zero-copy frames, writes from script memory, read requests. Empty without UnrealJs. */
public class WebGPUComputeJs : ModuleRules
{
	public WebGPUComputeJs(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PrivateDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "WebGPUCompute" });

		// UnrealJs's V8 module, when the project has it
		bool bHasUnrealJs = false;
		try
		{
			bHasUnrealJs = !string.IsNullOrEmpty(GetModuleDirectory("V8"));
		}
		catch (Exception)
		{
		}

		if (bHasUnrealJs)
		{
			PrivateDependencyModuleNames.Add("V8");
			if (Target.bBuildEditor)
			{
				// The JavaScript example map's play test (PIE)
				PrivateDependencyModuleNames.AddRange(new string[] { "UnrealEd", "RHI" });
			}
		}
		PublicDefinitions.Add("WITH_WEBGPU_UNREALJS=" + (bHasUnrealJs ? "1" : "0"));

		// webgpu.js and the examples, found by UnrealJs's per-plugin Content/Scripts search path
		if (bHasUnrealJs)
		{
			RuntimeDependencies.Add("$(PluginDir)/Content/Scripts/...", StagedFileType.NonUFS);
		}
	}
}
