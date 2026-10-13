// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "Modules/ModuleManager.h"

WEBGPUCOMPUTE_API DECLARE_LOG_CATEGORY_EXTERN(LogWebGPU, Log, All);

class FWebGPUComputeModule : public IModuleInterface
{
public:

	/** IModuleInterface implementation */
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};
