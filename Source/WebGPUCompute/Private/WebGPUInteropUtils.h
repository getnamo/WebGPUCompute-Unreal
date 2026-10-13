// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"

namespace WebGPUInterop
{
	/** LUID of the adapter UE's D3D12 RHI renders with. False when the RHI isn't D3D12 (e.g. -nullrhi). */
	bool GetRHIAdapterLUID(uint32& OutLowPart, int32& OutHighPart);
}
