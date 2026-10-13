// Copyright 2025-current Getnamo. All Rights Reserved.

// The Dawn nightly prebuilt is compiled with a newer MSVC STL than UE's toolchains (14.44-14.51),
// whose vectorized min/max_element got unsigned variants. Those helpers live in the static part of
// msvcprt.lib, so older toolchains can't resolve them. Provide equivalents until we build Dawn
// with UE's toolchain ourselves. Semantics match std::min_element/std::max_element (first match wins).

#include "CoreMinimal.h"

#if PLATFORM_WINDOWS && defined(_MSC_VER) && _MSC_VER <= 1951

namespace
{
	template <typename T>
	const void* MinElement(const void* First, const void* Last)
	{
		const T* It = static_cast<const T*>(First);
		const T* End = static_cast<const T*>(Last);
		if (It == End)
		{
			return End;
		}
		const T* Best = It;
		for (++It; It != End; ++It)
		{
			if (*It < *Best)
			{
				Best = It;
			}
		}
		return Best;
	}

	template <typename T>
	const void* MaxElement(const void* First, const void* Last)
	{
		const T* It = static_cast<const T*>(First);
		const T* End = static_cast<const T*>(Last);
		if (It == End)
		{
			return End;
		}
		const T* Best = It;
		for (++It; It != End; ++It)
		{
			if (*Best < *It)
			{
				Best = It;
			}
		}
		return Best;
	}
}

extern "C"
{
	const void* __stdcall __std_min_element_1u(const void* First, const void* Last) noexcept { return MinElement<uint8>(First, Last); }
	const void* __stdcall __std_min_element_2u(const void* First, const void* Last) noexcept { return MinElement<uint16>(First, Last); }
	const void* __stdcall __std_min_element_4u(const void* First, const void* Last) noexcept { return MinElement<uint32>(First, Last); }
	const void* __stdcall __std_min_element_8u(const void* First, const void* Last) noexcept { return MinElement<uint64>(First, Last); }
	const void* __stdcall __std_max_element_1u(const void* First, const void* Last) noexcept { return MaxElement<uint8>(First, Last); }
	const void* __stdcall __std_max_element_2u(const void* First, const void* Last) noexcept { return MaxElement<uint16>(First, Last); }
	const void* __stdcall __std_max_element_4u(const void* First, const void* Last) noexcept { return MaxElement<uint32>(First, Last); }
	const void* __stdcall __std_max_element_8u(const void* First, const void* Last) noexcept { return MaxElement<uint64>(First, Last); }
}

#endif
