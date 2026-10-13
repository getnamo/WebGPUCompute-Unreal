// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "WebGPUStructLayout.generated.h"

struct FWebGPUTypeLayout;
class FWebGPUSourceSpellings;

/** Which WGSL layout rules apply. Uniform buffers align nested structs and arrays to 16 bytes. */
UENUM(BlueprintType)
enum class EWebGPUAddressSpace : uint8
{
	Storage,
	Uniform,
};

/**
 * How a UStruct maps onto a WGSL struct: field types, WGSL offsets (host-shareable layout rules),
 * generated WGSL declarations, and packing of struct arrays to and from GPU bytes.
 *
 * Supported members: float, double (as f32), int32, uint32, int8/int16/uint8/uint16/enums (widened to
 * i32/u32), bool (u32 0/1), FVector2f/3f/4f, FVector2D/FVector/FVector4 (as f32), FLinearColor,
 * FQuat4f/FQuat, FRotator(3f) (pitch, yaw, roll), FIntPoint, FIntVector, FIntVector4, FColor (u32, bytes
 * B,G,R,A), FMatrix44f/FMatrix (mat4x4<f32>, UE's row-major memory), nested UStructs, and fixed C arrays.
 * When the UE memory layout already equals the WGSL layout, packing is a single memcpy.
 * Build layouts on the game thread (they read UObject reflection); packing is thread-safe.
 */
class WEBGPUCOMPUTE_API FWebGPUStructLayout
{
public:
	enum class EConvert : uint8
	{
		Copy,
		FloatFromDouble,
		I32FromInt8,
		I32FromInt16,
		U32FromUInt8,
		U32FromUInt16,
		U32FromBool,
		Nested,
	};

	struct FField
	{
		const FProperty* Property = nullptr;
		/** The property's name. Packaged builds don't keep FName case, so it can differ in case from the C++ name. */
		FString Name;
		FString WgslName;
		FString WgslType;
		/** WGSL offset of the first element, size of one element, and its alignment. */
		uint32 Offset = 0;
		uint32 Size = 0;
		uint32 Align = 0;
		/** Fixed C arrays become array<T, N>. */
		int32 ArrayDim = 1;
		uint32 ArrayStride = 0;
		/** UE side: bytes per element, and how many scalar components a conversion touches. */
		uint32 SourceElementSize = 0;
		uint32 CopySize = 0;
		uint32 Components = 1;
		EConvert Convert = EConvert::Copy;
		TSharedPtr<const FWebGPUStructLayout> Nested;
		/** Explicit @align needed in generated WGSL (uniform rules), or 0. */
		uint32 ExplicitAlign = 0;
	};

	/** Cached layout for a struct, or null (OutError says why: e.g. an unsupported member type). */
	static TSharedPtr<const FWebGPUStructLayout> Get(const UScriptStruct* Struct, EWebGPUAddressSpace Space, FString& OutError);
	static TSharedPtr<const FWebGPUStructLayout> Get(const UScriptStruct* Struct, EWebGPUAddressSpace Space = EWebGPUAddressSpace::Storage);

	const FString& GetWgslName() const { return WgslName; }
	const FString& GetStructName() const { return StructName; }
	EWebGPUAddressSpace GetAddressSpace() const { return Space; }
	uint32 GetSize() const { return Size; }
	uint32 GetAlign() const { return AlignBytes; }
	/** Bytes per element in an array<T> on the GPU. */
	uint32 GetStride() const { return Stride; }
	/** Bytes per element in UE memory (TArray stride). */
	uint32 GetSourceStride() const { return SourceStride; }
	/** True when UE memory already matches the GPU layout, so packing is one memcpy. */
	bool IsMemcpyCompatible() const { return bMemcpyCompatible; }
	const TArray<FField>& GetFields() const { return Fields; }

	/** WGSL declarations for this struct and the structs it nests, dependencies first. */
	FString GenerateWGSL() const;

	/** Declarations for several structs, each nested struct declared once. */
	static FString GenerateWGSL(TConstArrayView<TSharedPtr<const FWebGPUStructLayout>> Layouts);

	/**
	 * Declarations for a kernel's source: struct and field names take the source's spelling when they differ from it
	 * only in case. UE names come from FNames, whose case packaged builds don't keep ("Position" can come back as
	 * "position"), so a kernel that compiles in the editor keeps compiling in a packaged game.
	 */
	static FString GenerateWGSL(TConstArrayView<TSharedPtr<const FWebGPUStructLayout>> Layouts, const FString& Source);

	/** Packs Count elements (UE memory, SourceStride apart) into Dest (Count * GetStride() bytes). */
	void Pack(const void* Elements, int32 Count, uint8* Dest) const;
	TArray<uint8> Pack(const void* Elements, int32 Count) const;

	/** Unpacks into Count already-constructed UE structs. */
	void Unpack(const uint8* Source, int32 Count, void* Elements) const;

	/** Compares against a struct Tint reflected from a shader (names ignore case). False lists the differences. */
	bool Matches(const FWebGPUTypeLayout& Reflected, FString& OutDifferences) const;

	/** WGSL-safe identifier for a name (invalid characters, leading digits and keywords handled). */
	static FString SanitizeIdentifier(const FString& Name);

private:
	friend class FWebGPUStructLayoutBuilder;

	void PackElement(const uint8* Source, uint8* Dest) const;
	void UnpackElement(const uint8* Source, uint8* Dest) const;
	void GenerateWGSL(TArray<FString>& Declared, FString& Out, const FWebGPUSourceSpellings* Spellings) const;

	FString StructName;
	FString WgslName;
	EWebGPUAddressSpace Space = EWebGPUAddressSpace::Storage;
	uint32 Size = 0;
	uint32 AlignBytes = 1;
	uint32 Stride = 0;
	uint32 SourceStride = 0;
	bool bMemcpyCompatible = false;
	TArray<FField> Fields;
};
