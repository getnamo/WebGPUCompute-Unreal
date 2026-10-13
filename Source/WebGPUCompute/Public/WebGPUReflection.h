// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "WebGPUDevice.h"
#include "WebGPUReflection.generated.h"

/** What a shader binding is. */
UENUM(BlueprintType)
enum class EWebGPUBindingType : uint8
{
	UniformBuffer,
	StorageBuffer,
	ReadOnlyStorageBuffer,
	Sampler,
	SampledTexture,
	StorageTexture,
	Other,
};

/** Host-shareable layout of a WGSL type, as WGSL computes it. */
struct WEBGPUCOMPUTE_API FWebGPUTypeLayout
{
	enum class EKind : uint8
	{
		Scalar,
		Vector,
		Matrix,
		Array,
		Struct,
		Atomic,
		Other,
	};

	struct FField
	{
		FString Name;
		uint32 Offset = 0;
		uint32 Size = 0;
		TSharedPtr<const FWebGPUTypeLayout> Type;
	};

	EKind Kind = EKind::Other;

	/** WGSL spelling, e.g. "f32", "vec3<f32>", "Agent", "array<Agent>". */
	FString Name;
	uint32 Size = 0;
	uint32 Align = 0;

	/** Scalar, Vector and Matrix: the component scalar ("f32", "i32", "u32", "f16", "bool"). */
	FString ScalarName;
	/** Vector width, or matrix rows. */
	uint32 Rows = 1;
	/** Matrix columns. */
	uint32 Columns = 1;

	/** Array: element layout, stride, and element count (-1 for runtime-sized arrays). */
	TSharedPtr<const FWebGPUTypeLayout> Element;
	uint32 Stride = 0;
	int32 Count = -1;

	/** Struct members in declaration order. */
	TArray<FField> Fields;

	const FField* FindField(const FString& FieldName) const;
};

/** One resource an entry point uses. */
USTRUCT(BlueprintType)
struct WEBGPUCOMPUTE_API FWebGPUBindingInfo
{
	GENERATED_BODY()

	/** The WGSL variable name. */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU")
	FString Name;

	UPROPERTY(BlueprintReadOnly, Category = "WebGPU")
	int32 Group = 0;

	UPROPERTY(BlueprintReadOnly, Category = "WebGPU")
	int32 Binding = 0;

	UPROPERTY(BlueprintReadOnly, Category = "WebGPU")
	EWebGPUBindingType Type = EWebGPUBindingType::Other;

	/** Smallest buffer binding in bytes that the shader accepts (the fixed part plus one runtime-array element). */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU")
	int64 MinSize = 0;

	/** WGSL store type, e.g. "array<Agent>" or "Params". */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU")
	FString TypeName;

	/** Textures: "1d", "2d", "2d_array", "3d", "cube" or "cube_array". */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU")
	FString TextureDimension;

	/** Storage textures: the WGSL texel format, e.g. "rgba16float". */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU")
	FString TexelFormat;

	/** Storage textures: "write", "read" or "read_write". */
	UPROPERTY(BlueprintReadOnly, Category = "WebGPU")
	FString Access;

	/** Buffers: the full layout of the store type. */
	TSharedPtr<const FWebGPUTypeLayout> Layout;

	bool IsBuffer() const { return Type == EWebGPUBindingType::UniformBuffer || Type == EWebGPUBindingType::StorageBuffer || Type == EWebGPUBindingType::ReadOnlyStorageBuffer; }
	bool IsTexture() const { return Type == EWebGPUBindingType::SampledTexture || Type == EWebGPUBindingType::StorageTexture; }

	/** Runtime-sized arrays: the element layout (also for a struct whose last member is the runtime array). */
	TSharedPtr<const FWebGPUTypeLayout> GetRuntimeArrayElement() const;
};

struct WEBGPUCOMPUTE_API FWebGPUEntryPointInfo
{
	FString Name;
	bool bCompute = false;

	/** @workgroup_size; zero when it depends on override constants without defaults. */
	FIntVector WorkgroupSize = FIntVector::ZeroValue;

	/** Names of override constants the entry point uses. */
	TArray<FString> Overrides;

	TArray<FWebGPUBindingInfo> Bindings;

	const FWebGPUBindingInfo* FindBinding(const FString& BindingName) const;
	const FWebGPUBindingInfo* FindBinding(uint32 Group, uint32 Binding) const;
};

/** What a WGSL module declares and uses, computed by Tint (the WGSL compiler inside Dawn). */
struct WEBGPUCOMPUTE_API FWebGPUShaderReflection
{
	TArray<FWebGPUEntryPointInfo> EntryPoints;

	/** Every struct the module declares, by name. */
	TMap<FString, TSharedPtr<const FWebGPUTypeLayout>> Structs;

	const FWebGPUEntryPointInfo* FindEntryPoint(const FString& Name) const;

	/** Parses and resolves WGSL. On failure, OutMessages has the errors (line/column) and false is returned. */
	static bool Reflect(const FString& Source, FWebGPUShaderReflection& OutReflection, TArray<FWebGPUCompileMessage>& OutMessages);
};
