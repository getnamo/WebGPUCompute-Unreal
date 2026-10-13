// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUReflection.h"

// Tint (Dawn's WGSL compiler) is compiled into webgpu_dawn.lib; its headers come from Tools/BuildDawn.ps1.
// They must see the same configuration the lib was built with (see the Dawn build's tint compile defines).
#define DAWN_HAVE_UTILS_BUILD_CONFIG 1
#if !defined(NDEBUG)
#define NDEBUG 1
#define WEBGPU_DEFINED_NDEBUG 1
#endif
#define TINT_BUILD_GLSL_VALIDATOR 0
#define TINT_BUILD_GLSL_WRITER 0
#define TINT_BUILD_HLSL_WRITER 1
#define TINT_BUILD_IR_BINARY 0
#define TINT_BUILD_IS_LINUX 0
#define TINT_BUILD_IS_MAC 0
#define TINT_BUILD_IS_WIN 1
#define TINT_BUILD_MSL_WRITER 0
#define TINT_BUILD_NULL_WRITER 1
#define TINT_BUILD_SPV_READER 0
#define TINT_BUILD_SPV_WRITER 0
#define TINT_BUILD_WGSL_READER 1
#define TINT_BUILD_WGSL_WRITER 1
#define TINT_ENABLE_IR_DUMPING 1
#define TINT_ENABLE_IR_VALIDATION_ASSERTS 1

THIRD_PARTY_INCLUDES_START
#include "src/tint/lang/core/type/array.h"
#include "src/tint/lang/core/type/atomic.h"
#include "src/tint/lang/core/type/matrix.h"
#include "src/tint/lang/core/type/reference.h"
#include "src/tint/lang/core/type/scalar.h"
#include "src/tint/lang/core/type/struct.h"
#include "src/tint/lang/core/type/vector.h"
#include "src/tint/lang/wgsl/allowed_features.h"
#include "src/tint/lang/wgsl/ast/module.h"
#include "src/tint/lang/wgsl/ast/variable.h"
#include "src/tint/lang/wgsl/inspector/inspector.h"
#include "src/tint/lang/wgsl/program/program.h"
#include "src/tint/lang/wgsl/reader/reader.h"
#include "src/tint/lang/wgsl/sem/struct.h"
#include "src/tint/lang/wgsl/sem/variable.h"
THIRD_PARTY_INCLUDES_END

#if defined(WEBGPU_DEFINED_NDEBUG)
#undef NDEBUG
#undef WEBGPU_DEFINED_NDEBUG
#endif

namespace
{
	FString ToFString(const std::string& Str)
	{
		return FString(UTF8_TO_TCHAR(Str.c_str()));
	}

	/** Converts Tint types to our layout tree, sharing struct layouts by name. */
	class FLayoutBuilder
	{
	public:
		TMap<FString, TSharedPtr<const FWebGPUTypeLayout>>& Structs;

		explicit FLayoutBuilder(TMap<FString, TSharedPtr<const FWebGPUTypeLayout>>& InStructs)
			: Structs(InStructs)
		{
		}

		TSharedPtr<const FWebGPUTypeLayout> Build(const tint::core::type::Type* Type)
		{
			if (!Type)
			{
				return nullptr;
			}
			Type = Type->UnwrapRef();

			if (const tint::core::type::Struct* Struct = Type->As<tint::core::type::Struct>())
			{
				const FString StructName = ToFString(Struct->Name().Name());
				if (const TSharedPtr<const FWebGPUTypeLayout>* Existing = Structs.Find(StructName))
				{
					return *Existing;
				}

				TSharedPtr<FWebGPUTypeLayout> Layout = MakeShared<FWebGPUTypeLayout>();
				Layout->Kind = FWebGPUTypeLayout::EKind::Struct;
				Layout->Name = StructName;
				Layout->Size = Struct->Size();
				Layout->Align = Struct->Align();
				for (const tint::core::type::StructMember* Member : Struct->Members())
				{
					FWebGPUTypeLayout::FField& Field = Layout->Fields.AddDefaulted_GetRef();
					Field.Name = ToFString(Member->Name().Name());
					Field.Offset = Member->Offset();
					Field.Size = Member->Size();
					Field.Type = Build(Member->Type());
				}
				Structs.Add(StructName, Layout);
				return Layout;
			}

			TSharedPtr<FWebGPUTypeLayout> Layout = MakeShared<FWebGPUTypeLayout>();
			Layout->Name = ToFString(Type->FriendlyName());
			Layout->Size = Type->Size();
			Layout->Align = Type->Align();

			if (const tint::core::type::Array* Array = Type->As<tint::core::type::Array>())
			{
				Layout->Kind = FWebGPUTypeLayout::EKind::Array;
				Layout->Element = Build(Array->ElemType());
				Layout->Stride = Array->ImplicitStride();
				const std::optional<uint32_t> Count = Array->ConstantCount();
				Layout->Count = Count ? (int32)*Count : -1;
			}
			else if (const tint::core::type::Vector* Vector = Type->As<tint::core::type::Vector>())
			{
				Layout->Kind = FWebGPUTypeLayout::EKind::Vector;
				Layout->Rows = Vector->Width();
				Layout->ScalarName = ToFString(Vector->Type()->FriendlyName());
			}
			else if (const tint::core::type::Matrix* Matrix = Type->As<tint::core::type::Matrix>())
			{
				Layout->Kind = FWebGPUTypeLayout::EKind::Matrix;
				Layout->Rows = Matrix->Rows();
				Layout->Columns = Matrix->Columns();
				Layout->ScalarName = ToFString(Matrix->ColumnType()->Type()->FriendlyName());
			}
			else if (const tint::core::type::Atomic* Atomic = Type->As<tint::core::type::Atomic>())
			{
				Layout->Kind = FWebGPUTypeLayout::EKind::Atomic;
				Layout->ScalarName = ToFString(Atomic->Type()->FriendlyName());
			}
			else if (Type->Is<tint::core::type::Scalar>())
			{
				Layout->Kind = FWebGPUTypeLayout::EKind::Scalar;
				Layout->ScalarName = Layout->Name;
			}
			return Layout;
		}
	};

	EWebGPUBindingType ToBindingType(tint::inspector::ResourceBinding::ResourceType Type)
	{
		using EType = tint::inspector::ResourceBinding::ResourceType;
		switch (Type)
		{
		case EType::kUniformBuffer: return EWebGPUBindingType::UniformBuffer;
		case EType::kStorageBuffer: return EWebGPUBindingType::StorageBuffer;
		case EType::kReadOnlyStorageBuffer: return EWebGPUBindingType::ReadOnlyStorageBuffer;
		case EType::kSampler: return EWebGPUBindingType::Sampler;
		case EType::kSampledTexture:
		case EType::kMultisampledTexture:
		case EType::kDepthTexture:
		case EType::kDepthMultisampledTexture: return EWebGPUBindingType::SampledTexture;
		case EType::kWriteOnlyStorageTexture:
		case EType::kReadOnlyStorageTexture:
		case EType::kReadWriteStorageTexture: return EWebGPUBindingType::StorageTexture;
		default: return EWebGPUBindingType::Other;
		}
	}
}

const FWebGPUTypeLayout::FField* FWebGPUTypeLayout::FindField(const FString& FieldName) const
{
	return Fields.FindByPredicate([&FieldName](const FField& Field) { return Field.Name == FieldName; });
}

TSharedPtr<const FWebGPUTypeLayout> FWebGPUBindingInfo::GetRuntimeArrayElement() const
{
	if (!Layout)
	{
		return nullptr;
	}
	if (Layout->Kind == FWebGPUTypeLayout::EKind::Array && Layout->Count < 0)
	{
		return Layout->Element;
	}
	if (Layout->Kind == FWebGPUTypeLayout::EKind::Struct && Layout->Fields.Num() > 0)
	{
		const TSharedPtr<const FWebGPUTypeLayout>& Last = Layout->Fields.Last().Type;
		if (Last && Last->Kind == FWebGPUTypeLayout::EKind::Array && Last->Count < 0)
		{
			return Last->Element;
		}
	}
	return nullptr;
}

const FWebGPUBindingInfo* FWebGPUEntryPointInfo::FindBinding(const FString& BindingName) const
{
	return Bindings.FindByPredicate([&BindingName](const FWebGPUBindingInfo& Info) { return Info.Name == BindingName; });
}

const FWebGPUBindingInfo* FWebGPUEntryPointInfo::FindBinding(uint32 Group, uint32 Binding) const
{
	return Bindings.FindByPredicate([Group, Binding](const FWebGPUBindingInfo& Info) { return Info.Group == (int32)Group && Info.Binding == (int32)Binding; });
}

const FWebGPUEntryPointInfo* FWebGPUShaderReflection::FindEntryPoint(const FString& Name) const
{
	return EntryPoints.FindByPredicate([&Name](const FWebGPUEntryPointInfo& Info) { return Info.Name == Name; });
}

bool FWebGPUShaderReflection::Reflect(const FString& Source, FWebGPUShaderReflection& OutReflection, TArray<FWebGPUCompileMessage>& OutMessages)
{
	OutReflection = FWebGPUShaderReflection();

	const FTCHARToUTF8 Utf8(*Source);
	const tint::Source::File File("shader.wgsl", std::string_view(Utf8.Get(), Utf8.Length()));
	tint::wgsl::reader::Options Options;
	Options.allowed_features = tint::wgsl::AllowedFeatures::Everything();
	const tint::Program Program = tint::wgsl::reader::Parse(&File, Options);

	for (const tint::diag::Diagnostic& Diagnostic : Program.Diagnostics())
	{
		if (Diagnostic.severity == tint::diag::Severity::Note)
		{
			continue;
		}
		FWebGPUCompileMessage& Message = OutMessages.AddDefaulted_GetRef();
		Message.bIsError = Diagnostic.severity == tint::diag::Severity::Error;
		Message.Message = ToFString(Diagnostic.message.Plain());
		Message.Line = Diagnostic.source.range.begin.line;
		Message.Column = Diagnostic.source.range.begin.column;
	}
	if (!Program.IsValid())
	{
		return false;
	}

	FLayoutBuilder Builder(OutReflection.Structs);

	// Layouts of every declared struct, and of every resource variable by binding point
	for (const tint::ast::TypeDecl* Decl : Program.AST().TypeDecls())
	{
		if (const tint::sem::Struct* Struct = Program.Sem().Get<tint::sem::Struct>(Decl))
		{
			Builder.Build(Struct);
		}
	}

	TMap<TPair<uint32, uint32>, TSharedPtr<const FWebGPUTypeLayout>> ResourceLayouts;
	for (const tint::ast::Variable* Var : Program.AST().GlobalVariables())
	{
		const tint::sem::GlobalVariable* Global = Program.Sem().Get<tint::sem::GlobalVariable>(Var);
		if (Global && Global->Attributes().binding_point)
		{
			const tint::BindingPoint& Point = *Global->Attributes().binding_point;
			ResourceLayouts.Add(TPair<uint32, uint32>(Point.group, Point.binding), Builder.Build(Global->Type()));
		}
	}

	tint::inspector::Inspector Inspector(Program);
	for (const tint::inspector::EntryPoint& EntryPoint : Inspector.GetEntryPoints())
	{
		FWebGPUEntryPointInfo& Info = OutReflection.EntryPoints.AddDefaulted_GetRef();
		Info.Name = ToFString(EntryPoint.name);
		Info.bCompute = EntryPoint.stage == tint::inspector::PipelineStage::kCompute;
		if (EntryPoint.workgroup_size)
		{
			Info.WorkgroupSize = FIntVector((int32)EntryPoint.workgroup_size->x, (int32)EntryPoint.workgroup_size->y, (int32)EntryPoint.workgroup_size->z);
		}
		for (const tint::inspector::Override& Override : EntryPoint.overrides)
		{
			Info.Overrides.Add(ToFString(Override.name));
		}

		for (const tint::inspector::ResourceBinding& Resource : Inspector.GetResourceBindings(EntryPoint.name))
		{
			FWebGPUBindingInfo& Binding = Info.Bindings.AddDefaulted_GetRef();
			Binding.Name = ToFString(Resource.variable_name);
			Binding.Group = (int32)Resource.bind_group;
			Binding.Binding = (int32)Resource.binding;
			Binding.Type = ToBindingType(Resource.resource_type);
			if (const TSharedPtr<const FWebGPUTypeLayout>* Layout = ResourceLayouts.Find(TPair<uint32, uint32>(Resource.bind_group, Resource.binding)))
			{
				Binding.Layout = *Layout;
				Binding.TypeName = (*Layout)->Name;
			}

			if (Binding.IsBuffer() && Binding.Layout)
			{
				// Runtime-sized arrays report their fixed part; require at least one element
				uint64 MinSize = Binding.Layout->Size;
				if (TSharedPtr<const FWebGPUTypeLayout> Element = Binding.GetRuntimeArrayElement())
				{
					const uint64 Stride = Align(FMath::Max<uint64>(Element->Size, 1), FMath::Max<uint64>(Element->Align, 1));
					MinSize = FMath::Max<uint64>(MinSize, Binding.Layout->Kind == FWebGPUTypeLayout::EKind::Array ? Stride : Binding.Layout->Fields.Last().Offset + Stride);
				}
				Binding.MinSize = (int64)MinSize;
			}
		}
		Info.Bindings.Sort([](const FWebGPUBindingInfo& A, const FWebGPUBindingInfo& B)
		{
			return A.Group != B.Group ? A.Group < B.Group : A.Binding < B.Binding;
		});
	}
	return true;
}
