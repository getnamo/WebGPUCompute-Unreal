// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUStructLayout.h"
#include "WebGPUCompute.h"
#include "WebGPUReflection.h"
#include "Misc/ScopeLock.h"
#include "UObject/Package.h"
#include "UObject/UnrealType.h"

namespace
{
	/** A core math struct and how it maps onto WGSL. */
	struct FKnownStruct
	{
		const TCHAR* Name;
		const TCHAR* WgslType;
		uint32 Components;
		bool bDoubleSource;
		uint32 WgslSize;
		uint32 WgslAlign;
	};

	const FKnownStruct KnownStructs[] = {
		{ TEXT("Vector2f"),    TEXT("vec2<f32>"),   2,  false, 8,  8 },
		{ TEXT("Vector3f"),    TEXT("vec3<f32>"),   3,  false, 12, 16 },
		{ TEXT("Vector4f"),    TEXT("vec4<f32>"),   4,  false, 16, 16 },
		{ TEXT("LinearColor"), TEXT("vec4<f32>"),   4,  false, 16, 16 },
		{ TEXT("Quat4f"),      TEXT("vec4<f32>"),   4,  false, 16, 16 },
		{ TEXT("Rotator3f"),   TEXT("vec3<f32>"),   3,  false, 12, 16 },
		{ TEXT("Vector2D"),    TEXT("vec2<f32>"),   2,  true,  8,  8 },
		{ TEXT("Vector"),      TEXT("vec3<f32>"),   3,  true,  12, 16 },
		{ TEXT("Vector4"),     TEXT("vec4<f32>"),   4,  true,  16, 16 },
		{ TEXT("Quat"),        TEXT("vec4<f32>"),   4,  true,  16, 16 },
		{ TEXT("Rotator"),     TEXT("vec3<f32>"),   3,  true,  12, 16 },
		{ TEXT("IntPoint"),    TEXT("vec2<i32>"),   2,  false, 8,  8 },
		{ TEXT("IntVector"),   TEXT("vec3<i32>"),   3,  false, 12, 16 },
		{ TEXT("IntVector4"),  TEXT("vec4<i32>"),   4,  false, 16, 16 },
		{ TEXT("Color"),       TEXT("u32"),         1,  false, 4,  4 },
		{ TEXT("Matrix44f"),   TEXT("mat4x4<f32>"), 16, false, 64, 16 },
		{ TEXT("Matrix"),      TEXT("mat4x4<f32>"), 16, true,  64, 16 },
	};

	const FKnownStruct* FindKnownStruct(const UScriptStruct* Struct)
	{
		static const FName CoreUObjectPackage(TEXT("/Script/CoreUObject"));
		if (Struct->GetOutermost()->GetFName() != CoreUObjectPackage)
		{
			return nullptr;
		}
		const FString Name = Struct->GetName();
		for (const FKnownStruct& Known : KnownStructs)
		{
			if (Name == Known.Name)
			{
				return &Known;
			}
		}
		return nullptr;
	}

	const TCHAR* const WgslKeywords[] = {
		TEXT("alias"), TEXT("break"), TEXT("case"), TEXT("const"), TEXT("const_assert"), TEXT("continue"), TEXT("continuing"),
		TEXT("default"), TEXT("diagnostic"), TEXT("discard"), TEXT("else"), TEXT("enable"), TEXT("false"), TEXT("fn"), TEXT("for"),
		TEXT("if"), TEXT("let"), TEXT("loop"), TEXT("override"), TEXT("requires"), TEXT("return"), TEXT("struct"), TEXT("switch"),
		TEXT("true"), TEXT("var"), TEXT("while"), TEXT("f32"), TEXT("f16"), TEXT("i32"), TEXT("u32"), TEXT("bool"), TEXT("array"),
		TEXT("atomic"), TEXT("ptr"), TEXT("sampler"), TEXT("vec2"), TEXT("vec3"), TEXT("vec4"), TEXT("mat2x2"), TEXT("mat3x3"), TEXT("mat4x4"),
		TEXT("self"), TEXT("target"), TEXT("type"), TEXT("this"), TEXT("null"), TEXT("set"), TEXT("get"), TEXT("in"), TEXT("out"),
		TEXT("inout"), TEXT("static"), TEXT("public"), TEXT("private"), TEXT("uniform"), TEXT("storage"), TEXT("workgroup"), TEXT("function"),
	};

	struct FCacheKey
	{
		const UScriptStruct* Struct;
		EWebGPUAddressSpace Space;

		bool operator==(const FCacheKey& Other) const { return Struct == Other.Struct && Space == Other.Space; }
		friend uint32 GetTypeHash(const FCacheKey& Key) { return HashCombine(GetTypeHash(Key.Struct), (uint32)Key.Space); }
	};

	FCriticalSection CacheLock;
	TMap<FCacheKey, TSharedPtr<const FWebGPUStructLayout>> Cache;

	/** True when UE property ranges plus alignment padding explain every byte (no hidden non-UPROPERTY members). */
	bool PropertiesCoverStruct(const UScriptStruct* Struct)
	{
		TArray<const FProperty*> Properties;
		for (TFieldIterator<FProperty> It(Struct); It; ++It)
		{
			Properties.Add(*It);
		}
		Properties.Sort([](const FProperty& A, const FProperty& B) { return A.GetOffset_ForInternal() < B.GetOffset_ForInternal(); });

		int32 End = 0;
		for (const FProperty* Property : Properties)
		{
			const int32 Offset = Property->GetOffset_ForInternal();
			if (Offset < End || Offset - End >= FMath::Max(Property->GetMinAlignment(), 1))
			{
				return false;
			}
			End = Offset + Property->GetSize();
		}
		return Struct->GetStructureSize() - End < FMath::Max(Struct->GetMinAlignment(), 1);
	}
}

/** How a kernel's source spells its identifiers (comments skipped), keyed without case. */
class FWebGPUSourceSpellings
{
public:
	explicit FWebGPUSourceSpellings(const FString& Source)
	{
		const TCHAR* Text = *Source;
		const int32 Length = Source.Len();
		int32 Index = 0;
		while (Index < Length)
		{
			if (Text[Index] == TEXT('/') && Index + 1 < Length && Text[Index + 1] == TEXT('/'))
			{
				while (Index < Length && Text[Index] != TEXT('\n'))
				{
					Index++;
				}
				continue;
			}
			if (Text[Index] == TEXT('/') && Index + 1 < Length && Text[Index + 1] == TEXT('*'))
			{
				// WGSL block comments nest
				int32 Depth = 0;
				do
				{
					if (Text[Index] == TEXT('/') && Index + 1 < Length && Text[Index + 1] == TEXT('*'))
					{
						Depth++;
						Index += 2;
					}
					else if (Text[Index] == TEXT('*') && Index + 1 < Length && Text[Index + 1] == TEXT('/'))
					{
						Depth--;
						Index += 2;
					}
					else
					{
						Index++;
					}
				} while (Depth > 0 && Index < Length);
				continue;
			}
			if (FChar::IsAlpha(Text[Index]) || Text[Index] == TEXT('_'))
			{
				const int32 Start = Index;
				while (Index < Length && (FChar::IsAlnum(Text[Index]) || Text[Index] == TEXT('_')))
				{
					Index++;
				}
				int32 Before = Start - 1;
				while (Before >= 0 && FChar::IsWhitespace(Text[Before]))
				{
					Before--;
				}
				// FString keys compare without case: one entry per name, holding every spelling in source order
				FString Word = FString::ConstructFromPtrSize(Text + Start, Index - Start);
				TArray<FString>& Spellings = (Before >= 0 && Text[Before] == TEXT('.') ? Members : Others).FindOrAdd(Word);
				if (!Spellings.ContainsByPredicate([&Word](const FString& Spelling) { return Spelling.Equals(Word, ESearchCase::CaseSensitive); }))
				{
					Spellings.Add(MoveTemp(Word));
				}
				continue;
			}
			Index++;
		}
	}

	/** Name as the source spells it: unchanged when the source uses it as is (or not at all). Fields prefer member accesses. */
	FString Spell(const FString& Name, bool bField) const
	{
		const TArray<FString>* Preferred = (bField ? Members : Others).Find(Name);
		const TArray<FString>* Fallback = (bField ? Others : Members).Find(Name);
		for (const TArray<FString>* Spellings : { Preferred, Fallback })
		{
			if (Spellings && Spellings->ContainsByPredicate([&Name](const FString& Spelling) { return Spelling.Equals(Name, ESearchCase::CaseSensitive); }))
			{
				return Name;
			}
		}
		const TArray<FString>* Found = Preferred ? Preferred : Fallback;
		return Found && Found->Num() > 0 ? (*Found)[0] : Name;
	}

private:
	/** Identifiers right after a '.', and all the others */
	TMap<FString, TArray<FString>> Members;
	TMap<FString, TArray<FString>> Others;
};

/** Builds layouts from UObject reflection (game thread). */
class FWebGPUStructLayoutBuilder
{
public:
	static TSharedPtr<FWebGPUStructLayout> Build(const UScriptStruct* Struct, EWebGPUAddressSpace Space, FString& OutError, int32 Depth)
	{
		if (Depth > 16)
		{
			OutError = FString::Printf(TEXT("%s: struct nesting too deep"), *Struct->GetName());
			return nullptr;
		}

		TSharedPtr<FWebGPUStructLayout> Layout = MakeShared<FWebGPUStructLayout>();
		Layout->StructName = Struct->GetName();
		Layout->WgslName = FWebGPUStructLayout::SanitizeIdentifier(Struct->GetName());
		Layout->Space = Space;
		Layout->SourceStride = (uint32)Struct->GetStructureSize();

		uint32 Offset = 0;
		uint32 MaxAlign = 1;
		bool bMemcpy = true;
		for (TFieldIterator<FProperty> It(Struct); It; ++It)
		{
			const FProperty* Property = *It;
			FWebGPUStructLayout::FField Field;
			Field.Property = Property;
			Field.Name = Property->GetAuthoredName();
			Field.WgslName = FWebGPUStructLayout::SanitizeIdentifier(Field.Name);
			Field.ArrayDim = Property->ArrayDim;
			Field.SourceElementSize = (uint32)Property->GetElementSize();

			FString Error;
			if (!Classify(Property, Space, Field, Error, Depth))
			{
				OutError = FString::Printf(TEXT("%s.%s: %s"), *Struct->GetName(), *Property->GetAuthoredName(), *Error);
				return nullptr;
			}

			uint32 FieldAlign = Field.Align;
			uint32 FieldTotalSize = Field.Size;
			if (Field.ArrayDim > 1)
			{
				Field.ArrayStride = Align(Field.Size, Field.Align);
				FieldTotalSize = Field.ArrayStride * Field.ArrayDim;
				if (Space == EWebGPUAddressSpace::Uniform && Field.ArrayStride % 16 != 0)
				{
					OutError = FString::Printf(TEXT("%s.%s: uniform arrays need 16-byte elements (use vec4 or a struct)"), *Struct->GetName(), *Property->GetAuthoredName());
					return nullptr;
				}
				Field.WgslType = FString::Printf(TEXT("array<%s, %d>"), *Field.WgslType, Field.ArrayDim);
			}

			// Uniform buffers place nested structs and arrays at 16-byte boundaries
			if (Space == EWebGPUAddressSpace::Uniform && (Field.Convert == FWebGPUStructLayout::EConvert::Nested || Field.ArrayDim > 1))
			{
				const uint32 Required = FMath::Max<uint32>(16, FieldAlign);
				if (Required != FieldAlign)
				{
					Field.ExplicitAlign = Required;
				}
				FieldAlign = Required;
			}

			Offset = Align(Offset, FieldAlign);
			Field.Offset = Offset;
			Offset += FieldTotalSize;
			MaxAlign = FMath::Max(MaxAlign, FieldAlign);

			// Same bytes in the same place on both sides?
			const bool bSameArrayStride = Field.ArrayDim == 1 || Field.ArrayStride == Field.SourceElementSize;
			const bool bSameOffset = Field.Offset == (uint32)Property->GetOffset_ForInternal();
			if (Field.Convert == FWebGPUStructLayout::EConvert::Copy)
			{
				bMemcpy &= bSameOffset && bSameArrayStride && Field.CopySize == Field.SourceElementSize;
			}
			else if (Field.Convert == FWebGPUStructLayout::EConvert::Nested)
			{
				bMemcpy &= bSameOffset && bSameArrayStride && Field.Nested->IsMemcpyCompatible() && Field.Nested->GetStride() == Field.SourceElementSize;
			}
			else
			{
				bMemcpy = false;
			}

			Layout->Fields.Add(MoveTemp(Field));
		}

		if (Layout->Fields.Num() == 0)
		{
			OutError = FString::Printf(TEXT("%s has no properties"), *Struct->GetName());
			return nullptr;
		}

		Layout->AlignBytes = MaxAlign;
		Layout->Size = Align(Offset, MaxAlign);
		Layout->Stride = Layout->Size;
		Layout->bMemcpyCompatible = bMemcpy && Layout->Stride == Layout->SourceStride && PropertiesCoverStruct(Struct);
		return Layout;
	}

private:
	static void SetScalar(FWebGPUStructLayout::FField& Field, const TCHAR* Type, FWebGPUStructLayout::EConvert Convert, uint32 CopySize)
	{
		Field.WgslType = Type;
		Field.Size = 4;
		Field.Align = 4;
		Field.Convert = Convert;
		Field.CopySize = CopySize;
	}

	static bool Classify(const FProperty* Property, EWebGPUAddressSpace Space, FWebGPUStructLayout::FField& Field, FString& OutError, int32 Depth)
	{
		using EConvert = FWebGPUStructLayout::EConvert;

		const FProperty* Numeric = Property;
		if (const FEnumProperty* EnumProperty = CastField<FEnumProperty>(Property))
		{
			Numeric = EnumProperty->GetUnderlyingProperty();
		}

		if (Numeric->IsA<FFloatProperty>())
		{
			SetScalar(Field, TEXT("f32"), EConvert::Copy, 4);
		}
		else if (Numeric->IsA<FDoubleProperty>())
		{
			SetScalar(Field, TEXT("f32"), EConvert::FloatFromDouble, 0);
		}
		else if (Numeric->IsA<FIntProperty>())
		{
			SetScalar(Field, TEXT("i32"), EConvert::Copy, 4);
		}
		else if (Numeric->IsA<FUInt32Property>())
		{
			SetScalar(Field, TEXT("u32"), EConvert::Copy, 4);
		}
		else if (Numeric->IsA<FInt8Property>())
		{
			SetScalar(Field, TEXT("i32"), EConvert::I32FromInt8, 0);
		}
		else if (Numeric->IsA<FInt16Property>())
		{
			SetScalar(Field, TEXT("i32"), EConvert::I32FromInt16, 0);
		}
		else if (Numeric->IsA<FByteProperty>())
		{
			SetScalar(Field, TEXT("u32"), EConvert::U32FromUInt8, 0);
		}
		else if (Numeric->IsA<FUInt16Property>())
		{
			SetScalar(Field, TEXT("u32"), EConvert::U32FromUInt16, 0);
		}
		else if (Property->IsA<FBoolProperty>())
		{
			SetScalar(Field, TEXT("u32"), EConvert::U32FromBool, 0);
		}
		else if (const FStructProperty* StructProperty = CastField<FStructProperty>(Property))
		{
			if (const FKnownStruct* Known = FindKnownStruct(StructProperty->Struct))
			{
				Field.WgslType = Known->WgslType;
				Field.Size = Known->WgslSize;
				Field.Align = Known->WgslAlign;
				Field.Components = Known->Components;
				Field.Convert = Known->bDoubleSource ? EConvert::FloatFromDouble : EConvert::Copy;
				Field.CopySize = Known->bDoubleSource ? 0 : Known->Components * 4;
			}
			else
			{
				FString NestedError;
				TSharedPtr<FWebGPUStructLayout> Nested = FWebGPUStructLayoutBuilder::Build(StructProperty->Struct, Space, NestedError, Depth + 1);
				if (!Nested)
				{
					OutError = NestedError;
					return false;
				}
				Field.WgslType = Nested->GetWgslName();
				Field.Size = Nested->GetSize();
				Field.Align = Nested->GetAlign();
				Field.Convert = EConvert::Nested;
				Field.Nested = Nested;
			}
		}
		else
		{
			OutError = FString::Printf(TEXT("%s is not supported on the GPU (numbers, bools, enums, math structs, nested structs and fixed arrays are)"), *Property->GetCPPType());
			return false;
		}
		return true;
	}
};

FString FWebGPUStructLayout::SanitizeIdentifier(const FString& Name)
{
	FString Result;
	Result.Reserve(Name.Len() + 1);
	for (TCHAR Char : Name)
	{
		Result.AppendChar(FChar::IsAlnum(Char) && Char < 128 ? Char : TEXT('_'));
	}
	if (Result.IsEmpty() || FChar::IsDigit(Result[0]))
	{
		Result.InsertAt(0, TEXT("v"));
	}
	if (Result.StartsWith(TEXT("__")) || Result == TEXT("_"))
	{
		Result.InsertAt(0, TEXT("v"));
	}
	for (const TCHAR* Keyword : WgslKeywords)
	{
		if (Result.Equals(Keyword, ESearchCase::CaseSensitive))
		{
			Result.AppendChar(TEXT('_'));
			break;
		}
	}
	return Result;
}

TSharedPtr<const FWebGPUStructLayout> FWebGPUStructLayout::Get(const UScriptStruct* Struct, EWebGPUAddressSpace Space, FString& OutError)
{
	if (!Struct)
	{
		OutError = TEXT("No struct");
		return nullptr;
	}

	// Native structs never change layout; Blueprint structs can be edited, so they are rebuilt each time
	const bool bCacheable = (Struct->StructFlags & STRUCT_Native) != 0;
	const FCacheKey Key{ Struct, Space };
	if (bCacheable)
	{
		FScopeLock Lock(&CacheLock);
		if (const TSharedPtr<const FWebGPUStructLayout>* Cached = Cache.Find(Key))
		{
			return *Cached;
		}
	}

	TSharedPtr<const FWebGPUStructLayout> Layout = FWebGPUStructLayoutBuilder::Build(Struct, Space, OutError, 0);
	if (Layout && bCacheable)
	{
		FScopeLock Lock(&CacheLock);
		Cache.Add(Key, Layout);
	}
	return Layout;
}

TSharedPtr<const FWebGPUStructLayout> FWebGPUStructLayout::Get(const UScriptStruct* Struct, EWebGPUAddressSpace Space)
{
	FString Error;
	TSharedPtr<const FWebGPUStructLayout> Layout = Get(Struct, Space, Error);
	if (!Layout)
	{
		UE_LOG(LogWebGPU, Error, TEXT("No GPU layout for %s: %s"), Struct ? *Struct->GetName() : TEXT("null"), *Error);
	}
	return Layout;
}

FString FWebGPUStructLayout::GenerateWGSL() const
{
	TArray<FString> Declared;
	FString Out;
	GenerateWGSL(Declared, Out, nullptr);
	return Out;
}

FString FWebGPUStructLayout::GenerateWGSL(TConstArrayView<TSharedPtr<const FWebGPUStructLayout>> Layouts)
{
	TArray<FString> Declared;
	FString Out;
	for (const TSharedPtr<const FWebGPUStructLayout>& Layout : Layouts)
	{
		if (Layout)
		{
			Layout->GenerateWGSL(Declared, Out, nullptr);
		}
	}
	return Out;
}

FString FWebGPUStructLayout::GenerateWGSL(TConstArrayView<TSharedPtr<const FWebGPUStructLayout>> Layouts, const FString& Source)
{
	const FWebGPUSourceSpellings Spellings(Source);
	TArray<FString> Declared;
	FString Out;
	for (const TSharedPtr<const FWebGPUStructLayout>& Layout : Layouts)
	{
		if (Layout)
		{
			Layout->GenerateWGSL(Declared, Out, &Spellings);
		}
	}
	return Out;
}

void FWebGPUStructLayout::GenerateWGSL(TArray<FString>& Declared, FString& Out, const FWebGPUSourceSpellings* Spellings) const
{
	auto TypeName = [Spellings](const FWebGPUStructLayout& Layout)
	{
		return Spellings ? SanitizeIdentifier(Spellings->Spell(Layout.StructName, false)) : Layout.WgslName;
	};

	for (const FField& Field : Fields)
	{
		if (Field.Nested)
		{
			Field.Nested->GenerateWGSL(Declared, Out, Spellings);
		}
	}
	const FString Name = TypeName(*this);
	if (Declared.Contains(Name))
	{
		return;
	}
	Declared.Add(Name);

	Out += FString::Printf(TEXT("// UStruct %s, %s layout: %u bytes per element%s\n"), *StructName,
		Space == EWebGPUAddressSpace::Uniform ? TEXT("uniform") : TEXT("storage"), Stride, bMemcpyCompatible ? TEXT(" (same as UE memory)") : TEXT(""));
	Out += FString::Printf(TEXT("struct %s {\n"), *Name);
	for (const FField& Field : Fields)
	{
		FString Type = Field.WgslType;
		if (Field.Nested && Spellings)
		{
			Type = Field.ArrayDim > 1 ? FString::Printf(TEXT("array<%s, %d>"), *TypeName(*Field.Nested), Field.ArrayDim) : TypeName(*Field.Nested);
		}
		Out += TEXT("\t");
		if (Field.ExplicitAlign)
		{
			Out += FString::Printf(TEXT("@align(%u) "), Field.ExplicitAlign);
		}
		Out += FString::Printf(TEXT("%s: %s,\n"), Spellings ? *SanitizeIdentifier(Spellings->Spell(Field.Name, true)) : *Field.WgslName, *Type);
	}
	Out += TEXT("}\n\n");
}

void FWebGPUStructLayout::PackElement(const uint8* Source, uint8* Dest) const
{
	for (const FField& Field : Fields)
	{
		for (int32 Index = 0; Index < Field.ArrayDim; Index++)
		{
			const uint8* From = Field.Property->ContainerPtrToValuePtr<uint8>(Source, Index);
			uint8* To = Dest + Field.Offset + Index * Field.ArrayStride;
			switch (Field.Convert)
			{
			case EConvert::Copy:
				FMemory::Memcpy(To, From, Field.CopySize);
				break;
			case EConvert::FloatFromDouble:
				for (uint32 Component = 0; Component < Field.Components; Component++)
				{
					reinterpret_cast<float*>(To)[Component] = (float)reinterpret_cast<const double*>(From)[Component];
				}
				break;
			case EConvert::I32FromInt8:
				*reinterpret_cast<int32*>(To) = *reinterpret_cast<const int8*>(From);
				break;
			case EConvert::I32FromInt16:
				*reinterpret_cast<int32*>(To) = *reinterpret_cast<const int16*>(From);
				break;
			case EConvert::U32FromUInt8:
				*reinterpret_cast<uint32*>(To) = *From;
				break;
			case EConvert::U32FromUInt16:
				*reinterpret_cast<uint32*>(To) = *reinterpret_cast<const uint16*>(From);
				break;
			case EConvert::U32FromBool:
				*reinterpret_cast<uint32*>(To) = CastFieldChecked<FBoolProperty>(Field.Property)->GetPropertyValue(From) ? 1u : 0u;
				break;
			case EConvert::Nested:
				Field.Nested->PackElement(From, To);
				break;
			}
		}
	}
}

void FWebGPUStructLayout::UnpackElement(const uint8* Source, uint8* Dest) const
{
	for (const FField& Field : Fields)
	{
		for (int32 Index = 0; Index < Field.ArrayDim; Index++)
		{
			const uint8* From = Source + Field.Offset + Index * Field.ArrayStride;
			uint8* To = Field.Property->ContainerPtrToValuePtr<uint8>(Dest, Index);
			switch (Field.Convert)
			{
			case EConvert::Copy:
				FMemory::Memcpy(To, From, Field.CopySize);
				break;
			case EConvert::FloatFromDouble:
				for (uint32 Component = 0; Component < Field.Components; Component++)
				{
					reinterpret_cast<double*>(To)[Component] = (double)reinterpret_cast<const float*>(From)[Component];
				}
				break;
			case EConvert::I32FromInt8:
				*reinterpret_cast<int8*>(To) = (int8)*reinterpret_cast<const int32*>(From);
				break;
			case EConvert::I32FromInt16:
				*reinterpret_cast<int16*>(To) = (int16)*reinterpret_cast<const int32*>(From);
				break;
			case EConvert::U32FromUInt8:
				*To = (uint8)*reinterpret_cast<const uint32*>(From);
				break;
			case EConvert::U32FromUInt16:
				*reinterpret_cast<uint16*>(To) = (uint16)*reinterpret_cast<const uint32*>(From);
				break;
			case EConvert::U32FromBool:
				CastFieldChecked<FBoolProperty>(Field.Property)->SetPropertyValue(To, *reinterpret_cast<const uint32*>(From) != 0);
				break;
			case EConvert::Nested:
				Field.Nested->UnpackElement(From, To);
				break;
			}
		}
	}
}

void FWebGPUStructLayout::Pack(const void* Elements, int32 Count, uint8* Dest) const
{
	if (Count <= 0)
	{
		return;
	}
	if (bMemcpyCompatible)
	{
		FMemory::Memcpy(Dest, Elements, (SIZE_T)Count * Stride);
		return;
	}
	FMemory::Memzero(Dest, (SIZE_T)Count * Stride);
	const uint8* Source = static_cast<const uint8*>(Elements);
	for (int32 Index = 0; Index < Count; Index++)
	{
		PackElement(Source + (SIZE_T)Index * SourceStride, Dest + (SIZE_T)Index * Stride);
	}
}

TArray<uint8> FWebGPUStructLayout::Pack(const void* Elements, int32 Count) const
{
	TArray<uint8> Bytes;
	Bytes.SetNumUninitialized(FMath::Max(Count, 0) * (int32)Stride);
	Pack(Elements, Count, Bytes.GetData());
	return Bytes;
}

void FWebGPUStructLayout::Unpack(const uint8* Source, int32 Count, void* Elements) const
{
	if (Count <= 0)
	{
		return;
	}
	if (bMemcpyCompatible)
	{
		FMemory::Memcpy(Elements, Source, (SIZE_T)Count * Stride);
		return;
	}
	uint8* Dest = static_cast<uint8*>(Elements);
	for (int32 Index = 0; Index < Count; Index++)
	{
		UnpackElement(Source + (SIZE_T)Index * Stride, Dest + (SIZE_T)Index * SourceStride);
	}
}

bool FWebGPUStructLayout::Matches(const FWebGPUTypeLayout& Reflected, FString& OutDifferences) const
{
	TArray<FString> Differences;
	if (Reflected.Kind != FWebGPUTypeLayout::EKind::Struct)
	{
		OutDifferences = FString::Printf(TEXT("shader type %s is not a struct"), *Reflected.Name);
		return false;
	}
	if (Reflected.Size != Size)
	{
		Differences.Add(FString::Printf(TEXT("size: shader %u bytes, %s %u bytes"), Reflected.Size, *StructName, Size));
	}

	for (const FField& Field : Fields)
	{
		// FindField ignores case (FString ==): packaged builds don't keep the case of UE names
		const FWebGPUTypeLayout::FField* Other = Reflected.FindField(Field.WgslName);
		if (!Other)
		{
			Other = Reflected.FindField(Field.Name);
		}
		if (!Other)
		{
			Differences.Add(FString::Printf(TEXT("'%s' is not in shader struct %s"), *Field.WgslName, *Reflected.Name));
			continue;
		}
		if (Other->Offset != Field.Offset)
		{
			Differences.Add(FString::Printf(TEXT("'%s' offset: shader %u, %s %u"), *Field.WgslName, Other->Offset, *StructName, Field.Offset));
		}
		if (Field.Nested && Field.ArrayDim == 1)
		{
			FString NestedDifferences;
			if (!Other->Type || !Field.Nested->Matches(*Other->Type, NestedDifferences))
			{
				Differences.Add(FString::Printf(TEXT("'%s': %s"), *Field.WgslName, *NestedDifferences));
			}
		}
		else if (Other->Type && Field.ArrayDim == 1 && Other->Type->Name.Replace(TEXT(" "), TEXT("")) != Field.WgslType.Replace(TEXT(" "), TEXT("")))
		{
			Differences.Add(FString::Printf(TEXT("'%s' type: shader %s, %s %s"), *Field.WgslName, *Other->Type->Name, *StructName, *Field.WgslType));
		}
	}
	for (const FWebGPUTypeLayout::FField& Other : Reflected.Fields)
	{
		if (!Fields.ContainsByPredicate([&Other](const FField& Field) { return Field.WgslName == Other.Name || Field.Name == Other.Name; }))
		{
			Differences.Add(FString::Printf(TEXT("shader field '%s' is not in %s"), *Other.Name, *StructName));
		}
	}

	OutDifferences = FString::Join(Differences, TEXT("; "));
	return Differences.Num() == 0;
}
