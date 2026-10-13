// Copyright 2025-current Getnamo. All Rights Reserved.

// Binding sets: named resources and uniform values shared by kernels

#include "WebGPUContext.h"
#include "WebGPUCompute.h"

namespace
{
	/** Writes one value into a uniform's bytes at the member Path ("" is the uniform itself), using the shader's layout. */
	bool PackUniformValue(const FWebGPUTypeLayout& Root, const FString& Path, const FWebGPUUniformValue& Value, TArray<uint8>& Bytes, FString& OutError)
	{
		const FWebGPUTypeLayout* Type = &Root;
		uint32 Offset = 0;
		TArray<FString> Parts;
		Path.ParseIntoArray(Parts, TEXT("."));
		for (const FString& Part : Parts)
		{
			if (Type->Kind != FWebGPUTypeLayout::EKind::Struct)
			{
				OutError = FString::Printf(TEXT("'%s' is %s, which has no members"), *Path, *Type->Name);
				return false;
			}
			const FWebGPUTypeLayout::FField* Field = Type->FindField(Part);
			if (!Field || !Field->Type)
			{
				TArray<FString> Names;
				for (const FWebGPUTypeLayout::FField& Member : Type->Fields)
				{
					Names.Add(Member.Name);
				}
				OutError = FString::Printf(TEXT("%s has no member '%s' (members: %s)"), *Type->Name, *Part, *FString::Join(Names, TEXT(", ")));
				return false;
			}
			Offset += Field->Offset;
			Type = Field->Type.Get();
		}

		const bool bScalar = Type->Kind == FWebGPUTypeLayout::EKind::Scalar;
		if (!bScalar && Type->Kind != FWebGPUTypeLayout::EKind::Vector)
		{
			OutError = FString::Printf(TEXT("'%s' is %s; only scalars and vectors can be set"), *Path, *Type->Name);
			return false;
		}
		const FString& Scalar = Type->ScalarName;
		if (Scalar != TEXT("f32") && Scalar != TEXT("i32") && Scalar != TEXT("u32"))
		{
			OutError = FString::Printf(TEXT("'%s' is %s; only f32, i32 and u32 components can be set"), *Path, *Type->Name);
			return false;
		}
		const uint32 Components = bScalar ? 1 : FMath::Min<uint32>(Type->Rows, 4);
		if (Offset + Components * 4 > (uint32)Bytes.Num())
		{
			OutError = FString::Printf(TEXT("'%s' lies outside the buffer"), *Path);
			return false;
		}

		for (uint32 Component = 0; Component < Components; Component++)
		{
			// Scalars fill every component of a vector; a vector fills as many as the member has
			const int32 From = Value.Type == FWebGPUUniformValue::EType::Vector ? (int32)Component : 0;
			const bool bInt = Value.Type == FWebGPUUniformValue::EType::Int;
			uint8* Dest = Bytes.GetData() + Offset + Component * 4;
			if (Scalar == TEXT("f32"))
			{
				const float Number = bInt ? (float)Value.Ints[From] : Value.Floats[From];
				FMemory::Memcpy(Dest, &Number, 4);
			}
			else if (Scalar == TEXT("i32"))
			{
				const int32 Number = bInt ? Value.Ints[From] : FMath::TruncToInt32(Value.Floats[From]);
				FMemory::Memcpy(Dest, &Number, 4);
			}
			else
			{
				const uint32 Number = (uint32)FMath::Max(bInt ? Value.Ints[From] : FMath::TruncToInt32(Value.Floats[From]), 0);
				FMemory::Memcpy(Dest, &Number, 4);
			}
		}
		return true;
	}
}

TSharedRef<FWebGPUBindingSet> FWebGPUContext::CreateBindingSet(const FString& Label)
{
	TSharedRef<FWebGPUBindingSet> Set = MakeShared<FWebGPUBindingSet>();
	Set->Context = AsShared();
	Set->Label = Label.IsEmpty() ? TEXT("WebGPUBindings") : Label;
	return Set;
}

void FWebGPUContext::SetKernelBindingSet(const TSharedRef<FWebGPUKernel>& Kernel, const TSharedPtr<FWebGPUBindingSet>& Set)
{
	Enqueue([Kernel, Set]()
	{
		Kernel->BindingSet = Set;
		Kernel->BindingSetVersion = 0;
		Kernel->bBindGroupsDirty = true;
	});
}

void FWebGPUContext::SetSharedBuffer(const TSharedRef<FWebGPUBindingSet>& Set, const FString& Name, const TSharedPtr<FWebGPUBuffer>& Buffer, uint64 Offset, uint64 Size)
{
	Enqueue([Set, Name, Buffer, Offset, Size]()
	{
		if (Buffer)
		{
			FWebGPUResourceBinding& Entry = Set->Named.FindOrAdd(Name);
			Entry = FWebGPUResourceBinding();
			Entry.Buffer = Buffer;
			Entry.Offset = Offset;
			Entry.Size = Size;
		}
		else
		{
			Set->Named.Remove(Name);
		}
		Set->Version++;
	});
}

void FWebGPUContext::SetSharedTexture(const TSharedRef<FWebGPUBindingSet>& Set, const FString& Name, const TSharedPtr<FWebGPUTexture>& Texture)
{
	Enqueue([Set, Name, Texture]()
	{
		if (Texture)
		{
			FWebGPUResourceBinding& Entry = Set->Named.FindOrAdd(Name);
			Entry = FWebGPUResourceBinding();
			Entry.Texture = Texture;
		}
		else
		{
			Set->Named.Remove(Name);
		}
		Set->Version++;
	});
}

void FWebGPUContext::SetSharedSampler(const TSharedRef<FWebGPUBindingSet>& Set, const FString& Name, EWebGPUSampler Sampler)
{
	Enqueue([Set, Name, Sampler]()
	{
		FWebGPUResourceBinding& Entry = Set->Named.FindOrAdd(Name);
		Entry = FWebGPUResourceBinding();
		Entry.Sampler = Sampler;
		Set->Version++;
	});
}

void FWebGPUContext::SetUniformValue(const TSharedRef<FWebGPUBindingSet>& Set, const FString& Path, const FWebGPUUniformValue& Value)
{
	FString Binding = Path;
	FString Member;
	Path.Split(TEXT("."), &Binding, &Member);
	if (Binding.IsEmpty())
	{
		UE_LOG(LogWebGPU, Warning, TEXT("SetUniformValue(%s): expected \"binding.member\" or a uniform's name"), *Path);
		return;
	}
	Enqueue([Set, Binding, Member, Value]()
	{
		TSharedPtr<FWebGPUBindingSet::FUniformBlock>& Block = Set->Uniforms.FindOrAdd(Binding);
		if (!Block)
		{
			// A new uniform: kernels that use one by this name now get it from the set
			Block = MakeShared<FWebGPUBindingSet::FUniformBlock>();
			Set->Version++;
		}
		Block->Values.Add(Member, Value);
	});
}

FWebGPUBindingSet::FUniformBlock* FWebGPUContext::GetUniformBlock(FWebGPUKernel& Kernel, FWebGPUBindingSet& Set, const FWebGPUBindingInfo& Info)
{
	TSharedPtr<FWebGPUBindingSet::FUniformBlock>* Found = Set.Uniforms.Find(Info.Name);
	if (!Found || !Found->IsValid())
	{
		return nullptr;
	}
	FWebGPUBindingSet::FUniformBlock& Block = **Found;

	// Sized for the largest layout seen: a shader edit that grows the struct gets a bigger buffer
	const uint64 Needed = Align(FMath::Max<uint64>((uint64)Info.MinSize, Info.Layout ? (uint64)Info.Layout->Size : 0), 16);
	if (!Block.Binding.Buffer || Block.Binding.Buffer->GetSize() < Needed)
	{
		TSharedRef<FWebGPUBuffer> Buffer = MakeShared<FWebGPUBuffer>();
		Buffer->Context = AsShared();
		Buffer->Size = FMath::Max<uint64>(Needed, 16);
		Buffer->Usage = EWebGPUBufferUsage::Uniform;
		Buffer->Label = FString::Printf(TEXT("%s %s"), *Set.Label, *Info.Name);
		CountBuffer(*Buffer, true);
		LiveBuffers.Add(Buffer);
		CreateBufferHandle(*Buffer);
		Block.Binding = FWebGPUResourceBinding();
		Block.Binding.Buffer = Buffer;
		Block.Uploaded.Reset();
		Set.Version++;
	}
	return &Block;
}

void FWebGPUContext::UploadSetUniforms(FWebGPUKernel& Kernel)
{
	for (const TPair<const FWebGPUBindingInfo*, FWebGPUBindingSet::FUniformBlock*>& Pair : Kernel.SetUniforms)
	{
		const FWebGPUBindingInfo& Info = *Pair.Key;
		FWebGPUBindingSet::FUniformBlock& Block = *Pair.Value;
		if (!Info.Layout || !Block.Binding.Buffer || !Block.Binding.Buffer->Handle)
		{
			continue;
		}

		// Packed with this kernel's layout: kernels compiled from different versions of a struct each get theirs
		TArray<uint8> Bytes;
		Bytes.SetNumZeroed((int32)Block.Binding.Buffer->GetSize());
		for (const TPair<FString, FWebGPUUniformValue>& Value : Block.Values)
		{
			FString Error;
			if (!PackUniformValue(*Info.Layout, Value.Key, Value.Value, Bytes, Error))
			{
				WarnOnce(Kernel, FString::Printf(TEXT("Kernel '%s': uniform '%s': %s"), *Kernel.Label, *Info.Name, *Error));
			}
		}
		if (Bytes != Block.Uploaded || Block.UploadedGeneration != DeviceGeneration)
		{
			UploadOnWorker(*Block.Binding.Buffer, 0, Bytes.GetData(), (uint64)Bytes.Num());
			Block.Uploaded = MoveTemp(Bytes);
			Block.UploadedGeneration = DeviceGeneration;
		}
	}
}
