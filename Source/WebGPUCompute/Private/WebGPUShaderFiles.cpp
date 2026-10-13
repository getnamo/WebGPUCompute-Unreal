// Copyright 2025-current Getnamo. All Rights Reserved.

#include "WebGPUShaderFiles.h"
#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"

bool FWebGPUSourceMap::Resolve(uint64 Line, FString& OutFile, uint64& OutLine) const
{
	if (Line == 0 || LineFiles.Num() == 0)
	{
		return false;
	}
	// Errors at the very end of the source point one past the last line
	const int32 Index = (int32)FMath::Min<uint64>(Line, (uint64)LineFiles.Num()) - 1;
	OutFile = Files[LineFiles[Index]];
	OutLine = (uint64)LineNumbers[Index] + (Line > (uint64)LineFiles.Num() ? 1 : 0);
	return true;
}

void FWebGPUSourceMap::Apply(TArray<FWebGPUCompileMessage>& Messages) const
{
	for (FWebGPUCompileMessage& Message : Messages)
	{
		if (Message.File.IsEmpty())
		{
			Resolve(Message.Line, Message.File, Message.Line);
		}
	}
}

namespace
{
	/** Expands #include lines depth first, recording where every output line came from. */
	struct FWebGPUPreprocessor
	{
		FWebGPUShaderSource& Out;
		TSharedRef<FWebGPUSourceMap> Map = MakeShared<FWebGPUSourceMap>();
		TArray<FString> Stack;
		TSet<FString> Included;

		explicit FWebGPUPreprocessor(FWebGPUShaderSource& InOut) : Out(InOut) {}

		void AddError(const FString& File, int32 Line, const FString& Message)
		{
			FWebGPUCompileMessage& Error = Out.Errors.AddDefaulted_GetRef();
			Error.bIsError = true;
			Error.File = File;
			Error.Line = (uint64)FMath::Max(Line, 0);
			Error.Message = Message;
		}

		int32 AddFile(const FString& File)
		{
			int32 Index = Map->Files.Find(File);
			if (Index == INDEX_NONE)
			{
				// Timestamp before reading: a write that lands during the read still looks newer afterwards
				Index = Map->Files.Add(File);
				Out.FileTimes.Add(FWebGPUShaderFiles::GetTimestamp(File));
				Out.FileHashes.Add(0);
			}
			return Index;
		}

		/** The quoted path of an #include line; false (with OutPath empty) when the line is malformed. */
		static bool ParseInclude(const FString& Directive, FString& OutPath)
		{
			int32 Open = INDEX_NONE;
			int32 Close = INDEX_NONE;
			if (Directive.FindChar(TEXT('"'), Open))
			{
				Close = Directive.Find(TEXT("\""), ESearchCase::CaseSensitive, ESearchDir::FromStart, Open + 1);
			}
			if (Open == INDEX_NONE || Close == INDEX_NONE || Close <= Open + 1)
			{
				return false;
			}
			OutPath = Directive.Mid(Open + 1, Close - Open - 1);
			return true;
		}

		static FString ResolveInclude(const FString& IncludingFile, const FString& Path)
		{
			FString Normalized = Path;
			Normalized.ReplaceInline(TEXT("\\"), TEXT("/"));
			if (Normalized.StartsWith(TEXT("/")) || !FPaths::IsRelative(Normalized))
			{
				return FWebGPUShaderFiles::ResolvePath(Normalized);
			}
			FString Full = FPaths::GetPath(IncludingFile) / Normalized;
			FPaths::CollapseRelativeDirectories(Full);
			return FPaths::ConvertRelativePathToFull(Full);
		}

		void Expand(const FString& File, const FString& IncludedFrom, int32 IncludedAtLine)
		{
			const int32 FileIndex = AddFile(File);
			FString Text;
			if (!FFileHelper::LoadFileToString(Text, *File))
			{
				const FString Problem = FPaths::FileExists(File) ? TEXT("could not be read") : TEXT("not found");
				if (IncludedFrom.IsEmpty())
				{
					AddError(File, 0, FString::Printf(TEXT("shader file %s %s"), *File, *Problem));
				}
				else
				{
					AddError(IncludedFrom, IncludedAtLine, FString::Printf(TEXT("included file %s %s"), *File, *Problem));
				}
				return;
			}

			Out.FileHashes[FileIndex] = FCrc::StrCrc32(*Text);
			Included.Add(File);
			Stack.Push(File);
			TArray<FString> Lines;
			Text.ParseIntoArrayLines(Lines, false);
			for (int32 Index = 0; Index < Lines.Num(); Index++)
			{
				const FString& Line = Lines[Index];
				const FString Trimmed = Line.TrimStart();
				if (Trimmed.StartsWith(TEXT("#include")))
				{
					FString IncludePath;
					if (!ParseInclude(Trimmed, IncludePath))
					{
						AddError(File, Index + 1, TEXT("expected #include \"file.wgsl\""));
						continue;
					}
					const FString Resolved = ResolveInclude(File, IncludePath);
					if (Stack.Contains(Resolved))
					{
						TArray<FString> Chain;
						for (const FString& Entry : Stack)
						{
							Chain.Add(FPaths::GetCleanFilename(Entry));
						}
						Chain.Add(FPaths::GetCleanFilename(Resolved));
						AddError(File, Index + 1, FString::Printf(TEXT("include cycle: %s"), *FString::Join(Chain, TEXT(" -> "))));
						continue;
					}
					if (!Included.Contains(Resolved))
					{
						Expand(Resolved, File, Index + 1);
					}
					continue;
				}
				Out.Source += Line;
				Out.Source += TEXT('\n');
				Map->LineFiles.Add(FileIndex);
				Map->LineNumbers.Add(Index + 1);
			}
			Stack.Pop();
		}
	};
}

FString FWebGPUShaderFiles::ResolvePath(const FString& Path)
{
	FString Normalized = Path.TrimStartAndEnd();
	Normalized.ReplaceInline(TEXT("\\"), TEXT("/"));
	if (Normalized.IsEmpty())
	{
		return FString();
	}

	// Content style: the first folder is a mount point ("/Game/...", "/PluginName/...")
	if (Normalized.StartsWith(TEXT("/")) && !Normalized.StartsWith(TEXT("//")))
	{
		const FString Extension = FPaths::GetExtension(Normalized, true);
		FString Filename;
		if (FPackageName::TryConvertLongPackageNameToFilename(Normalized.LeftChop(Extension.Len()), Filename, Extension))
		{
			return FPaths::ConvertRelativePathToFull(Filename);
		}
	}
	if (FPaths::IsRelative(Normalized))
	{
		return FPaths::ConvertRelativePathToFull(FPaths::ProjectDir() / Normalized);
	}
	return FPaths::ConvertRelativePathToFull(Normalized);
}

FWebGPUShaderSource FWebGPUShaderFiles::Load(const FString& Path)
{
	FWebGPUShaderSource Result;
	FWebGPUPreprocessor Preprocessor(Result);
	const FString File = ResolvePath(Path);
	if (File.IsEmpty())
	{
		Preprocessor.AddError(FString(), 0, TEXT("no shader file given"));
	}
	else
	{
		Preprocessor.Expand(File, FString(), 0);
	}
	Result.Map = Preprocessor.Map;
	return Result;
}

FDateTime FWebGPUShaderFiles::GetTimestamp(const FString& Filename)
{
	return IFileManager::Get().GetTimeStamp(*Filename);
}
