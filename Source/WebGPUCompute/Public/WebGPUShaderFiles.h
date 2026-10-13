// Copyright 2025-current Getnamo. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "WebGPUDevice.h"

/** Where each line of a preprocessed WGSL source came from. */
struct WEBGPUCOMPUTE_API FWebGPUSourceMap
{
	/** Absolute paths of the files read: the main file first, then includes in the order they were reached. */
	TArray<FString> Files;

	/** Per line of the preprocessed source: the index into Files and the 1-based line in that file. */
	TArray<int32> LineFiles;
	TArray<int32> LineNumbers;

	/** Maps a 1-based line of the preprocessed source to its file and line. False when out of range. */
	bool Resolve(uint64 Line, FString& OutFile, uint64& OutLine) const;

	/** Rewrites Line/File of each message (lines of the preprocessed source) to the file and line it came from. */
	void Apply(TArray<FWebGPUCompileMessage>& Messages) const;
};

/** WGSL read from a file, with its includes expanded. */
struct WEBGPUCOMPUTE_API FWebGPUShaderSource
{
	FString Source;
	TSharedPtr<const FWebGPUSourceMap> Map;

	/** Last-modified time of each file in Map->Files when it was read (FDateTime::MinValue() if missing). */
	TArray<FDateTime> FileTimes;

	/** CRC of each file's text as read (0 if missing): file times have one second precision, so saves within a second differ only here. */
	TArray<uint32> FileHashes;

	/** Problems that stop the compile (a missing file or include, an include cycle), with file and line. */
	TArray<FWebGPUCompileMessage> Errors;

	bool IsValid() const { return Errors.Num() == 0; }
};

/**
 * .wgsl files for kernels (FWebGPUKernelDesc::SourceFile).
 *
 * Paths can be absolute, content style ("/Game/Shaders/crowd.wgsl" is Content/Shaders/crowd.wgsl and
 * "/SomePlugin/Shaders/x.wgsl" is in that plugin's Content folder), or relative to the project directory.
 *
 * A line `#include "common.wgsl"` pastes another file in its place. The path is relative to the including
 * file, or content style / absolute. Each file is pasted once per kernel (WGSL has no redeclarations), so
 * shared declarations can be included from several files. Compile messages name the file and line they
 * come from (FWebGPUCompileMessage::File).
 */
struct WEBGPUCOMPUTE_API FWebGPUShaderFiles
{
	/** The absolute filename for a path in any of the forms above. It need not exist. */
	static FString ResolvePath(const FString& Path);

	/** Reads Path and expands its includes. Any thread. */
	static FWebGPUShaderSource Load(const FString& Path);

	/** Last-modified time of a file; FDateTime::MinValue() if it doesn't exist. */
	static FDateTime GetTimestamp(const FString& Filename);
};
