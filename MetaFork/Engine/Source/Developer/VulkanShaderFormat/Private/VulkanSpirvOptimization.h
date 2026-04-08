// (c) 2023 Meta. All rights reserved.

#pragma once

#include "CoreMinimal.h"
#include "CrossCompilerCommon.h"

namespace CrossCompiler {
	class FShaderConductorContext;
	class FHlslccHeaderWriter;
};

struct FPackedGlobal
{
	FString Name;
	uint32 Offset;
	uint32 Size;
};

struct FEmulatedUBCopy
{
	uint16 FromBuffer;
	uint32 FromOffsetFloats;
	uint32 ToOffsetFloats;
	uint32 SizeFloats;
};

struct FOptimizedUniforms
{
	TArray<FString> EmulatedUBs;
	TArray<FPackedGlobal> PackedGlobals;
	TArray<FEmulatedUBCopy> EmulatedUBCopies;
	
	void AddPackedUB(const FString& Name) { EmulatedUBs.Add(Name); };
	void AddPackedGlobal(const FString& Name, uint32 Offset, uint32 Size) { PackedGlobals.Add({Name, Offset, Size}); }
	void AddPackedUBGlobalCopy(uint16 FromBuffer, uint32 FromOffsetFloats, uint32 ToOffsetFloats, uint32 SizeFloats)
		{ EmulatedUBCopies.Add({FromBuffer, FromOffsetFloats, ToOffsetFloats, SizeFloats}); }
};

bool OptimizeGlobalUniforms(
	const struct FShaderCompilerInput& Input,
	CrossCompiler::FShaderConductorContext& CompilerContext,
	struct FSpirv& Spirv,
	FOptimizedUniforms& OutUniforms,
	bool bDebugDump,
	bool bUseEmulatedUBs);