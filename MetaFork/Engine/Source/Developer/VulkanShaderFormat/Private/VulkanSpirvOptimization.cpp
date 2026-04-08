// (c) 2023 Meta. All rights reserved.

#include "VulkanSpirvOptimization.h"
#include "ShaderCompilerCore.h"
#include "ShaderCompilerCommon.h"
#include "ShaderConductorContext.h"
#include "SpirvCommon.h"

// This file contains the implementation of OptimizeGlobalUniforms.
// The function rewrites the spir-v in the following ways:
// - Moves uniforms from emulated UBs into the global UB
// - Splits arrays in emulated UBs into separate fields where possible
// - Splits matrix and vector fields into smaller fields based on how they are used
// - Reorders fields of the global UB to reduce its size
// - Generates a list of Packed UB copies in the Header
// These optimizations are important for mobile platforms, particularly adreno gpus.
//
// The optimization is split into three passes over the SpirV:
// 1. Data collection: FSpirvExtractedData collects information about all IDs in the spir-v and their types
// 2. Usage tracking: FUsageData figures out which uniforms (and which parts of those uniforms) are actually
// used by the shader.
// 3. Rewriting: FFieldMover optimizes the global data layout and updates any references to globals in the spir-v.


// 32 bit sentinel value used to represent an invalid index
static const uint32 None_u32 = ~uint32(0);

// 32 bit value used for uninitialized memory.
// Selected to be likely to cause an error when used as an index.
// Code should not check whether an object has this value.
static const uint32 Undefined_u32 = 0x7FFFFFFFU;

// This sentinel value is used for masks on values which are not vectors
static const uint8 NotVectorMask = 0xF0U;

// This sentinel value is used as a uniform buffer index to identify the global UB.
static const uint16 GlobalBufferSentinel = ~uint16(0);


template<typename T>
uint32 GetTypeHash(const TArrayView<const T>& View)
{
	uint32 Hash = GetTypeHash(View.Num());
	for (const T& Value : View)
	{
		Hash = HashCombine(Hash, GetTypeHash(Value));
	}
	return Hash;
}

static uint32 EncodeSpvOpWord(SpvOp Op, uint32 NumWords)
{
	check(NumWords <= 0xFFFFU);
	return NumWords << 16 | Op;
}

struct FSpirvOffset
{
	uint32 Offset = None_u32;

	FSpirvOffset() = default;
	FSpirvOffset(const FSpirvOffset&) = default;
	FSpirvOffset(FSpirvOffset&&) = default;
	FSpirvOffset& operator=(const FSpirvOffset&) = default;
	FSpirvOffset& operator=(FSpirvOffset&&) = default;

	explicit FSpirvOffset(uint32 InOffset) : Offset(InOffset) {}
	explicit operator uint32() const { return Offset; }

	bool operator==(const FSpirvOffset& Other) const { return Offset == Other.Offset; }
	bool operator!=(const FSpirvOffset& Other) const { return Offset != Other.Offset; }

	bool IsValid() const
	{
		return Offset != None_u32;
	}
};

static uint32 GetTypeHash(FSpirvOffset const& Offset)
{
	return GetTypeHash(Offset.Offset);
}

struct FSpirvId
{
	uint32 Id = None_u32;

	FSpirvId() = default;
	FSpirvId(const FSpirvId&) = default;
	FSpirvId(FSpirvId&&) = default;
	FSpirvId& operator=(const FSpirvId&) = default;
	FSpirvId& operator=(FSpirvId&&) = default;

	explicit FSpirvId(uint32 InId) : Id(InId) {}
	operator uint32() const { return Id; }

	bool operator==(const FSpirvId& Other) const { return Id == Other.Id; }
	bool operator!=(const FSpirvId& Other) const { return Id != Other.Id; }

	bool IsValid() const
	{
		return Id != None_u32;
	}
};

static uint32 GetTypeHash(FSpirvId const& Id)
{
	return GetTypeHash(Id.Id);
}

enum class EMetaType : uint8
{
	Scalar,
	Vector,
	Matrix,
	Array,
};

struct FTypeInfo
{
	EMetaType MetaType;
	uint8 VectorLength;
	uint8 MatrixHeight;
	uint8 LastRowSize; // always nonzero
	uint32 FullRows;    // may be zero
	uint32 ArrayLength;
	uint32 ArrayStrideBytes;
	FSpirvId ChildType;
};

struct FFieldInfo
{
	FSpirvId TypeId;
	uint32 Offset = None_u32;
	FString Name;
	TArray<FSpirvIterator> OtherDecorators;
	uint32 UsageInfoIndex = None_u32;
};

struct FUniformBufferInfo
{
	FSpirvId Id;
	FSpirvId StructTypeId;
	uint32 DescriptorSet = None_u32;
	uint32 Binding = None_u32;
	FString Name;
	TArray<FFieldInfo> FieldInfos;
};

struct FSpirvExtractedData
{
	FSpirv& SpirV;

	// Offsets of the definition instructions for all IDs
	TArray<FSpirvOffset> IdDefinitions;
	// Type IDs of all IDs.  IDs which represent types have their type set to themselves.
	TArray<FSpirvId> IdTypes;
	TBitArray<> RelaxedPrecision;

	TArray<FUniformBufferInfo> UniformBuffers;
	FUniformBufferInfo GlobalUniformBuffer;
	bool bFoundExistingGlobalsBuffer;

	// The Id of the unsigned 32-bit int type, or None_u32 if there isn't one.
	FSpirvId UintTypeId;

	// The Id of the OpFunction instruction
	FSpirvId FunctionStart;

	TMap<FSpirvId, FTypeInfo> TypeInfos;

	// E.g. key: %42 = OpTypeFloat 32, value: %56 = OpTypePointer %42 Uniform
	TMap<FSpirvId, FSpirvId> UniformPointerTypes;

	uint32 NextId;

	struct FVectorTypes
	{
		FSpirvId Ids[3];
		bool bGenerated[3];

		FVectorTypes()
		{
			for (FSpirvId& Id : Ids) Id = FSpirvId();
			for (bool& bGen : bGenerated) bGen = false;
		}
	};
	TMap<FSpirvId, FVectorTypes> VectorTypes;

	struct FGeneratedPointer
	{
		FSpirvId PointerTypeId;
		FSpirvId ChildTypeId;
	};
	TArray<FGeneratedPointer> GeneratedPointers;

	FSpirvExtractedData(FSpirv& InSpirV)
		: SpirV(InSpirV)
		, NextId(InSpirV.Data[3])
	{}

	FSpirvIterator GetInstruction(FSpirvOffset Offset)
	{
		check(Offset.IsValid());
		return FSpirvIterator(&SpirV.Data[Offset.Offset]);
	}

	FSpirvIterator GetInstruction(FSpirvId Id)
	{
		check(Id.IsValid());
		return GetInstruction(IdDefinitions[Id.Id]);
	}

	FSpirvOffset GetOffset(FSpirvId Id)
	{
		check(Id.IsValid());
		return IdDefinitions[Id.Id];
	}

	FSpirvOffset GetOffset(FSpirvIterator Inst)
	{
		return FSpirvOffset{ static_cast<uint32>(*Inst - SpirV.Data.GetData()) };
	}

	FSpirvId GetTypeId(FSpirvId Id)
	{
		check(Id.IsValid());
		return IdTypes[Id.Id];
	}

	FSpirvId GetOrCreateVectorType(FSpirvId BaseTypeId, uint32 VectorLength)
	{
		check(VectorLength >= 2);
		check(VectorLength <= 4);
		FVectorTypes& Vectors = VectorTypes.FindChecked(BaseTypeId);
		FSpirvId& Id = Vectors.Ids[VectorLength - 2];
		if (!Id.IsValid())
		{
			Id = AllocTypeId();
			Vectors.bGenerated[VectorLength - 2] = true;
			AddVectorTypeInfo(Id, BaseTypeId, VectorLength);
		}
		return Id;
	}

	FSpirvId GetOrCreatePointerType(FSpirvId BaseTypeId)
	{
		if (FSpirvId* ExistingId = UniformPointerTypes.Find(BaseTypeId))
		{
			return *ExistingId;
		}

		FSpirvId NewId = AllocTypeId();
		UniformPointerTypes.Add(BaseTypeId, NewId);
		GeneratedPointers.Add(FGeneratedPointer{ NewId, BaseTypeId });
		return NewId;
	}

	FSpirvId AllocId(FSpirvId ResultType, FSpirvId BasedOnId = FSpirvId())
	{
		check(NextId == IdDefinitions.Num());
		check(NextId == IdTypes.Num());
		check(NextId == RelaxedPrecision.Num());
		IdTypes.Add(ResultType);
		IdDefinitions.Add(FSpirvOffset());
		bool bRelaxedPrecision = BasedOnId.IsValid() && RelaxedPrecision[BasedOnId.Id];
		RelaxedPrecision.Add(bRelaxedPrecision);
		return FSpirvId(NextId++);
	}

	FSpirvId AllocTypeId()
	{
		return AllocId(FSpirvId(NextId));
	}

	bool IsType(FSpirvId Id)
	{
		// Types are equal to their own Id
		return IdTypes[Id.Id] == Id;
	}

	// Reads the spir-v code and extracts metadata about uniform buffers,
	// and other pieces of info we will need throughout the optimization
	// process.
	void ExtractData()
	{
		uint32 NumIds = SpirV.Data[3];
		IdDefinitions.Init(FSpirvOffset(), int32(NumIds));
		IdTypes.Init(FSpirvId(), int32(NumIds));
		RelaxedPrecision.Init(false, int32(NumIds));

		TArray<FSpirvId> PossibleUniformBlocks;
		TArray<FSpirvId> BlockDecorations;
		// Offsets of all decorator and name instructions
		TArray<FSpirvOffset> DecoratorsAndNames;

		FSpirvIterator const End = SpirV.end();
		for (FSpirvIterator Iter = SpirV.begin(); Iter != End; ++Iter)
		{
			SpvOp Opcode = Iter.Opcode();
			if (Opcode == SpvOpGroupDecorate ||
				Opcode == SpvOpGroupMemberDecorate)
			{
				check(false); // We don't correctly handle these opcodes
			}

			if (Opcode == SpvOpDecorate)
			{
				check(Iter.WordCount() >= 3);
				uint32 DecoratedId = Iter.Operand(1);
				uint32 Decoration = Iter.Operand(2);
				if (Decoration == SpvDecorationBlock)
				{
					BlockDecorations.Add(FSpirvId(DecoratedId));
				}
				if (Decoration == SpvDecorationRelaxedPrecision)
				{
					RelaxedPrecision[DecoratedId] = true;
				}
			}

			if (Opcode == SpvOpDecorate ||
				Opcode == SpvOpName ||
				Opcode == SpvOpMemberDecorate ||
				Opcode == SpvOpMemberName)
			{
				DecoratorsAndNames.Add(GetOffset(Iter));
			}

			if (Opcode == SpvOpTypeInt)
			{
				check(Iter.WordCount() == 4);
				FSpirvId Id(Iter.Operand(1));
				uint32 BitWidth = Iter.Operand(2);
				uint32 IsSigned = Iter.Operand(3);
				if (BitWidth == 32 && IsSigned == 0)
				{
					check(!UintTypeId.IsValid()); // There should be only one uint type declaration
					UintTypeId = Id;
				}
			}

			if (Opcode == SpvOpTypeInt ||
				Opcode == SpvOpTypeFloat ||
				Opcode == SpvOpTypeBool)
			{
				FSpirvId Id(Iter.Operand(1));
				VectorTypes.Add(Id, FVectorTypes());
			}

			if (Opcode == SpvOpTypeVector)
			{
				check(Iter.WordCount() == 4);
				FSpirvId Id(Iter.Operand(1));
				FSpirvId ChildId(Iter.Operand(2));
				uint32 Length = Iter.Operand(3);
				check(Length >= 2);
				check(Length <= 4);
				FVectorTypes& Vectors = VectorTypes.FindChecked(ChildId);
				check(!Vectors.Ids[Length - 2].IsValid());
				Vectors.Ids[Length - 2] = Id;
			}

			if (Opcode == SpvOpTypePointer)
			{
				check(Iter.WordCount() == 4);
				FSpirvId Id(Iter.Operand(1));
				uint32 StorageClass = Iter.Operand(2);
				FSpirvId ChildId(Iter.Operand(3));
				if (StorageClass == SpvStorageClassUniform)
				{
					UniformPointerTypes.Add(ChildId, Id);
				}
			}

			if (Opcode == SpvOpVariable)
			{
				check(Iter.WordCount() >= 4);
				uint32 Id = Iter.Operand(2);
				uint32 StorageClass = Iter.Operand(3);
				if (StorageClass == SpvStorageClassUniform)
				{
					PossibleUniformBlocks.Add(FSpirvId(Id));
				}
			}

			if (Opcode == SpvOpFunction)
			{
				check(!FunctionStart.IsValid());
				FunctionStart = FSpirvId(Iter.Operand(2));
			}
			
			bool bHasResultType = false;
			bool bHasResultId = false;
			SpvHasResultAndType(Opcode, &bHasResultId, &bHasResultType);
			if (bHasResultId)
			{
				uint32 ResultId;
				uint32 ResultType;
				if (bHasResultType)
				{
					ResultType = Iter.Operand(1);
					ResultId = Iter.Operand(2);
				}
				else
				{
					// Operations which produce an ID but not a type are considered to be of their own type.
					// All OpType* operations fall into this category, but four other ops do as well:
					// OpString, OpExtInstImport, OpDecorationGroup, OpLabel
					// For our purposes we will also consider these ops to define their own type.
					ResultType = ResultId = Iter.Operand(1);
					ComputeTypeInfo(Iter);
				}
				IdDefinitions[ResultId] = GetOffset(Iter);
				IdTypes[ResultId] = FSpirvId(ResultType);
			}
			else
			{
				check(!bHasResultType); // We can only have a result type if we also have a result Id
			}
		}

		check(FunctionStart.IsValid()); // Make sure we found the function
		// Need to find the declarations of all Ids, should be dense since it came out of the optimizer.
		// If these start failing, it should be safe to remove them.
		// Id 0 is often not used so we skip it.
		for (int Index = 1; Index < IdDefinitions.Num(); ++Index)
		{
			check(IdDefinitions[Index].IsValid());
			check(IdTypes[Index].IsValid());
		}

		// Now that we have seen all decorations, filter down to the set of actual uniform blocks
		for (FSpirvId VariableId : PossibleUniformBlocks)
		{
			FSpirvIterator Inst = GetInstruction(VariableId);
			check(Inst.Opcode() == SpvOpVariable);
			check(Inst.WordCount() >= 4);
			check(Inst.Operand(3) == SpvStorageClassUniform);
			FSpirvIterator PtrType = GetInstruction(GetTypeId(VariableId));
			check(PtrType.Opcode() == SpvOpTypePointer);
			check(PtrType.WordCount() >= 4);
			FSpirvId StructTypeId(PtrType.Operand(3));

			// Uniform buffers must have OpDecorate %StructType Block
			// Otherwise it could be a SSBO or some other sort of global data.
			// CacheFieldLayout will fail if it finds a OpRuntimeArray, nested struct,
			// or other sorts of data that only appear in uniform buffers, so
			// we want to make sure that we are looking at a uniform buffer first.
			if (!BlockDecorations.Contains(StructTypeId))
			{
				continue;
			}

			FSpirvIterator StructType = GetInstruction(StructTypeId);
			check(StructType.Opcode() == SpvOpTypeStruct);
			uint32 NumWords = StructType.WordCount();
			check(NumWords >= 2);
			uint32 NumFields = NumWords - 2;

			FUniformBufferInfo& Info = UniformBuffers.AddDefaulted_GetRef();
			Info.Id = VariableId;
			Info.StructTypeId = StructTypeId;
			Info.FieldInfos.Reserve(NumFields);

			for (uint32 Idx = 2; Idx < NumWords; ++Idx)
			{
				FSpirvId TypeId(StructType.Operand(Idx));
				check(TypeInfos.Contains(TypeId));

				FFieldInfo& Field = Info.FieldInfos.AddDefaulted_GetRef();
				Field.TypeId = TypeId;
			}
		}

		uint32 NextBinding = 0; // Also figure out the smallest unused binding
		{
			// And apply any names
			for (FSpirvOffset Offset : DecoratorsAndNames)
			{
				FSpirvIterator Inst = GetInstruction(Offset);
				switch (Inst.Opcode())
				{
				case SpvOpMemberDecorate:
				{
					check(Inst.WordCount() >= 4);
					FSpirvId DecoratedType(Inst.Operand(1));
					uint32 DecoratedField = Inst.Operand(2);
					uint32 DecorationAttr = Inst.Operand(3);
					for (FUniformBufferInfo& Info : UniformBuffers)
					{
						if (Info.StructTypeId == DecoratedType)
						{
							check(DecoratedField <= uint32(Info.FieldInfos.Num()));
							FFieldInfo& Field = Info.FieldInfos[DecoratedField];
							if (DecorationAttr == SpvDecorationOffset)
							{
								check(Inst.WordCount() >= 5);
								check(Field.Offset == None_u32);
								Field.Offset = Inst.Operand(4);
							}
							else
							{
								Field.OtherDecorators.Add(Inst);
							}
						}
					}
					break;
				}
				case SpvOpMemberName:
				{
					check(Inst.WordCount() >= 4);
					FSpirvId DecoratedType(Inst.Operand(1));
					uint32 DecoratedField = Inst.Operand(2);
					for (FUniformBufferInfo& Info : UniformBuffers)
					{
						if (Info.StructTypeId == DecoratedType)
						{
							check(DecoratedField <= uint32(Info.FieldInfos.Num()));
							FFieldInfo& Field = Info.FieldInfos[DecoratedField];
							check(Field.Name.IsEmpty());
							Field.Name = ANSI_TO_TCHAR(Inst.OperandAsString(3));
							check(!Field.Name.IsEmpty());
						}
					}
					break;
				}
				case SpvOpName:
				{
					check(Inst.WordCount() >= 3);
					FSpirvId NamedId(Inst.Operand(1));
					if (FUniformBufferInfo* Info = FindUniformBufferWithId(NamedId))
					{
						check(Info->Name.IsEmpty()); // should not have two names
						Info->Name = ANSI_TO_TCHAR(Inst.OperandAsString(2));
						check(!Info->Name.IsEmpty());
					}
					break;
				}
				case SpvOpDecorate:
				{
					check(Inst.WordCount() >= 3);
					FSpirvId DecoratedId(Inst.Operand(1));
					uint32 DecorationType = Inst.Operand(2);
					if (DecorationType == SpvDecorationBinding)
					{
						check(Inst.WordCount() >= 4);
						uint32 Binding = Inst.Operand(3);
						NextBinding = FMath::Max(NextBinding, Binding + 1);
						if (FUniformBufferInfo* Info = FindUniformBufferWithId(DecoratedId))
						{
							check(Info->Binding == None_u32);
							Info->Binding = Binding;
						}
					}
					if (DecorationType == SpvDecorationDescriptorSet)
					{
						check(Inst.WordCount() >= 4);
						uint32 DescriptorSet = Inst.Operand(3);
						if (FUniformBufferInfo* Info = FindUniformBufferWithId(DecoratedId))
						{
							check(Info->DescriptorSet == None_u32);
							Info->DescriptorSet = DescriptorSet;
						}
					}
					break;
				}
				default:
					check(false); // Unhandled op in DelayedInstructions
				}
			}

			// Finally, verify that we have all of the data
			checkCode(
				for (FUniformBufferInfo& Info : UniformBuffers)
				{
					check(!Info.Name.IsEmpty());
					check(Info.DescriptorSet != None_u32);
					check(Info.Binding != None_u32);
					for (FFieldInfo& Field : Info.FieldInfos)
					{
						check(!Field.Name.IsEmpty());
						check(Field.Offset != None_u32);
					}
				}
			);
		}

		// Ok, now we need to find the global uniform buffer (or create one)
		bFoundExistingGlobalsBuffer = false;
		for (uint32 Idx = 0; Idx < uint32(UniformBuffers.Num());)
		{
			if (UniformBuffers[Idx].Name == TEXT("$Globals"))
			{
				check(!bFoundExistingGlobalsBuffer);
				bFoundExistingGlobalsBuffer = true;
				GlobalUniformBuffer = MoveTemp(UniformBuffers[Idx]);
				UniformBuffers.RemoveAt(Idx);
				break;
			}
			++Idx;
		}

		// Start assigning IDs at the Bound ID from the spir-v
		if (!bFoundExistingGlobalsBuffer)
		{
			if (UniformBuffers.Num() != 0)
			{
				// We need to make an empty globals buffer
				FSpirvId StructTypeId = AllocTypeId();
				FSpirvId PointerTypeId = AllocTypeId();
				FSpirvId VariableId = AllocId(PointerTypeId);
				GlobalUniformBuffer.Id = VariableId;
				GlobalUniformBuffer.StructTypeId = StructTypeId;
				GlobalUniformBuffer.DescriptorSet = UniformBuffers[0].DescriptorSet; // All uniforms should use the same descriptor set right now.
				GlobalUniformBuffer.Binding = NextBinding++;
				GlobalUniformBuffer.Name = TEXT("$Globals");
			}
		}
	}

	FUniformBufferInfo* FindUniformBufferWithId(FSpirvId Id)
	{
		for (FUniformBufferInfo& Info : UniformBuffers)
		{
			if (Id == Info.Id)
			{
				return &Info;
			}
		}
		return nullptr;
	};

	const FTypeInfo& GetTypeInfo(FSpirvId TypeId)
	{
		return TypeInfos.FindChecked(TypeId);
	}

	void AddVectorTypeInfo(FSpirvId VectorTypeId, FSpirvId SubtypeId, uint32 Length)
	{
		const FTypeInfo& Subtype = GetTypeInfo(SubtypeId);
		check(Subtype.MetaType == EMetaType::Scalar);
		check(Length >= 2 && Length <= 4);

		FTypeInfo Info;
		Info.MetaType = EMetaType::Vector;
		Info.VectorLength = Length;
		Info.MatrixHeight = 1;
		Info.ArrayLength = 1;
		Info.ArrayStrideBytes = 16;
		Info.ChildType = SubtypeId;
		Info.FullRows = 0;
		Info.LastRowSize = Length;
		TypeInfos.Add(VectorTypeId, Info);
	}

	void ComputeTypeInfo(FSpirvIterator Inst)
	{
		check(!TypeInfos.Contains(FSpirvId(Inst.Operand(1))));

		switch (Inst.Opcode())
		{
		case SpvOpTypeBool:
		case SpvOpTypeInt:
		case SpvOpTypeFloat:
		{
			FSpirvId TypeId(Inst.Operand(1));

			FTypeInfo Info;
			Info.MetaType = EMetaType::Scalar;
			Info.VectorLength = 1;
			Info.MatrixHeight = 1;
			Info.ArrayLength = 1;
			Info.ArrayStrideBytes = 16;
			Info.ChildType = FSpirvId();
			Info.FullRows = 0;
			Info.LastRowSize = 1;
			TypeInfos.Add(TypeId, Info);
			break;
		}
		case SpvOpTypeVector:
		{
			check(Inst.WordCount() == 4);
			// opcode, id, subtype, length
			FSpirvId TypeId(Inst.Operand(1));
			FSpirvId SubtypeId(Inst.Operand(2));
			uint32 Length = Inst.Operand(3);
			AddVectorTypeInfo(TypeId, SubtypeId, Length);
			break;
		}
		case SpvOpTypeMatrix:
		{
			check(Inst.WordCount() == 4);
			// opcode, id, subtype, rows
			FSpirvId TypeId(Inst.Operand(1));
			FSpirvId RowTypeId(Inst.Operand(2));
			uint32 Rows = Inst.Operand(3);
			const FTypeInfo& RowType = GetTypeInfo(RowTypeId);
			check(RowType.MetaType == EMetaType::Vector);
			check(Rows >= 1 && Rows <= 4);

			FTypeInfo Info;
			Info.MetaType = EMetaType::Matrix;
			Info.VectorLength = RowType.VectorLength;
			Info.MatrixHeight = Rows;
			Info.ArrayLength = 1;
			Info.ArrayStrideBytes = Rows * 16;
			Info.ChildType = RowTypeId;
			Info.FullRows = Rows - 1;
			Info.LastRowSize = RowType.LastRowSize;
			TypeInfos.Add(TypeId, Info);
			break;
		}
		case SpvOpTypeArray:
		{
			check(Inst.WordCount() == 4);
			// opcode, id, subtype, elements
			FSpirvId TypeId(Inst.Operand(1));
			FSpirvId ElementTypeId(Inst.Operand(2));
			FSpirvId NumElementsId(Inst.Operand(3)); // For some reason arrays have an OpConstant instead of a literal
			const FTypeInfo* ElementType = TypeInfos.Find(ElementTypeId);
			// Arrays of arrays and arrays of structs may exist, but they can't be fields of uniform buffers.
			// We won't compute them here, usage code doesn't support them.
			if (ElementType == nullptr || ElementType->MetaType == EMetaType::Array) return;

			uint32 NumElements = GetPositiveIntegerConstantValueChecked(NumElementsId);
			uint32 WholeRows = (ElementType->FullRows + 1) * NumElements - 1;

			// TODO: ArrayStride should come from decorators, not from the child's std140 stride.
			// TODO: Should also update size calculation to honor array stride.

			FTypeInfo Info;
			Info.MetaType = EMetaType::Array;
			Info.VectorLength = ElementType->VectorLength;
			Info.MatrixHeight = ElementType->MatrixHeight;
			Info.ArrayLength = NumElements;
			Info.ArrayStrideBytes = ElementType->ArrayStrideBytes;
			Info.ChildType = ElementTypeId;
			Info.FullRows = WholeRows;
			Info.LastRowSize = ElementType->LastRowSize;
			TypeInfos.Add(TypeId, Info);
			break;
		}
		}
	}
	
	// If the Id corresponds to an integer constant with value >= 0,
	// returns that value.  Otherwise returns None_u32.
	uint32 GetPositiveIntegerConstantValue(FSpirvId ValueId)
	{
		FSpirvIterator ValueInst = GetInstruction(ValueId);
		if (ValueInst.Opcode() == SpvOpConstant)
		{
			check(ValueInst.WordCount() == 4); // opcode, type, id, value
			uint32 Value = ValueInst.Operand(3);

#if DO_CHECK
			uint32 TypeId = ValueInst.Operand(1);
			FSpirvIterator TypeInst = GetInstruction(FSpirvId(TypeId));
			check(TypeInst.Opcode() == SpvOpTypeInt);
			check(TypeInst.WordCount() == 4); // opcode, result, bits, signedness
			uint32 BitCount = TypeInst.Operand(2);
			bool bSigned = TypeInst.Operand(3) != 0;
			check(BitCount <= 32); // bit count
			uint32 InvalidMask = ~((1U << (BitCount - 1)) - 1); // -1 to identify sign confusion
			if (!bSigned) InvalidMask = InvalidMask << 1; // Unsigned gets one more valid bit
			check((Value & InvalidMask) == 0);
			check(Value >= 0);
#endif

			return Value;
		}

		return None_u32;
	}

	// Asserts that the Id corresponds to an integer constant with value >= 0,
	// and returns that value.
	uint32 GetPositiveIntegerConstantValueChecked(FSpirvId IndexValueId)
	{
		uint32 Result = GetPositiveIntegerConstantValue(IndexValueId);
		check(Result != None_u32);
		return Result;
	}
};

struct FArrayUsageInfo
{
	TBitArray<> UsedFields; // If this is empty, there is a whole variable index.
	uint32 UsageStride;
	uint32 FirstChildUsageInfo;

	FArrayUsageInfo(uint32 FirstChild, uint32 ArrayLength, uint32 InUsageStride)
		: UsedFields(false, ArrayLength)
		, UsageStride(InUsageStride)
		, FirstChildUsageInfo(FirstChild)
	{
		check(ArrayLength != 0);
	}
};

struct FVectorUsageInfo
{
	// Each element of this array is a mask of fields which are used together.
	// Mask layout is 0bwzyx, or (Bit = 1 << ChannelIndex)
	// Common settings might include:
	// 0b0011, 0b1100, 0b0000, 0b0000 // xy, zw
	// 0b0111, 0b1000, 0b0000, 0b0000 // xyz, w
	// 0b0001, 0b0010, 0b0100, 0b1000 // x, y, z, w
	// After calling CollapseVectorUsageMasks, all masks will be disjoint
	// and they will be sorted by their first set bit.
	uint8 UsageSets[4] = { 0, 0, 0, 0 };

	// Find which usage set contains a given channel.
	// If OutNewChannel is non-null, it will be set to the new index of the specified channel.
	// For example, with usage of 0b1001 and 0b0100, the results are:
	// Channel 0 => Sub Index 0, New Channel 0
	// Channel 1 => assert failure
	// Channel 2 => Sub Index 1, New Channel 0
	// Channel 3 => Sub Index 0, New Channel 1
	uint32 FindSubIndexForChannelChecked(uint32 Channel) const
	{
		check(Channel < 4);
		uint8 ChannelMask = 1U << Channel;
		return FindSubIndexForChannelMaskChecked(ChannelMask);
	}

	// Like FindSubIndexForChannelChecked, but works for a set of channels.
	// There must be an existing set with all of the specified channels together.
	uint32 FindSubIndexForChannelMaskChecked(uint8 ChannelMask) const
	{
		check((ChannelMask & 0xFU) == ChannelMask);
		uint32 MaskIndex = 0;
		while (MaskIndex < 4)
		{
			if ((UsageSets[MaskIndex] & ChannelMask) == ChannelMask) break;
			++MaskIndex;
		}
		check(MaskIndex < 4);
		return MaskIndex;
	}

	uint32 MapToNewChannel(uint32 SubIndex, uint32 OldChannel) const
	{
		check(SubIndex < 4);
		check(OldChannel < 4);
		uint32 ChannelMask = 1U << OldChannel;
		check(UsageSets[SubIndex] & ChannelMask); // channel must be present in the sub index
		uint8 BeforeChannelMask = ChannelMask - 1;
		uint8 BeforeChannelBits = BeforeChannelMask & UsageSets[SubIndex];
		return uint32(FPlatformMath::CountBits(BeforeChannelBits));
	}
};

struct FMatrixUsageInfo
{
	uint32 FirstVectorUsageInfo;
	bool bDynamicAccess = false;

	FMatrixUsageInfo(uint32 FirstVector)
		: FirstVectorUsageInfo(FirstVector)
	{ }
};

struct FUsageReference
{
	FSpirvId TypeId;
	uint32 UsageIndex; // Index into one of the Usages arrays, which one depends on type.
	uint8 WholeMask; // Only meaningful for vector types

	FUsageReference()
		: TypeId()
		, UsageIndex(None_u32)
		, WholeMask(NotVectorMask)
	{}
	FUsageReference(FSpirvId InTypeId, uint32 Index)
		: TypeId(InTypeId)
		, UsageIndex(Index)
		, WholeMask(NotVectorMask)
	{}
	FUsageReference(FSpirvId InTypeId, uint32 Index, uint32 VectorLength)
		: TypeId(InTypeId)
		, UsageIndex(Index)
		, WholeMask((1U << VectorLength) - 1)
	{
		check(VectorLength > 0 && VectorLength <= 4);
	}
	FUsageReference(FSpirvId InTypeId, uint32 Index, const FTypeInfo& TypeInfo)
		: FUsageReference(InTypeId, Index)
	{
		if (TypeInfo.MetaType == EMetaType::Vector)
		{
			check(TypeInfo.VectorLength > 0 && TypeInfo.VectorLength <= 4);
			WholeMask = (1U << TypeInfo.VectorLength) - 1;
		}
	}
};

struct FFieldRange
{
	uint32 FirstField;
	uint32 LastFieldExclusive;
};

struct FNewGlobal
{
	uint8 PaddingSize;
	uint8 FillSize;
	uint8 VectorMask;
	uint16 OriginalBuffer;
	uint32 OriginalOffset;
	uint32 OriginalIndex;
	uint32 NewOffset;
	uint32 NewSize;
	uint32 SortedIndex;
	FSpirvId OriginalTypeId;
	FSpirvId NewTypeId;
	FSpirvId NewPtrTypeId;
	TMap<FSpirvId, FSpirvId> LoadedIds;
	FString Name;
	const TArray<FSpirvIterator>* Decorators;

	FNewGlobal(
		uint16 Buffer, uint32 FieldIndex, const FString& InName,
		uint32 Offset, FSpirvId InOriginalTypeId, const FTypeInfo& OldLayout,
		uint8 InVectorMask, FSpirvId InNewTypeId, FSpirvId InNewPtrTypeId, const FTypeInfo& NewLayout,
		const TArray<FSpirvIterator>* Decorators
	)
		: PaddingSize((NewLayout.FullRows == 0) ? NewLayout.LastRowSize : 0)
		, FillSize(4 - NewLayout.LastRowSize)
		, VectorMask(InVectorMask)
		, OriginalBuffer(Buffer)
		, OriginalOffset(Offset)
		, OriginalIndex(FieldIndex)
		, NewOffset(None_u32)
		, NewSize(NewLayout.FullRows * 16 + NewLayout.LastRowSize * 4)
		, SortedIndex(Undefined_u32)
		, OriginalTypeId(InOriginalTypeId)
		, NewTypeId(InNewTypeId)
		, NewPtrTypeId(InNewPtrTypeId)
		, Name(InName)
		, Decorators(Decorators)
	{ }
	FNewGlobal(
		uint16 Buffer, uint32 FieldIndex, const FString& InName,
		uint32 Offset, FSpirvId TypeId, FSpirvId PtrTypeId, const FTypeInfo& Layout,
		const TArray<FSpirvIterator>* Decorators
	)
		: FNewGlobal(Buffer, FieldIndex,
			InName, Offset,
			TypeId, Layout,
			Layout.MetaType == EMetaType::Vector ? (1U << Layout.VectorLength) - 1 : NotVectorMask,
			TypeId, PtrTypeId, Layout,
			Decorators)
	{ }

	bool IsScalar() const
	{
		return FPlatformMath::CountBits(VectorMask) == 1;
	}

	// We will sort these in ascending order by their new offset.
	bool operator<(const FNewGlobal& Other) const
	{
		return NewOffset < Other.NewOffset;
	}
};

struct FUsageData
{
	FSpirvExtractedData& Data;
	TArray<FArrayUsageInfo> ArrayUsages;
	TArray<FMatrixUsageInfo> MatrixUsages;
	TArray<FVectorUsageInfo> VectorUsages;

	TMap<FSpirvId, FUsageReference> KnownReferences;
	TArray<FSpirvId> ForwardUsages; // A list of used references. Application is delayed until after the whole pass.

	FUsageData(FSpirvExtractedData& InData) : Data(InData) {}

	void CollectUsageInformation()
	{
		// Prefill all member splits that we care about.
		for (FUniformBufferInfo& Info : Data.UniformBuffers)
		{
			for (FFieldInfo& Field : Info.FieldInfos)
			{
				Field.UsageInfoIndex = AddEmptyUsageData(Field.TypeId);
			}
		}

		// Now look at all of the instructions and mark used fields.
		FSpirvIterator const End = Data.SpirV.end();
		FSpirvIterator Inst = Data.GetInstruction(Data.FunctionStart);
		for (; Inst.Opcode() != SpvOpFunctionEnd; ++Inst)
		{
			check(Inst != End); // No OpFunctionEnd found!
			switch (Inst.Opcode())
			{
			case SpvOpAccessChain:
			{
				const uint32 WordCount = Inst.WordCount();
				uint32 WordIndex = 3;

				check(WordIndex < WordCount);
				FSpirvId CompositeId(Inst.Operand(WordIndex++));

				FUsageReference Reference;
				bool bTrackedReference = false;

				if (FUniformBufferInfo* Buffer = Data.FindUniformBufferWithId(CompositeId))
				{
					check(WordIndex < WordCount);
					FSpirvId FieldIndexId(Inst.Operand(WordIndex++));

					uint32 FieldIndex = Data.GetPositiveIntegerConstantValueChecked(FieldIndexId);
					check(FieldIndex < uint32(Buffer->FieldInfos.Num()));
					FFieldInfo& Field = Buffer->FieldInfos[FieldIndex];

					if (Field.UsageInfoIndex != None_u32)
					{
						Reference = FUsageReference(Field.TypeId, Field.UsageInfoIndex, Data.GetTypeInfo(Field.TypeId));
						bTrackedReference = true;
					}
				}
				else if (FUsageReference* KnownReference = KnownReferences.Find(CompositeId))
				{
					Reference = *KnownReference;
					bTrackedReference = true;
				}

				while (bTrackedReference && WordIndex < WordCount)
				{
					FSpirvId AccessId(Inst.Operand(WordIndex++));
					bTrackedReference = TrackAccessId(Reference, AccessId);
				}

				if (bTrackedReference)
				{
					FSpirvId ResultId(Inst.Operand(2));
					KnownReferences.Add(ResultId, Reference);
				}
				break;
			}
			case SpvOpLoad:
			{
				FSpirvId PointerId(Inst.Operand(3));
				if (FUsageReference* KnownReference = KnownReferences.Find(PointerId))
				{
					FSpirvId ValueId(Inst.Operand(2));
					FSpirvId TypeId(Inst.Operand(1));
					check(TypeId == KnownReference->TypeId);
					FUsageReference NonClobberedRef = *KnownReference;
					KnownReferences.Add(ValueId, NonClobberedRef);
				}
				break;
			}
			case SpvOpVectorShuffle:
			{
				FSpirvId LeftId(Inst.Operand(3));
				FSpirvId RightId(Inst.Operand(4));
				uint32 Selections = Inst.WordCount() - 5;
				FSpirvId LeftTypeId = Data.GetTypeId(LeftId);
				const FTypeInfo& LeftType = Data.GetTypeInfo(LeftTypeId);
				check(LeftType.MetaType == EMetaType::Vector);
				uint32 LeftSideNum = LeftType.VectorLength;
				uint8 LeftSideMask = 0;
				uint8 RightSideMask = 0;

				bool bLeftAndRightAreSame = LeftId == RightId;

				uint32 WordCount = Inst.WordCount();
				for (uint32 WordIndex = 5; WordIndex < WordCount; ++WordIndex)
				{
					uint32 CombinedIndex = Inst.Operand(WordIndex);
					if (CombinedIndex == None_u32) continue; // sentinel for undefined output

					// If left and right operands are the same, normalize everything
					// to be referencing the left one.
					if (bLeftAndRightAreSame && CombinedIndex >= LeftSideNum)
					{
						CombinedIndex -= LeftSideNum;
						check(CombinedIndex < LeftSideNum);
					}

					if (CombinedIndex < LeftSideNum)
					{
						LeftSideMask |= 1U << CombinedIndex;
					}
					else
					{
						uint32 RightIndex = CombinedIndex - LeftSideNum;
						check(RightIndex < 4);
						RightSideMask |= 1U << RightIndex;
					}
				}

				if (LeftSideMask != 0)
				{
					if (FUsageReference* Ref = KnownReferences.Find(LeftId))
					{
						UseVectorMask(*Ref, LeftSideMask);
					}
				}
				if (RightSideMask != 0)
				{
					if (FUsageReference* Ref = KnownReferences.Find(RightId))
					{
						UseVectorMask(*Ref, RightSideMask);
					}
				}
				break;
			}
			case SpvOpCompositeExtract:
			{
				FSpirvId CompositeId(Inst.Operand(3));
				if (FUsageReference* Ref = KnownReferences.Find(CompositeId))
				{
					FUsageReference RefIter = *Ref;
					uint32 WordCount = Inst.WordCount();
					bool bValidRef = true;
					for (uint32 WordIndex = 4; bValidRef && WordIndex < WordCount; ++WordIndex)
					{
						bValidRef = TrackAccessIndex(RefIter, Inst.Operand(WordIndex));
					}

					if (bValidRef)
					{
						FSpirvId SelfId(Inst.Operand(2));
						KnownReferences.Add(SelfId, RefIter);
					}
				}
				break;
			}

			case SpvOpPhi:
			{
				// Phi nodes may use forward declared Ids, so we need a special sort of usage for them.
				uint32 const WordCount = Inst.WordCount();
				for (uint32 Index = 3; Index < WordCount; Index += 2)
				{
					UseMaybeFutureOperand(Index, Inst);
				}
				break;
			}

#pragma region Opcodes that count as a "Use"
			case SpvOpReturnValue:
				UseOperand(1, Inst);
			case SpvOpStore:
			case SpvOpCopyMemory:
			case SpvOpCopyMemorySized:
				UseOperand(2, Inst);
				break;
			case SpvOpInBoundsAccessChain:
			case SpvOpPtrAccessChain:
			case SpvOpInBoundsPtrAccessChain:
			case SpvOpConvertFToU:
			case SpvOpConvertFToS:
			case SpvOpConvertSToF:
			case SpvOpConvertUToF:
			case SpvOpUConvert:
			case SpvOpSConvert:
			case SpvOpFConvert:
			case SpvOpQuantizeToF16:
			case SpvOpConvertPtrToU: // Assume results are used if any pointer trickery happens
			case SpvOpSatConvertSToU:
			case SpvOpSatConvertUToS:
			case SpvOpConvertUToPtr:
			case SpvOpPtrCastToGeneric:
			case SpvOpGenericCastToPtr:
			case SpvOpGenericCastToPtrExplicit:
			case SpvOpBitcast:
			case SpvOpVectorExtractDynamic:
			case SpvOpVectorInsertDynamic:
			case SpvOpTranspose:
			case SpvOpSNegate:
			case SpvOpFNegate:
			case SpvOpNot:
			case SpvOpBitReverse:
			case SpvOpBitCount:
			case SpvOpAny:
			case SpvOpAll:
			case SpvOpIsNan:
			case SpvOpIsInf:
			case SpvOpIsFinite:
			case SpvOpIsNormal:
			case SpvOpSignBitSet:
			case SpvOpLogicalNot:
			case SpvOpDPdx:
			case SpvOpDPdy:
			case SpvOpFwidth:
			case SpvOpDPdxFine:
			case SpvOpDPdyFine:
			case SpvOpFwidthFine:
			case SpvOpDPdxCoarse:
			case SpvOpDPdyCoarse:
			case SpvOpFwidthCoarse:
			case SpvOpCopyObject:
				UseOperand(3, Inst);
				break;
			case SpvOpCompositeConstruct:
			case SpvOpIAdd:
			case SpvOpFAdd:
			case SpvOpISub:
			case SpvOpFSub:
			case SpvOpIMul:
			case SpvOpFMul:
			case SpvOpUDiv:
			case SpvOpSDiv:
			case SpvOpFDiv:
			case SpvOpUMod:
			case SpvOpSRem:
			case SpvOpSMod:
			case SpvOpFRem:
			case SpvOpFMod:
			case SpvOpVectorTimesScalar:
			case SpvOpMatrixTimesScalar:
			case SpvOpVectorTimesMatrix:
			case SpvOpMatrixTimesVector:
			case SpvOpMatrixTimesMatrix:
			case SpvOpOuterProduct:
			case SpvOpDot:
			case SpvOpIAddCarry:
			case SpvOpISubBorrow:
			case SpvOpUMulExtended:
			case SpvOpSMulExtended:
			case SpvOpShiftRightLogical:
			case SpvOpShiftRightArithmetic:
			case SpvOpShiftLeftLogical:
			case SpvOpBitwiseOr:
			case SpvOpBitwiseXor:
			case SpvOpBitwiseAnd:
			case SpvOpBitFieldInsert:
			case SpvOpBitFieldSExtract:
			case SpvOpBitFieldUExtract:
			case SpvOpLessOrGreater:
			case SpvOpOrdered:
			case SpvOpUnordered:
			case SpvOpLogicalEqual:
			case SpvOpLogicalNotEqual:
			case SpvOpLogicalOr:
			case SpvOpLogicalAnd:
			case SpvOpSelect:
			case SpvOpIEqual:
			case SpvOpINotEqual:
			case SpvOpUGreaterThan:
			case SpvOpSGreaterThan:
			case SpvOpUGreaterThanEqual:
			case SpvOpSGreaterThanEqual:
			case SpvOpULessThan:
			case SpvOpSLessThan:
			case SpvOpULessThanEqual:
			case SpvOpSLessThanEqual:
			case SpvOpFOrdEqual:
			case SpvOpFUnordEqual:
			case SpvOpFOrdNotEqual:
			case SpvOpFUnordNotEqual:
			case SpvOpFOrdLessThan:
			case SpvOpFUnordLessThan:
			case SpvOpFOrdGreaterThan:
			case SpvOpFUnordGreaterThan:
			case SpvOpFOrdLessThanEqual:
			case SpvOpFUnordLessThanEqual:
			case SpvOpFOrdGreaterThanEqual:
			case SpvOpFUnordGreaterThanEqual:
			case SpvOpBuildNDRange:
				UseOperandsAfter(3, Inst);
				break;
			case SpvOpImageTexelPointer:
			case SpvOpImageQueryLod:
			case SpvOpWritePipe:
			case SpvOpGroupNonUniformAll:
			case SpvOpGroupNonUniformAny:
			case SpvOpGroupNonUniformAllEqual:
			case SpvOpGroupNonUniformBroadcastFirst:
			case SpvOpGroupNonUniformBallot:
			case SpvOpGroupNonUniformInverseBallot:
			case SpvOpGroupNonUniformBallotFindLSB:
			case SpvOpGroupNonUniformBallotFindMSB:
				UseOperand(4, Inst);
				break;
			case SpvOpGroupAsyncCopy:
			case SpvOpReservedWritePipe:
			case SpvOpGroupNonUniformBallotBitCount:
				UseOperand(5, Inst);
				break;
			case SpvOpExtInst:
			case SpvOpGroupNonUniformIAdd:
			case SpvOpGroupNonUniformFAdd:
			case SpvOpGroupNonUniformIMul:
			case SpvOpGroupNonUniformFMul:
			case SpvOpGroupNonUniformSMin:
			case SpvOpGroupNonUniformUMin:
			case SpvOpGroupNonUniformFMin:
			case SpvOpGroupNonUniformSMax:
			case SpvOpGroupNonUniformUMax:
			case SpvOpGroupNonUniformFMax:
			case SpvOpGroupNonUniformBitwiseAnd:
			case SpvOpGroupNonUniformBitwiseOr:
			case SpvOpGroupNonUniformBitwiseXor:
			case SpvOpGroupNonUniformLogicalAnd:
			case SpvOpGroupNonUniformLogicalOr:
			case SpvOpGroupNonUniformLogicalXor:
				UseOperandsAfter(5, Inst);
				break;
			case SpvOpCompositeInsert:
				UseOperand(3, Inst); // inserted object, could be matrix row
				UseOperand(4, Inst); // overwritten composite
				break;
			case SpvOpGroupBroadcast:
			case SpvOpGroupNonUniformBroadcast:
			case SpvOpGroupNonUniformBallotBitExtract:
			case SpvOpGroupNonUniformShuffle:
			case SpvOpGroupNonUniformShuffleXor:
			case SpvOpGroupNonUniformShuffleUp:
			case SpvOpGroupNonUniformShuffleDown:
			case SpvOpGroupNonUniformQuadBroadcast:
			case SpvOpGroupNonUniformQuadSwap:
				UseOperand(4, Inst);
				UseOperand(5, Inst);
				break;
			case SpvOpImageSampleImplicitLod:
			case SpvOpImageSampleExplicitLod:
			case SpvOpImageSampleProjImplicitLod:
			case SpvOpImageSampleProjExplicitLod:
			case SpvOpImageFetch:
			case SpvOpImageRead:
			case SpvOpImageSparseSampleImplicitLod:
			case SpvOpImageSparseSampleExplicitLod:
			case SpvOpImageSparseFetch:
			case SpvOpImageSparseRead:
				UseOperand(4, Inst); // Coordinate
				UseOperandsAfter(6, Inst); // Extended operands
				break;
			case SpvOpImageSampleDrefImplicitLod:
			case SpvOpImageSampleDrefExplicitLod:
			case SpvOpImageSampleProjDrefImplicitLod:
			case SpvOpImageSampleProjDrefExplicitLod:
			case SpvOpImageGather:
			case SpvOpImageDrefGather:
			case SpvOpImageSparseSampleDrefImplicitLod:
			case SpvOpImageSparseSampleDrefExplicitLod:
			case SpvOpImageSparseGather:
			case SpvOpImageSparseDrefGather:
				UseOperand(4, Inst); // Coordinate
				UseOperandsAfter(7, Inst); // Extended operands
				break;
			case SpvOpImageWrite:
				UseOperand(2, Inst); // Coordinate
				UseOperand(3, Inst); // Value
				UseOperandsAfter(5, Inst); // Extended operands
				break;
#pragma endregion

#pragma region Opcodes that do not count as "Use"
			case SpvOpNop:
			case SpvOpUndef:
			case SpvOpSourceContinued:
			case SpvOpSource:
			case SpvOpSourceExtension:
			case SpvOpName:
			case SpvOpMemberName:
			case SpvOpString:
			case SpvOpLine:
			case SpvOpNoLine:
			case SpvOpDecorate:
			case SpvOpMemberDecorate:
			case SpvOpDecorationGroup:
			case SpvOpGroupDecorate:
			case SpvOpGroupMemberDecorate:
			case SpvOpExtension:
			case SpvOpExtInstImport:
			case SpvOpMemoryModel:
			case SpvOpEntryPoint:
			case SpvOpExecutionMode:
			case SpvOpCapability:
			case SpvOpTypeVoid:
			case SpvOpTypeBool:
			case SpvOpTypeInt:
			case SpvOpTypeFloat:
			case SpvOpTypeVector:
			case SpvOpTypeMatrix:
			case SpvOpTypeImage:
			case SpvOpTypeSampler:
			case SpvOpTypeSampledImage:
			case SpvOpTypeArray:
			case SpvOpTypeRuntimeArray:
			case SpvOpTypeStruct:
			case SpvOpTypeOpaque:
			case SpvOpTypePointer:
			case SpvOpTypeFunction:
			case SpvOpTypeEvent:
			case SpvOpTypeDeviceEvent:
			case SpvOpTypeReserveId:
			case SpvOpTypeQueue:
			case SpvOpTypePipe:
			case SpvOpTypeForwardPointer:
			case SpvOpConstantTrue:
			case SpvOpConstantFalse:
			case SpvOpConstant:
			case SpvOpConstantComposite:
			case SpvOpConstantSampler:
			case SpvOpConstantNull:
			case SpvOpSpecConstantTrue:
			case SpvOpSpecConstantFalse:
			case SpvOpSpecConstant:
			case SpvOpSpecConstantComposite:
			case SpvOpSpecConstantOp:
			case SpvOpVariable:
			case SpvOpArrayLength:
			case SpvOpGenericPtrMemSemantics:
			case SpvOpFunction:
			case SpvOpFunctionParameter:
			case SpvOpSampledImage:
			case SpvOpImage:
			case SpvOpImageQueryFormat:
			case SpvOpImageQueryOrder:
			case SpvOpImageQuerySizeLod:
			case SpvOpImageQuerySize:
			case SpvOpImageQueryLevels:
			case SpvOpImageQuerySamples:
			case SpvOpImageSparseTexelsResident:
			case SpvOpLoopMerge:
			case SpvOpSelectionMerge:
			case SpvOpLabel:
			case SpvOpBranch:
			case SpvOpBranchConditional: // condition is boolean type, so we can't be tracking it
			case SpvOpSwitch: // selector is scalar, so we can't be tracking it
			case SpvOpKill:
			case SpvOpReturn:
			case SpvOpUnreachable:
			case SpvOpLifetimeStart:
			case SpvOpLifetimeStop:
			case SpvOpAtomicLoad: // Atomics only operate on scalars so we can ignore them
			case SpvOpAtomicStore:
			case SpvOpAtomicExchange:
			case SpvOpAtomicCompareExchange:
			case SpvOpAtomicCompareExchangeWeak:
			case SpvOpAtomicIIncrement:
			case SpvOpAtomicIDecrement:
			case SpvOpAtomicIAdd:
			case SpvOpAtomicISub:
			case SpvOpAtomicSMin:
			case SpvOpAtomicUMin:
			case SpvOpAtomicSMax:
			case SpvOpAtomicUMax:
			case SpvOpAtomicAnd:
			case SpvOpAtomicOr:
			case SpvOpAtomicXor:
			case SpvOpAtomicFlagTestAndSet:
			case SpvOpAtomicFlagClear:
			case SpvOpEmitVertex: // This is a usage of all "out" vars, but we aren't tracking those.
			case SpvOpEndPrimitive:
			case SpvOpEmitStreamVertex:
			case SpvOpEndStreamPrimitive:
			case SpvOpControlBarrier:
			case SpvOpMemoryBarrier:
			case SpvOpGroupWaitEvents:
			case SpvOpGroupAll:
			case SpvOpGroupAny:
			case SpvOpGroupIAdd: // Group arithmetic instructions only operate on scalars, which we aren't tracking.
			case SpvOpGroupFAdd:
			case SpvOpGroupFMin:
			case SpvOpGroupUMin:
			case SpvOpGroupSMin:
			case SpvOpGroupFMax:
			case SpvOpGroupUMax:
			case SpvOpGroupSMax:
			case SpvOpEnqueueMarker:
			case SpvOpEnqueueKernel: // Parameters must be shoved into uint8 memory so should already have been marked as used
			case SpvOpGetKernelNDrangeSubGroupCount:
			case SpvOpGetKernelNDrangeMaxSubGroupSize:
			case SpvOpGetKernelWorkGroupSize:
			case SpvOpGetKernelPreferredWorkGroupSizeMultiple:
			case SpvOpRetainEvent:
			case SpvOpReleaseEvent:
			case SpvOpCreateUserEvent:
			case SpvOpIsValidEvent:
			case SpvOpSetUserEventStatus:
			case SpvOpCaptureEventProfilingInfo:
			case SpvOpGetDefaultQueue:
			case SpvOpReadPipe:
			case SpvOpReservedReadPipe:
			case SpvOpReserveReadPipePackets:
			case SpvOpReserveWritePipePackets:
			case SpvOpCommitReadPipe:
			case SpvOpCommitWritePipe:
			case SpvOpIsValidReserveId:
			case SpvOpGetNumPipePackets:
			case SpvOpGetMaxPipePackets:
			case SpvOpGroupReserveReadPipePackets:
			case SpvOpGroupReserveWritePipePackets:
			case SpvOpGroupCommitReadPipe:
			case SpvOpGroupCommitWritePipe:
			case SpvOpGroupNonUniformElect:
				break;
#pragma endregion

			case SpvOpFunctionEnd:
			case SpvOpFunctionCall:
			default:
				check(false);
				break;
			}
		}

		ApplyForwardUsages();
		for (FUniformBufferInfo& Info : Data.UniformBuffers)
		{
			for (FFieldInfo& Field : Info.FieldInfos)
			{
				CollapseDynamicAccessValues(Field.TypeId, Field.UsageInfoIndex);
			}
		}
		CollapseVectorUsageMasks();
	}

	uint32 AddEmptyUsageData(FSpirvId TypeId)
	{
		const FTypeInfo& Info = Data.GetTypeInfo(TypeId);
		switch (Info.MetaType)
		{
		case EMetaType::Scalar:
		{
			return None_u32;
		}
		case EMetaType::Vector:
		{
			uint32 Num = uint32(VectorUsages.Num());
			VectorUsages.Add(FVectorUsageInfo());
			return Num;
		}
		case EMetaType::Matrix:
		{
			uint32 VectorIdx = uint32(VectorUsages.Num());
			for (uint32 Idx = 0; Idx < Info.MatrixHeight; ++Idx)
			{
				VectorUsages.Add(FVectorUsageInfo());
			}
			return uint32(MatrixUsages.Add(FMatrixUsageInfo(VectorIdx)));
		}
		case EMetaType::Array:
		{
			uint32 BaseId = uint32(ArrayUsages.Num());
			uint32 FirstChild = AddEmptyUsageData(Info.ChildType);
			// Arrays of arrays have an even stride in the ArrayUsages array
			uint32 UsageStride = uint32(ArrayUsages.Num() - BaseId);
			if (UsageStride == 0) UsageStride = 1; // Matrices or vectors should have one entry per index
			for (uint32 Idx = 1; Idx < Info.ArrayLength; ++Idx)
			{
				uint32 ChildIdx = AddEmptyUsageData(Info.ChildType);
				check(ChildIdx == FirstChild + Idx * UsageStride);
			}
			return uint32(ArrayUsages.Add(FArrayUsageInfo(FirstChild, Info.ArrayLength, UsageStride)));
		}
		default:
			check(false); // Unhandled meta type in usage analysis
			return None_u32;
		}
	}

	static void MarkVectorUsage(FVectorUsageInfo& UniformMember, uint8 UsageMask)
	{
		check(UsageMask != 0);
		check((UsageMask & 0xFU) == UsageMask);

		// Combine that mask with any other simultaneously used fields including the same fields
		// If none are found, add an empty slot.  Note that this algorithm may leave multiple
		// usage sets that overlap.  That's handled in CollapseVectorUsageMasks, where we normalize
		// all of them in batch.
		for (uint8& Mask : UniformMember.UsageSets)
		{
			if (Mask == 0 || (Mask & UsageMask) != 0)
			{
				Mask |= UsageMask;
				return;
			}
		}
		check(false); // If we get here, all uniform usage slots are full and none of them overlapped this mask.
		// That shouldn't be possible, because there are only four bits, there are four slots, and each slot
		// must have had at least one unique bit set when it was created.  Bits are never unset so this
		// can't happen.  If this somehow does happen in release, just add the mask to group 0 I guess.
		UniformMember.UsageSets[0] |= UsageMask;
	}

	// Marks a whole vector reference as used.  The FUsageReference passed in
	// must correspond to a vector type.
	inline void UseWholeVector(const FUsageReference& Reference)
	{
		UseVectorMask(Reference, Reference.WholeMask);
	}

	// Marks a single channel of a vector as used
	inline void UseVectorChannel(const FUsageReference& Reference, uint32 Channel)
	{
		check(Channel < 4);
		UseVectorMask(Reference, 1U << Channel);
	}

	// Marks a set of channels of a vector as being used simultaneously
	inline void UseVectorMask(const FUsageReference& Reference, uint8 Mask)
	{
		check((Mask & 0xFU) == Mask); // Mask must only have 4 bits
		check((Reference.WholeMask & NotVectorMask) == 0U); // Reference must be to a vector
		check((Reference.WholeMask & Mask) == Mask); // Mask must not include extra fields (e.g. w on a vec3)
		MarkVectorUsage(VectorUsages[Reference.UsageIndex], Mask);
	}

	void UseOperand(uint32 Index, FSpirvIterator Inst)
	{
		check(Index < Inst.WordCount());
		FSpirvId Id(Inst.Operand(Index));
		if (FUsageReference* Reference = KnownReferences.Find(Id))
		{
			TrackWholeUsage(*Reference);
		}
	}

	void UseOperandsAfter(uint32 FirstIndex, FSpirvIterator Inst)
	{
		uint32 const WordCount = Inst.WordCount();
		for (uint32 Index = FirstIndex; Index < WordCount; ++Index)
		{
			FSpirvId Id(Inst.Operand(Index));
			if (FUsageReference* Reference = KnownReferences.Find(Id))
			{
				TrackWholeUsage(*Reference);
			}
		}
	}

	// Phi nodes in loops may use IDs that haven't been declared yet.
	// We need to handle those IDs specially to ensure that they get
	// properly marked as used.
	// For now, we consider a phi node to be a full use.
	void UseMaybeFutureOperand(uint32 Index, FSpirvIterator Inst)
	{
		FSpirvId Id(Inst.Operand(Index));
		if (FUsageReference* Reference = KnownReferences.Find(Id))
		{
			TrackWholeUsage(*Reference);
		}
		else
		{
			ForwardUsages.Add(Id);
		}
	}

	void ApplyForwardUsages()
	{
		for (FSpirvId Id : ForwardUsages)
		{
			if (FUsageReference* Reference = KnownReferences.Find(Id))
			{
				TrackWholeUsage(*Reference);
			}
		}
	}

	// If a vector's usage is a contiguous subset of its source, there's value in keeping it together.
	// It generates a simpler copy list.  Force those values to be combined.
	// Unless it's a vec3, in which case that could cause unwanted fragmentation.
	// E.g. Usage sets 0b1000 and 0b0100 should be combined, or 0b1000 + 0b0100 + 0b0011.
	// 0b1000 + 0b0001 should not be combined because this will be two copies no matter what.
	// 0b1000 + 0b0110 should not be combined because a vec3 would be harder to pack than a float + vec2.
	static inline bool IsDesirableVectorMask(uint8 Mask)
	{
		if (Mask == 0) return false;
		uint8 Condensed = Mask >> FMath::CountTrailingZeros(Mask);
		bool bIsContiguous = (Condensed & (Condensed + 1)) == 0;
		bool bIsVec3 = FMath::CountBits(Mask) == 3;
		return bIsContiguous && !bIsVec3;
	}

	// Normalizes vector usage masks to ensure that all usage groups are independent.
	// Also sorts all valid usage groups before all zeroes in the list, and sorts them
	// by the index of the first used channel (so e.g. xy comes before zw).
	void CollapseVectorUsageMasks()
	{
		for (FVectorUsageInfo& Vector : VectorUsages)
		{
			if (Vector.UsageSets[0] == 0) continue;
			uint8 AllUsage = Vector.UsageSets[0] | Vector.UsageSets[1] | Vector.UsageSets[2] | Vector.UsageSets[3];
			check(AllUsage != 0);

			if (IsDesirableVectorMask(AllUsage))
			{
				Vector.UsageSets[0] = AllUsage;
				Vector.UsageSets[1] = Vector.UsageSets[2] = Vector.UsageSets[3] = 0;
				continue;
			}

			// Pairwise combine all usage sets that overlap
			for (uint32 i = 0; i < 3; ++i)
			{
				if (Vector.UsageSets[i] != 0)
				{
					for (uint32 j = i + 1; j < 4; ++j)
					{
						if ((Vector.UsageSets[i] & Vector.UsageSets[j]) != 0)
						{
							Vector.UsageSets[i] |= Vector.UsageSets[j];
							Vector.UsageSets[j] = 0;
						}
					}
				}
			}

			// Sort usage sets by their lowest set bit.  So e.g. xy consistently comes before zw.
			// This also moves nonzero elements before zero elements.
			Algo::Sort(Vector.UsageSets, [](uint8 const& A, uint8 const& B)->bool
				{
					return FPlatformMath::CountTrailingZeros(A) < FPlatformMath::CountTrailingZeros(B);
				});
		}
	}

	void CollapseTreesRecursive(const FTypeInfo& TypeInfo, uint32 DestUsageIndex, uint32 SrcUsageIndex)
	{
		switch (TypeInfo.MetaType)
		{
		case EMetaType::Scalar:
		{
			break;
		}
		case EMetaType::Vector:
		{
			FVectorUsageInfo& DestUsage = VectorUsages[DestUsageIndex];
			FVectorUsageInfo& ComponentUsage = VectorUsages[SrcUsageIndex];
			for (uint8 Mask : ComponentUsage.UsageSets)
			{
				if (Mask == 0) break;
				MarkVectorUsage(DestUsage, Mask);
			}
			break;
		}
		case EMetaType::Matrix:
		{
			FMatrixUsageInfo& DestUsage = MatrixUsages[DestUsageIndex];
			FMatrixUsageInfo& SourceUsage = MatrixUsages[SrcUsageIndex];
			DestUsage.bDynamicAccess |= SourceUsage.bDynamicAccess;
			const FTypeInfo& RowType = Data.GetTypeInfo(TypeInfo.ChildType);
			// Collapse corresponding rows of the two matrices
			for (uint32 Row = 0; Row < TypeInfo.MatrixHeight; ++Row)
			{
				CollapseTreesRecursive(RowType, DestUsage.FirstVectorUsageInfo + Row, SourceUsage.FirstVectorUsageInfo + Row);
			}
			break;
		}
		case EMetaType::Array:
		{
			FArrayUsageInfo& DestUsage = ArrayUsages[DestUsageIndex];
			FArrayUsageInfo& SourceUsage = ArrayUsages[SrcUsageIndex];
			check(SourceUsage.UsageStride == DestUsage.UsageStride); // Same type, should have same stride
			if (DestUsage.UsedFields.Num() == 0 || SourceUsage.UsedFields.Num() == 0)
			{
				DestUsage.UsedFields.Empty();
			}
			else
			{
				check(DestUsage.UsedFields.Num() == SourceUsage.UsedFields.Num());
				check(DestUsage.UsedFields.Num() == TypeInfo.ArrayLength);
				DestUsage.UsedFields.CombineWithBitwiseOR(SourceUsage.UsedFields, EBitwiseOperatorFlags::MaintainSize);
			}
			const FTypeInfo& ElementType = Data.GetTypeInfo(TypeInfo.ChildType);
			// Collapse corresponding elements of the two arrays
			for (uint32 Index = 0; Index < TypeInfo.ArrayLength; ++Index)
			{
				// Perf note: This could be sped up significantly by inlining the array->matrix or array->vector edge.
				// Each array should have a contiguous chunk of arrays or vectors, so we could do one loop over those
				// values instead of a tree traversal.
				CollapseTreesRecursive(ElementType,
					SourceUsage.FirstChildUsageInfo + Index * SourceUsage.UsageStride,
					DestUsage.FirstChildUsageInfo + Index * DestUsage.UsageStride);
			}
			break;
		}
		default:
			check(false);
		}
	}

	// For each member with dynamic access, collapse all of its children into one shared usage mask.
	// This is kind of funky when children are trees, in which case it uses CollapseTreesRecursive
	// to associate each member of each tree with a corresponding member in the other.
	// It then recurses to handle dynamic access values in children.
	void CollapseDynamicAccessValues(FSpirvId TypeId, uint32 UsageIndex)
	{
		const FTypeInfo& TypeInfo = Data.GetTypeInfo(TypeId);
		switch (TypeInfo.MetaType)
		{
		case EMetaType::Scalar:
		case EMetaType::Vector:
			break; // No dynamic access for scalars or vectors
		case EMetaType::Matrix:
		{
			// Collapse all rows into row 0
			FMatrixUsageInfo& MatrixUsage = MatrixUsages[UsageIndex];
			if (MatrixUsage.bDynamicAccess)
			{
				FVectorUsageInfo& SharedUsage = VectorUsages[MatrixUsage.FirstVectorUsageInfo];
				for (uint32 Index = 1; Index < TypeInfo.MatrixHeight; ++Index)
				{
					FVectorUsageInfo& ComponentUsage = VectorUsages[MatrixUsage.FirstVectorUsageInfo + Index];
					for (uint8 Mask : ComponentUsage.UsageSets)
					{
						if (Mask == 0) break;
						MarkVectorUsage(SharedUsage, Mask);
					}
				}
			}
			break;
		}
		case EMetaType::Array:
		{
			FArrayUsageInfo& ArrayUsage = ArrayUsages[UsageIndex];
			if (ArrayUsage.UsedFields.Num() != 0)
			{
				for (TConstSetBitIterator Iter(ArrayUsage.UsedFields); Iter; ++Iter)
				{
					uint32 ChildUsageIndex = ArrayUsage.FirstChildUsageInfo + Iter.GetIndex() * ArrayUsage.UsageStride;
					CollapseDynamicAccessValues(TypeInfo.ChildType, ChildUsageIndex);
				}
			}
			else
			{
				// First, we need to collapse all child trees into their corresponding elements
				const FTypeInfo& ChildType = Data.GetTypeInfo(TypeInfo.ChildType);
				for (uint32 Index = 1; Index < TypeInfo.ArrayLength; ++Index)
				{
					CollapseTreesRecursive(ChildType, ArrayUsage.FirstChildUsageInfo,
						ArrayUsage.FirstChildUsageInfo + Index * ArrayUsage.UsageStride);
				}
				// Then, we need to recursively do the same thing to the combined child
				CollapseDynamicAccessValues(TypeInfo.ChildType, ArrayUsage.FirstChildUsageInfo);
			}
			break;
		}
		}
	}

	// Accesses do not themselves count as usages.  Instead they propagate reference information.
	// Passing an AccessConstant of None_u32 counts as a "dynamic index operation".
	// Accesses with dynamic indices will collapse all children into one combined usage.
	// If this access produces trackable data, InOutReference and InOutTypeId are updated to the
	// derived reference, and this function returns true.  Otherwise, InOutReference and InOutTypeId
	// are left unmodified and this function returns false.
	// Access
	bool TrackAccessIndex(FUsageReference& InOutReference, uint32 AccessConstant)
	{
		const FTypeInfo& Info = Data.GetTypeInfo(InOutReference.TypeId);
		switch (Info.MetaType)
		{
		case EMetaType::Scalar:
		{
			check(false); // Can't use access on a scalar, something has gone very wrong.
			return false;
		}
		case EMetaType::Vector:
		{
			if (AccessConstant != None_u32)
			{
				// we assume all scalars are used because the optimizer before this is decent.
				UseVectorChannel(InOutReference, AccessConstant);
			}
			else
			{
				UseWholeVector(InOutReference);
			}
			// Can never continue an access chain into a scalar,
			return false;
		}
		case EMetaType::Matrix:
		{
			FMatrixUsageInfo& MatrixUsage = MatrixUsages[InOutReference.UsageIndex];
			if (AccessConstant == None_u32)
			{
				MatrixUsage.bDynamicAccess = true;
				AccessConstant = 0;
			}
			check(AccessConstant < Info.MatrixHeight);
			InOutReference = FUsageReference(Info.ChildType, MatrixUsage.FirstVectorUsageInfo + AccessConstant, Info.VectorLength);
			return true;
		}
		case EMetaType::Array:
		{
			FArrayUsageInfo& ArrayUsage = ArrayUsages[InOutReference.UsageIndex];
			if (AccessConstant == None_u32)
			{
				ArrayUsage.UsedFields.Empty();
				AccessConstant = 0;
			}
			else if (ArrayUsage.UsedFields.Num() != 0)
			{
				ArrayUsage.UsedFields[AccessConstant] = true;
			}
			check(AccessConstant < Info.ArrayLength);
			InOutReference = FUsageReference(Info.ChildType,
				ArrayUsage.FirstChildUsageInfo + AccessConstant * ArrayUsage.UsageStride,
				Data.GetTypeInfo(Info.ChildType));
			return true;
		}
		}
		check(false);
		return false;
	}

	bool TrackAccessId(FUsageReference& InOutReference, FSpirvId AccessId)
	{
		uint32 AccessConstant = Data.GetPositiveIntegerConstantValue(AccessId);
		return TrackAccessIndex(InOutReference, AccessConstant);
	}

	// Track usage of an entire object by recording dynamic access at every level.
	void TrackWholeUsage(const FUsageReference& InReference)
	{
		FUsageReference UsageIter = InReference;
		while (TrackAccessIndex(UsageIter, None_u32));
	}
};

struct FTrackedAccess
{
	TArray<uint32, TInlineAllocator<4>> OriginalPath;
	uint32 UsageIndex;
	FSpirvId OldTypeId;
};

struct FFieldMover
{
	FSpirvExtractedData& Data;
	FUsageData& Usage;

	FFieldMover(FSpirvExtractedData& InData, FUsageData& InUsage)
		: Data(InData)
		, Usage(InUsage)
	{ }

	// List of all new global fields.  This array is "stable" and will not be reordered.
	// You may refer to elements by index and expect those indices not to change.
	TArray<FNewGlobal> NewGlobalFields;
	
	// See FindFieldChecked and FindFieldRangeChecked for detailed info about how this map works.
	TMap<TArray<uint32, TInlineAllocator<4>>, FFieldRange> FieldCopiesByPath;

	// List of indices of fields in NewGlobalFields, in optimized order.
	// Each index from 0 to NewGlobalFields.Num() exists exactly once in this array.
	TArray<uint32> NewFieldOrder;

	// Generate the set of new global fields, based on the usage data.
	void GenerateNewFields()
	{
		{
			uint32 FieldIndex = 0;
			for (FFieldInfo& Info : Data.GlobalUniformBuffer.FieldInfos)
			{
				FSpirvId PtrTypeId = Data.GetOrCreatePointerType(Info.TypeId);
				uint32 Index = uint32(NewGlobalFields.Add(FNewGlobal(
					GlobalBufferSentinel, FieldIndex,
					Info.Name, Info.Offset, Info.TypeId, PtrTypeId, Data.GetTypeInfo(Info.TypeId),
					&Info.OtherDecorators)));
				FieldCopiesByPath.Emplace({ Data.GlobalUniformBuffer.Id, FieldIndex }, { Index, Index + 1 });
				++FieldIndex;
			}
		}
		{
			uint16 BufferIndex = 0;
			for (FUniformBufferInfo& Buffer : Data.UniformBuffers)
			{
				uint32 FieldIndex = 0;
				for (FFieldInfo& Info : Buffer.FieldInfos)
				{
					TArray<uint32> Path{ Buffer.Id, FieldIndex };
					GenerateMovedFields(
						BufferIndex, FieldIndex, Path,
						Info.UsageInfoIndex, Info.Name, Info.Offset,
						Info.TypeId, &Info.OtherDecorators);
					check(Path.Num() == 2);
					++FieldIndex;
				}
				++BufferIndex;
			}
		}
	}

	void GenerateMovedFields(
		uint16 BufferIndex, uint32 FieldIndex, TArray<uint32>& ReferencePath,
		uint32 UsageIndex, FString Name, uint32 Offset,
		FSpirvId TypeId, TArray<FSpirvIterator>* Decorators)
	{
		uint32 FirstIndex = uint32(NewGlobalFields.Num());
		// References get invalidated when generating vector types,
		// so we need to do a full copy of the layout here.
		const FTypeInfo Layout = Data.GetTypeInfo(TypeId);
		switch (Layout.MetaType)
		{
		case EMetaType::Array:
		{
			FArrayUsageInfo& ArrayUsage = Usage.ArrayUsages[UsageIndex];
			if (ArrayUsage.UsedFields.Num() == 0)
			{
				// Dynamic access, copy the whole array
				// TODO: We should collapse array elements under dynamic arrays.
				// For example, most mobile editor shaders have a [2]float4x4, but only use element [EyeIndex][3][3].  This should be converted to [2]float.
				// We already have the usage data to back this up, but we need additional data in the field copies.
				FSpirvId PtrTypeId = Data.GetOrCreatePointerType(TypeId);
				NewGlobalFields.Add(FNewGlobal(
					BufferIndex, FieldIndex,
					Name, Offset,
					TypeId, PtrTypeId, Layout,
					Decorators
				));
			}
			else
			{
				for (TConstSetBitIterator Iter(ArrayUsage.UsedFields); Iter; ++Iter)
				{
					uint32 ArrayIndex = uint32(Iter.GetIndex());
					uint32 IndexedUsageIndex = ArrayUsage.FirstChildUsageInfo + ArrayIndex * ArrayUsage.UsageStride;
					FString IndexedName = FString::Printf(TEXT("%s$%d"), *Name, ArrayIndex);
					uint32 IndexedOffset = Offset + ArrayIndex * Layout.ArrayStrideBytes;
					uint32 ReferenceIndex = ReferencePath.Add(ArrayIndex);
					GenerateMovedFields(
						BufferIndex, FieldIndex, ReferencePath,
						IndexedUsageIndex, IndexedName, IndexedOffset,
						Layout.ChildType, Decorators);
					ReferencePath.RemoveAt(ReferenceIndex, 1, EAllowShrinking::No);
					check(ReferencePath.Num() == ReferenceIndex);
				}
			}
			break;
		}
		case EMetaType::Matrix:
		{
			const FMatrixUsageInfo& MatrixUsage = Usage.MatrixUsages[UsageIndex];
			if (MatrixUsage.bDynamicAccess)
			{
				// TODO: We should collapse array elements under matrices.
				// For example, if we do matrix[i].x, we should collapse to only x values.
				FSpirvId PtrTypeId = Data.GetOrCreatePointerType(TypeId);
				NewGlobalFields.Add(FNewGlobal(
					BufferIndex, FieldIndex,
					Name, Offset,
					TypeId, PtrTypeId, Layout,
					Decorators
				));
			}
			else
			{
				for (uint32 Row = 0; Row < Layout.MatrixHeight; ++Row)
				{
					uint32 RowUsageIndex = MatrixUsage.FirstVectorUsageInfo + Row;
					uint32 ReferenceIndex = ReferencePath.Add(Row);
					uint32 RowOffset = Offset + Row * 16; // TODO: Use matrix type info stride instead of 16 here
					FString RowName = FString::Printf(TEXT("%s$%d"), *Name, Row);
					GenerateMovedFields(
						BufferIndex, FieldIndex, ReferencePath,
						RowUsageIndex, RowName, RowOffset,
						Layout.ChildType, Decorators);
					ReferencePath.RemoveAt(ReferenceIndex, 1, EAllowShrinking::No);
					check(ReferencePath.Num() == ReferenceIndex);
				}
			}
			break;
		}
		case EMetaType::Vector:
		{
			const FVectorUsageInfo& VectorUsage = Usage.VectorUsages[UsageIndex];
			uint8 FullMask = (1U << Layout.VectorLength) - 1;
			uint8 UsedMasks = 0;
			for (uint32 MaskIndex = 0; MaskIndex < UE_ARRAY_COUNT(VectorUsage.UsageSets); ++MaskIndex)
			{
				uint8 Mask = VectorUsage.UsageSets[MaskIndex];
				check((Mask & ~FullMask) == 0);
				if (Mask == 0) break;
				check((Mask & UsedMasks) == 0);
				UsedMasks |= Mask;
				if (Mask == FullMask)
				{
					FSpirvId PtrTypeId = Data.GetOrCreatePointerType(TypeId);
					NewGlobalFields.Add(FNewGlobal(
						BufferIndex, FieldIndex,
						Name, Offset,
						TypeId, PtrTypeId, Layout,
						Decorators
					));
				}
				else
				{
					static const TCHAR* MaskNames[] = {
						TEXT("none"),
						TEXT("x"),
						TEXT("y"),
						TEXT("xy"),
						TEXT("z"),
						TEXT("xz"),
						TEXT("yz"),
						TEXT("xyz"),
						TEXT("w"),
						TEXT("xw"),
						TEXT("yw"),
						TEXT("xyw"),
						TEXT("zw"),
						TEXT("xzw"),
						TEXT("yzw"),
						TEXT("xyzw"),
					};
					static_assert(UE_ARRAY_COUNT(MaskNames) == 16);
					check(Mask < 16);

					FString MaskedName = FString::Printf(TEXT("%s$%s"), *Name, MaskNames[Mask]);
					uint32 NumChannels = uint32(FPlatformMath::CountBits(Mask));
					FSpirvId NewTypeId = NumChannels == 1 ? Layout.ChildType : Data.GetOrCreateVectorType(Layout.ChildType, NumChannels);
					const FTypeInfo& NewLayout = Data.GetTypeInfo(NewTypeId);
					FSpirvId NewPtrTypeId = Data.GetOrCreatePointerType(NewTypeId);
					uint32 Index = uint32(NewGlobalFields.Add(FNewGlobal(
						BufferIndex, FieldIndex,
						MaskedName, Offset, TypeId, Layout,
						Mask, NewTypeId, NewPtrTypeId, NewLayout,
						Decorators
					)));
					ReferencePath.Add(MaskIndex);
					FieldCopiesByPath.Emplace(ReferencePath, FFieldRange{ Index, Index + 1 });
					ReferencePath.Pop(EAllowShrinking::No);
				}
			}
			break;
		}
		case EMetaType::Scalar:
		{
			check(UsageIndex == None_u32);
			FSpirvId PtrTypeId = Data.GetOrCreatePointerType(TypeId);
			NewGlobalFields.Add(FNewGlobal(
				BufferIndex, FieldIndex,
				Name, Offset, TypeId, PtrTypeId, Layout,
				Decorators
			));
			break;
		}
		default:
			check(false); // unsupported meta type
			break;
		}

		uint32 LastIndexExclusive = uint32(NewGlobalFields.Num());
		if (LastIndexExclusive != FirstIndex)
		{
			FieldCopiesByPath.Emplace(ReferencePath, FFieldRange{ FirstIndex, LastIndexExclusive });
		}
	}

	// Reorder the global fields for optimal packing. Note that this generates NewFieldOrder,
	// it *does not* actually reorder NewGlobalFields.
	void OptimizeFieldOrder()
	{
		// At this point we can reorder the fields
		// For now, we won't mess with array strides.  That might be worth doing in the
		// future though, for example we have some [2]float from multivew that would benefit
		// from an array stride of 1.  That would require introducing new instructions though,
		// and possibly requiring more of the driver, so it's nontrivial for now.

		// The goal here is to preserve the original order as much as possible, but move
		// some items earlier in the array to fill in gaps.
		// We don't consider moving arrays yet, or filling gaps between array items,
		// though we could do that in the future.
		uint32 Fill1 = FindNextFill(1);
		uint32 Fill2 = FindNextFill(2);
		uint32 Fill3 = FindNextFill(3);
		uint32 Offset = 0;
		const uint32 NumFields = uint32(NewGlobalFields.Num());
		for (uint32 FieldIdx = 0; FieldIdx < NumFields; ++FieldIdx)
		{
			FNewGlobal& Field = NewGlobalFields[FieldIdx];
			if (Field.PaddingSize == 1) continue; // We will place all unused scalars at the end
			if (FieldIdx == Fill2) FindNextFill(2, &Fill2); // place 2s immediately, and also fill.
			if (FieldIdx == Fill3) FindNextFill(3, &Fill3); // place 3s immediately, and also fill.
			if (Field.NewOffset != None_u32) continue; // If we already placed the field, move to the next

			// Place the field
			Field.NewOffset = Offset;
			Offset += Field.NewSize;

			// Try to fill in gaps
			uint32 FillSize = Field.FillSize;
			auto TryFill = [&](uint32 TrySize, uint32& FillIter) -> bool
			{
				bool CanFill = FillSize >= TrySize && FillIter < NumFields;
				if (CanFill)
				{
					// Place fill and increment
					NewGlobalFields[FillIter].NewOffset = Offset;
					FillSize -= TrySize;
					Offset += TrySize * 4;
					FindNextFill(TrySize, &FillIter);
				}
				return CanFill;
			};

			TryFill(3, Fill3);            // Only one 3 can fit in any given gap
			TryFill(2, Fill2);            // Only one 2 can fit in any given gap (always <= 3)
			while (TryFill(1, Fill1)) {}  // Multiple 1s may fit in a gap, try to add as many as will fit.
			Offset += FillSize * 4;       // Round up to next full row
			check((Offset & 15) == 0);    // Offset must be a multiple of 16, or we did our math wrong.
		}
		// We may have Fill1 values that are still unplaced.  Place them all now.
		while (Fill1 < NumFields)
		{
			NewGlobalFields[Fill1].NewOffset = Offset;
			Offset += 4;
			FindNextFill(1, &Fill1);
		}
		// Make sure we placed all fields
		check(Fill2 == NumFields);
		check(Fill3 == NumFields);

		// Sort the field copies by new offset
		NewFieldOrder.Reset(NewGlobalFields.Num());
		for (uint32 i = 0; i < uint32(NewGlobalFields.Num()); ++i)
		{
			NewFieldOrder.Add(i);
		}
		NewFieldOrder.Sort([&](const uint32& IdxA, const uint32& IdxB) -> bool
			{
				return NewGlobalFields[IdxA] < NewGlobalFields[IdxB];
			});
		uint32 SortedIndex = 0;
		for (uint32 FieldIdx : NewFieldOrder)
		{
			NewGlobalFields[FieldIdx].SortedIndex = SortedIndex;
			++SortedIndex;
		}
	}

	// Starting after *Pos, or at 0 if Pos is null, finds the next
	// global field (in original order) that can exactly fill a
	// gap of the specified size. If no such field is found,
	// returns NewGlobalFields.Num(). If pos is nonnull, its value
	// is updated to the found index or Num(). The result is also returned.
	uint32 FindNextFill(uint16 FillSize, uint32* Pos = nullptr)
	{
		int32 Iter = Pos ? int32(*Pos + 1) : 0;
		while (Iter < NewGlobalFields.Num())
		{
			if (NewGlobalFields[Iter].PaddingSize == FillSize) break;
			++Iter;
		}
		if (Pos) *Pos = uint32(Iter);
		return uint32(Iter);
	}

	void GenerateGlobalDecl(TArray<uint32>& NewSpirvData, uint32 const NumFields)
	{
		// Starting with the struct declaration
		NewSpirvData.Add(EncodeSpvOpWord(SpvOpTypeStruct, NumFields + 2));
		NewSpirvData.Add(Data.GlobalUniformBuffer.StructTypeId);
		for (uint32 FieldIndex = 0; FieldIndex < NumFields; ++FieldIndex)
		{
			FNewGlobal& Field = NewGlobalFields[NewFieldOrder[FieldIndex]];
			check(Field.SortedIndex == FieldIndex);
			NewSpirvData.Add(Field.NewTypeId);
		}
		// And then the pointer
		FSpirvId PointerTypeId = Data.GetTypeId(Data.GlobalUniformBuffer.Id);
		NewSpirvData.Add(EncodeSpvOpWord(SpvOpTypePointer, 4));
		NewSpirvData.Add(PointerTypeId);
		NewSpirvData.Add(SpvStorageClassUniform);
		NewSpirvData.Add(Data.GlobalUniformBuffer.StructTypeId);
		// Then the actual global
		NewSpirvData.Add(EncodeSpvOpWord(SpvOpVariable, 4));
		NewSpirvData.Add(PointerTypeId);
		NewSpirvData.Add(Data.GlobalUniformBuffer.Id);
		NewSpirvData.Add(SpvStorageClassUniform);
	}

	// Generate new spir-v that uses the new fields
	void RewriteSpirv(TArray<uint32>& NewSpirvData)
	{
		// Ok, fields have been optimized.  Now for the hard part:
		// We need to generate new patched and valid spir-v.
		// We aren't going to try to remove the unused buffers,
		// instead we are just going to patch all references to them and then
		// run an actual optimizer to remove the unused stuff.
		bool bNeedsGlobalDecl = true;
		bool bNeedsGlobalNames = !Data.bFoundExistingGlobalsBuffer;
		bool bNeedsGlobalMembers = true;
		bool bNeedsGlobalBindings = !Data.bFoundExistingGlobalsBuffer;
		bool bNeedsGlobalMemberDecorations = true;
		bool bNeedsPointerTypes = true;

		// We need uint constants to patch field accesses.  For simplicity,
		// We're going to generate all new constants, where the ID of the
		// constant with value C is UintFieldIndexBase + C.  After we
		// finish the spirv, we'll run it through the optimizer to clean
		// up any duplicates this introduces.
		bool bNeedsUintConstants = true;
		bool bNeedsUintType = !Data.UintTypeId.IsValid();
		if (bNeedsUintType)
		{
			Data.UintTypeId = Data.AllocTypeId();
		}

		uint32 const NumFields = NewGlobalFields.Num();
		uint32 UintFieldIndexBase = Data.AllocId(Data.UintTypeId).Id;
		uint32 const NumUints = FMath::Max<uint32>(NumFields, 4);
		for (uint32 i = 1; i < NumUints; ++i)
		{
			uint32 FieldIndex = Data.AllocId(Data.UintTypeId).Id;
			check(FieldIndex == UintFieldIndexBase + i);
		}

		TMap<FSpirvId, FTrackedAccess> TrackedAccesses;

		uint32 RelaxedPrecisionBlockPos = None_u32;

		FSpirvIterator Iter = Data.SpirV.begin();
		FSpirvIterator const End = Data.SpirV.end();
		// Copy the header over verbatim (we will patch the Bound later)
		NewSpirvData.Append(Data.SpirV.Data.GetData(), *Iter - Data.SpirV.Data.GetData());
		while (Iter != End)
		{
			switch (Iter.Opcode())
			{
			case SpvOpName:
			{
				if (bNeedsGlobalNames)
				{
					bNeedsGlobalNames = false;
					// Patch in the global names
					NewSpirvData.Add(EncodeSpvOpWord(SpvOpName, 6));
					NewSpirvData.Add(Data.GlobalUniformBuffer.StructTypeId);
					NewSpirvData.Add('epyt'); // Fourcc codes are backwards grumble grumble
					NewSpirvData.Add('lG$.');
					NewSpirvData.Add('labo');
					NewSpirvData.Add('s');

					NewSpirvData.Add(EncodeSpvOpWord(SpvOpName, 5));
					NewSpirvData.Add(Data.GlobalUniformBuffer.Id);
					NewSpirvData.Add('olG$');
					NewSpirvData.Add('slab');
					NewSpirvData.Add('\0');
				}

				FSpirvId NamedId(Iter.Operand(1));
				if (!Data.FindUniformBufferWithId(NamedId))
				{
					NewSpirvData.Append(*Iter, Iter.WordCount());
				}
				break;
			}
			case SpvOpMemberName:
			{
				if (bNeedsGlobalMembers)
				{
					bNeedsGlobalMembers = false;
					for (uint32 FieldIndex = 0; FieldIndex < NumFields; ++FieldIndex)
					{
						FNewGlobal& Field = NewGlobalFields[NewFieldOrder[FieldIndex]];
						check(Field.SortedIndex == FieldIndex);
						auto AnsiFieldName = StringCast<ANSICHAR>(*Field.Name);
						uint32 Len = AnsiFieldName.Length();
						uint32 LenWords = (Len + 4) / 4; // + 4 to include the null
						NewSpirvData.Add(EncodeSpvOpWord(SpvOpMemberName, LenWords + 3));
						NewSpirvData.Add(Data.GlobalUniformBuffer.StructTypeId.Id);
						NewSpirvData.Add(FieldIndex);
						uint32 StringStart = NewSpirvData.Num();
						for (uint32 i = 0; i < LenWords; ++i) NewSpirvData.Add(0);
						FMemory::Memcpy(&NewSpirvData[StringStart], AnsiFieldName.Get(), Len);
					}
				}

				// Pass through any struct members for non-global structs.
				FSpirvId StructId(Iter.Operand(1));
				if (StructId != Data.GlobalUniformBuffer.StructTypeId)
				{
					NewSpirvData.Append(*Iter, Iter.WordCount());
				}
				break;
			}
			case SpvOpDecorate:
			{
				if (bNeedsGlobalBindings)
				{
					bNeedsGlobalBindings = false;

					NewSpirvData.Add(EncodeSpvOpWord(SpvOpDecorate, 4));
					NewSpirvData.Add(Data.GlobalUniformBuffer.Id);
					NewSpirvData.Add(SpvDecorationDescriptorSet);
					NewSpirvData.Add(Data.GlobalUniformBuffer.DescriptorSet);

					NewSpirvData.Add(EncodeSpvOpWord(SpvOpDecorate, 4));
					NewSpirvData.Add(Data.GlobalUniformBuffer.Id);
					NewSpirvData.Add(SpvDecorationBinding);
					NewSpirvData.Add(Data.GlobalUniformBuffer.Binding);

					NewSpirvData.Add(EncodeSpvOpWord(SpvOpDecorate, 3));
					NewSpirvData.Add(Data.GlobalUniformBuffer.StructTypeId);
					NewSpirvData.Add(SpvDecorationBlock);
				}

				FSpirvId DecoratedId(Iter.Operand(1));
				if (!Data.FindUniformBufferWithId(DecoratedId))
				{
					uint32 Decoration = Iter.Operand(2);
					if (Decoration == SpvDecorationRelaxedPrecision)
					{
						// We will generate a new relaxed precision block, don't pass through.
						if (RelaxedPrecisionBlockPos == None_u32)
						{
							RelaxedPrecisionBlockPos = NewSpirvData.Num();
						}
					}
					else
					{
						NewSpirvData.Append(*Iter, Iter.WordCount());
					}
				}
				break;
			}
			case SpvOpDecorateString:
			{
				// Pass through global decorations and non-removed ids
				FSpirvId DecoratedId(Iter.Operand(1));
				if (!Data.FindUniformBufferWithId(DecoratedId))
				{
					NewSpirvData.Append(*Iter, Iter.WordCount());
				}
				break;
			}
			case SpvOpMemberDecorate:
			{
				if (bNeedsGlobalMemberDecorations)
				{
					bNeedsGlobalMemberDecorations = false;
					for (uint32 FieldIndex = 0; FieldIndex < NumFields; ++FieldIndex)
					{
						FNewGlobal& Field = NewGlobalFields[NewFieldOrder[FieldIndex]];
						check(Field.SortedIndex == FieldIndex);
						NewSpirvData.Add(EncodeSpvOpWord(SpvOpMemberDecorate, 5));
						NewSpirvData.Add(Data.GlobalUniformBuffer.StructTypeId);
						NewSpirvData.Add(FieldIndex);
						NewSpirvData.Add(SpvDecorationOffset);
						NewSpirvData.Add(Field.NewOffset);

						// Pass through other decorators but patch the buffer + field
						for (FSpirvIterator Deco : *Field.Decorators)
						{
							check(Deco.WordCount() >= 3);
							uint32 Pos = NewSpirvData.Num();
							NewSpirvData.Append(*Deco, Deco.WordCount());
							NewSpirvData[Pos + 1] = Data.GlobalUniformBuffer.StructTypeId;
							NewSpirvData[Pos + 2] = FieldIndex;
						}
					}
				}

				// Only pass through existing decorations for non-removed buffers
				uint32 DecoratedId = Iter.Operand(1);
				if (DecoratedId != Data.GlobalUniformBuffer.StructTypeId)
				{
					NewSpirvData.Append(*Iter, Iter.WordCount());
				}
				break;
			}
			case SpvOpTypeInt:
			case SpvOpTypeFloat:
			case SpvOpTypeBool:
			{
				// Pass through
				NewSpirvData.Append(*Iter, Iter.WordCount());

				// Then generate new dependent types
				FSpirvId BaseId(Iter.Operand(1));
				if (const FSpirvExtractedData::FVectorTypes* Vecs = Data.VectorTypes.Find(BaseId))
				{
					static_assert(UE_ARRAY_COUNT(Vecs->Ids) == 3);
					static_assert(UE_ARRAY_COUNT(Vecs->bGenerated) == 3);
					for (uint32 i = 0; i < 3; ++i)
					{
						if (Vecs->bGenerated[i])
						{
							check(Vecs->Ids[i].IsValid());
							NewSpirvData.Add(EncodeSpvOpWord(SpvOpTypeVector, 4));
							NewSpirvData.Add(Vecs->Ids[i]);
							NewSpirvData.Add(BaseId);
							NewSpirvData.Add(i + 2); // length
						}
					}
				}
				break;
			}
			case SpvOpTypeStruct:
			{
				uint32 Id = Iter.Operand(1);
				if (Data.bFoundExistingGlobalsBuffer && Id == Data.GlobalUniformBuffer.StructTypeId)
				{
					check(bNeedsGlobalDecl);
					// We're going to move the global decl later in the file, so that all field types
					// are declared before the struct type.
				}
				else
				{
					NewSpirvData.Append(*Iter, Iter.WordCount());
				}
				break;
			}
			case SpvOpTypePointer:
			{
				FSpirvId Id(Iter.Operand(1));
				FSpirvId GlobalPointerTypeId = Data.GetTypeId(Data.GlobalUniformBuffer.Id);
				// Move the global pointer type down, but leave everything else in place
				if (Id != GlobalPointerTypeId)
				{
					NewSpirvData.Append(*Iter, Iter.WordCount());
				}
				break;
			}
			case SpvOpVariable:
			{
				// Don't pass through the uniform variable declarations we removed.
				// Do pass through the global uniform buffer, if there is one.
				FSpirvId Id(Iter.Operand(2));
				uint32 StorageClass = Iter.Operand(3);

				bool const bIsKnownUniformBuffer = StorageClass == SpvStorageClassUniform &&
					(Data.FindUniformBufferWithId(Id) != nullptr || Id == Data.GlobalUniformBuffer.Id);

				// Generate the global decl near other OpVariables
				if (bIsKnownUniformBuffer && bNeedsGlobalDecl)
				{
					bNeedsGlobalDecl = false;
					GenerateGlobalDecl(NewSpirvData, NumFields);
				}

				if (!bIsKnownUniformBuffer)
				{
					NewSpirvData.Append(*Iter, Iter.WordCount());
				}
				break;
			}
			case SpvOpFunction:
			{
				// If we didn't have global variables, generate the global decl here.
				if (bNeedsGlobalDecl)
				{
					bNeedsGlobalDecl = false;
					GenerateGlobalDecl(NewSpirvData, NumFields);
				}

				// Generate new pointer types right before the function
				if (bNeedsPointerTypes)
				{
					bNeedsPointerTypes = false;
					for (FSpirvExtractedData::FGeneratedPointer& GenPtr : Data.GeneratedPointers)
					{
						NewSpirvData.Add(EncodeSpvOpWord(SpvOpTypePointer, 4));
						NewSpirvData.Add(GenPtr.PointerTypeId);
						NewSpirvData.Add(SpvStorageClassUniform);
						NewSpirvData.Add(GenPtr.ChildTypeId);
					}
				}

				// Generate new uint constants right before the function
				if (bNeedsUintConstants)
				{
					bNeedsUintConstants = false;

					// We'll do a constant consolidation pass after this,
					// so we can spit out duplicates.
					// However, constant consolidation won't deduplicate types,
					// so we need to find and reuse the existing uint32 type.
					// If there isn't one, we will create it now.
					if (bNeedsUintType)
					{
						bNeedsUintType = false;
						NewSpirvData.Add(EncodeSpvOpWord(SpvOpTypeInt, 4));
						NewSpirvData.Add(Data.UintTypeId);
						NewSpirvData.Add(32); // bit width
						NewSpirvData.Add(0); // unsigned
					}

					for (uint32 c = 0; c < NumFields; ++c)
					{
						NewSpirvData.Add(EncodeSpvOpWord(SpvOpConstant, 4));
						NewSpirvData.Add(Data.UintTypeId);
						NewSpirvData.Add(UintFieldIndexBase + c);
						NewSpirvData.Add(c);
					}
				}
				// Then pass through the function
				NewSpirvData.Append(*Iter, Iter.WordCount());
				break;
			}
			case SpvOpAccessChain:
			{
				check(!bNeedsUintConstants); // We must generate uint constants before we can do these patches
				check(Iter.WordCount() >= 5);
				FSpirvId LhsId(Iter.Operand(3));
				FSpirvId RhsId(Iter.Operand(4));

				// Find the uniform buffer this is accessing
				const FUniformBufferInfo* LhsInfo = LhsId == Data.GlobalUniformBuffer.Id
					? &Data.GlobalUniformBuffer
					: Data.FindUniformBufferWithId(LhsId);

				bool bTracked = false;
				FTrackedAccess Access;

				if (LhsInfo)
				{
					// Patch the access.
					uint32 Rhs = Data.GetPositiveIntegerConstantValueChecked(RhsId);
					const FFieldInfo& Field = LhsInfo->FieldInfos[Rhs];
					Access.UsageIndex = Field.UsageInfoIndex;
					Access.OriginalPath.Add(LhsId);
					Access.OriginalPath.Add(Rhs);
					Access.OldTypeId = Field.TypeId;
					bTracked = true;
				}
				else if (const FTrackedAccess* ExistingAccess = TrackedAccesses.Find(LhsId))
				{
					Access = *ExistingAccess;
					bTracked = true;
				}

				if (bTracked)
				{
					const uint32 Words = Iter.WordCount();
					uint32 Index = 5;
					auto GetNextAccessIndex = [&](uint32& OutIndex) -> bool
					{
						if (Index >= Words) return false;
						OutIndex = Data.GetPositiveIntegerConstantValueChecked(FSpirvId(Iter.Operand(Index++)));
						return true;
					};

					FSpirvId AccessId(Iter.Operand(2));
					uint32 OutWithinFieldPath = None_u32;
					if (FNewGlobal* Field = FollowChain(Access, OutWithinFieldPath, GetNextAccessIndex))
					{
						check(Index <= Words);
						bool bHasWithinFieldPath = OutWithinFieldPath != None_u32;
						TArrayView<const uint32> RemainingWords(*Iter + Index, Words - Index);
						NewSpirvData.Add(EncodeSpvOpWord(SpvOpAccessChain, (bHasWithinFieldPath ? 6 : 5) + RemainingWords.Num()));
						NewSpirvData.Add(Iter.Operand(1)); // Result type
						NewSpirvData.Add(AccessId); // Result ID
						NewSpirvData.Add(Data.GlobalUniformBuffer.Id); // Buffer
						NewSpirvData.Add(UintFieldIndexBase + Field->SortedIndex); // Patched part of the path
						if (bHasWithinFieldPath)
						{
							check(RemainingWords.Num() == 0); // Can't have further dereferences after a scalar
							NewSpirvData.Add(UintFieldIndexBase + OutWithinFieldPath);
						}
						NewSpirvData.Append(RemainingWords); // Any further dereferences
					}
					else
					{
						// Add tracking data, but don't generate any access chains yet.
						// Basically just save the state for this loop.
						TrackedAccesses.Emplace(AccessId, MoveTemp(Access));
					}
				}
				else
				{
					NewSpirvData.Append(*Iter, Iter.WordCount());
				}
				break;
			}
			case SpvOpLoad:
			{
				// Convert a load into access chains and loads for every field
				FSpirvId PointerId(Iter.Operand(3));
				FSpirvId OldLoadId(Iter.Operand(2));
				if (const FTrackedAccess* Access = TrackedAccesses.Find(PointerId))
				{
					for (FNewGlobal& Field : FindFieldRangeChecked(Access->OriginalPath))
					{
						FSpirvId AccessId = Data.AllocId(Field.NewPtrTypeId, PointerId);
						FSpirvId LoadId = Data.AllocId(Field.NewTypeId, OldLoadId);

						NewSpirvData.Add(EncodeSpvOpWord(SpvOpAccessChain, 5));
						NewSpirvData.Add(Field.NewPtrTypeId); // Result type
						NewSpirvData.Add(AccessId); // Result ID
						NewSpirvData.Add(Data.GlobalUniformBuffer.Id); // Buffer
						NewSpirvData.Add(UintFieldIndexBase + Field.SortedIndex); // Field

						NewSpirvData.Add(EncodeSpvOpWord(SpvOpLoad, 4));
						NewSpirvData.Add(Field.NewTypeId); // Result type
						NewSpirvData.Add(LoadId); // Result ID
						NewSpirvData.Add(AccessId); // Pointer ID

						Field.LoadedIds.Add(OldLoadId, LoadId);
					}
					TrackedAccesses.Add(OldLoadId, *Access);
				}
				else
				{
					NewSpirvData.Append(*Iter, Iter.WordCount());
				}
				break;
			}
			case SpvOpCompositeExtract:
			{
				FSpirvId CompositeId(Iter.Operand(3));
				if (const FTrackedAccess* Access = TrackedAccesses.Find(CompositeId))
				{
					const uint32 Words = Iter.WordCount();
					uint32 Index = 4;
					auto GetNextAccessIndex = [&](uint32& OutIndex) -> bool
					{
						if (Index >= Words) return false;
						OutIndex = Iter.Operand(Index++);
						return true;
					};

					FSpirvId ResultId(Iter.Operand(2));
					FTrackedAccess NewAccess = *Access;
					uint32 OutWithinFieldPath = None_u32;
					if (FNewGlobal* Field = FollowChain(NewAccess, OutWithinFieldPath, GetNextAccessIndex))
					{
						FSpirvId LoadedId = Field->LoadedIds.FindChecked(CompositeId); // Must have previously issued an OpLoad for this composite
						FSpirvId ResultTypeId(Iter.Operand(1));
						bool bHasWithinFieldPath = OutWithinFieldPath != None_u32;
						TArrayView<const uint32> RemainingIndexers(*Iter + Index, Words - Index);
						// If there are no indexes to do (i.e. because of destructuring), we can't use OpCompositeExtract.
						// We need to patch it into OpCopyObject.  The optimizer should delete this later.
						if (!bHasWithinFieldPath && RemainingIndexers.Num() == 0)
						{
							NewSpirvData.Add(EncodeSpvOpWord(SpvOpCopyObject, 4));
							NewSpirvData.Add(ResultTypeId); // Result type (unchanged)
							NewSpirvData.Add(ResultId); // Result ID (unchanged)
							NewSpirvData.Add(LoadedId); // New loaded object ID
						}
						else
						{
							NewSpirvData.Add(EncodeSpvOpWord(SpvOpCompositeExtract, (bHasWithinFieldPath ? 5 : 4) + RemainingIndexers.Num()));
							NewSpirvData.Add(ResultTypeId);
							NewSpirvData.Add(ResultId);
							NewSpirvData.Add(LoadedId);
							if (bHasWithinFieldPath)
							{
								check(RemainingIndexers.Num() == 0); // Can't have more indexers into a scalar
								NewSpirvData.Add(OutWithinFieldPath);
							}
							NewSpirvData.Append(RemainingIndexers);
						}
					}
					else
					{
						// We are still tracking the value somehow, e.g. this is a row extracted from a matrix, which is not fully used.
						// Or a matrix extracted from an array value. In that case, we have effectively already done the extraction, in
						// the form of an OpLoad for the extracted piece.  We just need to save the access data so we can find it later.
						for (FNewGlobal& SubField : FindFieldRangeChecked(NewAccess.OriginalPath))
						{
							FSpirvId LoadedId = SubField.LoadedIds.FindChecked(CompositeId);
							SubField.LoadedIds.Add(ResultId, LoadedId);
						}
						TrackedAccesses.Add(ResultId, MoveTemp(NewAccess));
					}
				}
				else
				{
					NewSpirvData.Append(*Iter, Iter.WordCount());
				}
				break;
			}
			case SpvOpVectorShuffle:
			{
				FSpirvId LeftId(Iter.Operand(3));
				FSpirvId RightId(Iter.Operand(4));
				FSpirvId LeftTypeId = Data.GetTypeId(LeftId);
				const FTypeInfo& LeftType = Data.GetTypeInfo(LeftTypeId);
				check(LeftType.MetaType == EMetaType::Vector);
				uint32 LeftSideNum = LeftType.VectorLength;
				uint8 LeftSideMask = 0;
				uint8 RightSideMask = 0;

				bool bLeftAndRightAreSame = LeftId == RightId;

				const TArrayView<const uint32> Selections(*Iter + 5, Iter.WordCount() - 5);
				for (uint32 Selection : Selections)
				{
					if (Selection != None_u32) // sentinel meaning "undefined"
					{
						if (bLeftAndRightAreSame && Selection >= LeftSideNum)
						{
							Selection -= LeftSideNum;
							check(Selection < LeftSideNum);
						}

						if (Selection < LeftSideNum)
						{
							LeftSideMask |= 1U << Selection;
						}
						else
						{
							uint32 RightIndex = Selection - LeftSideNum;
							check(RightIndex < 4);
							RightSideMask |= 1U << RightIndex;
						}
					}
				}

				const FTrackedAccess* LeftRef = TrackedAccesses.Find(LeftId);
				const FTrackedAccess* RightRef = TrackedAccesses.Find(RightId);
				uint32 LeftSubIndex = None_u32;
				uint32 RightSubIndex = None_u32;
				const FVectorUsageInfo* LeftUsage = nullptr;
				const FVectorUsageInfo* RightUsage = nullptr;
				const FNewGlobal* LeftField = nullptr;
				const FNewGlobal* RightField = nullptr;
				FSpirvId LeftLoadedId;
				FSpirvId RightLoadedId;

				bool bNeedsRewriting = false;
				bool bNeedsCompositeConstruct = false;
				if (LeftRef)
				{
					const FTypeInfo& Info = Data.GetTypeInfo(LeftRef->OldTypeId);
					check(Info.MetaType == EMetaType::Vector);
					LeftUsage = &Usage.VectorUsages[LeftRef->UsageIndex];
					LeftSubIndex = LeftUsage->FindSubIndexForChannelMaskChecked(LeftSideMask);
					{
						TArray<uint32, TInlineAllocator<4>> Path = LeftRef->OriginalPath;
						Path.Add(LeftSubIndex);
						LeftField = &FindFieldChecked(Path);
					}
					LeftLoadedId = LeftField->LoadedIds.FindChecked(LeftId);
					if (LeftField->IsScalar()) bNeedsCompositeConstruct = true;
					bNeedsRewriting = true;
				}
				if (bLeftAndRightAreSame)
				{
					RightRef = LeftRef;
					RightUsage = LeftUsage;
					RightSubIndex = LeftSubIndex;
					RightField = LeftField;
					RightLoadedId = LeftLoadedId;
				}
				else if (RightRef)
				{
					const FTypeInfo& Info = Data.GetTypeInfo(RightRef->OldTypeId);
					check(Info.MetaType == EMetaType::Vector);
					RightUsage = &Usage.VectorUsages[RightRef->UsageIndex];
					RightSubIndex = RightUsage->FindSubIndexForChannelMaskChecked(RightSideMask);
					{
						TArray<uint32, TInlineAllocator<4>> Path = RightRef->OriginalPath;
						Path.Add(RightSubIndex);
						RightField = &FindFieldChecked(Path);
					}
					RightLoadedId = RightField->LoadedIds.FindChecked(RightId);
					if (RightField->IsScalar()) bNeedsCompositeConstruct = true;
					bNeedsRewriting = true;
				}

				if (bNeedsRewriting)
				{
					FSpirvId ResultTypeId(Iter.Operand(1));
					FSpirvId ResultId(Iter.Operand(2));
					FSpirvIterator ResultTypeInst = Data.GetInstruction(ResultTypeId);
					check(ResultTypeInst.Opcode() == SpvOpTypeVector);
					FSpirvId ScalarTypeId(ResultTypeInst.Operand(2));
					if (bNeedsCompositeConstruct)
					{
						// At least one of the operands was converted to a scalar, so we
						// can no longer use a vector shuffle.
						// We need to use OpCompositeExtract and OpCompositeConstruct to
						// create the vector instead.
						FSpirvId ChannelIds[8];
						for (FSpirvId& Id : ChannelIds) Id = FSpirvId();

						// Generate IDs for all of the channels we care about.
						FSpirvId AnyId;
						for (uint32 Selection : Selections)
						{
							if (Selection != None_u32)
							{
								if (Selection >= LeftSideNum && bLeftAndRightAreSame)
								{
									Selection -= LeftSideNum;
									check(Selection < LeftSideNum);
								}

								check(Selection < UE_ARRAY_COUNT(ChannelIds));
								if (ChannelIds[Selection].IsValid()) continue;

								bool bFromLeft = Selection < LeftSideNum;
								FSpirvId SourceId = bFromLeft ? LeftId : RightId;
								const FNewGlobal* Field = bFromLeft ? LeftField : RightField;
								const FVectorUsageInfo* VectorUsage = bFromLeft ? LeftUsage : RightUsage;
								uint32 SubIndex = bFromLeft ? LeftSubIndex : RightSubIndex;
								uint32 Channel = bFromLeft ? Selection : (Selection - LeftSideNum);
								FSpirvId LoadedId = bFromLeft ? LeftLoadedId : RightLoadedId;
								if (Field != nullptr)
								{
									if (Field->IsScalar())
									{
										ChannelIds[Selection] = LoadedId;
										AnyId = LoadedId;
									}
									else
									{
										check(VectorUsage != nullptr);
										check(SubIndex != None_u32);

										FSpirvId ExtractedId = Data.AllocId(ScalarTypeId, ResultId);
										NewSpirvData.Add(EncodeSpvOpWord(SpvOpCompositeExtract, 5));
										NewSpirvData.Add(ScalarTypeId);
										NewSpirvData.Add(ExtractedId);
										NewSpirvData.Add(LoadedId);
										NewSpirvData.Add(VectorUsage->MapToNewChannel(SubIndex, Channel));
										ChannelIds[Selection] = ExtractedId;
										AnyId = ExtractedId;
									}
								}
								else
								{
									FSpirvId ExtractedId = Data.AllocId(ScalarTypeId, ResultId);
									NewSpirvData.Add(EncodeSpvOpWord(SpvOpCompositeExtract, 5));
									NewSpirvData.Add(ScalarTypeId);
									NewSpirvData.Add(ExtractedId);
									NewSpirvData.Add(SourceId);
									NewSpirvData.Add(Channel);
									ChannelIds[Selection] = ExtractedId;
									AnyId = ExtractedId;
								}
							}
						}

						check(AnyId.IsValid());

						// Then spit out an OpCompositeConstruct
						NewSpirvData.Add(EncodeSpvOpWord(SpvOpCompositeConstruct, 3 + Selections.Num()));
						NewSpirvData.Add(ResultTypeId); // Result type (unchanged)
						NewSpirvData.Add(ResultId); // Result Id (unchanged)
						for (uint32 Selection : Selections)
						{
							if (Selection == None_u32)
							{
								NewSpirvData.Add(AnyId);
							}
							else
							{
								check(Selection < UE_ARRAY_COUNT(ChannelIds));
								check(ChannelIds[Selection].IsValid());
								NewSpirvData.Add(ChannelIds[Selection]);
							}
						}
					}
					else // if (!bNeedsCompositeConstruct)
					{
						NewSpirvData.Add(Iter.Operand(0)); // instruction/count (unchanged)
						NewSpirvData.Add(ResultTypeId); // result type (unchanged)
						NewSpirvData.Add(ResultId); // result id (unchanged)
						NewSpirvData.Add(LeftField ? LeftLoadedId : LeftId); // Left input
						NewSpirvData.Add(RightField ? RightLoadedId : RightId); // Right input

						uint32 NewLeftSideNum = LeftField ? Data.GetTypeInfo(LeftField->NewTypeId).VectorLength : LeftSideNum;
						for (uint32 Selection : Selections)
						{
							if (Selection == None_u32)
							{
								NewSpirvData.Add(Selection);
							}
							else
							{
								bool bFromLeft = Selection < LeftSideNum;
								// Convert to index into operand
								if (!bFromLeft)
								{
									Selection -= LeftSideNum;
								}
								// Map to new channel (e.g. if usage converted a vec3 to vec2, map .z to .y)
								const FVectorUsageInfo* VectorUsage = bFromLeft ? LeftUsage : RightUsage;
								if (VectorUsage)
								{
									uint32 SubIndex = bFromLeft ? LeftSubIndex : RightSubIndex;
									Selection = VectorUsage->MapToNewChannel(SubIndex, Selection);
								}
								// Move back to right side if needed
								if (!bFromLeft)
								{
									Selection += NewLeftSideNum;
								}
								NewSpirvData.Add(Selection);
							}
						}
					}
				}
				else // if (!bNeedsRewriting)
				{
					NewSpirvData.Append(*Iter, Iter.WordCount());
				}
				break;
			}
			default:
				// Copy the data over verbatim
				NewSpirvData.Append(*Iter, Iter.WordCount());
				break;
			}
			++Iter;
		}
		// Patch the max ID value
		NewSpirvData[3] = Data.NextId;
		// Make sure we generated everything
		check(!bNeedsGlobalDecl);
		check(!bNeedsGlobalNames);
		check(!bNeedsGlobalMembers);
		check(!bNeedsGlobalBindings);
		check(!bNeedsPointerTypes);
		check(!bNeedsUintConstants);
		check(!bNeedsUintType);
		check(!bNeedsGlobalMemberDecorations);

		// Generate a new relaxed precision block
		if (RelaxedPrecisionBlockPos != None_u32)
		{
			TArray<uint32> RPDecorators;
			for (TConstSetBitIterator RPIter(Data.RelaxedPrecision); RPIter; ++RPIter)
			{
				FSpirvId Id(RPIter.GetIndex());
				// Any "tracked" IDs were removed from the source, so we can't emit
				// decorators for them.  Their decorators should be copied onto any
				// new generated instructions by the BasedOnId parameter to AllocId.
				if (!TrackedAccesses.Contains(Id))
				{
					RPDecorators.Add(EncodeSpvOpWord(SpvOpDecorate, 3));
					RPDecorators.Add(Id);
					RPDecorators.Add(SpvDecorationRelaxedPrecision);
				}
			}
			NewSpirvData.Insert(RPDecorators, RelaxedPrecisionBlockPos);
		}
		else
		{
			check(Data.RelaxedPrecision.CountSetBits() == 0);
		}
	}

	// Follows a chain of constant accesses.  GetNextAccessIndex is used to fetch the next index.
	// If at any point in the chain, including before the first access or after the last access,
	// the chain encounters a value which is (or is part of) a new field, it returns a pointer to
	// that new field. If OutWithinFieldPath is not None_u32 at this point, that is modified
	// channel index into the new field that should be patched as well.
	// 
	// If the end of the chain is reached before finding a field, this function returns nullptr,
	// and the TrackedAccess state is left in a state that may be resumed by a further call to
	// FollowChain later when more pieces of the full chain are found.
	// 
	// The passed GetNextAccessIndex lambda should have the signature
	// bool GetNextAccessIndex(uint32& OutIndex). If the chain has another index and that index is
	// a constant, GetNextAccessIndex should set OutIndex to that constant and return true.  Otherwise
	// it should return false and set OutIndex to undefined.
	template<typename TGetNextAccessIndex>
	FNewGlobal* FollowChain(FTrackedAccess& Access, uint32& OutWithinFieldPath, TGetNextAccessIndex& GetNextAccessIndex)
	{
		OutWithinFieldPath = None_u32;
		while (true)
		{
			if (Access.UsageIndex == None_u32)
			{
				return &FindFieldChecked(Access.OriginalPath);
			}

			const FTypeInfo& Info = Data.GetTypeInfo(Access.OldTypeId);
			uint32 NextUsageBaseIndex = Undefined_u32;
			uint32 NextUsageStride = 0;
			switch (Info.MetaType)
			{
			case EMetaType::Array:
			{
				FArrayUsageInfo& ArrayUsage = Usage.ArrayUsages[Access.UsageIndex];
				if (ArrayUsage.UsedFields.Num() == 0)
				{
					return &FindFieldChecked(Access.OriginalPath);
				}
				else
				{
					NextUsageBaseIndex = ArrayUsage.FirstChildUsageInfo;
					NextUsageStride = ArrayUsage.UsageStride;
				}
				break;
			}
			case EMetaType::Matrix:
			{
				FMatrixUsageInfo& MatrixUsage = Usage.MatrixUsages[Access.UsageIndex];
				if (MatrixUsage.bDynamicAccess)
				{
					return &FindFieldChecked(Access.OriginalPath);
				}
				else
				{
					NextUsageBaseIndex = MatrixUsage.FirstVectorUsageInfo;
					NextUsageStride = 1;
				}
				break;
			}
			case EMetaType::Vector:
			{
				FVectorUsageInfo& VectorUsage = Usage.VectorUsages[Access.UsageIndex];
				uint8 FullMask = (1U << Info.VectorLength) - 1;
				bool bIsFullVector = FullMask == VectorUsage.UsageSets[0];
				if (bIsFullVector)
				{
					check(VectorUsage.UsageSets[1] == 0);
					return &FindFieldChecked(Access.OriginalPath);
				}
				else
				{
					uint32 OrigChannelIndex;
					if (!GetNextAccessIndex(OrigChannelIndex)) return nullptr;

					uint32 MaskIndex = VectorUsage.FindSubIndexForChannelChecked(OrigChannelIndex);
					uint32 NewChannelIndex = VectorUsage.MapToNewChannel(MaskIndex, OrigChannelIndex);
					Access.OriginalPath.Add(MaskIndex);
					FNewGlobal& Field = FindFieldChecked(Access.OriginalPath);
					check(Field.VectorMask == VectorUsage.UsageSets[MaskIndex]);
					// Save off the new in-vector indexer, but only if the new value is still a vector.
					if (Field.IsScalar())
					{
						check(NewChannelIndex == 0);
					}
					else
					{
						OutWithinFieldPath = NewChannelIndex;
					}
					return &Field;
				}
			}
			case EMetaType::Scalar:
			{
				return &FindFieldChecked(Access.OriginalPath);
			}
			default:
			{
				check(false);
				break;
			}
			}

			uint32 AccessIndex;
			if (!GetNextAccessIndex(AccessIndex)) return nullptr;

			Access.OriginalPath.Add(AccessIndex);
			Access.OldTypeId = Info.ChildType;
			Access.UsageIndex = NextUsageBaseIndex + NextUsageStride * AccessIndex;
		}
	}

	// Find the field associated with a reference path.  For example, with
	// OpName %4 "View"
	// %4 = uniform buffer { mat4 m; vec4 v; float f; vec4 u; }
	// As split into
	// global {
	//   vec4 View_m$1;
	//   float View_m$2$y;
	//   float View_m$2$w;
	//	 vec2 View_v$yz;
	//   vec4 View_u;
	// }
	// these paths are valid:
	//   { 4, 0, 1 }    => View_m$1
	//   { 4, 0, 2, 0 } => View_m$2$y // last index is usage set index
	//   { 4, 0, 2, 1 } => View_m$2$w
	//   { 4, 1, 0 }    => View_v$yz
	//   { 4, 3 }       => View_u
	// If the path was split into multiple fields, or did not generate a field, this function causes an assertion error.
	// For example, the paths { 4, 0 } and { 4, 2 } will trigger these errors, respectively.
	FNewGlobal& FindFieldChecked(const TArray<uint32, TInlineAllocator<4>>& Path)
	{
		FFieldRange Range = FieldCopiesByPath.FindChecked(Path);
		check(Range.FirstField + 1 == Range.LastFieldExclusive); // Must be a single field
		return NewGlobalFields[Range.FirstField];
	}

	// Find all fields associated with a reference path.  For example, with the example on FindFieldChecked,
	// the path { 4, 0 } will return the array { View_m$1, View_m$2$y, View_m$2$w }.
	// The requested path must have generated at least one field.  Requesting e.g. { 4, 2 } will trigger an assertion error.
	TArrayView<FNewGlobal> FindFieldRangeChecked(const TArray<uint32, TInlineAllocator<4>>& Path)
	{
		FFieldRange Range = FieldCopiesByPath.FindChecked(Path);
		return TArrayView<FNewGlobal>(&NewGlobalFields[Range.FirstField], Range.LastFieldExclusive - Range.FirstField);
	}

};

// See the comment at the top of this file for an overview.
bool OptimizeGlobalUniforms(
	const FShaderCompilerInput& Input,
	CrossCompiler::FShaderConductorContext& CompilerContext,
	FSpirv& Spirv,
	FOptimizedUniforms& OutUniforms,
	bool bDebugDump,
	bool bUseEmulatedUBs)
{
	if (bDebugDump)
	{
		DumpDebugShaderDisassembledSpirv(Input, Spirv.GetByteData(), Spirv.GetByteSize(), TEXT("unpatched.spvasm"));
	}

	// Do one pass over the spirv to collect metadata
	FSpirvExtractedData Data(Spirv);
	Data.ExtractData();
	{
		for (uint32 Idx = 0; Idx < uint32(Data.UniformBuffers.Num());)
		{
			const FUniformBufferEntry* Entry = Input.Environment.UniformBufferMap.Find(Data.UniformBuffers[Idx].Name);
			if ((Entry && EnumHasAnyFlags(Entry->Flags, ERHIUniformBufferFlags::NoEmulatedUniformBuffer | ERHIUniformBufferFlags::UniformView)) || !bUseEmulatedUBs)
			{
				Data.UniformBuffers.RemoveAt(Idx);
				continue;
			}

			++Idx;
		}
		check(bUseEmulatedUBs || Data.UniformBuffers.Num() == 0);

		if (!Data.bFoundExistingGlobalsBuffer && Data.UniformBuffers.Num() == 0)
		{
			return true; // Nothing to optimize
		}
	}

	// Now we can do array and composite splitting.
	// We can't safely split arrays in the global uniform buffer, because the copies we emit for it are not
	// fine-grained enough.  But any fields of emulated UBs can be packed.
	// Since it only applies to emulated UBs, we only need to do this if those are enabled.
	// We'll start by collecting data about what parts of fields are actually used.
	FUsageData Usage(Data);
	if (Data.UniformBuffers.Num() > 0)
	{
		Usage.CollectUsageInformation();
	}

	// Now we can reorder the fields and generate new Spirv
	FFieldMover Mover(Data, Usage);
	Mover.GenerateNewFields();
	Mover.OptimizeFieldOrder();
	TArray<uint32> NewSpirvData;
	Mover.RewriteSpirv(NewSpirvData);

	// Lastly, we will optimize the new shader a bit to clean up some of our messy generation.
	if (bDebugDump)
	{
		DumpDebugShaderDisassembledSpirv(Input, NewSpirvData.GetData(), NewSpirvData.GetTypeSize() * NewSpirvData.Num(), TEXT("patched.raw.spvasm"));
	}

	const char* OptArgs[] = {
		"--unify-const", // Remove duplicate uint instructions
		"--simplify-instructions", // Remove OpCopyObject and identity swizzles
		"--eliminate-dead-code-aggressive", // Remove any types or constants used only by removed uniform buffers
		"--compact-ids" // Reduce the Bound
	};
	if (!CompilerContext.OptimizeSpirv(NewSpirvData, OptArgs, UE_ARRAY_COUNT(OptArgs)))
	{
		return false;
	}

	if (bDebugDump)
	{
		DumpDebugShaderDisassembledSpirv(Input, NewSpirvData.GetData(), NewSpirvData.GetTypeSize() * NewSpirvData.Num(), TEXT("patched.spvasm"));
	}

	Spirv.Data = MoveTemp(NewSpirvData);

	// Finally, we can write our reorderings into the CC Header
	{
		struct FMaskCopy { uint32 SrcOffset; uint32 Words; };
		static const FMaskCopy Copies[] = {
			{ 0, 0 }, { 0, 0 }, // 0b0000
			{ 0, 1 }, { 0, 0 }, // 0b0001
			{ 1, 1 }, { 0, 0 }, // 0b0010
			{ 0, 2 }, { 0, 0 }, // 0b0011
			{ 2, 1 }, { 0, 0 }, // 0b0100
			{ 0, 1 }, { 2, 1 }, // 0b0101
			{ 1, 2 }, { 0, 0 }, // 0b0110
			{ 0, 3 }, { 0, 0 }, // 0b0111
			{ 3, 1 }, { 0, 0 }, // 0b1000
			{ 0, 1 }, { 3, 1 }, // 0b1001
			{ 1, 1 }, { 3, 1 }, // 0b1010
			{ 0, 2 }, { 3, 1 }, // 0b1011
			{ 2, 2 }, { 0, 0 }, // 0b1100
			{ 0, 1 }, { 2, 2 }, // 0b1101
			{ 1, 3 }, { 0, 0 }, // 0b1110
			{ 0, 4 }, { 0, 0 }, // 0b1111
		};
		static_assert(UE_ARRAY_COUNT(Copies) == 2 * 16, "Wrong number of copies!");

		for (FUniformBufferInfo& Info : Data.UniformBuffers)
		{
			OutUniforms.AddPackedUB(*Info.Name);
		}

		// Register all actual global uniform buffer members as loose data
		uint32 const NumFields = Mover.NewGlobalFields.Num();
		for (uint32 FieldIndex = 0; FieldIndex < NumFields; ++FieldIndex)
		{
			FNewGlobal& Field = Mover.NewGlobalFields[Mover.NewFieldOrder[FieldIndex]];
			check(Field.SortedIndex == FieldIndex);
			if (Field.OriginalBuffer == GlobalBufferSentinel)
			{
				check(Field.OriginalTypeId == Field.NewTypeId);
				OutUniforms.AddPackedGlobal(
					*Field.Name,
					Field.NewOffset,
					Field.NewSize);
			}
			else
			{
				FUniformBufferInfo& Buffer = Data.UniformBuffers[Field.OriginalBuffer];
				check(Field.OriginalOffset % 4 == 0);
				check(Field.NewOffset % 4 == 0);
				check(Field.NewSize % 4 == 0);
				if (Field.VectorMask == NotVectorMask)
				{
					check(Field.OriginalTypeId == Field.NewTypeId);
					OutUniforms.AddPackedUBGlobalCopy(
						Field.OriginalBuffer, Field.OriginalOffset / 4,
						Field.NewOffset / 4, Field.NewSize / 4);
				}
				else
				{
					check(Field.VectorMask < 16);
					const FMaskCopy* VecCopies = &Copies[Field.VectorMask * 2];
					for (uint32 Copy = 0; Copy < 2; ++Copy)
					{
						if (VecCopies[Copy].Words == 0) break;
						OutUniforms.AddPackedUBGlobalCopy(
							Field.OriginalBuffer, Field.OriginalOffset / 4 + VecCopies[Copy].SrcOffset,
							Field.NewOffset / 4, VecCopies[Copy].Words);
					}
				}
			}
		}
	}

	return true;
}
