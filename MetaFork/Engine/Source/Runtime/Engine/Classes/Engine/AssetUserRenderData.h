// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "CoreTypes.h"

/*----------------------------------------------------------------------------
	FAssetUserRenderData.
----------------------------------------------------------------------------*/
/**
 * Interface class for the render data associated with UAssetUserData
 */
class FAssetUserRenderData
{
private:
	inline static TArray<FName> RenderDataTypeNames;
	uint32 RenderType;
public:

	/** Constructor. */
	FAssetUserRenderData(uint32 InType) :RenderType(InType) { ; }

	virtual ~FAssetUserRenderData() { ; }

	FORCEINLINE uint32 GetType() const { return RenderType; }

	virtual FString GetDebugName() const { return FString(); }

	static uint32 RegisterRenderDataType(FName TypeName)
	{
		ensureMsgf(!RenderDataTypeNames.Contains(TypeName), TEXT("Render Data Type '%s' is already registered!"), *TypeName.ToString());
		uint32 TypeId = RenderDataTypeNames.Num();
		RenderDataTypeNames.Add(TypeName);
		return TypeId;
	}
};

template<typename T>
FORCEINLINE T* Cast(FAssetUserRenderData* Src)
{
	return Src && T::StaticType() == Src->GetType() ? (T*)Src : nullptr;
}

template<typename T>
FORCEINLINE T* CastChecked(FAssetUserRenderData* Src)
{
	T* RenderDataCastResult = Src && T::StaticType() == Src->GetType() ? (T*)Src : nullptr;
	check(RenderDataCastResult);
	return RenderDataCastResult;
}
