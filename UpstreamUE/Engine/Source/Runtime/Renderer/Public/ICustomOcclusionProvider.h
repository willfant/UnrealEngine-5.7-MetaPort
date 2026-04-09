// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Features/IModularFeature.h"
#include "UObject/NameTypes.h"

class FViewInfo;

class ICustomOcclusionProvider : public IModularFeature
{
public:
	static FName GetModularFeatureName()
	{
		static FName FeatureName(TEXT("CustomOcclusionProvider"));
		return FeatureName;
	}

	virtual ~ICustomOcclusionProvider() = default;

	virtual void SetupCustomOcclusion(FViewInfo& View) = 0;
};