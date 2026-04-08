// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Features/IModularFeature.h"
#include "UObject/NameTypes.h"
#include "ScreenPass.h"

// Custom Occlusion Provider
class ICustomOcclusionProvider : public IModularFeature
{
public:
	static FName GetModularFeatureName()
	{
		static FName FeatureName = FName(TEXT("CustomOcclusionProvider"));
		return FeatureName;
	}

	virtual void SetupCustomOcclusion(FViewInfo& View) = 0;
};
