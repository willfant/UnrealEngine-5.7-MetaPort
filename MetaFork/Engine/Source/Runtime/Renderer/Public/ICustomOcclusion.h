// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "ScreenPass.h"

// Custom Occlusion (e.g. Software Occlusion)
class FScene;
class ICustomOcclusion
{
public:
	virtual ~ICustomOcclusion() = default;
	virtual int32 Process(const FScene* Scene, FViewInfo& View) = 0;
	virtual void DebugDraw(FRDGBuilder& GraphBuilder, const FViewInfo& View, FScreenPassRenderTarget Output, int32 InX, int32 InY) = 0;
};
