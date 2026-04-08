// Copyright Epic Games, Inc. All Rights Reserved.

/*=============================================================================
 EnvironmentDepthRendering.cpp
=============================================================================*/

#include "EnvironmentDepthRendering.h"
#include "MobileBasePassRendering.h"
#include "SceneTextures.h"

void SetupEnvironmentDepthUniformParameters(const class FSceneView& View, const FRDGSystemTextures& SystemTextures, const FMobileBasePassTextures& MobileBasePassTextures, FEnvironmentDepthUniformParameters& OutParameters)
{
	if (MobileBasePassTextures.EnvironmentDepthTexture != nullptr)
	{
		OutParameters.EnvironmentDepthTexture = MobileBasePassTextures.EnvironmentDepthTexture;
		OutParameters.EnvironmentDepthMinMaxTexture = MobileBasePassTextures.EnvironmentDepthMinMaxTexture;
	}
	else
	{
		OutParameters.EnvironmentDepthTexture = SystemTextures.White;
		OutParameters.EnvironmentDepthMinMaxTexture = SystemTextures.Black;
	}
	OutParameters.EnvironmentDepthSampler = TStaticSamplerState<SF_Bilinear, AM_Clamp, AM_Clamp, AM_Clamp, 0, 0, 0, SCF_Less>::GetRHI();
	OutParameters.DepthFactors = MobileBasePassTextures.DepthFactors;
	for (int i = 0; i < 2; ++i)
	{
		OutParameters.ScreenToDepthMatrices[i] = MobileBasePassTextures.ScreenToDepthMatrices[i];
		OutParameters.DepthViewProjMatrices[i] = MobileBasePassTextures.DepthViewProjMatrices[i];
	}
}

void SetupEnvironmentDepthUniformParameters(const class FSceneView& View, const FRDGSystemTextures& SystemTextures, const FSceneTextures* SceneTextures, FEnvironmentDepthUniformParameters& OutParameters)
{
	if (SceneTextures != nullptr && SceneTextures->EnvironmentDepthTexture != nullptr)
	{
		OutParameters.EnvironmentDepthTexture = SceneTextures->EnvironmentDepthTexture;
		OutParameters.EnvironmentDepthMinMaxTexture = SceneTextures->EnvironmentDepthMinMaxTexture;
	}
	else
	{
		OutParameters.EnvironmentDepthTexture = SystemTextures.White;
		OutParameters.EnvironmentDepthMinMaxTexture = SystemTextures.Black;
	}
	OutParameters.EnvironmentDepthSampler = TStaticSamplerState<SF_Bilinear, AM_Clamp, AM_Clamp, AM_Clamp, 0, 0, 0, SCF_Less>::GetRHI();
	if (SceneTextures != nullptr)
	{
		OutParameters.DepthFactors = SceneTextures->DepthFactors;
		for (int i = 0; i < 2; ++i)
		{
			OutParameters.ScreenToDepthMatrices[i] = SceneTextures->ScreenToDepthMatrices[i];
			OutParameters.DepthViewProjMatrices[i] = SceneTextures->DepthViewProjMatrices[i];
		}
	}
}

IMPLEMENT_GLOBAL_SHADER_PARAMETER_STRUCT(FEnvironmentDepthUniformParameters, "EnvironmentDepthStruct");
