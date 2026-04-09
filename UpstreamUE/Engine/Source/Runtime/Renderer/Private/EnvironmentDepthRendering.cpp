// Copyright Epic Games, Inc. All Rights Reserved.

#include "EnvironmentDepthRendering.h"

#include "MobileBasePassRendering.h"
#include "SceneTextures.h"

void SetupEnvironmentDepthUniformParameters(
	const FSceneView& View,
	const FRDGSystemTextures& SystemTextures,
	const FMobileBasePassTextures& MobileBasePassTextures,
	FEnvironmentDepthUniformParameters& OutParameters)
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

	OutParameters.EnvironmentDepthSampler = TStaticSamplerState<>::GetRHI();
	OutParameters.DepthFactors = MobileBasePassTextures.DepthFactors;

	for (int32 i = 0; i < 2; ++i)
	{
		OutParameters.ScreenToDepthMatrices[i] = MobileBasePassTextures.ScreenToDepthMatrices[i];
		OutParameters.DepthViewProjMatrices[i] = MobileBasePassTextures.DepthViewProjMatrices[i];
	}
}

void SetupEnvironmentDepthUniformParameters(
	const FSceneView& View,
	const FRDGSystemTextures& SystemTextures,
	const FSceneTextures* SceneTextures,
	FEnvironmentDepthUniformParameters& OutParameters)
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

	OutParameters.EnvironmentDepthSampler = TStaticSamplerState<>::GetRHI();

	if (SceneTextures != nullptr)
	{
		OutParameters.DepthFactors = SceneTextures->DepthFactors;

		for (int32 i = 0; i < 2; ++i)
		{
			OutParameters.ScreenToDepthMatrices[i] = SceneTextures->ScreenToDepthMatrices[i];
			OutParameters.DepthViewProjMatrices[i] = SceneTextures->DepthViewProjMatrices[i];
		}
	}
	else
	{
		OutParameters.DepthFactors = FVector2f::ZeroVector;

		for (int32 i = 0; i < 2; ++i)
		{
			OutParameters.ScreenToDepthMatrices[i] = FMatrix44f::Identity;
			OutParameters.DepthViewProjMatrices[i] = FMatrix44f::Identity;
		}
	}
}

IMPLEMENT_GLOBAL_SHADER_PARAMETER_STRUCT(FEnvironmentDepthUniformParameters, "EnvironmentDepthStruct");