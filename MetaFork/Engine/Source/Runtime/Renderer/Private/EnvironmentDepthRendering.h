// Copyright Epic Games, Inc. All Rights Reserved.

/*=============================================================================
EnvironmentDepthRendering.h: environment depth rendering declarations.
=============================================================================*/

#pragma once

#include "CoreMinimal.h"
#include "ShaderParameters.h"
#include "UniformBuffer.h"
#include "Matrix3x4.h"
#include "SystemTextures.h"

class FShaderParameterMap;
class FSceneView;
struct FMobileBasePassTextures;
struct FSceneTextures;

BEGIN_GLOBAL_SHADER_PARAMETER_STRUCT(FEnvironmentDepthUniformParameters, )
	SHADER_PARAMETER_RDG_TEXTURE(Texture2DArray, EnvironmentDepthTexture)
	SHADER_PARAMETER_RDG_TEXTURE(Texture2DArray, EnvironmentDepthMinMaxTexture)
	SHADER_PARAMETER_SAMPLER(SamplerState, EnvironmentDepthSampler)
	SHADER_PARAMETER(FVector2f, DepthFactors)
	SHADER_PARAMETER_ARRAY(FMatrix44f, ScreenToDepthMatrices, [2])
	SHADER_PARAMETER_ARRAY(FMatrix44f, DepthViewProjMatrices, [2])
END_GLOBAL_SHADER_PARAMETER_STRUCT()

extern void SetupEnvironmentDepthUniformParameters(const class FSceneView& View, const FRDGSystemTextures& SystemTextures, const FMobileBasePassTextures& MobileBasePassTextures, FEnvironmentDepthUniformParameters& OutParameters);
extern void SetupEnvironmentDepthUniformParameters(const class FSceneView& View, const FRDGSystemTextures& SystemTextures, const FSceneTextures* SceneTextures, FEnvironmentDepthUniformParameters& OutParameters);
