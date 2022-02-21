/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "raycommon.glsl"
#include "wavefront.glsl"

hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 0) rayPayloadInEXT hitPayload prd;
layout(location = 1) rayPayloadEXT bool isShadowed;

layout(buffer_reference, scalar) buffer Vertices {Vertex v[]; }; // Positions of an object
layout(buffer_reference, scalar) buffer Indices {ivec3 i[]; }; // Triangle indices
layout(buffer_reference, scalar) buffer Materials {WaveFrontMaterial m[]; }; // Array of all materials on an object
layout(buffer_reference, scalar) buffer MatIndices {int i[]; }; // Material ID for each triangle
layout(set = 0, binding = eTlas) uniform accelerationStructureEXT topLevelAS;
layout(set = 1, binding = eObjDescs, scalar) buffer ObjDesc_ { ObjDesc i[]; } objDesc;
//layout(set = 1, binding = eTextures) uniform sampler2D textureSamplers[];
layout(set = 1, binding = eAtrInfo) uniform _AtrInfoUniforms { AtrInfo ai; };

layout(set = 1, binding = eAtrSamplerLinear) uniform sampler AtrSampLin;
layout(set = 1, binding = eAtrTexture) uniform texture3D atrTexture;

layout(push_constant) uniform _PushConstantRay { PushConstantRay pcRay; };
// clang-format on


layout(location = 3) callableDataEXT rayLight cLight;


void main()
{
  // Object data
  /*
  MatIndices matIndices  = MatIndices(objResource.materialIndexAddress);
  Materials  materials   = Materials(objResource.materialAddress);

  */
  ObjDesc    objResource = objDesc.i[gl_InstanceCustomIndexEXT];
  Vertices   vertices    = Vertices(objResource.vertexAddress);
  Indices    indices     = Indices(objResource.indexAddress);
  // Indices of the triangle
  ivec3 ind = indices.i[gl_PrimitiveID];

  // Vertex of the triangle
  Vertex v0 = vertices.v[ind.x];
  Vertex v1 = vertices.v[ind.y];
  Vertex v2 = vertices.v[ind.z];
  
  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
  vec3 pos = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;
  
  //********************refine hit with ray marching***********************
    vec3 debug_color = vec3(0);
#if 1

  vec3 rayDirInv = normalize(-gl_WorldRayDirectionEXT);
  float stepSize =  max(1.0f/ai.dimension.x, max(1.0f/ai.dimension.y, 1.0f/ai.dimension.z));
  vec3 lastPos = pos + rayDirInv * stepSize;

  for (int i=0; i<10; i++) {
    vec3 midPos = (pos + lastPos)/2.f;
    vec3 normalizedCoord = (midPos - ai.minPoint.xyz) / ai.dimension.xyz;
    float attri = texture(sampler3D(atrTexture, AtrSampLin), normalizedCoord).r;
    if (attri >= ai.ISOValue) {
      pos = midPos;
      debug_color += vec3(0, 0.1, 0);
      
    } else {
      lastPos = midPos;
      debug_color += vec3(0.1, 0, 0);
    }
  }

#else
    debug_color = vec3(1);
#endif
  //*****************************************************

  vec3 normalizedCoord = (pos - ai.minPoint.xyz) / ai.dimension.xyz;

  float f_XPlusDeltaX    = texture(sampler3D(atrTexture, AtrSampLin), clamp (normalizedCoord + vec3(1.0f/ai.dimension.x, 0.0f, 0.0f), 0.0f, 1.0f)).r;
  float f_XMinusDeltaX   = texture(sampler3D(atrTexture, AtrSampLin), clamp (normalizedCoord - vec3(1.0f/ai.dimension.x, 0.0f, 0.0f), 0.0f, 1.0f)).r;
  float f_YPlusDeltaY    = texture(sampler3D(atrTexture, AtrSampLin), clamp (normalizedCoord + vec3(0.0f, 1.0f/ai.dimension.y, 0.0f), 0.0f, 1.0f)).r;
  float f_YMinusDeltaY   = texture(sampler3D(atrTexture, AtrSampLin), clamp (normalizedCoord - vec3(0.0f, 1.0f/ai.dimension.y, 0.0f), 0.0f, 1.0f)).r;
  float f_ZPlusDeltaZ    = texture(sampler3D(atrTexture, AtrSampLin), clamp (normalizedCoord + vec3(0.0f, 0.0f, 1.0f/ai.dimension.z), 0.0f, 1.0f)).r;
  float f_ZMinusDeltaZ   = texture(sampler3D(atrTexture, AtrSampLin), clamp (normalizedCoord - vec3(0.0f, 0.0f, 1.0f/ai.dimension.z), 0.0f, 1.0f)).r;
  
  vec3 n = vec3 (f_XPlusDeltaX - f_XMinusDeltaX, f_YPlusDeltaY - f_YMinusDeltaY, f_ZPlusDeltaZ - f_ZMinusDeltaZ) * ai.dimension.xyz * 0.5f;
  vec3 normal;
  if(dot(n, n) < 0.001f)    // |n^2| < 0.001f
   normal = vec3(0.0f);
  else
   normal = normalize(n);

  //prd.hitValue = (normal + 1) / 2.0f;



  /*
  // Computing the normal at hit position
  vec3 normal = v0.nrm * barycentrics.x + v1.nrm * barycentrics.y + v2.nrm * barycentrics.z;
  // Transforming the normal to world space
  normal = normalize(vec3(normal * gl_WorldToObjectEXT));
  
  float attr = v0.atr * barycentrics.x + v1.atr * barycentrics.y + v2.atr * barycentrics.z;
  */

  // Computing the coordinates of the hit position
//  vec3 worldPos = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;
  // Transforming the position to world space
  //worldPos = vec3(gl_ObjectToWorldEXT * vec4(worldPos, 1.0));

  cLight.inHitPosition = pos;
#define DONT_USE_CALLABLE
#if defined(DONT_USE_CALLABLE)
  // Point light
  if(pcRay.lightType == 0)
  {
    vec3  lDir              = pcRay.lightPosition - cLight.inHitPosition;
    float lightDistance     = length(lDir);
    cLight.outIntensity     = pcRay.lightIntensity / (lightDistance * lightDistance);
    cLight.outLightDir      = normalize(lDir);
    cLight.outLightDistance = lightDistance;
  }
  else if(pcRay.lightType == 1) // spot light
  {
    vec3 lDir               = pcRay.lightPosition - cLight.inHitPosition;
    cLight.outLightDistance = length(lDir);
    cLight.outIntensity     = pcRay.lightIntensity / (cLight.outLightDistance * cLight.outLightDistance);
    cLight.outLightDir      = normalize(lDir);
    float theta             = dot(cLight.outLightDir, normalize(-pcRay.lightDirection));
    float epsilon           = pcRay.lightSpotCutoff - pcRay.lightSpotOuterCutoff;
    float spotIntensity     = clamp((theta - pcRay.lightSpotOuterCutoff) / epsilon, 0.0, 1.0);
    cLight.outIntensity *= spotIntensity;
  }
  else  // Directional light
  {
    cLight.outLightDir      = normalize(-pcRay.lightDirection);
    cLight.outIntensity     = 1.0;
    cLight.outLightDistance = 10000000;
  }
#else
  executeCallableEXT(pcRay.lightType, 3);
#endif

  // Material of the object
  //int               matIdx = matIndices.i[gl_PrimitiveID];
  //WaveFrontMaterial mat    = materials.m[matIdx];


  // Diffuse
  vec3 diffuse = computeDiffuse(cLight.outLightDir, normal);
//  if(mat.textureId >= 0)
//  {
//    uint txtId    = mat.textureId + objDesc.i[gl_InstanceCustomIndexEXT].txtOffset;
//    vec2 texCoord = v0.texCoord * barycentrics.x + v1.texCoord * barycentrics.y + v2.texCoord * barycentrics.z;
//    diffuse *= texture(textureSamplers[nonuniformEXT(txtId)], texCoord).xyz;
//  }

  vec3  specular    = vec3(0);
  float attenuation = 1;

  // Tracing shadow ray only if the light is visible from the surface
  if(dot(normal, cLight.outLightDir) > 0)
  {
    float tMin   = 0.001;
    float tMax   = cLight.outLightDistance;
    vec3  origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3  rayDir = cLight.outLightDir;
    uint  flags  = gl_RayFlagsSkipClosestHitShaderEXT;
    isShadowed   = true;
    traceRayEXT(topLevelAS,  // acceleration structure
                flags,       // rayFlags
                0xFF,        // cullMask
                0,           // sbtRecordOffset
                0,           // sbtRecordStride
                1,           // missIndex
                origin,      // ray origin
                tMin,        // ray min range
                rayDir,      // ray direction
                tMax,        // ray max range
                1            // payload (location = 1)
    );

    if(isShadowed)
    {
      attenuation = 0.3;
    }
    else
    {
      // Specular
      specular = computeSpecular(gl_WorldRayDirectionEXT, cLight.outLightDir, normal);
    }
  }

  // Reflection
  /*if(mat.illum == 3)
  {
    vec3 origin = worldPos;
    vec3 rayDir = reflect(gl_WorldRayDirectionEXT, normal);
    prd.attenuation *= mat.specular;
    prd.done      = 0;
    prd.rayOrigin = origin;
    prd.rayDir    = rayDir;
  }*/

  //prd.hitValue = barycentrics;


  prd.hitValue = debug_color * vec3(cLight.outIntensity * attenuation * (diffuse + specular));
}
