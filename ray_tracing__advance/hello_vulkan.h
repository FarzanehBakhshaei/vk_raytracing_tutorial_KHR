/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "nvvk/appbase_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "shaders/host_device.h"

// #VKRay
#include "nvvk/raytraceKHR_vk.hpp"
#include "offscreen.hpp"

#include "obj.hpp"
#include "raytrace.hpp"

#include <memory>

// Choosing the allocator to use
#define ALLOC_DMA
//#define ALLOC_DEDICATED
//#define ALLOC_VMA
#include <nvvk/resourceallocator_vk.hpp>

#if defined(ALLOC_DMA)
#include <nvvk/memallocator_dma_vk.hpp>
using Allocator = nvvk::ResourceAllocatorDma;
#elif defined(ALLOC_VMA)
#include <nvvk/memallocator_vma_vk.hpp>
using Allocator = nvvk::ResourceAllocatorVma;
#else
using Allocator = nvvk::ResourceAllocatorDedicated;
#endif

#define PT_IDXn(x, y, z) ((x) + (y) * (numCellsX + 1) + (z) * (numCellsY + 1) * (numCellsX + 1))

//--------------------------------------------------------------------------------------------------
// Simple rasterizer of OBJ objects
// - Each OBJ loaded are stored in an `ObjModel` and referenced by a `ObjInstance`
// - It is possible to have many `ObjInstance` referencing the same `ObjModel`
// - Rendering is done in an offscreen framebuffer
// - The image of the framebuffer is displayed in post-process in a full-screen quad
//
class HelloVulkan : public nvvk::AppBaseVk
{
public:
  void setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueFamily) override;
  void createDescriptorSetLayout();
  void createGraphicsPipeline();
  void loadModel(const std::string& filename, nvmath::mat4f transform = nvmath::mat4f(1));
  void updateDescriptorSet();
  void createUniformBuffer();
  void createObjDescriptionBuffer();
  void createAtrInfoBuffer();
  void createTextureImages(const VkCommandBuffer& cmdBuf, const std::vector<std::string>& textures);
  void updateUniformBuffer(const VkCommandBuffer& cmdBuf);
  void onResize(int /*w*/, int /*h*/) override;
  void destroyResources();
  void rasterize(const VkCommandBuffer& cmdBuff);
  void loadVolumetricData(const char* filePath, nvmath::mat4f transform = nvmath::mat4f(1)); // mat4f(1) is identity
  void createColormap();
  void updateColormap(const VkCommandBuffer& cmdBuff, std::vector<uint8_t> colormap);
  void HelloVulkan::fillRandomDirections();
 
  Offscreen& offscreen() { return m_offscreen; }
  Raytracer& raytracer() { return m_raytrace; }


  // Information pushed at each draw call
  PushConstantRaster m_pcRaster{
      {1},                    // Identity matrix
      {120.f, 70.f, 170.f},      // light position
      0,                      // instance Id
      {0.538f, 0.641f, 1.f},  //{0.07f, -0.33f, 0.56f},              // lightDirection;
      {cos(deg2rad(12.5f))},  // lightSpotCutoff;
      {cos(deg2rad(17.5f))},  // lightSpotOuterCutoff;
      1000.f,                  // light intensity
      2                       // light type
  };

  // Array of objects and instances in the scene
  std::vector<ObjModel>    m_objModel;   // Model on host
  std::vector<ObjDesc>     m_objDesc;    // Model description for device access
  std::vector<ObjInstance> m_instances;  // Scene model instances
  AtrInfo                  m_atrInfo;
  vec4                     m_center;    // center of geometry

  nvmath::mat4f calculateTransform(vec3 N, vec3 P, float scale);

  // Graphic pipeline
  VkPipelineLayout            m_pipelineLayout;
  VkPipeline                  m_graphicsPipeline;
  nvvk::DescriptorSetBindings m_descSetLayoutBind;
  VkDescriptorPool            m_descPool;
  VkDescriptorSetLayout       m_descSetLayout;
  VkDescriptorSet             m_descSet;

  int  m_maxFrames{500};
  void resetFrame();
  void updateFrame();

  nvvk::Buffer m_bGlobals;  // Device-Host of the camera matrices
  nvvk::Buffer m_bObjDesc;  // Device buffer of the OBJ descriptions
  nvvk::Buffer m_bAtrInfo;   // Device buffer of the attribute info

  std::vector<nvvk::Texture> m_textures;  // vector of all textures of the scene
  nvvk::Texture              m_colormapTexture;
  nvvk::Texture              m_atrInfoTexture;  // to save the model texture in loadVlumetricData(). I want to dont emplace_back the model to m_objModel, but I need the texture to use inupdateDescriptorSet()

  Allocator       m_alloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil m_debug;  // Utility to name objects

  // #Post
  Offscreen m_offscreen;
  void      initOffscreen();


  // #VKRay
  Raytracer m_raytrace;

  void initRayTracing();
  void raytrace(const VkCommandBuffer& cmdBuf, const nvmath::vec4f& clearColor);

  // Implicit
  ImplInst m_implObjects;

  void addImplSphere(nvmath::vec3f center, float radius, int matId);
  void addImplCube(nvmath::vec3f minumum, nvmath::vec3f maximum, int matId);
  void addImplMaterial(const MaterialObj& mat);
  void createImplictBuffers();

  VkSampler m_linearSampler     = VK_NULL_HANDLE;
  VkSampler m_MinMaxSampler     = VK_NULL_HANDLE;
  VkSampler m_ColormapSampler   = VK_NULL_HANDLE;
};
