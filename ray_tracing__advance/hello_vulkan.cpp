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


#include <sstream>


#define VMA_IMPLEMENTATION

#define STB_IMAGE_IMPLEMENTATION
#include "obj_loader.h"
#include "stb_image.h"


#include "hello_vulkan.h"
#include "nvh/cameramanipulator.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"

#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"

#include "nvvk/buffers_vk.hpp"
#include <SimVisData.hpp>

extern std::vector<std::string> defaultSearchPaths;

//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void HelloVulkan::setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueFamily)
{
  AppBaseVk::setup(instance, device, physicalDevice, queueFamily);
  m_alloc.init(instance, device, physicalDevice);
  m_debug.setup(m_device);


  m_offscreen.setup(device, physicalDevice, &m_alloc, queueFamily);
  m_raytrace.setup(device, physicalDevice, &m_alloc, queueFamily);
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void HelloVulkan::updateUniformBuffer(const VkCommandBuffer& cmdBuf)
{
  // Prepare new UBO contents on host.
  const float    aspectRatio = m_size.width / static_cast<float>(m_size.height);
  GlobalUniforms hostUBO     = {};
  const auto&    view        = CameraManip.getMatrix();
  const auto&    proj        = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);
  // proj[1][1] *= -1;  // Inverting Y for Vulkan (not needed with perspectiveVK).

  hostUBO.viewProj    = proj * view;
  hostUBO.viewInverse = nvmath::invert(view);
  hostUBO.projInverse = nvmath::invert(proj);

  // UBO on the device, and what stages access it.
  VkBuffer deviceUBO      = m_bGlobals.buffer;
  auto     uboUsageStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;

  // Ensure that the modified UBO is not visible to previous frames.
  VkBufferMemoryBarrier beforeBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  beforeBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  beforeBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  beforeBarrier.buffer        = deviceUBO;
  beforeBarrier.offset        = 0;
  beforeBarrier.size          = sizeof(hostUBO);
  vkCmdPipelineBarrier(cmdBuf, uboUsageStages, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &beforeBarrier, 0, nullptr);


  // Schedule the host-to-device upload. (hostUBO is copied into the cmd
  // buffer so it is okay to deallocate when the function returns).
  vkCmdUpdateBuffer(cmdBuf, m_bGlobals.buffer, 0, sizeof(GlobalUniforms), &hostUBO);
  
  vkCmdUpdateBuffer(cmdBuf, m_bAtrInfo.buffer, 0, sizeof(AtrInfo), &m_atrInfo);

  // Making sure the updated UBO will be visible.
  VkBufferMemoryBarrier afterBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  afterBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  afterBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  afterBarrier.buffer        = deviceUBO;
  afterBarrier.offset        = 0;
  afterBarrier.size          = sizeof(hostUBO);
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, uboUsageStages, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &afterBarrier, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Describing the layout pushed when rendering
//
void HelloVulkan::createDescriptorSetLayout()
{
  auto nbTxt = static_cast<uint32_t>(m_textures.size());

  // Camera matrices
  m_descSetLayoutBind.addBinding(SceneBindings::eGlobals, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  
  // Obj descriptions
  m_descSetLayoutBind.addBinding(SceneBindings::eObjDescs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
                                     | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR);
  // Textures
  m_descSetLayoutBind.addBinding(SceneBindings::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nbTxt,
                                 VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR);

  // Attributes
  m_descSetLayoutBind.addBinding(SceneBindings::eAtrTexture, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                                 VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR
                                     | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT);
  
  // AttributeSamplerLinear
  m_descSetLayoutBind.addBinding(SceneBindings::eAtrSamplerLinear, VK_DESCRIPTOR_TYPE_SAMPLER, 1,
                                 VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR
                                     | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT);
  
  // AttributeSamplerMinMax
  m_descSetLayoutBind.addBinding(SceneBindings::eAtrSamplerMinMax, VK_DESCRIPTOR_TYPE_SAMPLER, 1,
                                 VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR);

  // AttributesDimension
  m_descSetLayoutBind.addBinding(SceneBindings::eAtrInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                 VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR
                                | VK_SHADER_STAGE_COMPUTE_BIT);

  // Colormap Texture
  m_descSetLayoutBind.addBinding(SceneBindings::eColormapTexture, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                                 VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR
                                     | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT);

  // Storing implicit obj (binding = 3)
  m_descSetLayoutBind.addBinding(eImplicits, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                 VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR
                                     | VK_SHADER_STAGE_ANY_HIT_BIT_KHR);


  m_descSetLayout = m_descSetLayoutBind.createLayout(m_device);
  m_descPool      = m_descSetLayoutBind.createPool(m_device, 1);
  m_descSet       = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Setting up the buffers in the descriptor set
//
void HelloVulkan::updateDescriptorSet()
{
  std::vector<VkWriteDescriptorSet> writes;

  // Camera matrices and scene description
  VkDescriptorBufferInfo dbiUnif{m_bGlobals.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eGlobals, &dbiUnif));

  VkDescriptorBufferInfo dbiSceneDesc{m_bObjDesc.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eObjDescs, &dbiSceneDesc));

  // All texture samplers
  std::vector<VkDescriptorImageInfo> diit;
  for(auto& texture : m_textures)
  {
    diit.emplace_back(texture.descriptor);
  }
  writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, SceneBindings::eTextures, diit.data()));

  // Attribues
  VkDescriptorImageInfo dii{m_objModel[0].texture.descriptor};
  dii.sampler = VK_NULL_HANDLE;
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eAtrTexture, &dii));

  // AtrSampLinear
  VkDescriptorImageInfo diiS{m_linearSampler};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eAtrSamplerLinear, &diiS));


  // AtrSampMinMax
  VkDescriptorImageInfo diiSMM{m_MinMaxSampler};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eAtrSamplerMinMax, &diiSMM));


  // Attributes Info
  VkDescriptorBufferInfo adUni{m_bAtrInfo.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eAtrInfo, &adUni));

  // Colormap Texture
  VkDescriptorImageInfo diiC{m_colormapTexture.descriptor};
  diiC.sampler = VK_NULL_HANDLE;
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eColormapTexture, &diiC));


  VkDescriptorBufferInfo dbiImplDesc{m_implObjects.implBuf.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 3, &dbiImplDesc));

  // Writing the information
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Creating the pipeline layout
//
void HelloVulkan::createGraphicsPipeline()
{
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantRaster)};

  // Creating the Pipeline Layout
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = 1;
  createInfo.pSetLayouts            = &m_descSetLayout;
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_pipelineLayout);


  // Creating the Pipeline
  std::vector<std::string>                paths = defaultSearchPaths;
  nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, m_pipelineLayout, m_offscreen.renderPass());
  gpb.depthStencilState.depthTestEnable = true;
  gpb.addShader(nvh::loadFile("spv/vert_shader.vert.spv", true, paths, true), VK_SHADER_STAGE_VERTEX_BIT);
  gpb.addShader(nvh::loadFile("spv/frag_shader.frag.spv", true, paths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
  gpb.addBindingDescription({0, sizeof(VertexObj)});
  gpb.addAttributeDescriptions({
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, pos))},
      {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, nrm))},
      {2, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, color))},
      {3, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, texCoord))},
  });

  m_graphicsPipeline = gpb.createPipeline();
  m_debug.setObjectName(m_graphicsPipeline, "Graphics");
}

//--------------------------------------------------------------------------------------------------
// Loading the OBJ file and setting up all buffers
//
void HelloVulkan::loadModel(const std::string& filename, nvmath::mat4f transform)
{
  LOGI("Loading File:  %s \n", filename.c_str());
  ObjLoader loader;
  loader.loadModel(filename);

  // Converting from Srgb to linear
  for(auto& m : loader.m_materials)
  {
    m.ambient  = nvmath::pow(m.ambient, 2.2f);
    m.diffuse  = nvmath::pow(m.diffuse, 2.2f);
    m.specular = nvmath::pow(m.specular, 2.2f);
  }

  ObjModel model;


  model.nbIndices  = static_cast<uint32_t>(loader.m_indices.size());
  model.nbVertices = static_cast<uint32_t>(loader.m_vertices.size());

  // Create the buffers on Device and copy vertices, indices and materials
  nvvk::CommandPool  cmdBufGet(m_device, m_graphicsQueueIndex);
  VkCommandBuffer    cmdBuf          = cmdBufGet.createCommandBuffer();
  VkBufferUsageFlags flag            = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  VkBufferUsageFlags rayTracingFlags =  // used also for building acceleration structures
      flag | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  model.vertexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | rayTracingFlags);
  model.indexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | rayTracingFlags);
  model.matColorBuffer = m_alloc.createBuffer(cmdBuf, loader.m_materials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | flag);
  model.matIndexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_matIndx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | flag);
  // Creates all textures found and find the offset for this model
  auto txtOffset = static_cast<uint32_t>(m_textures.size());
  createTextureImages(cmdBuf, loader.m_textures);
  cmdBufGet.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();

  std::string objNb = std::to_string(m_objModel.size());
  m_debug.setObjectName(model.vertexBuffer.buffer, (std::string("vertex_" + objNb)));
  m_debug.setObjectName(model.indexBuffer.buffer, (std::string("index_" + objNb)));
  m_debug.setObjectName(model.matColorBuffer.buffer, (std::string("mat_" + objNb)));
  m_debug.setObjectName(model.matIndexBuffer.buffer, (std::string("matIdx_" + objNb)));

  // Keeping transformation matrix of the instance
  ObjInstance instance;
  instance.transform = transform;
  instance.objIndex  = static_cast<uint32_t>(m_objModel.size());
  m_instances.push_back(instance);

  // Creating information for device access
  ObjDesc desc;
  desc.txtOffset            = txtOffset;
  desc.vertexAddress        = nvvk::getBufferDeviceAddress(m_device, model.vertexBuffer.buffer);
  desc.indexAddress         = nvvk::getBufferDeviceAddress(m_device, model.indexBuffer.buffer);
  desc.materialAddress      = nvvk::getBufferDeviceAddress(m_device, model.matColorBuffer.buffer);
  desc.materialIndexAddress = nvvk::getBufferDeviceAddress(m_device, model.matIndexBuffer.buffer);

  // Keeping the obj host model and device description
  m_objModel.emplace_back(model);
  m_objDesc.emplace_back(desc);
}


// -------------------------------------------------------------------------------------------------
// Loading the DAT file and setting up all buffers
//

void HelloVulkan::loadVolumetricData(const char* filePath, nvmath::mat4f transform)
{
  //return;
  SimVisDataPtr simVisPtr = SimVisData::loadFromFile(filePath);
  //SimVisDataPtr simVisPtr = SimVisData::loadSphere(32);
  std::vector<MaterialObj> materials(128, MaterialObj());
  std::vector<std::string> textures;

  int numCellsX = simVisPtr->numCellsX;
  int numCellsY = simVisPtr->numCellsY;
  int numCellsZ = simVisPtr->numCellsZ;
  
  
  std::vector<int32_t> matIndx((numCellsZ + 1) * (numCellsY + 1) * (numCellsX + 1), 0);

  std::vector<VertexObj> vertices(simVisPtr->vertices.size());
  for(size_t i = 0; i < simVisPtr->vertices.size(); ++i)
  {
    VertexObj& v = vertices[i];
        
    v.pos.x = simVisPtr->vertices[i].x;
    v.pos.y = simVisPtr->vertices[i].y;
    v.pos.z = simVisPtr->vertices[i].z;    
    
    v.color = vec3(1, 0, 0);
    v.nrm   = vec3(0, 1, 0);
    v.texCoord = vec2(1, 1);

  }
  float max = -FLT_MAX, min = FLT_MAX;

  for(uint32_t z = 0; z < numCellsZ; z++)
  {
    for(uint32_t y = 0; y < numCellsY; y++)
    {
      for(uint32_t x = 0; x < numCellsX; x++)
      {
        VertexObj& v = vertices[PT_IDXn(x, y, z)];
        v.atr        = simVisPtr->attributesList[0][z * numCellsY * numCellsX + y * numCellsX + x];

        if(v.atr > max)
        {
          max = v.atr;
        }
        if(v.atr < min)
        {
          min = v.atr;
        }
      }
    }
  }

  //constexpr size_t maxCellIndices = 64 * 64 ;//  64 + 16 + 1;

  size_t s = simVisPtr->cellIndices.size();
  //std::min(simVisPtr->cellIndices.size(), maxCellIndices);
  std::vector<int> indices;
  for(int i = 0; i < s; i += 8)
  {
    indices.insert(indices.end(), {simVisPtr->cellIndices[i + 0], simVisPtr->cellIndices[i + 2], simVisPtr->cellIndices[i + 1]});  ///
    indices.insert(indices.end(), {simVisPtr->cellIndices[i + 0], simVisPtr->cellIndices[i + 3], simVisPtr->cellIndices[i + 2]});  ///
    indices.insert(indices.end(), {simVisPtr->cellIndices[i + 0], simVisPtr->cellIndices[i + 4], simVisPtr->cellIndices[i + 3]});  ///
    indices.insert(indices.end(), {simVisPtr->cellIndices[i + 3], simVisPtr->cellIndices[i + 4], simVisPtr->cellIndices[i + 7]});  ///
    indices.insert(indices.end(), {simVisPtr->cellIndices[i + 0], simVisPtr->cellIndices[i + 1], simVisPtr->cellIndices[i + 4]});  ///
    indices.insert(indices.end(), {simVisPtr->cellIndices[i + 1], simVisPtr->cellIndices[i + 5], simVisPtr->cellIndices[i + 4]});  ///
    indices.insert(indices.end(), {simVisPtr->cellIndices[i + 4], simVisPtr->cellIndices[i + 5], simVisPtr->cellIndices[i + 6]});  ///
    indices.insert(indices.end(), {simVisPtr->cellIndices[i + 4], simVisPtr->cellIndices[i + 6], simVisPtr->cellIndices[i + 7]});  ///
    indices.insert(indices.end(), {simVisPtr->cellIndices[i + 1], simVisPtr->cellIndices[i + 2], simVisPtr->cellIndices[i + 6]});  ///
    indices.insert(indices.end(), {simVisPtr->cellIndices[i + 1], simVisPtr->cellIndices[i + 6], simVisPtr->cellIndices[i + 5]});  ///
    indices.insert(indices.end(), {simVisPtr->cellIndices[i + 2], simVisPtr->cellIndices[i + 3], simVisPtr->cellIndices[i + 7]});  ///
    indices.insert(indices.end(), {simVisPtr->cellIndices[i + 2], simVisPtr->cellIndices[i + 7], simVisPtr->cellIndices[i + 6]});  ///
  }

  ObjModel model;
  model.nbIndices  = static_cast<uint32_t>(indices.size());
  model.nbVertices = static_cast<uint32_t>(simVisPtr->vertices.size());
  
  // Create the buffers on Device and copy vertices, indices and materials
  nvvk::CommandPool  cmdBufGet(m_device, m_graphicsQueueIndex);
  VkCommandBuffer    cmdBuf          = cmdBufGet.createCommandBuffer();
  VkBufferUsageFlags flag            = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  VkBufferUsageFlags rayTracingFlags =  // used also for building acceleration structures
      flag | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  model.vertexBuffer = m_alloc.createBuffer(cmdBuf, vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | rayTracingFlags);
  model.indexBuffer = m_alloc.createBuffer(cmdBuf, indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | rayTracingFlags);
  model.matColorBuffer = m_alloc.createBuffer(cmdBuf, materials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | flag);
  model.matIndexBuffer = m_alloc.createBuffer(cmdBuf, matIndx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | flag);
  
  // Creates all textures found and find the offset for this model
  auto txtOffset = 0;
  createTextureImages(cmdBuf, textures);
  
  VkFormat format               = VK_FORMAT_R32_SFLOAT;
  VkExtent3D imgSize            = VkExtent3D{(uint32_t)numCellsX, (uint32_t)numCellsY, (uint32_t)numCellsZ};
  VkDeviceSize bufferSize       = static_cast<uint64_t>(numCellsX * numCellsY * numCellsZ * sizeof(float));
  auto imageCreateInfo          = nvvk::makeImage3DCreateInfo(imgSize, format);
  VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerCreateInfo.minFilter   = VK_FILTER_LINEAR;
  samplerCreateInfo.magFilter   = VK_FILTER_LINEAR;
  samplerCreateInfo.mipmapMode  = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerCreateInfo.maxLod      = 1;
  samplerCreateInfo.anisotropyEnable = VK_TRUE;
  samplerCreateInfo.maxAnisotropy    = 16;

  // create linear sampler
  if(vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_linearSampler) != VK_SUCCESS)
  {
    throw std::runtime_error("cannot create linear sampler.\n");
  }

  // create min_max sampler
  VkSamplerReductionModeCreateInfo reductionCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO};
  reductionCreateInfo.reductionMode = VK_SAMPLER_REDUCTION_MODE_MAX;

  VkSamplerCreateInfo minMaxSampCreateInfo {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  minMaxSampCreateInfo.minFilter = VK_FILTER_LINEAR;
  minMaxSampCreateInfo.magFilter = VK_FILTER_LINEAR;
  minMaxSampCreateInfo.maxLod    = 1;
  minMaxSampCreateInfo.pNext     = &reductionCreateInfo;
  minMaxSampCreateInfo.anisotropyEnable = VK_TRUE;
  minMaxSampCreateInfo.maxAnisotropy    = 16;

  if(vkCreateSampler(m_device, &minMaxSampCreateInfo, nullptr, &m_MinMaxSampler) != VK_SUCCESS)
  {
    throw std::runtime_error("cannot create min_max sampler.\n");
  }

  nvvk::Image image             = m_alloc.createImage(cmdBuf, bufferSize, simVisPtr->attributesList[0].data(), imageCreateInfo);
  VkImageViewCreateInfo ivInfo  = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
  model.texture = m_alloc.createTexture(image, ivInfo);
  
  
  cmdBufGet.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();

  std::string objNb = std::to_string(m_objModel.size());
  m_debug.setObjectName(model.vertexBuffer.buffer, (std::string("vertex_" + objNb)));
  m_debug.setObjectName(model.indexBuffer.buffer, (std::string("index_" + objNb)));
  m_debug.setObjectName(model.matColorBuffer.buffer, (std::string("mat_" + objNb)));
  m_debug.setObjectName(model.matIndexBuffer.buffer, (std::string("matIdx_" + objNb)));
  m_debug.setObjectName(model.texture.image, (std::string("texture_" + objNb)));

  // Keeping transformation matrix of the instance
  ObjInstance instance;
  instance.transform = transform;
  instance.objIndex  = static_cast<uint32_t>(m_objModel.size());
  m_instances.push_back(instance);

  // Creating information for device access
  ObjDesc desc;
  desc.txtOffset            = txtOffset;
  desc.vertexAddress        = nvvk::getBufferDeviceAddress(m_device, model.vertexBuffer.buffer);
  desc.indexAddress         = nvvk::getBufferDeviceAddress(m_device, model.indexBuffer.buffer);
  desc.materialAddress      = nvvk::getBufferDeviceAddress(m_device, model.matColorBuffer.buffer);
  desc.materialIndexAddress = nvvk::getBufferDeviceAddress(m_device, model.matIndexBuffer.buffer);

  // Keeping the obj host model and device description
  m_objModel.emplace_back(model);
  m_objDesc.emplace_back(desc);

  m_atrInfo.dimension = vec4(numCellsX, numCellsY, numCellsZ, 1);
  m_atrInfo.minPoint  = vec4(0, 0, 0, 1);
  m_atrInfo.ISOValue  = 0.077f;
  m_atrInfo.minAtrValue = min;
  m_atrInfo.maxAtrValue = max;
}



//-------------------------------------------------------------------------------------------------

void HelloVulkan::createColormap()
{
  VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  imageInfo.imageType            = VK_IMAGE_TYPE_1D;
  imageInfo.extent.width         = static_cast<uint32_t>(720/4);
  imageInfo.extent.height        = static_cast<uint32_t>(1);
  imageInfo.extent.depth         = 1;
  imageInfo.mipLevels            = 1;
  imageInfo.arrayLayers          = 1;
  imageInfo.format               = VK_FORMAT_R8G8B8A8_UNORM;
  imageInfo.tiling               = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.samples              = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.initialLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.sharingMode          = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.usage                = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

  nvvk::Image           tex    = m_alloc.createImage(imageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VkImageViewCreateInfo ivInfo   = nvvk::makeImageViewCreateInfo(tex.image, imageInfo);
  m_colormapTexture            = m_alloc.createTexture(tex, ivInfo);
}

void HelloVulkan::updateColormap(const VkCommandBuffer& cmdBuff, std::vector<uint8_t> colormap) 
{    
  VkOffset3D               offset      = {0};
  VkImageSubresourceLayers subresource = {0};
  subresource.aspectMask               = VK_IMAGE_ASPECT_COLOR_BIT;
  subresource.layerCount               = 1;
  VkExtent3D extent                    = {static_cast<uint32_t>(720 / 4), static_cast<uint32_t>(1), 1};
  VkImageLayout imgLayout              = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Copy buffer to image
  VkImageSubresourceRange subresourceRange{};
  subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  subresourceRange.baseArrayLayer = 0;
  subresourceRange.baseMipLevel   = 0;
  subresourceRange.layerCount     = 1;
  subresourceRange.levelCount     = 1;


  static VkImageLayout currentImgLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  // doing these transitions per copy is not efficient, should do in bulk for many images
  nvvk::cmdBarrierImageLayout(cmdBuff, m_colormapTexture.image, currentImgLayout,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange);

  currentImgLayout                      = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

  m_alloc.getStaging()->cmdToImage(cmdBuff, m_colormapTexture.image, offset, extent, subresource,
                                   (VkDeviceSize)720,
                                   colormap.data());


  // doing these transitions per copy is not efficient, should do in bulk for many images
  nvvk::cmdBarrierImageLayout(cmdBuff, m_colormapTexture.image, currentImgLayout,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresourceRange);

  currentImgLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the camera matrices
// - Buffer is host visible
//
void HelloVulkan::createUniformBuffer()
{
  m_bGlobals = m_alloc.createBuffer(sizeof(GlobalUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_bGlobals.buffer, "Globals");
}

//--------------------------------------------------------------------------------------------------
// Create a storage buffer containing the description of the scene elements
// - Which geometry is used by which instance
// - Transformation
// - Offset for texture
//
void HelloVulkan::createObjDescriptionBuffer()
{
  nvvk::CommandPool cmdGen(m_device, m_graphicsQueueIndex);

  auto cmdBuf = cmdGen.createCommandBuffer();
  m_bObjDesc  = m_alloc.createBuffer(cmdBuf, m_objDesc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  cmdGen.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();
  m_debug.setObjectName(m_bObjDesc.buffer, "ObjDescs");
}

//--------------------------------------------------------------------------------------------------
// Creating the m_bAtrDim buffer holding the attributes dimension
// - 
//
void HelloVulkan::createAtrInfoBuffer()
{
  m_bAtrInfo = m_alloc.createBuffer(sizeof(AtrInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_bAtrInfo.buffer, "AtrInfo");
}

//--------------------------------------------------------------------------------------------------
// Creating all textures and samplers
//
void HelloVulkan::createTextureImages(const VkCommandBuffer& cmdBuf, const std::vector<std::string>& textures)
{
  VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerCreateInfo.minFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.magFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerCreateInfo.maxLod     = FLT_MAX;

  VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;

  // If no textures are present, create a dummy one to accommodate the pipeline layout
  if(textures.empty() && m_textures.empty())
  {
    nvvk::Texture texture;

    std::array<uint8_t, 4> color{255u, 255u, 255u, 255u};
    VkDeviceSize           bufferSize      = sizeof(color);
    auto                   imgSize         = VkExtent2D{1, 1};
    auto                   imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format);

    // Creating the dummy texture
    nvvk::Image           image  = m_alloc.createImage(cmdBuf, bufferSize, color.data(), imageCreateInfo);
    VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    texture                      = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

    // The image format must be in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    nvvk::cmdBarrierImageLayout(cmdBuf, texture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_textures.push_back(texture);
  }
  else
  {
    // Uploading all images
    for(const auto& texture : textures)
    {
      std::stringstream o;
      int               texWidth, texHeight, texChannels;
      o << "media/textures/" << texture;
      std::string txtFile = nvh::findFile(o.str(), defaultSearchPaths, true);

      stbi_uc* stbi_pixels = stbi_load(txtFile.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

      std::array<stbi_uc, 4> color{255u, 0u, 255u, 255u};

      stbi_uc* pixels = stbi_pixels;
      // Handle failure
      if(!stbi_pixels)
      {
        texWidth = texHeight = 1;
        texChannels          = 4;
        pixels               = reinterpret_cast<stbi_uc*>(color.data());
      }

      VkDeviceSize bufferSize      = static_cast<uint64_t>(texWidth) * texHeight * sizeof(uint8_t) * 4;
      auto         imgSize         = VkExtent2D{(uint32_t)texWidth, (uint32_t)texHeight};
      auto         imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);

      {
        nvvk::Image image = m_alloc.createImage(cmdBuf, bufferSize, pixels, imageCreateInfo);
        nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
        VkImageViewCreateInfo ivInfo  = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
        nvvk::Texture         texture = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

        m_textures.push_back(texture);
      }

      stbi_image_free(stbi_pixels);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void HelloVulkan::destroyResources()
{
  vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

  m_alloc.destroy(m_bGlobals);
  m_alloc.destroy(m_bObjDesc);
  m_alloc.destroy(m_bAtrInfo);
  m_alloc.destroy(m_implObjects.implBuf);
  m_alloc.destroy(m_implObjects.implMatBuf);

  for(auto& m : m_objModel)
  {
    m_alloc.destroy(m.vertexBuffer);
    m_alloc.destroy(m.indexBuffer);
    m_alloc.destroy(m.matColorBuffer);
    m_alloc.destroy(m.matIndexBuffer);
    m_alloc.destroy(m.texture);
  }

  for(auto& t : m_textures)
  {
    m_alloc.destroy(t);
  }

  //#Post
  m_offscreen.destroy();

  // #VKRay
  m_raytrace.destroy();

  m_alloc.deinit();
}

//--------------------------------------------------------------------------------------------------
// Drawing the scene in raster mode
//
void HelloVulkan::rasterize(const VkCommandBuffer& cmdBuf)
{
  VkDeviceSize offset{0};

  m_debug.beginLabel(cmdBuf, "Rasterize");

  // Dynamic Viewport
  setViewport(cmdBuf);

  // Drawing all triangles
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descSet, 0, nullptr);

  auto nbInst = static_cast<uint32_t>(m_instances.size() - 1);  // Remove the implicit object
  for(uint32_t i = 0; i < nbInst; ++i)
  {
    auto& inst             = m_instances[i];
    auto& model            = m_objModel[inst.objIndex];
    m_pcRaster.objIndex    = inst.objIndex;  // Telling which object is drawn
    m_pcRaster.modelMatrix = inst.transform;

    vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(PushConstantRaster), &m_pcRaster);
    vkCmdBindVertexBuffers(cmdBuf, 0, 1, &model.vertexBuffer.buffer, &offset);
    vkCmdBindIndexBuffer(cmdBuf, model.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmdBuf, model.nbIndices, 1, 0, 0, 0);
  }
  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void HelloVulkan::onResize(int /*w*/, int /*h*/)
{
  m_offscreen.createFramebuffer(m_size);
  m_offscreen.updateDescriptorSet();
  m_raytrace.updateRtDescriptorSet(m_offscreen.colorTexture().descriptor.imageView);
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Initialize offscreen rendering
//
void HelloVulkan::initOffscreen()
{
  m_offscreen.createFramebuffer(m_size);
  m_offscreen.createDescriptor();
  m_offscreen.createPipeline(m_renderPass);
  m_offscreen.updateDescriptorSet();
}

//--------------------------------------------------------------------------------------------------
// Initialize Vulkan ray tracing
//
void HelloVulkan::initRayTracing()
{
  m_raytrace.createBottomLevelAS(m_objModel, m_implObjects);
  m_raytrace.createTopLevelAS(m_instances, m_implObjects);
  m_raytrace.createRtDescriptorSet(m_offscreen.colorTexture().descriptor.imageView);
  m_raytrace.createRtPipeline(m_descSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Ray trace the scene
//
void HelloVulkan::raytrace(const VkCommandBuffer& cmdBuf, const nvmath::vec4f& clearColor)
{
  updateFrame();
  //if(m_pcRaster.frame >= m_maxFrames)
  //  return;

  m_raytrace.raytrace(cmdBuf, clearColor, m_descSet, m_size, m_pcRaster);
}

//--------------------------------------------------------------------------------------------------
// If the camera matrix has changed, resets the frame.
// otherwise, increments frame.
//
void HelloVulkan::updateFrame()
{
  static nvmath::mat4f refCamMatrix;
  static float         refFov{CameraManip.getFov()};

  const auto& m   = CameraManip.getMatrix();
  const auto  fov = CameraManip.getFov();

  //if(memcmp(&refCamMatrix.a00, &m.a00, sizeof(nvmath::mat4f)) != 0 || refFov != fov)
  {
    resetFrame();
    refCamMatrix = m;
    refFov       = fov;
  }
  m_pcRaster.frame++;
}

void HelloVulkan::resetFrame()
{
  m_pcRaster.frame = -1;
}


void HelloVulkan::addImplSphere(nvmath::vec3f center, float radius, int matId)
{
  ObjImplicit impl;
  impl.minimum = center - radius;
  impl.maximum = center + radius;
  impl.objType = EObjType::eSphere;
  impl.matId   = matId;
  m_implObjects.objImpl.push_back(impl);
}

void HelloVulkan::addImplCube(nvmath::vec3f minumum, nvmath::vec3f maximum, int matId)
{
  ObjImplicit impl;
  impl.minimum = minumum;
  impl.maximum = maximum;
  impl.objType = EObjType::eCube;
  impl.matId   = matId;
  m_implObjects.objImpl.push_back(impl);
}

void HelloVulkan::addImplMaterial(const MaterialObj& mat)
{
  m_implObjects.implMat.push_back(mat);
}

//--------------------------------------------------------------------------------------------------
// Create a storage buffer containing the description of the scene elements
// - Which geometry is used by which instance
// - Transformation
// - Offset for texture
//
void HelloVulkan::createImplictBuffers()
{
  using vkBU = VkBufferUsageFlagBits;
  nvvk::CommandPool cmdGen(m_device, m_graphicsQueueIndex);

  // Not allowing empty buffers
  if(m_implObjects.objImpl.empty())
    m_implObjects.objImpl.push_back({});
  if(m_implObjects.implMat.empty())
    m_implObjects.implMat.push_back({});

  auto cmdBuf              = cmdGen.createCommandBuffer();
  m_implObjects.implBuf    = m_alloc.createBuffer(cmdBuf, m_implObjects.objImpl,
                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR
                                                   | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                   | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_implObjects.implMatBuf = m_alloc.createBuffer(cmdBuf, m_implObjects.implMat,
                                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  cmdGen.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();
  m_debug.setObjectName(m_implObjects.implBuf.buffer, "implicitObj");
  m_debug.setObjectName(m_implObjects.implMatBuf.buffer, "implicitMat");


  // Adding an extra instance to get access to the material buffers
  ObjDesc objDesc{};
  objDesc.materialAddress = nvvk::getBufferDeviceAddress(m_device, m_implObjects.implMatBuf.buffer);
  m_objDesc.emplace_back(objDesc);

  ObjInstance instance{};
  instance.objIndex = static_cast<uint32_t>(m_objModel.size());
  m_instances.emplace_back(instance);
}
