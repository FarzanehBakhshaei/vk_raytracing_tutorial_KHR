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


// ImGui - standalone example application for Glfw + Vulkan, using programmable
// pipeline If you are new to ImGui, see examples/README.txt and documentation
// at the top of imgui.cpp.

#include <array>

#include "backends/imgui_impl_glfw.h"
#include "imgui.h"

#include "hello_vulkan.h"
#include "imgui/imgui_camera_widget.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvpsystem.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/context_vk.hpp"
#include "transfer_function_widget.h"
#include "imGuIZMOquat.h"

#include <random>

namespace tfnw {
uint8_t rainbow2[] = {0x89, 0x50, 0x4e, 0x47, 0xd,  0xa,  0x1a, 0xa,  0x0,  0x0,  0x0,  0xd,  0x49, 0x48, 0x44, 0x52,
                     0x0,  0x0,  0x0,  0xb4, 0x0,  0x0,  0x0,  0x1,  0x8,  0x6,  0x0,  0x0,  0x0,  0x77, 0xca, 0x84,
                     0xf4, 0x0,  0x0,  0x0,  0x46, 0x49, 0x44, 0x41, 0x54, 0x78, 0x5e, 0x63, 0x64, 0x60, 0xf8, 0xff,
                     0x9f, 0x81, 0x85, 0x81, 0x1,  0x8c, 0x39, 0x70, 0xd0, 0x20, 0xf9, 0xc1, 0x24, 0xc7, 0xfe, 0x1f,
                     0xe8, 0xd0, 0x3f, 0x40, 0xfc, 0x3,  0x8d, 0xc6, 0x26, 0x6,  0x53, 0x33, 0xb0, 0x72, 0x4c, 0xc,
                     0xff, 0x18, 0xd0, 0x83, 0x70, 0xb0, 0x5,  0x2b, 0xba, 0xfb, 0x18, 0x7f, 0x42, 0x83, 0x77, 0xf0,
                     0x6,  0x2b, 0x3,  0x3,  0xb2, 0xdb, 0xfe, 0x30, 0x30, 0x0,  0x0,  0x85, 0x9e, 0x5d, 0x0,  0x63,
                     0x5b, 0x7,  0x6f, 0x0,  0x0,  0x0,  0x0,  0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};
}

//////////////////////////////////////////////////////////////////////////
#define UNUSED(x) (void)(x)
//////////////////////////////////////////////////////////////////////////

// Default search path for shaders
std::vector<std::string> defaultSearchPaths;


// GLFW Callback functions
static void onErrorCallback(int error, const char* description)
{
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Extra UI
void renderUI(HelloVulkan& helloVk)
{
  bool changed = false;

  changed |= ImGuiH::CameraWidget();
  if(ImGui::CollapsingHeader("Light"))
  {
    auto& pc = helloVk.m_pcRaster;

    changed |= ImGui::RadioButton("Point", &pc.lightType, 0);
    ImGui::SameLine();
    changed |= ImGui::RadioButton("Spot", &pc.lightType, 1);
    ImGui::SameLine();
    changed |= ImGui::RadioButton("Infinite", &pc.lightType, 2);
    

    if(pc.lightType < 2)
    {
      changed |= ImGui::SliderFloat3("Light Position", &pc.lightPosition.x, -1000.f, 1000.f);
    }
    if(pc.lightType > 0)
    {
      changed |= ImGui::SliderFloat3("Light Direction", &pc.lightDirection.x, -1.f, 1.f);
    }
    if(pc.lightType < 2)
    {
      changed |= ImGui::SliderFloat("Light Intensity", &pc.lightIntensity, 0.f, 500.f);
    }

    if(pc.lightType == 1)
    {
      float dCutoff    = rad2deg(acos(pc.lightSpotCutoff));
      float dOutCutoff = rad2deg(acos(pc.lightSpotOuterCutoff));
      changed |= ImGui::SliderFloat("Cutoff", &dCutoff, 0.f, 45.f);
      changed |= ImGui::SliderFloat("OutCutoff", &dOutCutoff, 0.f, 45.f);
      dCutoff = dCutoff > dOutCutoff ? dOutCutoff : dCutoff;

      pc.lightSpotCutoff      = cos(deg2rad(dCutoff));
      pc.lightSpotOuterCutoff = cos(deg2rad(dOutCutoff));
    }
  }

  //if(ImGui::CollapsingHeader("ISO Surface rendering"))
  //{
    //ISOValue
    changed |= ImGui::SliderFloat("ISOValue", &helloVk.m_atrInfo.ISOValue, 0.0f, 2.0f);

    static bool enableRefinement = true;
    changed |= ImGui::Checkbox("Enable Refinement", &enableRefinement);
    helloVk.m_atrInfo.enableRefinement = enableRefinement ? 1 : 0;
  //}      
  static float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  changed |= ImGui::ColorEdit3("AO color", reinterpret_cast<float*>(&color));
  helloVk.m_atrInfo.ambientColor = {color[0], color[1], color[2], color[3]};

  changed |= ImGui::SliderFloat("Step Size Scale", &helloVk.m_atrInfo.stepSize, 0.1f, 2.0f);

  static bool useAmbinetOcclusion = false;
  changed |= ImGui::Checkbox("Ambient Occlusion", &useAmbinetOcclusion);
  helloVk.m_atrInfo.useAmbinetOcclusion = useAmbinetOcclusion ? 1 : 0;
  //changed |=  ImGui::ColorEdit3("color", color);
   
  vgm::Vec3 dir(helloVk.m_atrInfo.planeNormal.x, helloVk.m_atrInfo.planeNormal.y, helloVk.m_atrInfo.planeNormal.z);
  changed |= ImGui::gizmo3D("Plane Normal", dir, 100, imguiGizmo::modeDirPlane);
  helloVk.m_atrInfo.planeNormal.x = dir.x;
  helloVk.m_atrInfo.planeNormal.y = dir.y;
  helloVk.m_atrInfo.planeNormal.z = dir.z;
  helloVk.m_atrInfo.planeNormal.w = 0.f;
  
  //changed |= ImGui::SliderFloat3("Plane Normal", (float*)&helloVk.m_atrInfo.planeNormal, 0.0f, 360.0f);

  vgm::Vec3 pos(helloVk.m_atrInfo.planePosition.x, helloVk.m_atrInfo.planePosition.y, helloVk.m_atrInfo.planePosition.z);
  changed |= ImGui::SliderFloat3("Plane Position", (float*)&pos, -helloVk.m_center.x * 3.f, helloVk.m_center.x * 3.f);
  helloVk.m_atrInfo.planePosition.x = pos.x;
  helloVk.m_atrInfo.planePosition.y = pos.y;
  helloVk.m_atrInfo.planePosition.z = pos.z;
  helloVk.m_atrInfo.planePosition.w = 1.f;
    
  static bool hideClipPlane = false;
  changed |= ImGui::Checkbox("Hide ClipPlane", &hideClipPlane);
  helloVk.m_atrInfo.hideClipPlane = hideClipPlane ? 1 : 0;

  ImGui::SameLine();
  //ImGUi::
  static bool useHeadLight = true;
  changed |= ImGui::Checkbox("Head Light", &useHeadLight);
  helloVk.m_atrInfo.useHeadLight = useHeadLight ? 1 : 0;

  static bool shadowRay = false;
  changed |= ImGui::Checkbox("Shadow ray", &shadowRay);
  helloVk.m_atrInfo.shadowRay = shadowRay ? 1 : 0;

  changed |= ImGui::SliderInt("Max Frames", &helloVk.m_maxFrames, 1, 1000);
  if(changed)
    helloVk.resetFrame();
  }


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
static int const SAMPLE_WIDTH  = 1280;
static int const SAMPLE_HEIGHT = 720;


//--------------------------------------------------------------------------------------------------
// Application Entry
//
int main(int argc, char** argv)
{
  UNUSED(argc);

  // Setup GLFW window
  glfwSetErrorCallback(onErrorCallback);
  if(!glfwInit())
  {
    return 1;
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window = glfwCreateWindow(SAMPLE_WIDTH, SAMPLE_HEIGHT, PROJECT_NAME, nullptr, nullptr);

  // Setup camera
  CameraManip.setWindowSize(SAMPLE_WIDTH, SAMPLE_HEIGHT);
  CameraManip.setLookat({164, -347, 91}, {166, 78, 51}, {0.000, 1.000, 0.000});

  // Setup Vulkan
  if(!glfwVulkanSupported())
  {
    printf("GLFW: Vulkan Not Supported\n");
    return 1;
  }

  // setup some basic things for the sample, logging file for example
  NVPSystem system(PROJECT_NAME);

  // Search path for shaders and other media
  defaultSearchPaths = {
      NVPSystem::exePath() + PROJECT_RELDIRECTORY,
      NVPSystem::exePath() + PROJECT_RELDIRECTORY "..",
      std::string(PROJECT_NAME),
  };

  // Vulkan required extensions
  assert(glfwVulkanSupported() == 1);
  uint32_t count{0};
  auto     reqExtensions = glfwGetRequiredInstanceExtensions(&count);

  // Requesting Vulkan extensions and layers
  nvvk::ContextCreateInfo contextInfo;
  contextInfo.setVersion(1, 2);                       // Using Vulkan 1.2
  for(uint32_t ext_id = 0; ext_id < count; ext_id++)  // Adding required extensions (surface, win32, linux, ..)
    contextInfo.addInstanceExtension(reqExtensions[ext_id]);
  contextInfo.addInstanceLayer("VK_LAYER_LUNARG_monitor", true);              // FPS in titlebar
  contextInfo.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, true);  // Allow debug names
  contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);            // Enabling ability to present rendering

  // #VKRay: Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  contextInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accelFeature);  // To build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  contextInfo.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rtPipelineFeature);  // To use vkCmdTraceRaysKHR
  contextInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline
  contextInfo.addDeviceExtension(VK_EXT_SAMPLER_FILTER_MINMAX_EXTENSION_NAME);     // Required by min max sampler

  // Creating Vulkan base application
  nvvk::Context vkctx{};
  vkctx.initInstance(contextInfo);
  // Find all compatible devices
  auto compatibleDevices = vkctx.getCompatibleDevices(contextInfo);
  assert(!compatibleDevices.empty());
  // Use a compatible device
  vkctx.initDevice(compatibleDevices[0], contextInfo);

  // Create example
  HelloVulkan helloVk;

  // Window need to be opened to get the surface on which to draw
  const VkSurfaceKHR surface = helloVk.getVkSurface(vkctx.m_instance, window);
  vkctx.setGCTQueueWithPresent(surface);

  helloVk.setup(vkctx.m_instance, vkctx.m_device, vkctx.m_physicalDevice, vkctx.m_queueGCT.familyIndex);
  helloVk.createSwapchain(surface, SAMPLE_WIDTH, SAMPLE_HEIGHT);
  helloVk.createDepthBuffer();
  helloVk.createRenderPass();
  helloVk.createFrameBuffers();

  // Setup Imgui
  helloVk.initGUI(0);  // Using sub-pass 0
  /*helloVk.m_atrInfo.planePosition.x = 0.;
  helloVk.m_atrInfo.planePosition.y = 50.;
  helloVk.m_atrInfo.planePosition.z = 100;*/

  
  bool kitten = false;
  // Creation of the example
  //helloVk.loadVolumetricData(nvh::findFile("media/scenes/testData1_snapshot.dat", defaultSearchPaths, true).c_str());
  helloVk.loadVolumetricData(nvh::findFile("media/scenes/kitten_132_115_200_its500.dat", defaultSearchPaths, true).c_str());kitten = true;
  //helloVk.loadVolumetricData(nvh::findFile("media/scenes/testData.dat", defaultSearchPaths, true).c_str());
  //helloVk.loadVolumetricData(nvh::findFile("media/scenes/femur3D_144_96_184_A_its400.dat", defaultSearchPaths, true).c_str());
  //helloVk.loadVolumetricData(nvh::findFile("media/scenes/cantilever3D_256_128_128_iLoad3_R20_its500.dat", defaultSearchPaths, true).c_str());

  vec3 camLookPoint = helloVk.m_atrInfo.minPoint + helloVk.m_atrInfo.dimension / 2.0f; 

  CameraManip.setLookat({164, -347, 91}, camLookPoint, {0.000, 0.000, 1.000});

 /* nvmath::mat4f transform = helloVk.calculateTransform(
                                               vec3(helloVk.m_atrInfo.planeNormal.x, 
                                                    helloVk.m_atrInfo.planeNormal.y, 
                                                    helloVk.m_atrInfo.planeNormal.z),
                                               vec3(helloVk.m_center.x, 
                                                    helloVk.m_center.y, 
                                                    helloVk.m_center.z));*/
  helloVk.loadModel(nvh::findFile("media/scenes/clipPlane2.obj", defaultSearchPaths, true));//, transform);
  //helloVk.loadModel(nvh::findFile("media/scenes/clipPlane.obj", defaultSearchPaths, true));  //, transform);
  helloVk.loadModel(nvh::findFile("media/scenes/normal.obj", defaultSearchPaths, true));
  //helloVk.loadModel(nvh::findFile("media/scenes/wuson.obj", defaultSearchPaths, true),
                    //nvmath::scale_mat4(nvmath::vec3f(0.5f)) * nvmath::translation_mat4(nvmath::vec3f(0.0f, 0.0f, 6.0f)));

  nvvk::AccelKHR tlas;

  std::random_device              rd;         // Will be used to obtain a seed for the random number engine
  std::mt19937                    gen(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::normal_distribution<float> dis(2.0f, 2.0f);
  std::normal_distribution<float> disn(0.5f, 0.2f);
  auto                            wusonIndex = static_cast<int>(helloVk.m_objModel.size() - 1);

  /* for(int n = 0; n < 50; ++n)
  {
    ObjInstance inst;
    inst.objIndex       = wusonIndex;
    float         scale = fabsf(disn(gen));
    nvmath::mat4f mat   = nvmath::translation_mat4(nvmath::vec3f{dis(gen), 0.f, dis(gen) + 6});
    //    mat              = mat * nvmath::rotation_mat4_x(dis(gen));
    mat            = mat * nvmath::scale_mat4(nvmath::vec3f(scale));
    inst.transform = mat;

    helloVk.m_instances.push_back(inst);
  }
  */
  // Creation of implicit geometry
  MaterialObj mat;
  // Reflective
  mat.diffuse   = nvmath::vec3f(0, 0, 0);
  mat.specular  = nvmath::vec3f(1.f);
  mat.shininess = 0.0;
  mat.illum     = 3;
  helloVk.addImplMaterial(mat);
  // Transparent
  mat.diffuse  = nvmath::vec3f(0.4, 0.4, 1);
  mat.illum    = 4;
  mat.dissolve = 0.5;
  helloVk.addImplMaterial(mat);
  //helloVk.addImplCube({-6.1, 0, -6}, {-6, 10, 6}, 0);
  //helloVk.addImplSphere({1, 2, 4}, 1.f, 1);

  tfnw::TransferFunctionWidget tfn_widget;
  //tfn_widget.add_colormap(tfnw::Colormap("fary", std::vector<uint8_t>(tfnw::rainbow2, tfnw::rainbow2 + sizeof(tfnw::rainbow2)),
   //                                      tfnw::ColorSpace::LINEAR));
  if(kitten)
  {
    tfn_widget.setKittenAlpha_control_pts();
  }
  helloVk.initOffscreen();
  Offscreen& offscreen = helloVk.offscreen();

  helloVk.createImplictBuffers();


  helloVk.createDescriptorSetLayout();
  helloVk.createGraphicsPipeline();
  helloVk.createUniformBuffer();
  helloVk.createObjDescriptionBuffer();
  helloVk.createAtrInfoBuffer();
  helloVk.createColormap();
  helloVk.updateDescriptorSet();

  // #VKRay
  helloVk.initRayTracing();


  nvmath::vec4f clearColor   = nvmath::vec4f(1, 1, 1, 1.00f);
  bool          useRaytracer = true;


  helloVk.setupGlfwCallbacks(window);
  ImGui_ImplGlfw_InitForVulkan(window, true);



  // Main loop
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();
    if(helloVk.isMinimized())
      continue;

    // Start the Dear ImGui frame
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();


    // Show UI window.
    if(helloVk.showGui())
    {
      ImGuiH::Panel::Begin();
      bool changed = false;

      if(ImGui::CollapsingHeader("Background color"))
      {
        // Edit 3 floats representing a color
        changed |= ImGui::ColorEdit3("Background color", reinterpret_cast<float*>(&clearColor));
        // Switch between raster and ray tracing
        changed |= ImGui::Checkbox("Ray Tracer mode", &useRaytracer);
      }
      if(changed)
        helloVk.resetFrame();

      renderUI(helloVk);
      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
      ImGuiH::Control::Info("", "", "(F10) Toggle Pane", ImGuiH::Control::Flags::Disabled);
            
      // tfnw
      tfn_widget.draw_ui();
      
      ImGuiH::Panel::End();
    }

    // Start rendering the scene
    helloVk.prepareFrame();

    // Start command buffer of this frame
    auto                   curFrame = helloVk.getCurFrame();
    const VkCommandBuffer& cmdBuf   = helloVk.getCommandBuffers()[curFrame];

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    // tfnw and updating colomap
    //tfn_widget.update_gpu_image(cmdBuf, &helloVk.m_alloc);
  
    std::vector<uint8_t> colormap = tfn_widget.get_colormap();
    helloVk.updateColormap(cmdBuf, colormap);
     
    

    // Updating camera buffer
    helloVk.updateUniformBuffer(cmdBuf);

    // Clearing screen
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color        = {{clearColor[0], clearColor[1], clearColor[2], clearColor[3]}};
    clearValues[1].depthStencil = {1.0f, 0};

    // Offscreen render pass
    {
      VkRenderPassBeginInfo offscreenRenderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      offscreenRenderPassBeginInfo.clearValueCount = 2;
      offscreenRenderPassBeginInfo.pClearValues    = clearValues.data();
      offscreenRenderPassBeginInfo.renderPass      = offscreen.renderPass();
      offscreenRenderPassBeginInfo.framebuffer     = offscreen.frameBuffer();
      offscreenRenderPassBeginInfo.renderArea      = {{0, 0}, helloVk.getSize()};

      // Rendering Scene
      if(useRaytracer)
      {
        helloVk.raytrace(cmdBuf, clearColor);
      }
      else
      {
        vkCmdBeginRenderPass(cmdBuf, &offscreenRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        helloVk.rasterize(cmdBuf);
        vkCmdEndRenderPass(cmdBuf);
      }
    }

    // 2nd rendering pass: tone mapper, UI
    {
      VkRenderPassBeginInfo postRenderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      postRenderPassBeginInfo.clearValueCount = 2;
      postRenderPassBeginInfo.pClearValues    = clearValues.data();
      postRenderPassBeginInfo.renderPass      = helloVk.getRenderPass();
      postRenderPassBeginInfo.framebuffer     = helloVk.getFramebuffers()[curFrame];
      postRenderPassBeginInfo.renderArea      = {{0, 0}, helloVk.getSize()};

      // Rendering tonemapper
      vkCmdBeginRenderPass(cmdBuf, &postRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
      offscreen.draw(cmdBuf, helloVk.getSize());

      // Rendering UI
      ImGui::Render();
      ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuf);
      vkCmdEndRenderPass(cmdBuf);
    }

    // Submit for display
    vkEndCommandBuffer(cmdBuf);
    helloVk.submitFrame();
  }

  // Cleanup
  vkDeviceWaitIdle(helloVk.getDevice());

  helloVk.destroyResources();
  helloVk.destroy();
  vkctx.deinit();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
