#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "imgui.h"
#include "hello_vulkan.h"

namespace tfnw {

enum ColorSpace { LINEAR, SRGB };

struct Colormap {
    std::string name;
    // An RGBA8 1D image
    std::vector<uint8_t> colormap;
    ColorSpace color_space;

    Colormap(const std::string &name,
             const std::vector<uint8_t> &img,
             const ColorSpace color_space);
};

class TransferFunctionWidget {
    struct vec2f {
        float x, y;

        vec2f(float c = 0.f);
        vec2f(float x, float y);
        vec2f(const ImVec2 &v);

        float length() const;

        vec2f operator+(const vec2f &b) const;
        vec2f operator-(const vec2f &b) const;
        vec2f operator/(const vec2f &b) const;
        vec2f operator*(const vec2f &b) const;
        operator ImVec2() const;
    };

    std::vector<Colormap> colormaps;
    size_t selected_colormap = 0;
    std::vector<uint8_t> current_colormap;

    std::vector<vec2f> alpha_control_pts = {vec2f(0.f),          vec2f(0.14f, 0.f), vec2f(0.2f, 0.19f),
                                            vec2f(0.33f, 0.19f), vec2f(0.41f, 0.02f), vec2f(0.58f, 0.02f),
                                            vec2f(0.65f, 0.39f), vec2f(0.79f, 0.38f), vec2f(0.8f, 0.f),
                                            vec2f(1.0f, 0.f)};
    size_t selected_point = -1;

    bool clicked_on_item = false;
    bool gpu_image_stale = true;
    bool colormap_changed = true;
    //GLuint colormap_img = -1;
    

public:
    TransferFunctionWidget();

    // Add a colormap preset. The image should be a 1D RGBA8 image, if the image
    // is provided in sRGBA colorspace it will be linearized
    void add_colormap(const Colormap &map);

    // Add the transfer function UI into the currently active window
    void draw_ui();

    // Returns true if the colormap was updated since the last
    // call to draw_ui
    bool changed() const;

    // Get back the RGBA8 color data for the transfer function
    std::vector<uint8_t> get_colormap();

    // Get back the RGBA32F color data for the transfer function
    std::vector<float> get_colormapf();

    // Get back the RGBA32F color data for the transfer function
    // as separate color and opacity vectors
    void get_colormapf(std::vector<float> &color, std::vector<float> &opacity);

    void update_gpu_image(VkCommandBuffer cmdBuff, Allocator* alloc);

    VkImage texture;
    void    setKittenAlpha_control_pts()
    {
      alpha_control_pts = {vec2f(0.f),          vec2f(0.17f, 0.56f),  vec2f(0.52f, 0.74f),
                           vec2f(0.71f, 0.37f), vec2f(0.89f, 0.76f),  vec2f(1.0f, 0.f)
      };
      selected_colormap = 1;
      update_colormap();
    }
  private:
    

    void update_colormap();

    void load_embedded_preset(const uint8_t *buf, size_t size, const std::string &name);
};
}

