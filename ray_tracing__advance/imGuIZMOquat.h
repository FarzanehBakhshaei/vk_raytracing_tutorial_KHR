//------------------------------------------------------------------------------
//  Copyright (c) 2018-2020 Michele Morrone
//  All rights reserved.
//
//  https://michelemorrone.eu - https://BrutPitt.com
//
//  twitter: https://twitter.com/BrutPitt - github: https://github.com/BrutPitt
//
//  mailto:brutpitt@gmail.com - mailto:me@michelemorrone.eu
//  
//  This software is distributed under the terms of the BSD 2-Clause license
/*BSD 2-Clause License

Copyright (c) 2018-2020 Michele Morrone
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.*/
//------------------------------------------------------------------------------
#pragma once
#include <algorithm>


#include "vGizmo.h"

#if !defined(IMGUIZMO_IMGUI_FOLDER)
    #define IMGUIZMO_IMGUI_FOLDER imgui/
#endif

#define GET_PATH(P) P
#define INC_PATH(X) <GET_PATH(IMGUIZMO_IMGUI_FOLDER)X>

#define IMGUI_DEFINE_MATH_OPERATORS

#include "imgui.h"
#include "imgui_internal.h"

////////////////////////////////////////////////////////////////////////////
//
//    NEED TO BUILD :
//
//    - Using glm (OpenGL Mathematics) include files: need version 0.9.9 or later
//        and GLM_ENABLE_EXPERIMENTAL compile option to enable "quaternion" class
//    - Using include virtualGizmo.h (my virtualGizmoClass) to get imput controls
//          and apply rotations
//    - use c++11 standard
//
//    Use preprocessor define imguiGizmo_INTERPOLATE_NORMALS, to a phong color effect
//    for solid components, but there are more vertex memorized and more operations,
//    even if the visual effect is better... the computational difference is:
//    one normal for vertex  VS  one normal for plain figure (triangle or quad)
//    
//    comment/uncomment below or add as directive to compiler
////////////////////////////////////////////////////////////////////////////

#define imguiGizmo_INTERPOLATE_NORMALS
#define STARTING_ALPHA_PLANE .75f

////////////////////////////////////////////////////////////////////////////
//
//  imguiGizmo 3D
//
//      Buttons:
//          Left:   drag to free rotate axes
//          Right:  drag to free rotate spot (only dualmode!)
//          Right & Left: drag to rotate together
//              for ImGui features, right button drag is active only INSIDE the frame
//      Modifiers:
//          Shift:  rotation only around X
//          Ctrl:   rotation only around Y
//          Alt:    rotation only around Z
////////////////////////////////////////////////////////////////////////////

// The data structure that holds the orientation among other things
struct imguiGizmo
{

    vgm::Quat qtV  = vgm::Quat(1.0f, vgm::Vec3(0.0f));  // Quaternion value
  vgm::Quat qtV2 = vgm::Quat(1.0f, vgm::Vec3(0.0f));  // Quaternion value
#ifndef IMGUIZMO_USE_ONLY_ROT
  vgm::Vec3 posPanDolly    = vgm::Vec3(0.f);
    vgButtons buttonPanDolly = vg::evLeftButton;
#endif

    enum      {                              //0b0000'0000, //C++14 notation
                mode3Axes          = 0x0001, //0b0000'0001, 
                modeDirection      = 0x0002, //0b0000'0010,
                modeDirPlane       = 0x0004, //0b0000'0100,
                modeDual           = 0x0008, //0b0000'1000,
                modePanDolly       = 0x0010, //0b0001'0000,
                modeMask           = 0x00ff, 
                

                cubeAtOrigin       = 0x0100, //0b0000'0000, 
                sphereAtOrigin     = 0x0200, //0b0001'0000,
                noSolidAtOrigin    = 0x0400, //0b0010'0000,
                modeFullAxes       = 0x0800,
                axesModeMask       = 0xff00  
    };

    enum { sphereTess16, sphereTess8, sphereTess4, sphereTess2 };
    enum { CONE_SURF, CONE_CAP, CYL_SURF, CYL_CAP };
    //enum { SOLID_SURF, SOLID_CAP }
    //enum { 
    enum { axisIsX, axisIsY, axisIsZ };

    enum solidSides{ backSide, frontSide  }; // or viceversa... 
    static ImVector<vgm::Vec3> sphereVtx;
    static ImVector<int>  sphereTess;
    static ImVector<vgm::Vec3> cubeVtx;
    static ImVector<vgm::Vec3> cubeNorm;
    static ImVector<vgm::Vec3> planeVtx;
    static ImVector<vgm::Vec3> planeNorm;
    static ImVector<vgm::Vec3> arrowVtx[4];
    static ImVector<vgm::Vec3> arrowNorm[4];
    static void buildPlane   (const float size, const float thickness = planeThickness) {
        buildPolygon(vgm::Vec3(thickness,size,size), planeVtx, planeNorm);
    }
    static void buildCube    (const float size) {
        buildPolygon(vgm::Vec3(size), cubeVtx, cubeNorm);
    }
    static void buildPolygon (const vgm::Vec3& size,ImVector<vgm::Vec3>& vtx,ImVector<vgm::Vec3>& norm);
    static void buildSphere  (const float radius, const int tessFactor);
    static void buildCone    (const float x0, const float x1, const float radius, const int slices);
    static void buildCylinder(const float x0, const float x1, const float radius, const int slices);
    
    
    // helper functions
    ///////////////////////////////////////
    static void resizeAxesOf(const vgm::Vec3 &newSize) {
        savedAxesResizeFactor = axesResizeFactor; axesResizeFactor = newSize; }
    static void restoreAxesSize() {
        axesResizeFactor = savedAxesResizeFactor; }

    static void resizeSolidOf(float newSize) {
        savedSolidResizeFactor = solidResizeFactor; solidResizeFactor = newSize; }
    static void restoreSolidSize() {
        solidResizeFactor = savedSolidResizeFactor; }

    static void setDirectionColor(ImU32 dColor, const ImU32 pColor) {
        setDirectionColor(ImGui::ColorConvertU32ToFloat4(dColor), ImGui::ColorConvertU32ToFloat4(pColor)); }
    static void setDirectionColor(const ImVec4 &dColor, const ImVec4 &pColor) {
        savedDirectionColor = directionColor; savedPlaneColor = planeColor; 
        directionColor = dColor; planeColor = pColor;
    }
    static void setDirectionColor(ImU32 color) { setDirectionColor(ImGui::ColorConvertU32ToFloat4(color)); } 
    static void setDirectionColor(const ImVec4& color) { setDirectionColor(color,ImVec4(color.x, color.y, color.z, STARTING_ALPHA_PLANE));  }
    static void restoreDirectionColor() {
        directionColor = savedDirectionColor; 
        planeColor     = savedPlaneColor;     }

    static void setSphereColors(const ImVec4& a, const ImVec4& b) {
        setSphereColors( ImGui::ColorConvertFloat4ToU32(a), ImGui::ColorConvertFloat4ToU32(b)); }    
    static void setSphereColors(ImU32 a, ImU32 b) {
        savedSphereColors[0] = sphereColors[0]; savedSphereColors[1] = sphereColors[1];
        sphereColors[0] = a; sphereColors[1] = b; }
    static void restoreSphereColors() {
        sphereColors[0] = savedSphereColors[0]; sphereColors[1] = savedSphereColors[1]; }


    //  gizmo mouse/key settings
    ////////////////////////////////////////////////////////////////////////////
    // Call it once, to set all widgets... or if you need it 
    static void setGizmoFeelingRot(float f) { gizmoFeelingRot = f; } // default 1.0, >1 more mouse sensitivity, <1 less mouse sensitivity
    static float getGizmoFeelingRot() { return gizmoFeelingRot; }

#ifndef IMGUIZMO_USE_ONLY_ROT
// available vgModifiers values:
//      evShiftModifier   -> Shift
//      evControlModifier -> Ctrl
//      evAltModifier     -> Alt
//      evSuperModifier   -> Super
    static void setPanModifier(vgModifiers v) { panMod = v; }    // Change default assignment for Pan
    static void setDollyModifier(vgModifiers v) { panMod = v; }  // Change default assignment for Dolly

    //  Set the mouse response for the dolly operation...  also wheel
    static void setDollyScale(float  scale) { dollyScale = scale;  } // default 1.0, >1 more, <1 less 
    static float getDollyScale() { return dollyScale;  }
    //  Set the mouse response for pan    
    static void setPanScale(float scale) { panScale = scale; } // default 1.0, >1 more, <1 less 
    static float getPanScale() { return panScale; }
#endif


    //  internals
    ////////////////////////////////////////////////////////////////////////////
    static bool solidAreBuilded;
    static bool dragActivate;

    int drawMode = mode3Axes;
    int axesOriginType = cubeAtOrigin;
    bool showFullAxes = false;

    bool drawFunc(const char* label, float size);

    void modeSettings(int mode) {
        drawMode = int(mode & modeMask); axesOriginType = int(mode & axesModeMask); showFullAxes = bool(modeFullAxes & mode); }

    void setDualMode(const int mode) { modeSettings((imguiGizmo::modeDual | imguiGizmo::axesModeMask) & (mode | imguiGizmo::modeDual)); }

    // vgm::Vec3 -> vgm::Quat -> trackbalTransforms -> vgm::Quat -> vgm::Vec3
    ////////////////////////////////////////////////////////////////////////////
    bool getTransforms(vgm::Quat& q, const char* label, vgm::Vec3& dir, float size) {
        float len = length(dir);

        if(len<1.0 && len>= FLT_EPSILON) { normalize(dir); len = 1.0; }
        else if(len< FLT_EPSILON) { dir = vgm::Vec3(1.f, 0.f, 0.f); len = 1.0; }

        q = angleAxis(acosf(dir.x/len), normalize(vgm::Vec3(FLT_EPSILON, -dir.z, dir.y)));

        bool ret = drawFunc(label, size);
        if(ret)
          dir = (q * vgm::Vec3(1.0f, 0.0f, 0.0f)) * len;  //return vector with original lenght

        return ret;
    }
    // Vec4 (xyz axis, w angle) -> vgm::Quat -> trackbalTransforms -> vgm::Quat -> vgm::Vec4
    ////////////////////////////////////////////////////////////////////////////
    bool getTransforms(vgm::Quat& q, const char* label, vgm::Vec4& axis_angle, float size) {
        q = angleAxis(axis_angle.w,vgm::Vec3(axis_angle)); //g.ConvertFromAxisAngle();
   
        bool ret = drawFunc(label, size);
        if (ret) axis_angle = vgm::Vec4(vgm::Vec3(axis(q)),angle(q));

        return ret; 
    }

    //
    //  Settings
    //
    //      axes/arrow are composed of cone (or pyramid) and cylinder 
    //      (or parallelepiped): this solid are builded at first instance
    //      and will have same slices/radius/length for all controls in your 
    //      applications but can be  resized proportionally with a reductin
    //      factor: resizeAxesOf and resizeSolidOf (below).
    //      Also the colors of sphere tessellation are set at buil time, 
    //      while colors of axes and cube are fixed
    //
    //      if you want change solids attributes, change the static variables.
    //      If you need to resize solid and axes use resizeAxesOf and 
    //      resizeSolidOf, they works like push/pop stack (without buffer!) 
    //      with respective restoreAxesSize and restoreSolidSize.
    //      for example:
    //          // reDim axes ... same lenght, 
    //          imguiGizmo::resizeAxesOf(vgm::Vec3(imguiGizmo::axesResizeFactor.x, 2.0, 2.0)); 
    //          imguiGizmo::resizeSolidOf(1.25); // sphere bigger
    //          ImGui::gizmo3D("##RotB", b,sz);   
    //          imguiGizmo::restoreSolidSize(); // restore at default
    //          imguiGizmo::restoreAxesSize();
    ////////////////////////////////////////////////////////////////////////////

    //
    //  Build solid components
    //
    ////////////////////////////////////////////////////////////////////////////

    // arrow/axes components
    ///////////////////////////////////////
    static int   coneSlices ;
    static float coneRadius;
    static float coneLength;

    static int   cylSlices  ;
    static float cylRadius ;  // sizeCylLength ... defined in base to control size    

    // Sphere components
    ///////////////////////////////////////
    static float sphereRadius;
    static int sphereTessFactor;

    // Cube components
    ///////////////////////////////////////
    static float cubeSize;

    // Plane components
    ///////////////////////////////////////
    static float planeSize;
    static float planeThickness;


    //
    //  Resizing and color settings
    //
    ////////////////////////////////////////////////////////////////////////////


    // Axes reduction
    ///////////////////////////////////////
    static vgm::Vec3 axesResizeFactor;
    static vgm::Vec3 savedAxesResizeFactor;

    // solid reduction
    ///////////////////////////////////////
    static float solidResizeFactor;
    static float savedSolidResizeFactor;


    static ImU32 sphereColors[2]; // Tessellation colors
    static ImU32 savedSphereColors[2];
    //ImU32 spherecolorA=0xff005cc0, spherecolorB=0xffc05c00;
    
    static ImVec4 directionColor;
    static ImVec4 savedDirectionColor;

    static ImVec4 planeColor;
    static ImVec4 savedPlaneColor;

    // Gizmo mouse settings
    ///////////////////////////////////////
    static float gizmoFeelingRot; // >1 more mouse sensibility, <1 less mouse sensibility
#ifndef IMGUIZMO_USE_ONLY_ROT
    static float panScale, dollyScale;
    static vgModifiers panMod, dollyMod;
#endif

    static const int imguiGizmoDefaultSize;

};

#define IMGUIZMO_DEF_SIZE (ImGui::GetFrameHeightWithSpacing()*4 - (ImGui::GetStyle().ItemSpacing.y*2))

namespace ImGui
{

IMGUI_API bool gizmo3D(const char*, vgm::Quat&, float=IMGUIZMO_DEF_SIZE, const int=imguiGizmo::mode3Axes|imguiGizmo::cubeAtOrigin);
IMGUI_API bool gizmo3D(const char*, vgm::Vec4&, float=IMGUIZMO_DEF_SIZE, const int=imguiGizmo::mode3Axes|imguiGizmo::cubeAtOrigin);
IMGUI_API bool gizmo3D(const char*, vgm::Vec3&, float=IMGUIZMO_DEF_SIZE, const int=imguiGizmo::modeDirection);

IMGUI_API bool gizmo3D(const char*, vgm::Quat&, vgm::Quat&, float=IMGUIZMO_DEF_SIZE, const int=imguiGizmo::modeDual|imguiGizmo::cubeAtOrigin);
IMGUI_API bool gizmo3D(const char*, vgm::Quat&, vgm::Vec4&, float=IMGUIZMO_DEF_SIZE, const int=imguiGizmo::modeDual|imguiGizmo::cubeAtOrigin);
IMGUI_API bool gizmo3D(const char*, vgm::Quat&, vgm::Vec3&, float=IMGUIZMO_DEF_SIZE, const int=imguiGizmo::modeDual|imguiGizmo::cubeAtOrigin);

#ifndef IMGUIZMO_USE_ONLY_ROT

//with Pan & Dolly feature
IMGUI_API bool gizmo3D(const char*, vgm::Vec3&, vgm::Quat&, float=IMGUIZMO_DEF_SIZE, const int=imguiGizmo::mode3Axes|imguiGizmo::cubeAtOrigin);
IMGUI_API bool gizmo3D(const char*, vgm::Vec3&, vgm::Vec4&, float=IMGUIZMO_DEF_SIZE, const int=imguiGizmo::mode3Axes|imguiGizmo::cubeAtOrigin);
IMGUI_API bool gizmo3D(const char*, vgm::Vec3&, vgm::Vec3&, float=IMGUIZMO_DEF_SIZE, const int=imguiGizmo::modeDirection);

IMGUI_API bool gizmo3D(const char*, vgm::Vec3&, vgm::Quat&, vgm::Quat&, float=IMGUIZMO_DEF_SIZE, const int=imguiGizmo::modeDual|imguiGizmo::cubeAtOrigin);
IMGUI_API bool gizmo3D(const char*, vgm::Vec3&, vgm::Quat&, vgm::Vec4&, float=IMGUIZMO_DEF_SIZE, const int=imguiGizmo::modeDual|imguiGizmo::cubeAtOrigin);
IMGUI_API bool gizmo3D(const char*, vgm::Vec3&, vgm::Quat&, vgm::Vec3&, float=IMGUIZMO_DEF_SIZE, const int=imguiGizmo::modeDual|imguiGizmo::cubeAtOrigin);

#endif
}

//#undef imguiGizmo_DEF_SIZE