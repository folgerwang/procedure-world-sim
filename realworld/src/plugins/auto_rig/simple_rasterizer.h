#pragma once
#include "rig_types.h"
#include <vector>
#include <glm/glm.hpp>

namespace plugins {
namespace auto_rig {

// ---------------------------------------------------------------------------
//  SimpleRasterizer – lightweight CPU triangle rasterizer.
//
//  Renders a TriangleMesh into per-pixel buffers:
//    • depth (linear, in model space)
//    • world-space normals
//    • silhouette mask
//    • simple diffuse-shaded colour
//
//  Used to produce multi-view captures of a character model without
//  touching the Vulkan renderer.  Not fast, but self-contained.
// ---------------------------------------------------------------------------
class SimpleRasterizer {
public:
    SimpleRasterizer() = default;
    ~SimpleRasterizer() = default;

    // Render the mesh from a specific camera (opaque).
    // Returns a fully populated ViewCapture.
    ViewCapture render(
        const TriangleMesh& mesh,
        int   width,
        int   height,
        const glm::mat4& view,
        const glm::mat4& proj,
        float azimuth_deg  = 0.0f,
        float elevation_deg = 0.0f) const;

    // Render with Weighted Blended Order-Independent Transparency.
    // mesh_alpha in [0,1] controls the mesh opacity.  All fragments
    // contribute (no depth-test discard) and are composited using the
    // McGuire-Bavoil 2013 weighted-blend formula.  Result goes into
    // ViewCapture::color_rgba (RGBA).  color (RGB) is also filled with
    // the opaque pass for training-data export.
    ViewCapture renderOIT(
        const TriangleMesh& mesh,
        int   width,
        int   height,
        const glm::mat4& view,
        const glm::mat4& proj,
        float mesh_alpha   = 0.6f,
        float azimuth_deg  = 0.0f,
        float elevation_deg = 0.0f) const;

    // Generate an orbit of cameras looking at the mesh bounding-box centre.
    // Returns `num_views` ViewCaptures evenly spaced in azimuth.
    // `radius_mult` controls camera distance as a multiplier of mesh extent.
    std::vector<ViewCapture> captureOrbit(
        const TriangleMesh& mesh,
        int   num_views,
        int   resolution,
        float elevation_deg = 15.0f,
        float radius_mult   = 1.2f) const;

private:
    // Rasterise a single triangle into the depth buffer + attribute buffers.
    struct Fragment {
        int   x, y;
        float depth;
        float bary[3];
    };

    void rasterTriangle(
        const glm::vec4& v0_clip,
        const glm::vec4& v1_clip,
        const glm::vec4& v2_clip,
        int width, int height,
        std::vector<Fragment>& out_frags) const;
};

}  // namespace auto_rig
}  // namespace plugins