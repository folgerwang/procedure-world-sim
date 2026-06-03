#include "simple_rasterizer.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <limits>
#include <unordered_map>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>

#include "tiny_gltf.h"                 // glTF / GLB geometry (decls; impl elsewhere)
#include "third_parties/fbx/ufbx.h"   // FBX geometry (decls; impl in ufbx.c)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace plugins {
namespace auto_rig {

// ── SimpleTexture::sample ──────────────────────────────────────────────────

glm::vec3 SimpleTexture::sample(const glm::vec2& uv) const {
    if (empty()) return glm::vec3(1.0f);  // white fallback

    // Wrap UVs to [0,1).
    float u = uv.x - std::floor(uv.x);
    float v = uv.y - std::floor(uv.y);

    // Bilinear sample coordinates.
    float fx = u * (width  - 1);
    float fy = v * (height - 1);

    int x0 = std::clamp((int)std::floor(fx), 0, width  - 1);
    int y0 = std::clamp((int)std::floor(fy), 0, height - 1);
    int x1 = std::min(x0 + 1, width  - 1);
    int y1 = std::min(y0 + 1, height - 1);

    float sx = fx - x0;
    float sy = fy - y0;

    auto fetch = [&](int x, int y) -> glm::vec3 {
        int idx = (y * width + x) * channels;
        return glm::vec3(
            pixels[idx + 0] / 255.0f,
            pixels[idx + 1] / 255.0f,
            pixels[idx + 2] / 255.0f);
    };

    glm::vec3 c00 = fetch(x0, y0);
    glm::vec3 c10 = fetch(x1, y0);
    glm::vec3 c01 = fetch(x0, y1);
    glm::vec3 c11 = fetch(x1, y1);

    return glm::mix(glm::mix(c00, c10, sx),
                    glm::mix(c01, c11, sx), sy);
}

// ── Helpers ─────────────────────────────────────────────────────────────────

static glm::vec3 perspectiveDivide(const glm::vec4& clip) {
    return glm::vec3(clip) / clip.w;
}

static glm::vec3 ndcToScreen(const glm::vec3& ndc, int w, int h) {
    return {
        (ndc.x * 0.5f + 0.5f) * w,
        (ndc.y * 0.5f + 0.5f) * h,     // y down = flip later
        ndc.z * 0.5f + 0.5f             // [0, 1] depth
    };
}

static float edgeFunction(const glm::vec2& a, const glm::vec2& b, const glm::vec2& p) {
    return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
}

// ── rasterTriangle ──────────────────────────────────────────────────────────

void SimpleRasterizer::rasterTriangle(
    const glm::vec4& v0_clip,
    const glm::vec4& v1_clip,
    const glm::vec4& v2_clip,
    int width, int height,
    std::vector<Fragment>& out_frags) const
{
    // Near-plane cull (any vertex behind camera).
    if (v0_clip.w <= 0.0f || v1_clip.w <= 0.0f || v2_clip.w <= 0.0f) return;

    glm::vec3 ndc0 = perspectiveDivide(v0_clip);
    glm::vec3 ndc1 = perspectiveDivide(v1_clip);
    glm::vec3 ndc2 = perspectiveDivide(v2_clip);

    glm::vec3 s0 = ndcToScreen(ndc0, width, height);
    glm::vec3 s1 = ndcToScreen(ndc1, width, height);
    glm::vec3 s2 = ndcToScreen(ndc2, width, height);

    // Bounding box.
    float fminx = std::min({s0.x, s1.x, s2.x});
    float fminy = std::min({s0.y, s1.y, s2.y});
    float fmaxx = std::max({s0.x, s1.x, s2.x});
    float fmaxy = std::max({s0.y, s1.y, s2.y});

    int minx = std::max(0, (int)std::floor(fminx));
    int miny = std::max(0, (int)std::floor(fminy));
    int maxx = std::min(width  - 1, (int)std::ceil(fmaxx));
    int maxy = std::min(height - 1, (int)std::ceil(fmaxy));

    float area = edgeFunction(glm::vec2(s0), glm::vec2(s1), glm::vec2(s2));
    if (std::abs(area) < 1e-6f) return;  // degenerate
    float inv_area = 1.0f / area;

    for (int y = miny; y <= maxy; ++y) {
        for (int x = minx; x <= maxx; ++x) {
            glm::vec2 p(x + 0.5f, y + 0.5f);

            float w0 = edgeFunction(glm::vec2(s1), glm::vec2(s2), p) * inv_area;
            float w1 = edgeFunction(glm::vec2(s2), glm::vec2(s0), p) * inv_area;
            float w2 = 1.0f - w0 - w1;

            if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;

            float depth = w0 * s0.z + w1 * s1.z + w2 * s2.z;

            Fragment frag;
            frag.x = x;
            frag.y = y;
            frag.depth = depth;
            frag.bary[0] = w0;
            frag.bary[1] = w1;
            frag.bary[2] = w2;
            out_frags.push_back(frag);
        }
    }
}

// ── render ──────────────────────────────────────────────────────────────────

ViewCapture SimpleRasterizer::render(
    const TriangleMesh& mesh,
    int width, int height,
    const glm::mat4& view,
    const glm::mat4& proj,
    float azimuth_deg,
    float elevation_deg) const
{
    ViewCapture cap;
    cap.width  = width;
    cap.height = height;
    cap.view   = view;
    cap.proj   = proj;
    cap.view_proj    = proj * view;
    cap.azimuth_deg  = azimuth_deg;
    cap.elevation_deg = elevation_deg;

    const int npix = width * height;
    cap.depth.assign(npix, 1.0f);
    cap.normal_map.assign(npix * 3, 0.0f);
    cap.silhouette.assign(npix, 0);
    cap.color.assign(npix * 3, 0);

    // Depth buffer for z-test.
    std::vector<float> zbuf(npix, 1.0f);

    // Per-pixel normal accumulator (written in the z-test pass).
    // We store the winning triangle's interpolated normal.
    struct PixelData {
        glm::vec3 normal{0.0f};
    };
    std::vector<PixelData> pixel_data(npix);

    glm::mat4 vp = proj * view;

    // Camera forward direction in world space (for double-sided lighting).
    glm::mat4 inv_view = glm::inverse(view);
    glm::vec3 cam_fwd  = -glm::vec3(inv_view[2]);

    const size_t vert_count = mesh.positions.size();
    const bool has_normals  = mesh.normals.size() == vert_count;
    const bool has_uvs      = mesh.texcoords.size() == vert_count;
    const bool has_tex      = !mesh.base_color_texture.empty();
    const bool has_vcol     = mesh.vertex_colors.size() == vert_count;

    std::vector<Fragment> frags;

    const size_t tri_count = mesh.indices.size() / 3;
    for (size_t t = 0; t < tri_count; ++t) {
        uint32_t i0 = mesh.indices[t * 3 + 0];
        uint32_t i1 = mesh.indices[t * 3 + 1];
        uint32_t i2 = mesh.indices[t * 3 + 2];
        if (i0 >= vert_count || i1 >= vert_count || i2 >= vert_count) continue;

        glm::vec4 c0 = vp * glm::vec4(mesh.positions[i0], 1.0f);
        glm::vec4 c1 = vp * glm::vec4(mesh.positions[i1], 1.0f);
        glm::vec4 c2 = vp * glm::vec4(mesh.positions[i2], 1.0f);

        frags.clear();
        rasterTriangle(c0, c1, c2, width, height, frags);

        // Compute face normal if per-vertex normals unavailable.
        glm::vec3 n0, n1, n2;
        if (has_normals) {
            n0 = mesh.normals[i0];
            n1 = mesh.normals[i1];
            n2 = mesh.normals[i2];
        } else {
            glm::vec3 face_n = glm::normalize(
                glm::cross(mesh.positions[i1] - mesh.positions[i0],
                           mesh.positions[i2] - mesh.positions[i0]));
            n0 = n1 = n2 = face_n;
        }

        // Fetch per-vertex UVs for texture sampling.
        glm::vec2 uv0(0.0f), uv1(0.0f), uv2(0.0f);
        if (has_uvs) {
            uv0 = mesh.texcoords[i0];
            uv1 = mesh.texcoords[i1];
            uv2 = mesh.texcoords[i2];
        }

        // Fetch per-vertex colors.
        glm::vec3 vc0(1.0f), vc1(1.0f), vc2(1.0f);
        if (has_vcol) {
            vc0 = mesh.vertex_colors[i0];
            vc1 = mesh.vertex_colors[i1];
            vc2 = mesh.vertex_colors[i2];
        }

        for (auto& f : frags) {
            int idx = f.y * width + f.x;
            if (idx < 0 || idx >= npix) continue;
            if (f.depth < zbuf[idx]) {
                zbuf[idx] = f.depth;
                cap.depth[idx] = f.depth;
                cap.silhouette[idx] = 255;

                glm::vec3 n = glm::normalize(
                    f.bary[0] * n0 + f.bary[1] * n1 + f.bary[2] * n2);
                pixel_data[idx].normal = n;

                cap.normal_map[idx * 3 + 0] = n.x;
                cap.normal_map[idx * 3 + 1] = n.y;
                cap.normal_map[idx * 3 + 2] = n.z;

                // Double-sided: flip normal for back faces.
                float ndot = glm::dot(n, cam_fwd);
                bool is_front = (ndot >= 0.0f);
                if (!is_front) n = -n;

                float diffuse = std::clamp(glm::dot(n, cam_fwd), 0.0f, 1.0f);
                float brightness = 0.4f + 0.6f * diffuse;

                // Base color: texture > vertex color > normal-mapped fallback.
                glm::vec3 col;
                if (has_tex && has_uvs) {
                    glm::vec2 uv = f.bary[0] * uv0 + f.bary[1] * uv1 + f.bary[2] * uv2;
                    col = mesh.base_color_texture.sample(uv);
                    if (has_vcol) {
                        glm::vec3 vc = f.bary[0] * vc0 + f.bary[1] * vc1 + f.bary[2] * vc2;
                        col *= vc;
                    }
                } else if (has_vcol) {
                    col = f.bary[0] * vc0 + f.bary[1] * vc1 + f.bary[2] * vc2;
                } else {
                    col = n * 0.5f + 0.5f;
                }

                col *= brightness;

                cap.color[idx * 3 + 0] = static_cast<uint8_t>(
                    std::clamp(col.r * 255.0f, 0.0f, 255.0f));
                cap.color[idx * 3 + 1] = static_cast<uint8_t>(
                    std::clamp(col.g * 255.0f, 0.0f, 255.0f));
                cap.color[idx * 3 + 2] = static_cast<uint8_t>(
                    std::clamp(col.b * 255.0f, 0.0f, 255.0f));
            }
        }
    }

    return cap;
}

// ── renderOIT ──────────────────────────────────────────────────────────────
//
//  Weighted Blended Order-Independent Transparency (McGuire & Bavoil 2013).
//
//  Every fragment contributes to per-pixel accumulation buffers, weighted
//  by a depth-dependent function.  No depth-test discard means overlapping
//  triangles blend correctly regardless of draw order.
//
//  Accumulation:
//    accum.rgb += C_i * alpha * w(z_i)
//    accum.a   += alpha * w(z_i)
//    reveal    *= (1 - alpha)
//
//  Resolve:
//    final.rgb = accum.rgb / max(accum.a, 1e-4)
//    final.a   = 1 - reveal
//
//  Weight function w(z) dampens distant fragments so near surfaces
//  dominate the blend.
// ───────────────────────────────────────────────────────────────────────────

ViewCapture SimpleRasterizer::renderOIT(
    const TriangleMesh& mesh,
    int width, int height,
    const glm::mat4& view,
    const glm::mat4& proj,
    float mesh_alpha,
    float azimuth_deg,
    float elevation_deg) const
{
    // First do a normal opaque render (fills color, depth, silhouette, normals).
    ViewCapture cap = render(mesh, width, height, view, proj,
                             azimuth_deg, elevation_deg);

    const int npix = width * height;
    mesh_alpha = std::clamp(mesh_alpha, 0.0f, 1.0f);

    // At full opacity, skip OIT and use the opaque depth-tested result directly.
    if (mesh_alpha >= 0.99f) {
        cap.color_rgba.resize(npix * 4);
        for (int i = 0; i < npix; ++i) {
            cap.color_rgba[i * 4 + 0] = cap.color[i * 3 + 0];
            cap.color_rgba[i * 4 + 1] = cap.color[i * 3 + 1];
            cap.color_rgba[i * 4 + 2] = cap.color[i * 3 + 2];
            cap.color_rgba[i * 4 + 3] = cap.silhouette[i]; // 255 where mesh, 0 bg
        }
        return cap;
    }

    // Now build the OIT composite into color_rgba.

    // Accumulation buffers (float precision).
    struct OITPixel {
        float accum_r = 0.0f;
        float accum_g = 0.0f;
        float accum_b = 0.0f;
        float accum_a = 0.0f;  // sum of alpha * w(z)
        float reveal  = 1.0f;  // product of (1 - alpha)
    };
    std::vector<OITPixel> oit(npix);

    glm::mat4 vp = proj * view;

    // Camera forward direction in world space (for front/back shading).
    glm::mat4 inv_view = glm::inverse(view);
    glm::vec3 cam_fwd  = -glm::vec3(inv_view[2]);  // camera looks along -Z

    const size_t vert_count = mesh.positions.size();
    const bool has_normals  = mesh.normals.size() == vert_count;
    const bool has_uvs      = mesh.texcoords.size() == vert_count;
    const bool has_tex      = !mesh.base_color_texture.empty();
    const bool has_vcol     = mesh.vertex_colors.size() == vert_count;

    std::vector<Fragment> frags;

    const size_t tri_count = mesh.indices.size() / 3;
    for (size_t t = 0; t < tri_count; ++t) {
        uint32_t i0 = mesh.indices[t * 3 + 0];
        uint32_t i1 = mesh.indices[t * 3 + 1];
        uint32_t i2 = mesh.indices[t * 3 + 2];
        if (i0 >= vert_count || i1 >= vert_count || i2 >= vert_count) continue;

        glm::vec4 c0 = vp * glm::vec4(mesh.positions[i0], 1.0f);
        glm::vec4 c1 = vp * glm::vec4(mesh.positions[i1], 1.0f);
        glm::vec4 c2 = vp * glm::vec4(mesh.positions[i2], 1.0f);

        frags.clear();
        rasterTriangle(c0, c1, c2, width, height, frags);

        // Normals.
        glm::vec3 n0, n1, n2;
        if (has_normals) {
            n0 = mesh.normals[i0]; n1 = mesh.normals[i1]; n2 = mesh.normals[i2];
        } else {
            glm::vec3 face_n = glm::normalize(
                glm::cross(mesh.positions[i1] - mesh.positions[i0],
                           mesh.positions[i2] - mesh.positions[i0]));
            n0 = n1 = n2 = face_n;
        }

        glm::vec2 uv0(0.0f), uv1(0.0f), uv2(0.0f);
        if (has_uvs) {
            uv0 = mesh.texcoords[i0]; uv1 = mesh.texcoords[i1]; uv2 = mesh.texcoords[i2];
        }

        glm::vec3 vc0(1.0f), vc1(1.0f), vc2(1.0f);
        if (has_vcol) {
            vc0 = mesh.vertex_colors[i0]; vc1 = mesh.vertex_colors[i1]; vc2 = mesh.vertex_colors[i2];
        }

        for (auto& f : frags) {
            int idx = f.y * width + f.x;
            if (idx < 0 || idx >= npix) continue;

            // Compute fragment color (same logic as opaque render).
            glm::vec3 n = glm::normalize(
                f.bary[0] * n0 + f.bary[1] * n1 + f.bary[2] * n2);
            glm::vec3 col;
            if (has_tex && has_uvs) {
                glm::vec2 uv = f.bary[0] * uv0 + f.bary[1] * uv1 + f.bary[2] * uv2;
                col = mesh.base_color_texture.sample(uv);
                if (has_vcol) {
                    glm::vec3 vc = f.bary[0] * vc0 + f.bary[1] * vc1 + f.bary[2] * vc2;
                    col *= vc;
                }
            } else if (has_vcol) {
                col = f.bary[0] * vc0 + f.bary[1] * vc1 + f.bary[2] * vc2;
            } else {
                col = n * 0.5f + 0.5f;
            }

            // Double-sided: detect front/back, then flip normal for
            // back faces so both sides get proper lighting.
            float ndot = glm::dot(n, cam_fwd);
            bool is_front = (ndot >= 0.0f);

            // Flip normal for back faces so lighting is symmetric.
            if (!is_front) n = -n;

            // Simple diffuse brightness using the (now always front-facing) normal.
            float diffuse = std::clamp(glm::dot(n, cam_fwd), 0.0f, 1.0f);
            float brightness = 0.4f + 0.6f * diffuse;
            col *= brightness;

            // Color tint: green for front faces, red for back faces.
            if (is_front) {
                col.g = std::clamp(col.g + 0.20f, 0.0f, 1.0f);
            } else {
                col.r = std::clamp(col.r + 0.20f, 0.0f, 1.0f);
                col.b *= 0.7f;  // slightly desaturate blue on back faces
            }

            float frag_alpha = mesh_alpha;

            // Weighted blend: w(z) = clamp(1e3 * (1-z)^3, 1e-2, 3e3)
            float one_minus_z = std::clamp(1.0f - f.depth, 0.0f, 1.0f);
            float w = std::clamp(1000.0f * one_minus_z * one_minus_z * one_minus_z,
                                 0.01f, 3000.0f);

            float aw = frag_alpha * w;
            oit[idx].accum_r += col.r * aw;
            oit[idx].accum_g += col.g * aw;
            oit[idx].accum_b += col.b * aw;
            oit[idx].accum_a += aw;
            oit[idx].reveal  *= (1.0f - frag_alpha);
        }
    }

    // Resolve OIT into RGBA.
    cap.color_rgba.resize(npix * 4);
    for (int i = 0; i < npix; ++i) {
        auto& p = oit[i];
        float final_alpha = 1.0f - p.reveal;

        if (final_alpha < 1e-4f) {
            // No fragments hit this pixel — transparent background.
            cap.color_rgba[i * 4 + 0] = 0;
            cap.color_rgba[i * 4 + 1] = 0;
            cap.color_rgba[i * 4 + 2] = 0;
            cap.color_rgba[i * 4 + 3] = 0;
        } else {
            float inv_a = 1.0f / std::max(p.accum_a, 1e-4f);
            float r = std::clamp(p.accum_r * inv_a, 0.0f, 1.0f);
            float g = std::clamp(p.accum_g * inv_a, 0.0f, 1.0f);
            float b = std::clamp(p.accum_b * inv_a, 0.0f, 1.0f);
            float a = std::clamp(final_alpha, 0.0f, 1.0f);

            cap.color_rgba[i * 4 + 0] = static_cast<uint8_t>(r * 255.0f);
            cap.color_rgba[i * 4 + 1] = static_cast<uint8_t>(g * 255.0f);
            cap.color_rgba[i * 4 + 2] = static_cast<uint8_t>(b * 255.0f);
            cap.color_rgba[i * 4 + 3] = static_cast<uint8_t>(a * 255.0f);
        }
    }

    return cap;
}

// ── captureOrbit ────────────────────────────────────────────────────────────

std::vector<ViewCapture> SimpleRasterizer::captureOrbit(
    const TriangleMesh& mesh,
    int num_views,
    int resolution,
    float elevation_deg,
    float radius_mult) const
{
    std::vector<ViewCapture> captures;
    captures.reserve(num_views);

    // Orbit centre and radius from the mesh bounds.
    glm::vec3 centre = (mesh.bbox_min + mesh.bbox_max) * 0.5f;
    float extent = glm::length(mesh.bbox_max - mesh.bbox_min);
    float radius = extent * radius_mult;

    fprintf(stderr, "[SimpleRasterizer] captureOrbit: radius=%.4f (ext=%.4f * mult=%.4f) res=%d elev=%.1f\n",
            radius, extent, radius_mult, resolution, elevation_deg);

    float elev_rad = glm::radians(elevation_deg);

    // Perspective projection that fits the bounding sphere.
    float fov = glm::radians(45.0f);
    float aspect = 1.0f;
    float z_near = radius * 0.01f;
    float z_far  = radius * 4.0f;
    glm::mat4 proj = glm::perspective(fov, aspect, z_near, z_far);
    proj[1][1] *= -1.0f;   // Vulkan y-flip convention (kept for consistency)

    for (int i = 0; i < num_views; ++i) {
        float azimuth_deg = 360.0f * i / num_views;
        float az_rad = glm::radians(azimuth_deg);

        glm::vec3 eye;
        eye.x = centre.x + radius * cosf(elev_rad) * cosf(az_rad);
        eye.y = centre.y + radius * sinf(elev_rad);
        eye.z = centre.z + radius * cosf(elev_rad) * sinf(az_rad);

        glm::mat4 view = glm::lookAt(eye, centre, glm::vec3(0, 1, 0));

        captures.push_back(
            render(mesh, resolution, resolution, view, proj,
                   azimuth_deg, elevation_deg));
    }

    return captures;
}

// ── TriangleMesh helpers ────────────────────────────────────────────────────

void TriangleMesh::recomputeBounds() {
    bbox_min = glm::vec3( 1e30f);
    bbox_max = glm::vec3(-1e30f);
    for (auto& p : positions) {
        bbox_min = glm::min(bbox_min, p);
        bbox_max = glm::max(bbox_max, p);
    }
}

void TriangleMesh::recomputeNormals() {
    normals.assign(positions.size(), glm::vec3(0.0f));
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        uint32_t i0 = indices[i], i1 = indices[i + 1], i2 = indices[i + 2];
        glm::vec3 e1 = positions[i1] - positions[i0];
        glm::vec3 e2 = positions[i2] - positions[i0];
        glm::vec3 fn = glm::cross(e1, e2);
        normals[i0] += fn;
        normals[i1] += fn;
        normals[i2] += fn;
    }
    for (auto& n : normals) {
        float len = glm::length(n);
        if (len > 1e-8f) n /= len;
    }
}

// ── Geometry-only loaders for thumbnail rendering ───────────────────────────
namespace {

// Local transform of a glTF node (matrix, or T*R*S).
glm::mat4 gltfNodeLocal(const tinygltf::Node& n) {
    if (!n.matrix.empty()) {
        glm::mat4 m(1.0f);
        for (int i = 0; i < 16; ++i) (&m[0][0])[i] = (float)n.matrix[i];
        return m;
    }
    glm::mat4 T(1.0f), R(1.0f), S(1.0f);
    if (!n.translation.empty())
        T = glm::translate(glm::mat4(1.0f),
            glm::vec3((float)n.translation[0], (float)n.translation[1],
                      (float)n.translation[2]));
    if (!n.rotation.empty()) {
        glm::quat q((float)n.rotation[3], (float)n.rotation[0],
                    (float)n.rotation[1], (float)n.rotation[2]);
        R = glm::mat4_cast(q);
    }
    if (!n.scale.empty())
        S = glm::scale(glm::mat4(1.0f),
            glm::vec3((float)n.scale[0], (float)n.scale[1], (float)n.scale[2]));
    return T * R * S;
}

bool loadGltfGeometry(const std::string& path, TriangleMesh& out) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    bool ok = (path.size() >= 4 && path.substr(path.size() - 4) == ".glb")
        ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
        : loader.LoadASCIIFromFile(&model, &err, &warn, path);
    if (!ok) {
        std::fprintf(stderr, "[Thumb] glTF load failed (%s): %s\n",
                     path.c_str(), err.c_str());
        return false;
    }

    // World transform per node.
    std::vector<glm::mat4> node_world(model.nodes.size(), glm::mat4(1.0f));
    {
        std::function<void(int, const glm::mat4&)> walk =
            [&](int idx, const glm::mat4& parent) {
                node_world[idx] = parent * gltfNodeLocal(model.nodes[idx]);
                for (int c : model.nodes[idx].children) walk(c, node_world[idx]);
            };
        std::vector<bool> is_child(model.nodes.size(), false);
        for (auto& n : model.nodes)
            for (int c : n.children) is_child[c] = true;
        for (int i = 0; i < (int)model.nodes.size(); ++i)
            if (!is_child[i]) walk(i, glm::mat4(1.0f));
    }
    std::unordered_map<int, glm::mat4> mesh_to_world;
    for (int i = 0; i < (int)model.nodes.size(); ++i)
        if (model.nodes[i].mesh >= 0) mesh_to_world[model.nodes[i].mesh] = node_world[i];

    for (int mi = 0; mi < (int)model.meshes.size(); ++mi) {
        glm::mat4 world(1.0f);
        auto it = mesh_to_world.find(mi);
        if (it != mesh_to_world.end()) world = it->second;

        for (auto& prim : model.meshes[mi].primitives) {
            if (prim.mode != TINYGLTF_MODE_TRIANGLES && prim.mode != -1) continue;
            auto pit = prim.attributes.find("POSITION");
            if (pit == prim.attributes.end()) continue;

            const uint32_t base = (uint32_t)out.positions.size();
            const auto& acc = model.accessors[pit->second];
            const auto& bv  = model.bufferViews[acc.bufferView];
            const auto& buf = model.buffers[bv.buffer];
            const float* p  = reinterpret_cast<const float*>(
                buf.data.data() + bv.byteOffset + acc.byteOffset);
            const size_t stride = bv.byteStride ? bv.byteStride / sizeof(float) : 3;
            for (size_t i = 0; i < acc.count; ++i) {
                glm::vec4 wp = world * glm::vec4(p[i * stride + 0],
                                                 p[i * stride + 1],
                                                 p[i * stride + 2], 1.0f);
                out.positions.push_back(glm::vec3(wp));
            }

            if (prim.indices >= 0) {
                const auto& ia = model.accessors[prim.indices];
                const auto& ib = model.bufferViews[ia.bufferView];
                const auto& ibf = model.buffers[ib.buffer];
                const uint8_t* raw = ibf.data.data() + ib.byteOffset + ia.byteOffset;
                for (size_t i = 0; i < ia.count; ++i) {
                    uint32_t idx = 0;
                    switch (ia.componentType) {
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                            idx = reinterpret_cast<const uint16_t*>(raw)[i]; break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                            idx = reinterpret_cast<const uint32_t*>(raw)[i]; break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                            idx = raw[i]; break;
                        default: break;
                    }
                    out.indices.push_back(base + idx);
                }
            } else {
                for (uint32_t i = 0; i < (uint32_t)acc.count; ++i)
                    out.indices.push_back(base + i);
            }
        }
    }
    return !out.positions.empty() && !out.indices.empty();
}

bool loadFbxGeometry(const std::string& path, TriangleMesh& out) {
    ufbx_load_opts opts{};
    opts.target_axes      = ufbx_axes_right_handed_y_up;
    opts.target_unit_meters = 1.0f;
    ufbx_error err{};
    ufbx_scene* scene = ufbx_load_file(path.c_str(), &opts, &err);
    if (!scene) {
        std::fprintf(stderr, "[Thumb] FBX load failed (%s): %s\n",
                     path.c_str(), err.description.data);
        return false;
    }
    for (size_t mi = 0; mi < scene->meshes.count; ++mi) {
        const ufbx_mesh* m = scene->meshes.data[mi];
        const uint32_t base = (uint32_t)out.positions.size();
        // Positions in the mesh's vertex array (geometry space).
        for (size_t v = 0; v < m->vertices.count; ++v) {
            const ufbx_vec3& p = m->vertices.data[v];
            out.positions.push_back(glm::vec3((float)p.x, (float)p.y, (float)p.z));
        }
        // Triangulate each face into the index buffer.
        std::vector<uint32_t> tri(64 * 3);
        for (size_t fi = 0; fi < m->faces.count; ++fi) {
            const ufbx_face face = m->faces.data[fi];
            const size_t need = (size_t)(face.num_indices >= 3 ? face.num_indices - 2 : 0) * 3;
            if (need == 0) continue;
            if (tri.size() < need) tri.resize(need);
            uint32_t ntri = ufbx_triangulate_face(tri.data(), tri.size(), m, face);
            for (uint32_t t = 0; t < ntri * 3; ++t) {
                // tri[] holds *index-buffer* offsets; map to vertex indices.
                uint32_t vidx = m->vertex_indices.data[tri[t]];
                out.indices.push_back(base + vidx);
            }
        }
    }
    ufbx_free_scene(scene);
    return !out.positions.empty() && !out.indices.empty();
}

}  // anonymous namespace

bool loadMeshForThumbnail(const std::string& path, TriangleMesh& out) {
    out = TriangleMesh{};
    std::string ext;
    {
        const size_t dot = path.find_last_of('.');
        if (dot != std::string::npos) ext = path.substr(dot);
        for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
    }

    bool ok = false;
    if (ext == ".gltf" || ext == ".glb") ok = loadGltfGeometry(path, out);
    else if (ext == ".fbx")              ok = loadFbxGeometry(path, out);
    if (!ok || out.empty()) return false;

    out.recomputeBounds();
    out.recomputeNormals();
    return true;
}

}  // namespace auto_rig
}  // namespace plugins

