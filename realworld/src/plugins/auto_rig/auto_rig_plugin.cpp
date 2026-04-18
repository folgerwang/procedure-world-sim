// ---------------------------------------------------------------------------
//  auto_rig_plugin.cpp – full auto-rigging pipeline implementation.
// ---------------------------------------------------------------------------
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

#include "auto_rig_plugin.h"
#include "imgui.h"
#include "tiny_gltf.h"
#include "stb_image_write.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <functional>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace plugins {
namespace auto_rig {

// ============================================================================
//  Rig Editing Constants
// ============================================================================

// Standard 19-joint humanoid for editing.
constexpr int kNumEditJoints = 19;

struct JointDef { float lr; float fb; float ry; };
static const JointDef kJointDefs[kNumEditJoints] = {
    { 0.00f,  0.00f, 0.53f},  //  0 hips
    { 0.00f,  0.00f, 0.43f},  //  1 spine
    { 0.00f,  0.00f, 0.33f},  //  2 chest
    { 0.00f,  0.00f, 0.22f},  //  3 neck
    { 0.00f,  0.00f, 0.08f},  //  4 head
    {-0.10f,  0.00f, 0.28f},  //  5 L shoulder
    {-0.22f, -0.01f, 0.38f},  //  6 L upper arm
    {-0.32f, -0.02f, 0.48f},  //  7 L lower arm
    {-0.38f, -0.02f, 0.55f},  //  8 L hand
    { 0.10f,  0.00f, 0.28f},  //  9 R shoulder
    { 0.22f,  0.01f, 0.38f},  // 10 R upper arm
    { 0.32f,  0.02f, 0.48f},  // 11 R lower arm
    { 0.38f,  0.02f, 0.55f},  // 12 R hand
    {-0.06f,  0.00f, 0.58f},  // 13 L upper leg
    {-0.06f,  0.01f, 0.76f},  // 14 L lower leg
    {-0.06f,  0.03f, 0.95f},  // 15 L foot
    { 0.06f,  0.00f, 0.58f},  // 16 R upper leg
    { 0.06f,  0.01f, 0.76f},  // 17 R lower leg
    { 0.06f,  0.03f, 0.95f},  // 18 R foot
};

static const int kJointParents[kNumEditJoints] = {
    -1, 0, 1, 2, 3,
     2, 5, 6, 7,
     2, 9, 10, 11,
     0, 13, 14,
     0, 16, 17,
};

static const char* kJointNames[kNumEditJoints] = {
    "hips", "spine", "chest", "neck", "head",
    "left_shoulder", "left_upper_arm", "left_lower_arm", "left_hand",
    "right_shoulder", "right_upper_arm", "right_lower_arm", "right_hand",
    "left_upper_leg", "left_lower_leg", "left_foot",
    "right_upper_leg", "right_lower_leg", "right_foot",
};

// ============================================================================
//  Ctor / Dtor
// ============================================================================

AutoRigPlugin::AutoRigPlugin()  = default;
AutoRigPlugin::~AutoRigPlugin() { shutdown(); }

// ============================================================================
//  IPlugin lifecycle
// ============================================================================

bool AutoRigPlugin::init(
    const std::shared_ptr<engine::renderer::Device>& /*device*/)
{
    rasterizer_      = std::make_unique<SimpleRasterizer>();
    diffusion_model_ = std::make_unique<RigDiffusionModel>();

    // Load diffusion model.  If no .pt file exists yet, the stub runs.
    diffusion_model_->load("assets/models/rig_diffusion.pt", "cpu");

    state_ = PluginState::kLoaded;
    return true;
}

void AutoRigPlugin::shutdown() {
    rasterizer_.reset();
    diffusion_model_.reset();
    state_ = PluginState::kUnloaded;
}

void AutoRigPlugin::reportProgress(int step, int total, const std::string& msg) {
    ui_progress_ = static_cast<float>(step) / std::max(total, 1);
    ui_status_   = msg;
    if (progress_cb_) progress_cb_(step, total, msg);
    fprintf(stderr, "[AutoRig] %d/%d  %s\n", step, total, msg.c_str());
}

// ============================================================================
//  loadMesh – load raw triangle data from a glTF/glb file.
// ============================================================================

bool AutoRigPlugin::loadMesh(const std::string& path) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    bool ok = false;
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".glb") {
        ok = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    } else {
        ok = loader.LoadASCIIFromFile(&model, &err, &warn, path);
    }
    if (!ok) {
        fprintf(stderr, "[AutoRig] loadMesh failed: %s\n", err.c_str());
        return false;
    }

    source_mesh_path_ = path;

    // Check if the model already has skinning data.
    if (!model.skins.empty()) {
        fprintf(stderr, "[AutoRig] model '%s' already has %zu skin(s) — skipping.\n",
            path.c_str(), model.skins.size());
        ui_status_ = "Already skinned";
        state_ = PluginState::kFinished;
        return false;
    }

    mesh_ = TriangleMesh{};

    // ---- Helper: compute a node's LOCAL transform matrix -------------------
    auto nodeLocalMatrix = [](const tinygltf::Node& n) -> glm::mat4 {
        if (!n.matrix.empty()) {
            // Column-major 4x4.
            glm::mat4 m;
            for (int i = 0; i < 16; ++i)
                (&m[0][0])[i] = static_cast<float>(n.matrix[i]);
            return m;
        }
        glm::mat4 T(1.0f), R(1.0f), S(1.0f);
        if (!n.translation.empty())
            T = glm::translate(glm::mat4(1.0f),
                glm::vec3((float)n.translation[0],
                           (float)n.translation[1],
                           (float)n.translation[2]));
        if (!n.rotation.empty()) {
            glm::quat q((float)n.rotation[3],   // w
                        (float)n.rotation[0],    // x
                        (float)n.rotation[1],    // y
                        (float)n.rotation[2]);   // z
            R = glm::mat4_cast(q);
        }
        if (!n.scale.empty())
            S = glm::scale(glm::mat4(1.0f),
                glm::vec3((float)n.scale[0],
                           (float)n.scale[1],
                           (float)n.scale[2]));
        return T * R * S;
    };

    // ---- Compute world transforms for every node (walk the scene tree) ----
    std::vector<glm::mat4> node_world(model.nodes.size(), glm::mat4(1.0f));
    {
        // Recursive lambda via std::function.
        std::function<void(int, const glm::mat4&)> walkNode =
            [&](int idx, const glm::mat4& parent_world) {
            node_world[idx] = parent_world * nodeLocalMatrix(model.nodes[idx]);
            for (int child : model.nodes[idx].children)
                walkNode(child, node_world[idx]);
        };

        // Walk from scene roots (or all root-level nodes).
        std::vector<bool> is_child(model.nodes.size(), false);
        for (auto& n : model.nodes)
            for (int c : n.children) is_child[c] = true;

        for (int i = 0; i < (int)model.nodes.size(); ++i) {
            if (!is_child[i])
                walkNode(i, glm::mat4(1.0f));
        }
    }

    // ---- Build mesh-index → node-world-transform map ----------------------
    //  Find which node owns each mesh so we can apply its world transform.
    std::unordered_map<int, glm::mat4> mesh_to_world;
    for (int i = 0; i < (int)model.nodes.size(); ++i) {
        if (model.nodes[i].mesh >= 0) {
            mesh_to_world[model.nodes[i].mesh] = node_world[i];
        }
    }

    // ---- Store the mesh node's world transform for skinning export ----------
    //  glTF skinning skips the mesh node's own transform, so the inverse bind
    //  matrices must incorporate it.  We use the first mesh node's transform.
    mesh_node_world_transform_ = glm::mat4(1.0f);
    if (mesh_to_world.count(0)) {
        mesh_node_world_transform_ = mesh_to_world[0];
    } else if (!mesh_to_world.empty()) {
        mesh_node_world_transform_ = mesh_to_world.begin()->second;
    }

    // ---- Gather positions and indices from all meshes / primitives ---------
    for (int mi = 0; mi < (int)model.meshes.size(); ++mi) {
        auto& gltf_mesh = model.meshes[mi];
        glm::mat4 world_xform = glm::mat4(1.0f);
        auto it = mesh_to_world.find(mi);
        if (it != mesh_to_world.end()) world_xform = it->second;
        glm::mat3 normal_xform = glm::transpose(glm::inverse(glm::mat3(world_xform)));

        for (auto& prim : gltf_mesh.primitives) {
            if (prim.mode != TINYGLTF_MODE_TRIANGLES &&
                prim.mode != -1 /* default */) continue;

            uint32_t base_vertex = static_cast<uint32_t>(mesh_.positions.size());

            // Positions.
            auto pos_it = prim.attributes.find("POSITION");
            if (pos_it == prim.attributes.end()) continue;
            const auto& pos_acc = model.accessors[pos_it->second];
            const auto& pos_bv  = model.bufferViews[pos_acc.bufferView];
            const auto& pos_buf = model.buffers[pos_bv.buffer];
            const float* pos_ptr = reinterpret_cast<const float*>(
                pos_buf.data.data() + pos_bv.byteOffset + pos_acc.byteOffset);

            size_t stride = pos_bv.byteStride ? pos_bv.byteStride / sizeof(float) : 3;
            for (size_t i = 0; i < pos_acc.count; ++i) {
                glm::vec3 p(pos_ptr[i * stride + 0],
                            pos_ptr[i * stride + 1],
                            pos_ptr[i * stride + 2]);
                // Apply the node world transform so positions are in world space.
                glm::vec4 wp = world_xform * glm::vec4(p, 1.0f);
                mesh_.positions.push_back(glm::vec3(wp));
            }

            // Normals (optional).
            auto nrm_it = prim.attributes.find("NORMAL");
            if (nrm_it != prim.attributes.end()) {
                const auto& nrm_acc = model.accessors[nrm_it->second];
                const auto& nrm_bv  = model.bufferViews[nrm_acc.bufferView];
                const auto& nrm_buf = model.buffers[nrm_bv.buffer];
                const float* nrm_ptr = reinterpret_cast<const float*>(
                    nrm_buf.data.data() + nrm_bv.byteOffset + nrm_acc.byteOffset);
                size_t nstride = nrm_bv.byteStride ? nrm_bv.byteStride / sizeof(float) : 3;
                for (size_t i = 0; i < nrm_acc.count; ++i) {
                    glm::vec3 n(nrm_ptr[i * nstride + 0],
                                nrm_ptr[i * nstride + 1],
                                nrm_ptr[i * nstride + 2]);
                    mesh_.normals.push_back(glm::normalize(normal_xform * n));
                }
            }

            // Texture coordinates (TEXCOORD_0).
            // Must stay aligned with positions — pad with (0,0) if absent.
            auto uv_it = prim.attributes.find("TEXCOORD_0");
            if (uv_it != prim.attributes.end()) {
                const auto& uv_acc = model.accessors[uv_it->second];
                const auto& uv_bv  = model.bufferViews[uv_acc.bufferView];
                const auto& uv_buf = model.buffers[uv_bv.buffer];
                const float* uv_ptr = reinterpret_cast<const float*>(
                    uv_buf.data.data() + uv_bv.byteOffset + uv_acc.byteOffset);
                size_t uvstride = uv_bv.byteStride ? uv_bv.byteStride / sizeof(float) : 2;
                for (size_t i = 0; i < uv_acc.count; ++i) {
                    mesh_.texcoords.push_back(glm::vec2(
                        uv_ptr[i * uvstride + 0],
                        uv_ptr[i * uvstride + 1]));
                }
            } else {
                // Pad so texcoords stays aligned with positions.
                mesh_.texcoords.resize(mesh_.positions.size(), glm::vec2(0.0f));
            }

            // Vertex colors (COLOR_0).
            // Must stay aligned with positions — pad with (1,1,1) if absent.
            auto col_it = prim.attributes.find("COLOR_0");
            if (col_it != prim.attributes.end()) {
                const auto& col_acc = model.accessors[col_it->second];
                const auto& col_bv  = model.bufferViews[col_acc.bufferView];
                const auto& col_buf = model.buffers[col_bv.buffer];
                const uint8_t* col_raw = col_buf.data.data() +
                    col_bv.byteOffset + col_acc.byteOffset;

                int num_components = (col_acc.type == TINYGLTF_TYPE_VEC4) ? 4 : 3;

                for (size_t i = 0; i < col_acc.count; ++i) {
                    glm::vec3 vc(1.0f);
                    if (col_acc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                        const float* fp = reinterpret_cast<const float*>(col_raw);
                        size_t cs = col_bv.byteStride
                            ? col_bv.byteStride / sizeof(float)
                            : (size_t)num_components;
                        vc = glm::vec3(fp[i * cs + 0], fp[i * cs + 1], fp[i * cs + 2]);
                    } else if (col_acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        size_t cs = col_bv.byteStride
                            ? col_bv.byteStride
                            : (size_t)num_components;
                        vc = glm::vec3(
                            col_raw[i * cs + 0] / 255.0f,
                            col_raw[i * cs + 1] / 255.0f,
                            col_raw[i * cs + 2] / 255.0f);
                    } else if (col_acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        const uint16_t* sp = reinterpret_cast<const uint16_t*>(col_raw);
                        size_t cs = col_bv.byteStride
                            ? col_bv.byteStride / sizeof(uint16_t)
                            : (size_t)num_components;
                        vc = glm::vec3(
                            sp[i * cs + 0] / 65535.0f,
                            sp[i * cs + 1] / 65535.0f,
                            sp[i * cs + 2] / 65535.0f);
                    }
                    mesh_.vertex_colors.push_back(vc);
                }
            }

            // ── Bake per-primitive diffuse color into vertex colors ──
            //  Each primitive may have its own texture. Rather than supporting
            //  multi-texture in the rasterizer, we sample the texture at each
            //  vertex's UV and store the result as vertex colors.
            if (prim.material >= 0 && prim.material < (int)model.materials.size()) {
                const auto& mat = model.materials[prim.material];
                const auto& pbr = mat.pbrMetallicRoughness;

                // Find diffuse texture index.
                int texIdx = pbr.baseColorTexture.index;

                // Fallback: KHR_materials_pbrSpecularGlossiness diffuseTexture
                if (texIdx < 0) {
                    auto ext_it = mat.extensions.find("KHR_materials_pbrSpecularGlossiness");
                    if (ext_it != mat.extensions.end() && ext_it->second.Has("diffuseTexture")) {
                        auto& dt = ext_it->second.Get("diffuseTexture");
                        if (dt.Has("index"))
                            texIdx = dt.Get("index").GetNumberAsInt();
                    }
                }

                // Find diffuse factor (solid tint).
                glm::vec3 diffuse_factor(1.0f);
                if (pbr.baseColorFactor.size() >= 3) {
                    diffuse_factor = glm::vec3(
                        (float)pbr.baseColorFactor[0],
                        (float)pbr.baseColorFactor[1],
                        (float)pbr.baseColorFactor[2]);
                } else {
                    // Try specular-glossiness diffuseFactor
                    auto ext_it = mat.extensions.find("KHR_materials_pbrSpecularGlossiness");
                    if (ext_it != mat.extensions.end() && ext_it->second.Has("diffuseFactor")) {
                        auto& df = ext_it->second.Get("diffuseFactor");
                        if (df.ArrayLen() >= 3) {
                            diffuse_factor = glm::vec3(
                                (float)df.Get(0).GetNumberAsDouble(),
                                (float)df.Get(1).GetNumberAsDouble(),
                                (float)df.Get(2).GetNumberAsDouble());
                        }
                    }
                }

                // Decode texture image (if any) into a temporary SimpleTexture.
                SimpleTexture prim_tex;
                if (texIdx >= 0 && texIdx < (int)model.textures.size()) {
                    int imgIdx = model.textures[texIdx].source;
                    if (imgIdx >= 0 && imgIdx < (int)model.images.size()) {
                        const auto& img = model.images[imgIdx];
                        if (img.width > 0 && img.height > 0 &&
                            img.component >= 3 && !img.image.empty()) {
                            prim_tex.width    = img.width;
                            prim_tex.height   = img.height;
                            prim_tex.channels = 3;
                            if (img.component == 3) {
                                prim_tex.pixels = img.image;
                            } else {
                                prim_tex.pixels.resize((size_t)img.width * img.height * 3);
                                for (int p = 0; p < img.width * img.height; ++p) {
                                    prim_tex.pixels[p * 3 + 0] = img.image[p * img.component + 0];
                                    prim_tex.pixels[p * 3 + 1] = img.image[p * img.component + 1];
                                    prim_tex.pixels[p * 3 + 2] = img.image[p * img.component + 2];
                                }
                            }
                            fprintf(stderr, "[AutoRig]   material '%s': loaded diffuse tex %dx%d\n",
                                mat.name.c_str(), img.width, img.height);
                        }
                    }
                }

                // Ensure vertex_colors array is padded up to base_vertex.
                if (mesh_.vertex_colors.size() < base_vertex)
                    mesh_.vertex_colors.resize(base_vertex, glm::vec3(1.0f));

                // Bake: for each vertex in this primitive, sample the texture
                // at its UV and store as vertex color.
                size_t prim_vert_count = mesh_.positions.size() - base_vertex;
                bool has_prim_uvs = mesh_.texcoords.size() >= mesh_.positions.size();

                for (size_t i = 0; i < prim_vert_count; ++i) {
                    glm::vec3 col = diffuse_factor;
                    if (!prim_tex.empty() && has_prim_uvs) {
                        glm::vec2 uv = mesh_.texcoords[base_vertex + i];
                        col *= prim_tex.sample(uv);
                    }
                    mesh_.vertex_colors.push_back(col);
                }
            }

            // Indices.
            if (prim.indices >= 0) {
                const auto& idx_acc = model.accessors[prim.indices];
                const auto& idx_bv  = model.bufferViews[idx_acc.bufferView];
                const auto& idx_buf = model.buffers[idx_bv.buffer];
                const uint8_t* raw  = idx_buf.data.data() +
                    idx_bv.byteOffset + idx_acc.byteOffset;

                for (size_t i = 0; i < idx_acc.count; ++i) {
                    uint32_t idx = 0;
                    switch (idx_acc.componentType) {
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                            idx = reinterpret_cast<const uint16_t*>(raw)[i]; break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                            idx = reinterpret_cast<const uint32_t*>(raw)[i]; break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                            idx = raw[i]; break;
                        default: break;
                    }
                    mesh_.indices.push_back(base_vertex + idx);
                }
            } else {
                // Non-indexed: sequential triangles.
                for (uint32_t i = 0; i < (uint32_t)pos_acc.count; ++i) {
                    mesh_.indices.push_back(base_vertex + i);
                }
            }
        }
    }

    mesh_.recomputeBounds();
    if (mesh_.normals.size() != mesh_.positions.size()) {
        mesh_.recomputeNormals();
    }

    // ---- Ensure texcoords & vertex_colors are either empty or fully aligned ----
    //  Partial arrays cause out-of-bounds in the rasterizer.
    const size_t nv = mesh_.positions.size();
    if (!mesh_.texcoords.empty() && mesh_.texcoords.size() != nv) {
        mesh_.texcoords.resize(nv, glm::vec2(0.0f));
    }
    if (!mesh_.vertex_colors.empty() && mesh_.vertex_colors.size() != nv) {
        mesh_.vertex_colors.resize(nv, glm::vec3(1.0f));
    }

    size_t total_tris = mesh_.indices.size() / 3;
    fprintf(stderr, "[AutoRig] loaded mesh: %zu verts, %zu tris\n",
        mesh_.positions.size(), total_tris);
    fprintf(stderr, "[AutoRig]   UVs: %zu/%zu, vertex colors: %zu/%zu, base texture: %s\n",
        mesh_.texcoords.size(), nv,
        mesh_.vertex_colors.size(), nv,
        mesh_.base_color_texture.empty() ? "no" : "yes");

    // ---- Remove floor/base geometry -----------------------------------------
    //  Detect flat triangles at the bottom of the bounding box whose normals
    //  point mostly upward (Y+).  These are ground planes / pedestals that
    //  confuse the auto-rig silhouette analysis.
    {
        float floor_y = mesh_.bbox_min.y;
        float bbox_h  = mesh_.bbox_max.y - mesh_.bbox_min.y;
        float y_tol   = bbox_h * 0.02f;  // within 2% of bottom = floor

        std::vector<uint32_t> kept_indices;
        kept_indices.reserve(mesh_.indices.size());

        size_t removed = 0;
        for (size_t t = 0; t < total_tris; ++t) {
            uint32_t i0 = mesh_.indices[t * 3 + 0];
            uint32_t i1 = mesh_.indices[t * 3 + 1];
            uint32_t i2 = mesh_.indices[t * 3 + 2];

            const glm::vec3& p0 = mesh_.positions[i0];
            const glm::vec3& p1 = mesh_.positions[i1];
            const glm::vec3& p2 = mesh_.positions[i2];

            // Check if all 3 vertices are near the floor
            bool at_floor = (p0.y - floor_y < y_tol) &&
                            (p1.y - floor_y < y_tol) &&
                            (p2.y - floor_y < y_tol);

            if (at_floor) {
                // Also check if the triangle normal is roughly upward
                glm::vec3 fn = glm::cross(p1 - p0, p2 - p0);
                float len = glm::length(fn);
                if (len > 1e-8f) {
                    fn /= len;
                    if (std::abs(fn.y) > 0.7f) {
                        // Flat floor triangle — skip it
                        ++removed;
                        continue;
                    }
                }
            }

            kept_indices.push_back(i0);
            kept_indices.push_back(i1);
            kept_indices.push_back(i2);
        }

        if (removed > 0) {
            mesh_.indices = std::move(kept_indices);
            mesh_.recomputeBounds();
            fprintf(stderr, "[AutoRig] removed %zu floor triangles, %zu remain\n",
                removed, mesh_.indices.size() / 3);
        }
    }

    return !mesh_.empty();
}

// ============================================================================
//  captureViews
// ============================================================================

bool AutoRigPlugin::captureViews(int num_views, int resolution) {
    if (mesh_.empty()) return false;
    num_views_ = num_views;
    capture_resolution_ = resolution;
    captures_ = rasterizer_->captureOrbit(mesh_, num_views, resolution);
    fprintf(stderr, "[AutoRig] captured %d views @ %dx%d\n",
        (int)captures_.size(), resolution, resolution);
    return !captures_.empty();
}

// ============================================================================
//  initEditableJointsForView – place heuristic joints on a single capture
// ============================================================================

void AutoRigPlugin::initEditableJointsForView(
    ViewEditState& state, const ViewCapture& cap)
{
    int W = cap.width, H = cap.height;
    state.joints.resize(kNumEditJoints);
    state.any_edited = false;
    state.azimuth_deg = cap.azimuth_deg;
    state.elevation_deg = cap.elevation_deg;

    // Silhouette bounding box
    int sil_min_x = W, sil_max_x = 0, sil_min_y = H, sil_max_y = 0;
    if (!cap.silhouette.empty()) {
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                if (cap.silhouette[y * W + x]) {
                    sil_min_x = std::min(sil_min_x, x);
                    sil_max_x = std::max(sil_max_x, x);
                    sil_min_y = std::min(sil_min_y, y);
                    sil_max_y = std::max(sil_max_y, y);
                }
    }

    float sil_cx = (sil_min_x + sil_max_x) * 0.5f;
    float sil_h  = (float)(sil_max_y - sil_min_y);
    if (sil_h < 1.0f) sil_h = 1.0f;

    float az_rad = cap.azimuth_deg * 3.14159265f / 180.0f;
    float ca = std::cos(az_rad), sa = std::sin(az_rad);

    for (int j = 0; j < kNumEditJoints; ++j) {
        float ax = kJointDefs[j].lr * ca + kJointDefs[j].fb * sa;
        float px = sil_cx + ax * sil_h;
        float py = sil_min_y + kJointDefs[j].ry * sil_h;
        state.joints[j].uv = glm::clamp(
            glm::vec2((px + 0.5f) / W, (py + 0.5f) / H),
            glm::vec2(0.0f), glm::vec2(1.0f));
        state.joints[j].edited = false;
    }
}

// ============================================================================
//  initEditableJoints – initialize heuristic joints for all capture views
// ============================================================================

void AutoRigPlugin::initEditableJoints() {
    view_edits_.clear();
    view_edits_.resize(captures_.size());
    for (int v = 0; v < (int)captures_.size(); ++v)
        initEditableJointsForView(view_edits_[v], captures_[v]);
    drag_joint_ = -1;
    fprintf(stderr, "[AutoRig] initialized editable joints for %d views\n",
        (int)view_edits_.size());
}

// ============================================================================
//  predictJoints
// ============================================================================

bool AutoRigPlugin::predictJoints() {
    if (captures_.empty()) {
        fprintf(stderr, "[AutoRig] predictJoints: no captures available\n");
        return false;
    }
    if (!diffusion_model_) {
        fprintf(stderr, "[AutoRig] predictJoints: diffusion_model_ is null\n");
        return false;
    }
    if (!diffusion_model_->isLoaded()) {
        fprintf(stderr, "[AutoRig] predictJoints: model not loaded\n");
        return false;
    }

    fprintf(stderr, "[AutoRig] predictJoints: %d captures, each %dx%d, model has %d joints\n",
        (int)captures_.size(),
        captures_[0].width, captures_[0].height,
        diffusion_model_->numJoints());

    view_predictions_ = diffusion_model_->predictBatch(captures_);

    fprintf(stderr, "[AutoRig] predictJoints: got %d view predictions\n",
        (int)view_predictions_.size());

    // Log per-view summary
    for (int v = 0; v < (int)view_predictions_.size(); ++v) {
        const auto& vp = view_predictions_[v];
        fprintf(stderr, "[AutoRig]   view %d: %d joints, %d bones\n",
            vp.view_idx, (int)vp.joints.size(), (int)vp.bones.size());
        for (int j = 0; j < (int)vp.joints.size(); ++j) {
            const auto& jt = vp.joints[j];
            fprintf(stderr, "[AutoRig]     [%2d] %-18s  uv=(%.3f, %.3f)  conf=%.4f\n",
                j, jt.name.c_str(), jt.peak_uv.x, jt.peak_uv.y, jt.confidence);
        }
    }

    return !view_predictions_.empty();
}

// ============================================================================
//  fuseAndBuildSkeleton – back-project 2D predictions into 3D, then fuse.
// ============================================================================

bool AutoRigPlugin::fuseAndBuildSkeleton() {
    if (view_predictions_.empty()) return false;

    const auto& names   = getStandardJointNames();
    const auto& parents = getStandardJointParents();
    int J = diffusion_model_->numJoints();

    // For each joint, accumulate weighted 3D positions from all views.
    struct JointAccum {
        glm::vec3 pos_sum{0.0f};
        float     weight_sum = 0.0f;
    };
    std::vector<JointAccum> accum(J);

    for (auto& vp : view_predictions_) {
        if (vp.view_idx < 0 || vp.view_idx >= (int)captures_.size()) continue;
        const auto& cap = captures_[vp.view_idx];

        glm::mat4 inv_vp = glm::inverse(cap.view_proj);

        for (int j = 0; j < (int)vp.joints.size() && j < J; ++j) {
            auto& jh = vp.joints[j];
            if (jh.confidence < 0.05f) continue;

            // UV -> NDC.
            float ndc_x = jh.peak_uv.x * 2.0f - 1.0f;
            float ndc_y = jh.peak_uv.y * 2.0f - 1.0f;

            // Sample depth from the depth buffer at the peak location.
            int px = std::clamp((int)(jh.peak_uv.x * cap.width),  0, cap.width  - 1);
            int py = std::clamp((int)(jh.peak_uv.y * cap.height), 0, cap.height - 1);
            float depth_val = cap.depth[py * cap.width + px];

            // If the peak falls on background (depth == 1.0), search locally.
            if (depth_val >= 0.999f) {
                float best_d = 1.0f;
                int search_r = cap.width / 8;
                for (int dy = -search_r; dy <= search_r; ++dy) {
                    for (int dx = -search_r; dx <= search_r; ++dx) {
                        int sx = std::clamp(px + dx, 0, cap.width  - 1);
                        int sy = std::clamp(py + dy, 0, cap.height - 1);
                        float d = cap.depth[sy * cap.width + sx];
                        if (d < best_d) {
                            best_d = d;
                            // Also update NDC x/y to this valid pixel.
                            ndc_x = (sx + 0.5f) / cap.width  * 2.0f - 1.0f;
                            ndc_y = (sy + 0.5f) / cap.height * 2.0f - 1.0f;
                        }
                    }
                }
                depth_val = best_d;
            }

            // NDC depth: [0,1] mapped to [-1,1] for back-projection.
            float ndc_z = depth_val * 2.0f - 1.0f;

            glm::vec4 clip_pos = inv_vp * glm::vec4(ndc_x, ndc_y, ndc_z, 1.0f);
            glm::vec3 world_pos = glm::vec3(clip_pos) / clip_pos.w;

            float w = jh.confidence;
            accum[j].pos_sum    += world_pos * w;
            accum[j].weight_sum += w;
        }
    }

    // Build skeleton.
    skeleton_ = Skeleton{};
    skeleton_.root = 0;
    skeleton_.joints.resize(J);

    // Compute default humanoid joint positions from actual mesh vertex
    // distribution.  We analyse vertex positions at each height slice to
    // find where the mesh body actually is, rather than using bounding box
    // ratios that fail for non-T-pose characters.
    //
    // Axis convention: glTF spec mandates Y-up after node transforms are
    // applied.  We hardcode Y=up, X=right, Z=forward.  Do NOT auto-detect
    // from bounding box extents — that breaks for T-pose characters where
    // arm span (X) exceeds standing height (Y).
    glm::vec3 bmin = mesh_.bbox_min;
    glm::vec3 bmax = mesh_.bbox_max;
    glm::vec3 center = (bmin + bmax) * 0.5f;
    glm::vec3 extent = bmax - bmin;

    constexpr int up_ax    = 1;   // Y = up   (glTF standard)
    constexpr int right_ax = 0;   // X = right
    constexpr int fwd_ax   = 2;   // Z = forward

    float height = extent[up_ax];
    if (height < 1e-4f) height = 1.7f;
    float up_base = bmin[up_ax];

    detected_up_axis_ = up_ax;   // always 1 (Y-up per glTF spec)

    fprintf(stderr, "[AutoRig] bbox min=(%.3f, %.3f, %.3f) max=(%.3f, %.3f, %.3f)\n",
        bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z);
    fprintf(stderr, "[AutoRig] extent=(%.3f, %.3f, %.3f)  height=%.3f\n",
        extent.x, extent.y, extent.z, height);

    // ---- Build height-slice profile of the mesh geometry --------------------
    //  For several normalised height bands (0..1 of the character height),
    //  compute: median of the right-axis, min/max of right-axis, median of
    //  fwd-axis.  This tells us where the mesh body actually is at each
    //  height, so we can place arm, leg, and spine joints accurately.
    constexpr int kSlices = 20;
    struct Slice {
        float right_min =  1e9f;
        float right_max = -1e9f;
        float right_sum = 0.0f;
        float fwd_sum   = 0.0f;
        int   count     = 0;
    };
    Slice slices[kSlices];

    for (auto& p : mesh_.positions) {
        float t = (p[up_ax] - up_base) / height;        // normalised height [0..1]
        int si = std::clamp(static_cast<int>(t * kSlices), 0, kSlices - 1);
        slices[si].right_min = std::min(slices[si].right_min, p[right_ax]);
        slices[si].right_max = std::max(slices[si].right_max, p[right_ax]);
        slices[si].right_sum += p[right_ax];
        slices[si].fwd_sum   += p[fwd_ax];
        slices[si].count++;
    }

    // Compute per-slice centres and half-widths.
    auto sliceCenter = [&](int si, int ax) -> float {
        if (slices[si].count == 0) return center[ax];
        if (ax == right_ax) return slices[si].right_sum / slices[si].count;
        return slices[si].fwd_sum / slices[si].count;
    };
    auto sliceHalfWidth = [&](int si) -> float {
        if (slices[si].count == 0) return extent[right_ax] * 0.5f;
        return (slices[si].right_max - slices[si].right_min) * 0.5f;
    };
    auto sliceRightMin = [&](int si) -> float {
        return slices[si].count ? slices[si].right_min : center[right_ax];
    };
    auto sliceRightMax = [&](int si) -> float {
        return slices[si].count ? slices[si].right_max : center[right_ax];
    };

    // Height-to-slice-index helper.
    auto hSlice = [&](float norm_h) -> int {
        return std::clamp(static_cast<int>(norm_h * kSlices), 0, kSlices - 1);
    };

    // ---- Compute mesh-adaptive joint positions ------------------------------
    //  Spine / torso: centroid at each height.
    //  Arms: actual left/right extent at shoulder height, interpolated
    //         down toward the body for non-T-pose models.
    //  Legs: centroid ± small offset from hip-height slice.

    // Torso heights (normalised).
    float h_hips  = 0.53f, h_spine = 0.60f, h_chest  = 0.70f;
    float h_neck  = 0.82f, h_head  = 0.91f, h_shoulder = 0.78f;

    // Arm placement: sample the mesh at shoulder height to get actual extent.
    int si_shoulder = hSlice(h_shoulder);
    float shoulder_cx_r = sliceCenter(si_shoulder, right_ax);
    float shoulder_cx_f = sliceCenter(si_shoulder, fwd_ax);
    float shoulder_left  = sliceRightMin(si_shoulder);   // leftmost vertex at shoulder height
    float shoulder_right = sliceRightMax(si_shoulder);    // rightmost vertex at shoulder height
    float shoulder_hw    = (shoulder_right - shoulder_left) * 0.5f;
    float shoulder_mid   = (shoulder_right + shoulder_left) * 0.5f;

    // Ensure minimum arm spread (at least 20% of character height).
    shoulder_hw = std::max(shoulder_hw, height * 0.20f);

    // Leg placement: sample at hip height.
    int si_hips = hSlice(h_hips);
    float hip_cx_r = sliceCenter(si_hips, right_ax);
    float hip_cx_f = sliceCenter(si_hips, fwd_ax);
    float hip_hw   = sliceHalfWidth(si_hips);
    float hip_gap  = std::max(hip_hw * 0.35f, height * 0.04f);  // leg separation

    fprintf(stderr, "[AutoRig] bbox extent=(%.3f, %.3f, %.3f)  "
        "up=%d right=%d fwd=%d  height=%.3f\n",
        extent.x, extent.y, extent.z, up_ax, right_ax, fwd_ax, height);
    fprintf(stderr, "[AutoRig] shoulder slice: left=%.3f right=%.3f hw=%.3f mid=%.3f\n",
        shoulder_left, shoulder_right, shoulder_hw, shoulder_mid);
    fprintf(stderr, "[AutoRig] hip slice: cx_r=%.3f hw=%.3f gap=%.3f\n",
        hip_cx_r, hip_hw, hip_gap);

    // Helper: build a vec3 from per-axis values.
    auto makePos = [&](float up_val, float right_val, float fwd_val) -> glm::vec3 {
        glm::vec3 p;
        p[up_ax]    = up_val;
        p[right_ax] = right_val;
        p[fwd_ax]   = fwd_val;
        return p;
    };

    // Arm joints: interpolate from shoulder centre to the mesh edge.
    // shoulder (30%), upper_arm (55%), lower_arm (80%), hand (100% of hw).
    auto armPos = [&](float norm_h, float spread_frac, bool is_left) -> glm::vec3 {
        // For non-T-pose: arm joints below shoulder can droop.  Sample the
        // actual mesh extent at that height and blend with the shoulder extent.
        int si = hSlice(norm_h);
        float local_left  = sliceRightMin(si);
        float local_right = sliceRightMax(si);
        float local_mid   = (local_left + local_right) * 0.5f;
        float local_hw    = (local_right - local_left) * 0.5f;

        // Blend: mainly use shoulder extent for upper joints, local extent
        // for lower joints (hand).  This handles A-pose gracefully.
        float blend = spread_frac;  // 0.3→mostly shoulder, 1.0→mostly local
        float eff_hw  = shoulder_hw * (1.0f - blend * 0.3f) + local_hw * (blend * 0.3f);
        float eff_mid = shoulder_mid * (1.0f - blend * 0.3f) + local_mid * (blend * 0.3f);
        float eff_fwd = shoulder_cx_f;

        float pos_r = is_left
            ? eff_mid - eff_hw * spread_frac
            : eff_mid + eff_hw * spread_frac;
        return makePos(up_base + height * norm_h, pos_r, eff_fwd);
    };

    // Leg heights (normalised) — sample mesh at each height for fwd centering.
    auto legPos = [&](float norm_h, bool is_left) -> glm::vec3 {
        int si = hSlice(norm_h);
        float cx_f = sliceCenter(si, fwd_ax);
        float pos_r = is_left ? hip_cx_r - hip_gap : hip_cx_r + hip_gap;
        return makePos(up_base + height * norm_h, pos_r, cx_f);
    };

    auto spinePos = [&](float norm_h) -> glm::vec3 {
        int si = hSlice(norm_h);
        return makePos(up_base + height * norm_h,
                       sliceCenter(si, right_ax),
                       sliceCenter(si, fwd_ax));
    };

    glm::vec3 default_pos[19] = {
        /* 0  hips            */ spinePos(h_hips),
        /* 1  spine           */ spinePos(h_spine),
        /* 2  chest           */ spinePos(h_chest),
        /* 3  neck            */ spinePos(h_neck),
        /* 4  head            */ spinePos(h_head),
        /* 5  left_shoulder   */ armPos(h_shoulder, 0.30f, true),
        /* 6  left_upper_arm  */ armPos(h_shoulder, 0.55f, true),
        /* 7  left_lower_arm  */ armPos(h_shoulder * 0.97f, 0.80f, true),
        /* 8  left_hand       */ armPos(h_shoulder * 0.94f, 1.00f, true),
        /* 9  right_shoulder  */ armPos(h_shoulder, 0.30f, false),
        /* 10 right_upper_arm */ armPos(h_shoulder, 0.55f, false),
        /* 11 right_lower_arm */ armPos(h_shoulder * 0.97f, 0.80f, false),
        /* 12 right_hand      */ armPos(h_shoulder * 0.94f, 1.00f, false),
        /* 13 left_upper_leg  */ legPos(0.50f, true),
        /* 14 left_lower_leg  */ legPos(0.26f, true),
        /* 15 left_foot       */ legPos(0.03f, true),
        /* 16 right_upper_leg */ legPos(0.50f, false),
        /* 17 right_lower_leg */ legPos(0.26f, false),
        /* 18 right_foot      */ legPos(0.03f, false),
    };

    for (int j = 0; j < J; ++j) {
        Joint& jt = skeleton_.joints[j];
        jt.name   = (j < (int)names.size()) ? names[j] : ("joint_" + std::to_string(j));
        jt.parent = (j < (int)parents.size()) ? parents[j] : -1;

        if (accum[j].weight_sum > 1e-6f) {
            jt.position = accum[j].pos_sum / accum[j].weight_sum;
        } else if (j < 19) {
            // Fallback: anatomically correct default position.
            jt.position = default_pos[j];
        } else {
            jt.position = center;
        }
    }

    // Compute inverse bind matrices incorporating the mesh node's world
    // transform.  In glTF, the skinning equation is:
    //
    //   skinned = sum( weight_j * jointWorldMatrix_j * IBM_j * vertex )
    //
    // where `vertex` is in mesh-node-local space and the mesh node's own
    // transform is SKIPPED for skinned meshes.  To reproduce the unskinned
    // appearance at bind pose we need:
    //
    //   jointWorld * IBM * vertex = meshNodeWorld * vertex
    //   => IBM = inverse(jointWorld) * meshNodeWorld
    //
    // Our joint positions are already in world space, so jointWorld at bind
    // pose is translate(joint.position).
    for (auto& j : skeleton_.joints) {
        glm::mat4 joint_world = glm::translate(glm::mat4(1.0f), j.position);
        j.inverse_bind_matrix = glm::inverse(joint_world) * mesh_node_world_transform_;
    }

    fprintf(stderr, "[AutoRig] skeleton: %d joints, root=%d\n",
        (int)skeleton_.joints.size(), skeleton_.root);
    for (int j = 0; j < (int)skeleton_.joints.size(); ++j) {
        auto& jt = skeleton_.joints[j];
        fprintf(stderr, "  [%2d] %-18s pos=(%.3f, %.3f, %.3f)  parent=%d\n",
            j, jt.name.c_str(), jt.position.x, jt.position.y, jt.position.z, jt.parent);
    }
    return true;
}

// ============================================================================
//  computeSkinWeights – nearest-bone distance-based skinning.
// ============================================================================

bool AutoRigPlugin::computeSkinWeights() {
    if (skeleton_.empty() || mesh_.empty()) return false;

    int nv = static_cast<int>(mesh_.positions.size());
    int nj = static_cast<int>(skeleton_.joints.size());
    skin_weights_ = SkinWeights{};
    skin_weights_.per_vertex.resize(nv);

    // Pre-compute bone segments (parent -> child).
    struct BoneSeg {
        int joint_idx;
        glm::vec3 a, b;   // a = parent pos, b = child pos
    };
    std::vector<BoneSeg> bones;
    for (int j = 0; j < nj; ++j) {
        if (skeleton_.joints[j].parent >= 0) {
            BoneSeg seg;
            seg.joint_idx = j;
            seg.a = skeleton_.joints[skeleton_.joints[j].parent].position;
            seg.b = skeleton_.joints[j].position;
            bones.push_back(seg);
        }
    }

    // For each vertex, find the 4 closest bones and compute weights.
    for (int v = 0; v < nv; ++v) {
        const glm::vec3& p = mesh_.positions[v];

        struct BoneDist { int joint; float dist; };
        std::vector<BoneDist> dists;
        dists.reserve(bones.size());

        for (auto& bone : bones) {
            // Closest point on segment a-b to point p.
            glm::vec3 ab = bone.b - bone.a;
            float len2 = glm::dot(ab, ab);
            float t = (len2 > 1e-8f)
                ? std::clamp(glm::dot(p - bone.a, ab) / len2, 0.0f, 1.0f)
                : 0.0f;
            glm::vec3 closest = bone.a + t * ab;
            float dist = glm::length(p - closest);
            dists.push_back({bone.joint_idx, dist});
        }

        // Sort by distance, take top 4.
        std::sort(dists.begin(), dists.end(),
            [](const BoneDist& a, const BoneDist& b) { return a.dist < b.dist; });

        auto& vsd = skin_weights_.per_vertex[v];
        float total_w = 0.0f;
        int count = std::min(4, (int)dists.size());

        for (int i = 0; i < count; ++i) {
            vsd.joint_indices[i] = dists[i].joint;
            // Inverse-distance weighting with a small epsilon.
            float w = 1.0f / (dists[i].dist + 0.001f);
            vsd.weights[i] = w;
            total_w += w;
        }

        // Normalise.
        if (total_w > 1e-8f) {
            for (int i = 0; i < 4; ++i) vsd.weights[i] /= total_w;
        }
    }

    fprintf(stderr, "[AutoRig] skinning: %d vertices weighted to %d bones\n",
        nv, (int)bones.size());
    return true;
}

// ============================================================================
//  Skeleton::computeInverseBindMatrices
//
//  Default implementation — no mesh node transform compensation.
//  AutoRigPlugin overrides this via computeInverseBindMatricesWithMeshTransform.
// ============================================================================

void Skeleton::computeInverseBindMatrices() {
    for (auto& j : joints) {
        glm::mat4 bind = glm::translate(glm::mat4(1.0f), j.position);
        j.inverse_bind_matrix = glm::inverse(bind);
    }
}

// ============================================================================
//  exportTrainingData – save edited views + joints as training dataset
// ============================================================================

bool AutoRigPlugin::exportTrainingData(const std::string& output_dir) {
    // Use the output file stem (e.g. "knight-skinned") as the base name
    // so every file is  <base>_view<N>_color.png, etc.
    std::string base_name = std::filesystem::path(output_path_buf_).stem().string();
    if (base_name.empty())
        base_name = std::filesystem::path(source_mesh_path_).stem().string();
    std::string data_dir = output_dir + "/" + base_name;

    // Create directory structure
    try {
        std::filesystem::create_directories(data_dir);
    } catch (const std::exception& e) {
        fprintf(stderr, "[AutoRig] Failed to create training data directory: %s\n", e.what());
        return false;
    }

    int views_saved = 0;

    // Process each saved edit view
    for (int v = 0; v < (int)saved_edits_.size(); ++v) {
        const auto& cap = saved_edits_[v].capture;
        const auto& edit_state = saved_edits_[v].edit;
        if (!edit_state.any_edited) continue;
        int W = cap.width;
        int H = cap.height;

        char filename_buf[512];

        // ---- Save color image ----
        if (!cap.color.empty()) {
            std::snprintf(filename_buf, sizeof(filename_buf),
                "%s/%s_view%d_color.png", data_dir.c_str(), base_name.c_str(), v);
            if (!stbi_write_png(filename_buf, W, H, 3,
                    cap.color.data(), W * 3)) {
                fprintf(stderr, "[AutoRig] Failed to write %s\n", filename_buf);
            }
        }

        // ---- Save silhouette ----
        if (!cap.silhouette.empty()) {
            std::snprintf(filename_buf, sizeof(filename_buf),
                "%s/%s_view%d_silhouette.png", data_dir.c_str(), base_name.c_str(), v);
            if (!stbi_write_png(filename_buf, W, H, 1,
                    cap.silhouette.data(), W)) {
                fprintf(stderr, "[AutoRig] Failed to write %s\n", filename_buf);
            }
        }

        // ---- Generate combined joint heatmap (all joints in one image) ----
        //  Single-channel image: each joint contributes a Gaussian blob,
        //  max-pooled into one heatmap for visualization.  The per-joint
        //  UV coordinates in the JSON are the actual training labels.
        {
            std::vector<uint8_t> heatmap(W * H, 0);
            float sigma_px = 0.02f * W;  // 2% of width
            float sigma_sq = sigma_px * sigma_px;
            int gauss_radius = static_cast<int>(3.0f * sigma_px);

            for (int j = 0; j < kNumEditJoints; ++j) {
                const auto& joint = edit_state.joints[j];
                int cx = static_cast<int>(joint.uv.x * W);
                int cy = static_cast<int>(joint.uv.y * H);

                for (int dy = -gauss_radius; dy <= gauss_radius; ++dy) {
                    for (int dx = -gauss_radius; dx <= gauss_radius; ++dx) {
                        int x = cx + dx;
                        int y = cy + dy;
                        if (x < 0 || x >= W || y < 0 || y >= H) continue;

                        float dist_sq = (float)(dx * dx + dy * dy);
                        float gaussian = 255.0f * std::exp(-dist_sq / (2.0f * sigma_sq));
                        int idx = y * W + x;
                        uint8_t val = static_cast<uint8_t>(
                            std::clamp(gaussian, 0.0f, 255.0f));
                        heatmap[idx] = std::max(heatmap[idx], val);
                    }
                }
            }

            std::snprintf(filename_buf, sizeof(filename_buf),
                "%s/%s_view%d_heatmap.png", data_dir.c_str(), base_name.c_str(), v);
            if (!stbi_write_png(filename_buf, W, H, 1,
                    heatmap.data(), W)) {
                fprintf(stderr, "[AutoRig] Failed to write %s\n", filename_buf);
            }
        }

        // ---- Save metadata JSON ----
        std::snprintf(filename_buf, sizeof(filename_buf),
            "%s/%s_view%d_meta.json", data_dir.c_str(), base_name.c_str(), v);
        FILE* meta_file = std::fopen(filename_buf, "w");
        if (!meta_file) {
            fprintf(stderr, "[AutoRig] Failed to open %s for writing\n", filename_buf);
            continue;
        }

        // Write JSON metadata
        fprintf(meta_file, "{\n");
        fprintf(meta_file, "  \"view_index\": %d,\n", v);
        fprintf(meta_file, "  \"azimuth_deg\": %.1f,\n", cap.azimuth_deg);
        fprintf(meta_file, "  \"elevation_deg\": %.1f,\n", cap.elevation_deg);
        fprintf(meta_file, "  \"width\": %d,\n", W);
        fprintf(meta_file, "  \"height\": %d,\n", H);
        fprintf(meta_file, "  \"joints\": [\n");

        for (int j = 0; j < kNumEditJoints; ++j) {
            const auto& joint = edit_state.joints[j];
            fprintf(meta_file, "    {\n");
            fprintf(meta_file, "      \"name\": \"%s\",\n", kJointNames[j]);
            fprintf(meta_file, "      \"uv\": [%.6f, %.6f],\n",
                joint.uv.x, joint.uv.y);
            fprintf(meta_file, "      \"edited\": %s\n",
                joint.edited ? "true" : "false");
            fprintf(meta_file, "    }%s\n", (j < kNumEditJoints - 1) ? "," : "");
        }

        fprintf(meta_file, "  ],\n");
        fprintf(meta_file, "  \"parents\": [");
        for (int j = 0; j < kNumEditJoints; ++j) {
            fprintf(meta_file, "%d%s",
                kJointParents[j], (j < kNumEditJoints - 1) ? ", " : "");
        }
        fprintf(meta_file, "]\n");
        fprintf(meta_file, "}\n");
        std::fclose(meta_file);

        ++views_saved;
    }

    fprintf(stderr, "[AutoRig] Exported training data for %d edited views to %s\n",
        views_saved, data_dir.c_str());

    return views_saved > 0;
}

// ============================================================================
//  exportGltf – reload original model, inject skeleton + skin, re-save.
//
//  This preserves all original data (textures, materials, UVs, normals,
//  animations, etc.) and only adds: joint nodes, a skin, JOINTS_0 /
//  WEIGHTS_0 attributes, and inverse-bind-matrix data.
// ============================================================================

bool AutoRigPlugin::exportGltf(const std::string& output_path) {
    if (skeleton_.empty() || mesh_.empty()) return false;

    // ---- Reload the original model so textures/materials are intact --------
    tinygltf::Model out;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    bool loaded = false;
    if (source_mesh_path_.size() >= 4 &&
        source_mesh_path_.substr(source_mesh_path_.size() - 4) == ".glb") {
        loaded = loader.LoadBinaryFromFile(&out, &err, &warn, source_mesh_path_);
    } else {
        loaded = loader.LoadASCIIFromFile(&out, &err, &warn, source_mesh_path_);
    }
    if (!loaded) {
        fprintf(stderr, "[AutoRig] exportGltf: failed to reload source: %s\n",
            err.c_str());
        return false;
    }

    fprintf(stderr, "[AutoRig] source model: %zu images, %zu textures, "
        "%zu materials, %zu meshes, %zu buffers, %zu bufferViews\n",
        out.images.size(), out.textures.size(), out.materials.size(),
        out.meshes.size(), out.buffers.size(), out.bufferViews.size());

    // Log the mesh node world transform for debugging.
    {
        const float* m = glm::value_ptr(mesh_node_world_transform_);
        fprintf(stderr, "[AutoRig] mesh node world transform:\n"
            "  [%7.3f %7.3f %7.3f %7.3f]\n"
            "  [%7.3f %7.3f %7.3f %7.3f]\n"
            "  [%7.3f %7.3f %7.3f %7.3f]\n"
            "  [%7.3f %7.3f %7.3f %7.3f]\n",
            m[0], m[4], m[8],  m[12],
            m[1], m[5], m[9],  m[13],
            m[2], m[6], m[10], m[14],
            m[3], m[7], m[11], m[15]);
    }
    for (size_t i = 0; i < out.images.size(); ++i) {
        auto& img = out.images[i];
        fprintf(stderr, "  image[%zu]: uri='%s' bufferView=%d mime='%s' pixels=%zu\n",
            i, img.uri.c_str(), img.bufferView, img.mimeType.c_str(),
            img.image.size());
    }

    // ---- Ensure at least one buffer exists ---------------------------------
    if (out.buffers.empty()) {
        out.buffers.emplace_back();
    }
    int skin_buf_idx = 0;   // append new data to buffer 0
    auto& buf = out.buffers[skin_buf_idx];

    // ---- Embed URI-referenced images into buffer 0 -------------------------
    //
    //  .gltf files reference textures as external URIs (e.g. "textures/a.png").
    //  Since the output is in a different directory, those relative paths break.
    //  .glb files already have images as buffer views — those are fine.
    //
    //  We embed external images as raw file bytes BEFORE appending skin data,
    //  and then write with embedImages=false so tinygltf doesn't re-process
    //  anything (which would corrupt our appended offsets).
    {
        auto source_dir = std::filesystem::path(source_mesh_path_).parent_path();
        for (auto& img : out.images) {
            // Already a buffer view (GLB-embedded) — skip.
            if (img.bufferView >= 0) continue;
            // No URI or data-URI — skip.
            if (img.uri.empty()) continue;
            if (img.uri.rfind("data:", 0) == 0) continue;

            // Read the raw image file bytes.
            auto img_path = source_dir / img.uri;
            std::ifstream ifs(img_path.string(), std::ios::binary | std::ios::ate);
            if (!ifs.is_open()) {
                fprintf(stderr, "[AutoRig] warning: cannot open image '%s'\n",
                    img_path.string().c_str());
                continue;
            }

            auto file_size = static_cast<size_t>(ifs.tellg());
            ifs.seekg(0);

            // Pad to 4-byte alignment.
            while (buf.data.size() % 4 != 0) buf.data.push_back(0);
            size_t offset = buf.data.size();
            buf.data.resize(offset + file_size);
            ifs.read(reinterpret_cast<char*>(buf.data.data() + offset), file_size);

            // Create a buffer view for this image.
            tinygltf::BufferView bv;
            bv.buffer     = skin_buf_idx;
            bv.byteOffset = offset;
            bv.byteLength = file_size;
            int bv_idx = static_cast<int>(out.bufferViews.size());
            out.bufferViews.push_back(bv);

            // Determine MIME type from extension.
            auto ext = std::filesystem::path(img.uri).extension().string();
            for (auto& c : ext) c = static_cast<char>(std::tolower(c));
            if (ext == ".png")                    img.mimeType = "image/png";
            else if (ext == ".jpg" || ext == ".jpeg") img.mimeType = "image/jpeg";
            else                                  img.mimeType = "application/octet-stream";

            img.bufferView = bv_idx;
            img.uri.clear();   // now embedded — clear the URI

            fprintf(stderr, "[AutoRig] embedded image '%s' (%zu bytes)\n",
                img_path.filename().string().c_str(), file_size);
        }
    }

    auto appendData = [&](const void* data, size_t bytes) -> size_t {
        // Pad to 4-byte alignment first.
        while (buf.data.size() % 4 != 0) buf.data.push_back(0);
        size_t offset = buf.data.size();
        buf.data.resize(offset + bytes);
        memcpy(buf.data.data() + offset, data, bytes);
        return offset;
    };

    // ---- Count total vertices across all mesh primitives -------------------
    //  Must match the vertex count we computed skin weights for.
    size_t total_verts = mesh_.positions.size();

    // ---- Joint indices (JOINTS_0: uvec4 as unsigned short) -----------------
    std::vector<uint16_t> joint_data(total_verts * 4, 0);
    for (size_t v = 0; v < total_verts; ++v) {
        for (int i = 0; i < 4; ++i)
            joint_data[v * 4 + i] = static_cast<uint16_t>(
                skin_weights_.per_vertex[v].joint_indices[i]);
    }
    size_t joints_offset = appendData(
        joint_data.data(), joint_data.size() * sizeof(uint16_t));
    size_t joints_size = joint_data.size() * sizeof(uint16_t);

    // ---- Weights (WEIGHTS_0: vec4 float) -----------------------------------
    std::vector<float> weight_data(total_verts * 4, 0.0f);
    for (size_t v = 0; v < total_verts; ++v) {
        for (int i = 0; i < 4; ++i)
            weight_data[v * 4 + i] = skin_weights_.per_vertex[v].weights[i];
    }
    size_t weights_offset = appendData(
        weight_data.data(), weight_data.size() * sizeof(float));
    size_t weights_size = weight_data.size() * sizeof(float);

    // ---- Inverse bind matrices ---------------------------------------------
    std::vector<float> ibm_data;
    ibm_data.reserve(skeleton_.joints.size() * 16);
    for (auto& j : skeleton_.joints) {
        const float* m = glm::value_ptr(j.inverse_bind_matrix);
        ibm_data.insert(ibm_data.end(), m, m + 16);
    }
    size_t ibm_offset = appendData(
        ibm_data.data(), ibm_data.size() * sizeof(float));
    size_t ibm_size = ibm_data.size() * sizeof(float);

    // ---- Buffer views for the new data -------------------------------------
    auto addBV = [&](size_t offset, size_t length, int target = 0) -> int {
        tinygltf::BufferView bv;
        bv.buffer     = skin_buf_idx;
        bv.byteOffset = offset;
        bv.byteLength = length;
        bv.target     = target;
        int idx = static_cast<int>(out.bufferViews.size());
        out.bufferViews.push_back(bv);
        return idx;
    };

    int bv_joints  = addBV(joints_offset,  joints_size,  34962);  // ARRAY_BUFFER
    int bv_weights = addBV(weights_offset, weights_size, 34962);
    int bv_ibm     = addBV(ibm_offset,     ibm_size);

    // ---- Accessors for the new data ----------------------------------------
    auto addAcc = [&](int bv, int compType, int type, size_t count) -> int {
        tinygltf::Accessor acc;
        acc.bufferView    = bv;
        acc.byteOffset    = 0;
        acc.componentType = compType;
        acc.type          = type;
        acc.count         = count;
        int idx = static_cast<int>(out.accessors.size());
        out.accessors.push_back(acc);
        return idx;
    };

    int acc_joints  = addAcc(bv_joints, TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT,
                             TINYGLTF_TYPE_VEC4, total_verts);
    int acc_weights = addAcc(bv_weights, TINYGLTF_COMPONENT_TYPE_FLOAT,
                             TINYGLTF_TYPE_VEC4, total_verts);
    int acc_ibm     = addAcc(bv_ibm, TINYGLTF_COMPONENT_TYPE_FLOAT,
                             TINYGLTF_TYPE_MAT4, skeleton_.joints.size());

    // ---- Inject JOINTS_0 / WEIGHTS_0 into every existing primitive ---------
    //
    //  We track a running vertex offset so each primitive gets the right
    //  slice of the joint/weight arrays. For each primitive, create
    //  per-primitive accessors that reference the correct sub-range.
    {
        size_t vertex_offset = 0;
        for (auto& gltf_mesh : out.meshes) {
            for (auto& prim : gltf_mesh.primitives) {
                // Determine how many vertices this primitive has.
                size_t prim_verts = 0;
                auto pos_it = prim.attributes.find("POSITION");
                if (pos_it != prim.attributes.end() &&
                    pos_it->second >= 0 &&
                    pos_it->second < (int)out.accessors.size()) {
                    prim_verts = out.accessors[pos_it->second].count;
                }
                if (prim_verts == 0) continue;

                // JOINTS_0 accessor for this primitive's vertex slice.
                {
                    tinygltf::Accessor acc;
                    acc.bufferView    = bv_joints;
                    acc.byteOffset    = vertex_offset * 4 * sizeof(uint16_t);
                    acc.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT;
                    acc.type          = TINYGLTF_TYPE_VEC4;
                    acc.count         = prim_verts;
                    int idx = static_cast<int>(out.accessors.size());
                    out.accessors.push_back(acc);
                    prim.attributes["JOINTS_0"] = idx;
                }

                // WEIGHTS_0 accessor for this primitive's vertex slice.
                {
                    tinygltf::Accessor acc;
                    acc.bufferView    = bv_weights;
                    acc.byteOffset    = vertex_offset * 4 * sizeof(float);
                    acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
                    acc.type          = TINYGLTF_TYPE_VEC4;
                    acc.count         = prim_verts;
                    int idx = static_cast<int>(out.accessors.size());
                    out.accessors.push_back(acc);
                    prim.attributes["WEIGHTS_0"] = idx;
                }

                vertex_offset += prim_verts;
            }
        }
    }

    // ---- Add joint nodes ---------------------------------------------------
    //
    // glTF node translations are LOCAL (relative to parent).
    int joint_node_base = static_cast<int>(out.nodes.size());

    for (int j = 0; j < (int)skeleton_.joints.size(); ++j) {
        auto& jt = skeleton_.joints[j];
        tinygltf::Node node;
        node.name = jt.name;

        glm::vec3 local_pos;
        if (jt.parent >= 0) {
            local_pos = jt.position - skeleton_.joints[jt.parent].position;
        } else {
            local_pos = jt.position;
        }
        node.translation = {
            static_cast<double>(local_pos.x),
            static_cast<double>(local_pos.y),
            static_cast<double>(local_pos.z) };

        out.nodes.push_back(node);
    }

    // Set joint parent-child relationships.
    for (int j = 0; j < (int)skeleton_.joints.size(); ++j) {
        int p = skeleton_.joints[j].parent;
        if (p >= 0) {
            out.nodes[joint_node_base + p].children.push_back(joint_node_base + j);
        }
    }

    // ---- Create skin -------------------------------------------------------
    tinygltf::Skin skin;
    skin.name = "auto_rig_skin";
    skin.inverseBindMatrices = acc_ibm;
    skin.skeleton = joint_node_base + skeleton_.root;
    for (int j = 0; j < (int)skeleton_.joints.size(); ++j) {
        skin.joints.push_back(joint_node_base + j);
    }
    int skin_idx = static_cast<int>(out.skins.size());
    out.skins.push_back(skin);

    // ---- Assign skin to all mesh nodes -------------------------------------
    for (auto& node : out.nodes) {
        if (node.mesh >= 0) {
            node.skin = skin_idx;
        }
    }

    // ---- Add root joint node(s) to the scene -------------------------------
    if (!out.scenes.empty()) {
        auto& scene = out.scenes[out.defaultScene >= 0 ? out.defaultScene : 0];
        for (int j = 0; j < (int)skeleton_.joints.size(); ++j) {
            if (skeleton_.joints[j].parent < 0) {
                scene.nodes.push_back(joint_node_base + j);
            }
        }
    }

    // ---- Write -------------------------------------------------------------
    {
        auto parent_dir = std::filesystem::path(output_path).parent_path();
        if (!parent_dir.empty()) {
            std::filesystem::create_directories(parent_dir);
        }
    }

    // Clear decoded pixel data from images — the original compressed bytes
    // are already in the buffer via buffer views.  Leaving image.image
    // populated causes tinygltf to re-encode and duplicate image data,
    // bloating the output file.
    for (auto& img : out.images) {
        if (img.bufferView >= 0) {
            img.image.clear();
            img.width = img.height = img.component = 0;
        }
    }

    // Write with embedImages=false — we already embedded URI images above,
    // and GLB-source images are already buffer views.  Setting this to true
    // would cause tinygltf to re-encode images and rebuild buffer views,
    // corrupting our appended skin data offsets.
    tinygltf::TinyGLTF writer;
    bool is_glb = (output_path.size() >= 4 &&
                   output_path.substr(output_path.size() - 4) == ".glb");

    bool ok = is_glb
        ? writer.WriteGltfSceneToFile(&out, output_path, false, true, false, true)
        : writer.WriteGltfSceneToFile(&out, output_path, false, true, true, false);

    fprintf(stderr, "[AutoRig] export %s: %s\n",
        output_path.c_str(), ok ? "OK" : "FAILED");
    return ok;
}

// ============================================================================
//  rigCharacter – full pipeline in one call.
// ============================================================================

bool AutoRigPlugin::rigCharacter(
    const std::string& mesh_path,
    const std::string& output_gltf_path)
{
    state_ = PluginState::kRunning;
    const int kTotalSteps = 6;

    reportProgress(1, kTotalSteps, "Loading mesh...");
    if (!loadMesh(mesh_path)) {
        // ui_status_ may already be set (e.g. "Already skinned").
        if (state_ == PluginState::kRunning) state_ = PluginState::kError;
        return false;
    }

    reportProgress(2, kTotalSteps, "Capturing multi-view renders...");
    if (!captureViews(num_views_, capture_resolution_)) {
        state_ = PluginState::kError; return false;
    }

    initEditableJoints();

    reportProgress(3, kTotalSteps, "Running diffusion model...");
    if (!predictJoints()) { state_ = PluginState::kError; return false; }

    reportProgress(4, kTotalSteps, "Fusing 3D skeleton...");
    if (!fuseAndBuildSkeleton()) { state_ = PluginState::kError; return false; }

    reportProgress(5, kTotalSteps, "Computing skin weights...");
    if (!computeSkinWeights()) { state_ = PluginState::kError; return false; }

    reportProgress(6, kTotalSteps, "Exporting glTF...");
    if (!exportGltf(output_gltf_path)) { state_ = PluginState::kError; return false; }

    state_ = PluginState::kFinished;
    reportProgress(kTotalSteps, kTotalSteps, "Done!");
    return true;
}

// ============================================================================
//  ImGui panel
// ============================================================================

// ---------------------------------------------------------------------------
void AutoRigPlugin::refreshMeshFileList() {
    mesh_file_list_.clear();
    const char* scan_dirs[] = { "assets/characters" };
    const char* mesh_exts[] = { ".glb", ".gltf", ".obj", ".fbx" };

    for (auto dir : scan_dirs) {
        if (!std::filesystem::exists(dir)) continue;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            auto ext = entry.path().extension().string();
            // lower-case compare
            for (auto& c : ext) c = static_cast<char>(std::tolower(c));
            for (auto mext : mesh_exts) {
                if (ext == mext) {
                    mesh_file_list_.push_back(entry.path().string());
                    break;
                }
            }
        }
    }
    std::sort(mesh_file_list_.begin(), mesh_file_list_.end());
}

// ---------------------------------------------------------------------------
//  Helper: draw 3D model preview on an ImDrawList canvas with drag-rotation.
//  Returns true if the canvas was hovered.
// ---------------------------------------------------------------------------
static bool drawModel3DPreview(
    const TriangleMesh& mesh, const Skeleton& skeleton,
    float canvas_size, ImVec2 canvas_pos, ImDrawList* dl,
    float& yaw, float& pitch, bool& dragging, ImVec2& drag_start)
{
    // Dark background.
    dl->AddRectFilled(canvas_pos,
        ImVec2(canvas_pos.x + canvas_size, canvas_pos.y + canvas_size),
        IM_COL32(25, 25, 30, 255));
    dl->AddRect(canvas_pos,
        ImVec2(canvas_pos.x + canvas_size, canvas_pos.y + canvas_size),
        IM_COL32(80, 80, 80, 255));

    // Bounding box.
    glm::vec3 bmin( 1e9f), bmax(-1e9f);
    if (!mesh.empty()) { bmin = glm::min(bmin, mesh.bbox_min); bmax = glm::max(bmax, mesh.bbox_max); }
    if (!skeleton.empty()) { for (auto& jt : skeleton.joints) { bmin = glm::min(bmin, jt.position); bmax = glm::max(bmax, jt.position); } }
    if (bmin.x > bmax.x) { bmin = glm::vec3(0); bmax = glm::vec3(1); }

    glm::vec3 center = (bmin + bmax) * 0.5f;
    float max_ext = std::max({bmax.x-bmin.x, bmax.y-bmin.y, bmax.z-bmin.z, 0.001f});
    float proj_scale = (canvas_size * 0.85f) / max_ext;

    float cy = std::cos(yaw), sy = std::sin(yaw);
    float cp = std::cos(pitch), sp = std::sin(pitch);

    auto project = [&](const glm::vec3& p) -> ImVec2 {
        glm::vec3 q = p - center;
        glm::vec3 r;
        r.x =  cy * q.x + sy * sp * q.y + sy * cp * q.z;
        r.y =             cp * q.y       - sp * q.z;
        r.z = -sy * q.x + cy * sp * q.y + cy * cp * q.z;
        float sx = r.x * proj_scale + canvas_size * 0.5f;
        float sy2 = -r.y * proj_scale + canvas_size * 0.5f;
        return ImVec2(canvas_pos.x + sx, canvas_pos.y + sy2);
    };

    dl->PushClipRect(canvas_pos,
        ImVec2(canvas_pos.x + canvas_size, canvas_pos.y + canvas_size), true);

    // Draw mesh triangles.
    if (!mesh.empty()) {
        size_t tri_count = mesh.indices.size() / 3;
        size_t step = std::max<size_t>(1, tri_count / 800);
        for (size_t t = 0; t < tri_count; t += step) {
            ImVec2 v0 = project(mesh.positions[mesh.indices[t * 3 + 0]]);
            ImVec2 v1 = project(mesh.positions[mesh.indices[t * 3 + 1]]);
            ImVec2 v2 = project(mesh.positions[mesh.indices[t * 3 + 2]]);
            dl->AddTriangleFilled(v0, v1, v2, IM_COL32(80, 130, 180, 76));
            dl->AddTriangle(v0, v1, v2, IM_COL32(100, 160, 220, 50), 1.0f);
        }
    }

    // Draw skeleton.
    if (!skeleton.empty()) {
        for (int j = 0; j < (int)skeleton.joints.size(); ++j) {
            auto& jt = skeleton.joints[j];
            if (jt.parent >= 0)
                dl->AddLine(project(skeleton.joints[jt.parent].position),
                            project(jt.position), IM_COL32(255, 200, 50, 255), 2.5f);
        }
        for (int j = 0; j < (int)skeleton.joints.size(); ++j) {
            auto& jt = skeleton.joints[j];
            ImVec2 p = project(jt.position);
            bool root = (jt.parent < 0);
            dl->AddCircleFilled(p, root ? 5.0f : 3.5f,
                root ? IM_COL32(80,255,80,255) : IM_COL32(255,80,80,255));
            dl->AddCircle(p, root ? 5.0f : 3.5f, IM_COL32(0,0,0,200));
        }
    }

    dl->AddText(ImVec2(canvas_pos.x + 4, canvas_pos.y + canvas_size - 16),
        IM_COL32(160, 160, 160, 120), "Drag to rotate");

    dl->PopClipRect();

    // Handle drag rotation.
    bool hovered = ImGui::IsItemHovered();
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        dragging = true;
        drag_start = ImGui::GetMousePos();
    }
    if (dragging) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            ImVec2 m = ImGui::GetMousePos();
            yaw   += (m.x - drag_start.x) * 0.01f;
            pitch += (m.y - drag_start.y) * 0.01f;
            pitch  = std::clamp(pitch, -1.5f, 1.5f);
            drag_start = m;
        } else {
            dragging = false;
        }
    }
    return hovered;
}

// ---------------------------------------------------------------------------
//  Helper: draw 2D view image (color pixels from a ViewCapture).
//  If the capture has OIT color_rgba, uses that (with native per-pixel alpha).
//  Otherwise falls back to opaque RGB from cap.color.
// ---------------------------------------------------------------------------
static void drawViewPixels(
    ImDrawList* dl, ImVec2 cpos, float display_w, float display_h,
    const ViewCapture& cap)
{
    int W = cap.width, H = cap.height;
    int max_blocks = 600;
    int block = std::max(1, std::max(W, H) / max_blocks);
    float sx = display_w / W, sy = display_h / H;

    bool use_rgba = !cap.color_rgba.empty() &&
                    (int)cap.color_rgba.size() >= W * H * 4;

    for (int by = 0; by < H; by += block) {
        for (int bx = 0; bx < W; bx += block) {
            int px = std::min(bx + block / 2, W - 1);
            int py = std::min(by + block / 2, H - 1);
            int idx = py * W + px;

            uint8_t r, g, b, a;
            if (use_rgba) {
                r = cap.color_rgba[idx * 4 + 0];
                g = cap.color_rgba[idx * 4 + 1];
                b = cap.color_rgba[idx * 4 + 2];
                a = cap.color_rgba[idx * 4 + 3];
                if (a == 0) continue;  // fully transparent — skip
            } else {
                if (!cap.color.empty() && idx * 3 + 2 < (int)cap.color.size()) {
                    r = cap.color[idx * 3 + 0];
                    g = cap.color[idx * 3 + 1];
                    b = cap.color[idx * 3 + 2];
                } else {
                    uint8_t s = (!cap.silhouette.empty()) ? cap.silhouette[idx] : 0;
                    r = g = b = (s > 0) ? 140 : 25;
                }
                a = 255;
                if (r == 0 && g == 0 && b == 0) continue;
            }

            float x0 = cpos.x + bx * sx, y0 = cpos.y + by * sy;
            float x1 = x0 + block * sx,  y1 = y0 + block * sy;
            dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), IM_COL32(r, g, b, a));
        }
    }
}

// ---------------------------------------------------------------------------
void AutoRigPlugin::drawImGui() {
    // Track open/close transition (must run before early return).
    bool just_opened = show_window_ && !show_window_prev_;
    show_window_prev_ = show_window_;

    if (!show_window_) return;

    ImGuiIO& io = ImGui::GetIO();

    // Size to 80% of the display, centred.  Use Appearing so it re-centres
    // each time the window is toggled open.
    ImGui::SetNextWindowSize(
        ImVec2(io.DisplaySize.x * 0.8f, io.DisplaySize.y * 0.85f),
        ImGuiCond_Appearing);
    ImGui::SetNextWindowPos(
        ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f),
        ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    // Bring to front on open (only once, not every frame).
    if (just_opened)
        ImGui::SetNextWindowFocus();

    // Force this window into its own OS-level viewport (separate window).
    ImGui::SetNextWindowViewport(0);
    ImGuiWindowClass popup_class;
    popup_class.ViewportFlagsOverrideSet = ImGuiViewportFlags_NoAutoMerge;
    ImGui::SetNextWindowClass(&popup_class);

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoDocking;

    if (!ImGui::Begin("Auto Rig", &show_window_, flags)) {
        ImGui::End();
        return;
    }

    // ---- Mesh file selection (shared) ----
    if (!files_scanned_) { refreshMeshFileList(); files_scanned_ = true; }

    if (ImGui::Button("Refresh")) refreshMeshFileList();
    ImGui::SameLine();
    ImGui::Text("Mesh Files (%d found)", (int)mesh_file_list_.size());

    {
        float list_h = ImGui::GetTextLineHeightWithSpacing() * std::min((int)mesh_file_list_.size(), 5) + 4.0f;
        if (list_h < ImGui::GetTextLineHeightWithSpacing() * 2) list_h = ImGui::GetTextLineHeightWithSpacing() * 2;
        if (ImGui::BeginChild("MeshFileList", ImVec2(-1, list_h), true)) {
            for (int i = 0; i < (int)mesh_file_list_.size(); ++i) {
                std::string dname = "  " + std::filesystem::path(mesh_file_list_[i]).filename().string();
                if (ImGui::Selectable(dname.c_str(), selected_mesh_idx_ == i)) {
                    selected_mesh_idx_ = i;
                    selected_mesh_path_ = mesh_file_list_[i];
                    auto p = std::filesystem::path(selected_mesh_path_);
                    auto out = p.parent_path() / (p.stem().string() + "-skinned" + p.extension().string());
                    std::snprintf(output_path_buf_, sizeof(output_path_buf_), "%s", out.string().c_str());

                    // Pre-load mesh for the rig editor 3D preview.
                    if (mesh_.empty() || source_mesh_path_ != selected_mesh_path_) {
                        loadMesh(selected_mesh_path_);
                        // Render a default orbit so we have captures ready.
                        captures_ = rasterizer_->captureOrbit(mesh_, num_views_, capture_resolution_);
                        initEditableJoints();
                        edit_capture_valid_ = false;
                    }
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", mesh_file_list_[i].c_str());
            }
        }
        ImGui::EndChild();
    }

    if (selected_mesh_idx_ >= 0) {
        ImGui::Text("Source: %s", std::filesystem::path(selected_mesh_path_).filename().string().c_str());
    } else {
        ImGui::TextDisabled("No mesh selected");
    }

    ImGui::Separator();

    // ---- Mode tabs ----
    if (ImGui::BeginTabBar("##RigModeTabs")) {

        // ================================================================
        //  TAB 1: AUTO RIG
        // ================================================================
        if (ImGui::BeginTabItem("Auto Rig")) {
            ui_mode_ = 0;

            ImGui::SliderInt("Views", &ui_num_views_, 4, 16);
            ImGui::SliderInt("Resolution", &ui_resolution_, 128, 2048);
            ImGui::Text("Output: %s", std::filesystem::path(output_path_buf_).filename().string().c_str());

            bool is_running = (state_ == PluginState::kRunning);
            bool no_mesh = selected_mesh_path_.empty();
            if (is_running || no_mesh) ImGui::BeginDisabled();
            if (ImGui::Button("  Run Auto-Rig  ")) {
                num_views_ = ui_num_views_;
                capture_resolution_ = ui_resolution_;
                rigCharacter(selected_mesh_path_, output_path_buf_);
            }
            if (is_running || no_mesh) ImGui::EndDisabled();
            ImGui::SameLine(); ImGui::Text("%s", ui_status_.c_str());
            if (ui_progress_ > 0.0f) ImGui::ProgressBar(ui_progress_, ImVec2(-1, 0));

            // ---- 3D Preview ----
            if (!mesh_.empty() || !skeleton_.empty()) {
                ImGui::Separator();
                constexpr float kCanvasSize = 500.0f;

                // Info column
                float info_w = ImGui::GetContentRegionAvail().x - kCanvasSize - 12.0f;
                if (info_w < 100.0f) info_w = 100.0f;
                ImGui::BeginChild("##ar_info", ImVec2(info_w, kCanvasSize), false);
                if (!skeleton_.empty()) {
                    ImGui::Text("Skeleton: %d joints", (int)skeleton_.joints.size());
                    if (ImGui::TreeNode("Joint list")) {
                        for (int j = 0; j < (int)skeleton_.joints.size(); ++j)
                            ImGui::Text("[%2d] %-14s p=%2d", j, skeleton_.joints[j].name.c_str(), skeleton_.joints[j].parent);
                        ImGui::TreePop();
                    }
                }
                if (!skin_weights_.empty()) ImGui::Text("Skin: %d verts", (int)skin_weights_.per_vertex.size());
                if (!mesh_.empty()) ImGui::Text("Mesh: %zuk tris", mesh_.indices.size() / 3000);
                ImGui::EndChild();
                ImGui::SameLine();

                // 3D canvas
                ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
                ImGui::InvisibleButton("##ar_preview", ImVec2(kCanvasSize, kCanvasSize));
                drawModel3DPreview(mesh_, skeleton_, kCanvasSize, canvas_pos,
                    ImGui::GetWindowDrawList(),
                    preview_yaw_, preview_pitch_, preview_dragging_, preview_drag_start_);
            }

            // ---- Debug multi-view (off by default) ----
            ImGui::Checkbox("Show Debug Views", &show_debug_views_);
            if (show_debug_views_ && !captures_.empty() && !view_edits_.empty()) {
                if (ImGui::CollapsingHeader("Debug: Per-View Captures")) {
                    static float dbg_scale = 0.35f;
                    ImGui::SliderFloat("Scale", &dbg_scale, 0.15f, 1.0f, "%.2f");

                    for (int v = 0; v < (int)captures_.size() && v < (int)view_edits_.size(); ++v) {
                        const auto& cap = captures_[v];
                        int W = cap.width, H = cap.height;
                        float tw = W * dbg_scale, th = H * dbg_scale;
                        ImGui::PushID(v);
                        char lbl[64]; std::snprintf(lbl, sizeof(lbl), "View %d (az=%.0f)", v, cap.azimuth_deg);
                        if (ImGui::TreeNode(lbl)) {
                            ImVec2 cp = ImGui::GetCursorScreenPos();
                            ImGui::InvisibleButton("##dbg_view", ImVec2(tw, th));
                            ImDrawList* vdl = ImGui::GetWindowDrawList();
                            vdl->PushClipRect(cp, ImVec2(cp.x + tw, cp.y + th), true);
                            drawViewPixels(vdl, cp, tw, th, cap);

                            // Draw bones + joints from view_edits_
                            auto& es = view_edits_[v];
                            auto uv2s = [&](const glm::vec2& uv) -> ImVec2 {
                                return ImVec2(cp.x + uv.x * tw, cp.y + uv.y * th);
                            };
                            for (int j = 0; j < kNumEditJoints; ++j) {
                                int pj = kJointParents[j];
                                if (pj >= 0) {
                                    vdl->AddLine(uv2s(es.joints[pj].uv), uv2s(es.joints[j].uv),
                                        IM_COL32(0,220,220,180), 1.5f);
                                }
                            }
                            for (int j = 0; j < kNumEditJoints; ++j) {
                                vdl->AddCircleFilled(uv2s(es.joints[j].uv), 3.0f,
                                    IM_COL32(0,240,240,220));
                            }
                            vdl->AddRect(cp, ImVec2(cp.x + tw, cp.y + th), IM_COL32(80,80,80,200));
                            vdl->PopClipRect();
                            ImGui::TreePop();
                        }
                        ImGui::PopID();
                    }
                }
            }

            ImGui::EndTabItem();
        }

        // ================================================================
        //  TAB 2: RIG EDITOR
        // ================================================================
        if (ImGui::BeginTabItem("Rig Editor")) {
            ui_mode_ = 1;

            if (mesh_.empty()) {
                ImGui::TextDisabled("Select a mesh above to begin editing.");
                ImGui::EndTabItem();
                ImGui::EndTabBar();
                ImGui::End();
                return;
            }

            // Layout: left = live 2D rasterized preview, right = 2D edit canvas
            // The edit canvas fills all remaining width/height so no scrolling.
            constexpr float k3DSize = 600.0f;
            float avail_w = ImGui::GetContentRegionAvail().x;
            float edit_size = avail_w - k3DSize - 20.0f;
            if (edit_size < 300.0f) edit_size = 300.0f;

            // Right panel height = edit image + header text + checkbox line + button row
            float right_panel_h = edit_size + 90.0f;
            // Left panel matches its own content, not the right panel
            float left_panel_h = k3DSize + 150.0f;

            // ---- Left: live rasterized preview (re-rendered on angle change) ----
            ImGui::BeginChild("##re_3d", ImVec2(k3DSize, left_panel_h), true);
            ImGui::Text("Preview  (drag to rotate)");

            float canvas_dim = k3DSize - 16.0f;
            ImVec2 c3d_pos = ImGui::GetCursorScreenPos();
            ImGui::InvisibleButton("##re_3d_canvas", ImVec2(canvas_dim, canvas_dim));
            ImDrawList* pdl = ImGui::GetWindowDrawList();

            // Handle drag rotation on the invisible button.
            {
                bool prev_hov = ImGui::IsItemHovered();
                if (prev_hov && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    preview_dragging_ = true;
                    preview_drag_start_ = ImGui::GetMousePos();
                }
                if (preview_dragging_) {
                    if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                        ImVec2 m = ImGui::GetMousePos();
                        preview_yaw_   += (m.x - preview_drag_start_.x) * 0.01f;
                        preview_pitch_ += (m.y - preview_drag_start_.y) * 0.01f;
                        preview_pitch_  = std::clamp(preview_pitch_, -1.5f, 1.5f);
                        preview_drag_start_ = m;
                    } else {
                        preview_dragging_ = false;
                    }
                }
            }

            // Re-render the rasterized preview when the angle or size changes.
            int render_res = std::max(64, (int)canvas_dim);
            if (!mesh_.empty() && rasterizer_) {
                bool angle_changed =
                    std::abs(preview_yaw_ - preview_render_yaw_) > 0.001f ||
                    std::abs(preview_pitch_ - preview_render_pitch_) > 0.001f;
                bool size_changed = (preview_render_.width != render_res);

                if (angle_changed || size_changed || preview_render_.width == 0) {
                    float az = -glm::degrees(preview_yaw_);
                    float el = -glm::degrees(preview_pitch_);

                    glm::vec3 centre = (mesh_.bbox_min + mesh_.bbox_max) * 0.5f;
                    float ext = glm::length(mesh_.bbox_max - mesh_.bbox_min);
                    float radius = ext * camera_distance_;
                    float fov = glm::radians(45.0f);
                    float z_near = radius * 0.01f, z_far = radius * 10.0f;
                    glm::mat4 proj = glm::perspective(fov, 1.0f, z_near, z_far);
                    proj[1][1] *= -1.0f;

                    float az_rad = glm::radians(az);
                    float el_rad = glm::radians(el);
                    glm::vec3 eye;
                    eye.x = centre.x + radius * cosf(el_rad) * cosf(az_rad);
                    eye.y = centre.y + radius * sinf(el_rad);
                    eye.z = centre.z + radius * cosf(el_rad) * sinf(az_rad);
                    glm::mat4 view_mat = glm::lookAt(eye, centre, glm::vec3(0, 1, 0));

                    preview_render_ = rasterizer_->render(
                        mesh_, render_res, render_res,
                        view_mat, proj, az, el);
                    preview_render_yaw_   = preview_yaw_;
                    preview_render_pitch_ = preview_pitch_;
                }
            }

            // Draw the rasterized preview image.
            pdl->PushClipRect(c3d_pos,
                ImVec2(c3d_pos.x + canvas_dim, c3d_pos.y + canvas_dim), true);
            pdl->AddRectFilled(c3d_pos,
                ImVec2(c3d_pos.x + canvas_dim, c3d_pos.y + canvas_dim),
                IM_COL32(25, 25, 30, 255));
            if (preview_render_.width > 0)
                drawViewPixels(pdl, c3d_pos, canvas_dim, canvas_dim, preview_render_);
            pdl->AddRect(c3d_pos,
                ImVec2(c3d_pos.x + canvas_dim, c3d_pos.y + canvas_dim),
                IM_COL32(80, 80, 80, 255));
            pdl->AddText(ImVec2(c3d_pos.x + 4, c3d_pos.y + canvas_dim - 16),
                IM_COL32(160, 160, 160, 180), "Drag to rotate");
            pdl->PopClipRect();

            // Show current angle.
            float az_deg = -glm::degrees(preview_yaw_);
            float el_deg = -glm::degrees(preview_pitch_);
            ImGui::Text("Azimuth: %.0f  Elevation: %.0f", az_deg, el_deg);

            // Zoom slider (camera distance).
            float old_dist = camera_distance_;
            ImGui::SetNextItemWidth(canvas_dim);
            ImGui::SliderFloat("Zoom", &camera_distance_, 0.5f, 1.5f, "%.2f");
            if (camera_distance_ != old_dist) {
                preview_render_yaw_ = -999.0f;  // force re-render
            }

            // "Capture View" button — captures at full resolution for editing.
            if (ImGui::Button("  Capture View for Editing  ")) {
                float az = -glm::degrees(preview_yaw_);
                float el = -glm::degrees(preview_pitch_);

                glm::vec3 centre = (mesh_.bbox_min + mesh_.bbox_max) * 0.5f;
                float ext = glm::length(mesh_.bbox_max - mesh_.bbox_min);
                float radius = ext * camera_distance_;
                float fov = glm::radians(45.0f);
                float z_near = radius * 0.01f, z_far = radius * 4.0f;
                glm::mat4 proj = glm::perspective(fov, 1.0f, z_near, z_far);
                proj[1][1] *= -1.0f;

                float az_rad = glm::radians(az);
                float el_rad = glm::radians(el);
                glm::vec3 eye;
                eye.x = centre.x + radius * cosf(el_rad) * cosf(az_rad);
                eye.y = centre.y + radius * sinf(el_rad);
                eye.z = centre.z + radius * cosf(el_rad) * sinf(az_rad);
                glm::mat4 view_mat = glm::lookAt(eye, centre, glm::vec3(0, 1, 0));

                // Render with OIT for translucent mesh (joints visible behind body).
                edit_capture_ = rasterizer_->renderOIT(
                    mesh_, capture_resolution_, capture_resolution_,
                    view_mat, proj, mesh_opacity_, az, el);
                initEditableJointsForView(edit_view_, edit_capture_);
                edit_capture_valid_ = true;
                drag_joint_ = -1;

                fprintf(stderr, "[AutoRig] Captured edit view: az=%.1f el=%.1f\n", az, el);
            }

            ImGui::EndChild();

            ImGui::SameLine();

            // ---- Right: 2D edit canvas ----
            ImGui::BeginChild("##re_2d", ImVec2(edit_size + 16, right_panel_h), true);

            if (!edit_capture_valid_) {
                ImGui::Text("Rotate the 3D model, then click 'Capture View for Editing'");
            } else {
                float disp_w = edit_size;
                float disp_h = edit_size;

                // Show info
                ImGui::Text("Edit View  (az=%.0f, el=%.0f)  %dx%d  %s",
                    edit_capture_.azimuth_deg, edit_capture_.elevation_deg,
                    edit_capture_.width, edit_capture_.height,
                    edit_view_.any_edited ? "[EDITED]" : "");

                static bool show_labels = true;
                ImGui::SameLine(); ImGui::Checkbox("Labels", &show_labels);

                ImVec2 cpos = ImGui::GetCursorScreenPos();
                ImGui::InvisibleButton("##edit_canvas", ImVec2(disp_w, disp_h));
                bool canvas_hovered = ImGui::IsItemHovered();
                ImDrawList* edl = ImGui::GetWindowDrawList();

                edl->PushClipRect(cpos, ImVec2(cpos.x + disp_w, cpos.y + disp_h), true);

                // UV <-> screen helpers
                auto uvToScreen = [&](const glm::vec2& uv) -> ImVec2 {
                    return ImVec2(cpos.x + uv.x * disp_w, cpos.y + uv.y * disp_h);
                };
                auto screenToUv = [&](const ImVec2& s) -> glm::vec2 {
                    return glm::clamp(glm::vec2((s.x - cpos.x) / disp_w, (s.y - cpos.y) / disp_h),
                        glm::vec2(0.0f), glm::vec2(1.0f));
                };

                // ---- Drag detection ----
                if (canvas_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    ImVec2 mouse = ImGui::GetMousePos();
                    float best_dist = 15.0f;
                    int best_j = -1;
                    for (int j = 0; j < kNumEditJoints; ++j) {
                        ImVec2 jp = uvToScreen(edit_view_.joints[j].uv);
                        float d = std::sqrt((mouse.x-jp.x)*(mouse.x-jp.x) + (mouse.y-jp.y)*(mouse.y-jp.y));
                        if (d < best_dist) { best_dist = d; best_j = j; }
                    }
                    drag_joint_ = best_j;
                }
                if (drag_joint_ >= 0 && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                    edit_view_.joints[drag_joint_].uv = screenToUv(ImGui::GetMousePos());
                    edit_view_.joints[drag_joint_].edited = true;
                    edit_view_.any_edited = true;
                } else if (drag_joint_ >= 0 && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                    drag_joint_ = -1;
                }

                // ---- OIT compositing ----
                //  The edit_capture_ was rendered with OIT (weighted blended
                //  transparency), so color_rgba already has correct per-pixel
                //  alpha.  We just draw: background → mesh (OIT RGBA) → skeleton.
                //  Joints behind the mesh are visible through the translucent body.

                // Background fill.
                edl->AddRectFilled(cpos, ImVec2(cpos.x + disp_w, cpos.y + disp_h),
                    IM_COL32(25, 25, 30, 255));

                // Translucent mesh (OIT-resolved RGBA).
                drawViewPixels(edl, cpos, disp_w, disp_h, edit_capture_);

                // Skeleton on top.
                for (int j = 0; j < kNumEditJoints; ++j) {
                    int pj = kJointParents[j];
                    if (pj >= 0) {
                        ImVec2 a = uvToScreen(edit_view_.joints[pj].uv);
                        ImVec2 b = uvToScreen(edit_view_.joints[j].uv);
                        edl->AddLine(a, b, IM_COL32(0, 0, 0, 120), 3.5f);
                        edl->AddLine(a, b, IM_COL32(220, 170, 80, 220), 2.0f);
                    }
                }

                for (int j = 0; j < kNumEditJoints; ++j) {
                    ImVec2 p = uvToScreen(edit_view_.joints[j].uv);
                    bool edited = edit_view_.joints[j].edited;
                    bool hovered = false;
                    if (canvas_hovered) {
                        ImVec2 m = ImGui::GetMousePos();
                        float d = std::sqrt((m.x-p.x)*(m.x-p.x) + (m.y-p.y)*(m.y-p.y));
                        hovered = (d < 15.0f);
                    }

                    ImU32 jcol = edited ? IM_COL32(255, 150, 0, 255) : IM_COL32(0, 200, 255, 255);
                    float rad = (hovered || drag_joint_ == j) ? 7.0f : 5.0f;

                    edl->AddCircleFilled(p, rad, jcol);
                    edl->AddCircle(p, rad, IM_COL32(0, 0, 0, 220), 0, 1.5f);

                    if (show_labels && (hovered || edited || drag_joint_ == j)) {
                        edl->AddText(ImVec2(p.x + rad + 3, p.y - 7),
                            IM_COL32(0,0,0,200), kJointNames[j]);
                        edl->AddText(ImVec2(p.x + rad + 2, p.y - 8),
                            IM_COL32(255,255,255,240), kJointNames[j]);
                    }
                }

                // Border + angle tag
                edl->AddRect(cpos, ImVec2(cpos.x + disp_w, cpos.y + disp_h),
                    IM_COL32(120, 120, 120, 200));
                {
                    char ang[64];
                    std::snprintf(ang, sizeof(ang), "az:%.0f el:%.0f",
                        edit_capture_.azimuth_deg, edit_capture_.elevation_deg);
                    edl->AddText(ImVec2(cpos.x + 4, cpos.y + 3), IM_COL32(0,0,0,200), ang);
                    edl->AddText(ImVec2(cpos.x + 3, cpos.y + 2), IM_COL32(255,255,0,255), ang);
                }

                edl->PopClipRect();

                // ---- Buttons below the edit canvas ----
                if (ImGui::Button("Reset Joints")) {
                    initEditableJointsForView(edit_view_, edit_capture_);
                }
                ImGui::SameLine();
                if (ImGui::Button("Flip L/R")) {
                    // Swap UV positions of left<->right joint pairs.
                    // Standard indices: L_shoulder(5)<->R_shoulder(9),
                    // L_upper_arm(6)<->R_upper_arm(10), etc.
                    static const int kLRPairs[][2] = {
                        { 5,  9}, // left_shoulder  <-> right_shoulder
                        { 6, 10}, // left_upper_arm <-> right_upper_arm
                        { 7, 11}, // left_lower_arm <-> right_lower_arm
                        { 8, 12}, // left_hand      <-> right_hand
                        {13, 16}, // left_upper_leg <-> right_upper_leg
                        {14, 17}, // left_lower_leg <-> right_lower_leg
                        {15, 18}, // left_foot      <-> right_foot
                    };
                    int nj = (int)edit_view_.joints.size();
                    for (auto& pr : kLRPairs) {
                        if (pr[0] < nj && pr[1] < nj) {
                            std::swap(edit_view_.joints[pr[0]].uv,
                                      edit_view_.joints[pr[1]].uv);
                            edit_view_.joints[pr[0]].edited = true;
                            edit_view_.joints[pr[1]].edited = true;
                        }
                    }
                    edit_view_.any_edited = true;
                }
                ImGui::SameLine();
                if (ImGui::Button("Save This View")) {
                    saved_edits_.push_back({edit_view_, edit_capture_});
                    fprintf(stderr, "[AutoRig] Saved edit view #%d (az=%.0f)\n",
                        (int)saved_edits_.size(), edit_capture_.azimuth_deg);
                }
                ImGui::SameLine();
                ImGui::Text("  Saved views: %d", (int)saved_edits_.size());
            }

            ImGui::EndChild();

            // ---- Bottom: saved views list + export ----
            ImGui::Separator();
            if (!saved_edits_.empty()) {
                ImGui::Text("Saved Training Views:");
                for (int i = 0; i < (int)saved_edits_.size(); ++i) {
                    auto& sv = saved_edits_[i];
                    ImGui::BulletText("View %d: az=%.0f el=%.0f  (%d edited joints)",
                        i, sv.edit.azimuth_deg, sv.edit.elevation_deg,
                        (int)std::count_if(sv.edit.joints.begin(), sv.edit.joints.end(),
                            [](const EditableJoint& ej) { return ej.edited; }));
                    ImGui::SameLine();
                    ImGui::PushID(i);
                    if (ImGui::SmallButton("X")) {
                        saved_edits_.erase(saved_edits_.begin() + i);
                        --i;
                    }
                    ImGui::PopID();
                }

                if (ImGui::Button("  Export Training Data  ")) {
                    if (exportTrainingData("assets/rigs_training")) {
                        ui_status_ = "Training data exported!";
                    } else {
                        ui_status_ = "Export failed (no edited views?)";
                    }
                }
                ImGui::SameLine();
                ImGui::Text("-> assets/rigs_training/");
            }

            if (ImGui::Button("Clear All Saved Views")) {
                saved_edits_.clear();
            }

            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}

} // namespace auto_rig
} // namespace plugins
