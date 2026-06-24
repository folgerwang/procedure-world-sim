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
#include "renderer/renderer.h"       // engine::renderer::Helper, Device, enums
#include "helper/engine_helper.h"    // engine::helper::createTextureImage
#include <source_location>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>      // std::cout → editor Output Log window (see main.cpp)
#include <algorithm>
#include <numeric>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <functional>
#include <queue>          // geodesic skinning (Dijkstra over the mesh surface)
#include <thread>
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
    const std::shared_ptr<engine::renderer::Device>& device)
{
    device_          = device;   // kept for lazy-loading launcher icon textures
    rasterizer_      = std::make_unique<SimpleRasterizer>();
    diffusion_model_ = std::make_unique<RigDiffusionModel>();

    // Resolve the models directory once.
    {
        std::string exe_dir = std::filesystem::current_path().string();
        for (const auto& candidate : {
            std::string("assets/ml_models"),
            exe_dir + "/assets/ml_models",
            exe_dir + "/../realworld/assets/ml_models",
            std::string("assets/models"),                 // legacy fallback
            exe_dir + "/assets/models",
            exe_dir + "/../realworld/assets/models",
        }) {
            if (std::filesystem::exists(candidate)) {
                model_dir_ = std::filesystem::canonical(candidate).string();
                break;
            }
        }
        if (model_dir_.empty()) {
            model_dir_ = exe_dir + "/assets/ml_models";
            std::filesystem::create_directories(model_dir_);
        }
        fprintf(stderr, "[AutoRig] Model directory: %s\n", model_dir_.c_str());
    }

    // Scan for versioned models and load the latest.
    scanModelVersions();
    loadModelByIndex(-1);  // -1 = latest

    state_ = PluginState::kLoaded;
    return true;
}

void AutoRigPlugin::shutdown() {
    rasterizer_.reset();
    diffusion_model_.reset();
    state_ = PluginState::kUnloaded;
}

// ============================================================================
//  Model versioning helpers
//
//  Models are stored as:  <model_dir_>/rig_diffusion_v001.pt
//                         <model_dir_>/rig_diffusion_v002.pt  ...
//  Also accepts the legacy unversioned  rig_diffusion.pt  (treated as v0).
// ============================================================================

std::string AutoRigPlugin::modelPathForVersion(int v) const {
    // Legacy compat — searches by version number only (returns first match).
    for (auto& e : model_versions_)
        if (e.version == v) return e.path;
    if (v <= 0) return {};
    // Fallback to legacy naming
    char buf[64];
    std::snprintf(buf, sizeof(buf), "/rig_diffusion_v%03d.pt", v);
    std::string p = model_dir_ + buf;
    if (std::filesystem::exists(p)) return p;
    if (v == 1) {
        p = model_dir_ + "/rig_diffusion.pt";
        if (std::filesystem::exists(p)) return p;
    }
    return {};
}

std::string AutoRigPlugin::modelPathForArchVersion(const std::string& arch, int v) const {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "/rig_diffusion_%s_v%03d.pt", arch.c_str(), v);
    return model_dir_ + buf;
}

int AutoRigPlugin::nextVersionForArch(const std::string& arch) const {
    int max_ver = 0;
    for (auto& e : model_versions_) {
        if (e.arch == arch && e.version > max_ver)
            max_ver = e.version;
    }
    return max_ver + 1;
}

void AutoRigPlugin::scanModelVersions() {
    model_versions_.clear();

    if (model_dir_.empty() || !std::filesystem::exists(model_dir_))
        return;

    // Scan for all .pt files matching our naming patterns.
    for (auto& entry : std::filesystem::directory_iterator(model_dir_)) {
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        if (name.size() < 4 || name.substr(name.size() - 3) != ".pt") continue;
        // Skip checkpoint files
        if (name.find("_checkpoint") != std::string::npos) continue;

        std::string stem = name.substr(0, name.size() - 3);  // strip ".pt"

        // New format: rig_diffusion_<arch>_v###
        // Legacy format: rig_diffusion_v###
        const std::string prefix = "rig_diffusion_";
        if (stem.size() < prefix.size() ||
            stem.substr(0, prefix.size()) != prefix) continue;
        std::string rest = stem.substr(prefix.size());  // after "rig_diffusion_"

        ModelEntry me;
        me.path = entry.path().string();
        std::error_code ec;
        me.mtime = std::filesystem::last_write_time(entry.path(), ec);

        // Try new format: <arch>_v###
        auto vpos = rest.rfind("_v");
        if (vpos != std::string::npos && vpos > 0) {
            std::string arch_part = rest.substr(0, vpos);
            std::string num_str = rest.substr(vpos + 2);
            // Check if arch_part is a known architecture
            bool known = false;
            for (int i = 0; i < kNumModelArchs; ++i) {
                if (arch_part == kModelArchNames[i]) { known = true; break; }
            }
            if (known) {
                try {
                    me.version = std::stoi(num_str);
                    me.arch = arch_part;
                    if (me.version > 0) {
                        model_versions_.push_back(me);
                        continue;
                    }
                } catch (...) {}
            }
        }

        // Legacy format: v###  (no arch prefix — treat as "unet")
        if (rest.size() >= 1 && rest[0] == 'v') {
            try {
                me.version = std::stoi(rest.substr(1));
                me.arch = "unet";  // legacy models are all unet
                if (me.version > 0)
                    model_versions_.push_back(me);
            } catch (...) {}
        }
    }

    // Legacy unversioned rig_diffusion.pt counts as unet v001
    if (std::filesystem::exists(model_dir_ + "/rig_diffusion.pt")) {
        bool has_legacy_v1 = false;
        for (auto& e : model_versions_) {
            if (e.arch == "unet" && e.version == 1) { has_legacy_v1 = true; break; }
        }
        if (!has_legacy_v1) {
            ModelEntry me;
            me.version = 1;
            me.arch = "unet";
            me.path = model_dir_ + "/rig_diffusion.pt";
            std::error_code ec;
            me.mtime = std::filesystem::last_write_time(me.path, ec);
            model_versions_.push_back(me);
        }
    }

    // Sort by version number (ascending).
    std::sort(model_versions_.begin(), model_versions_.end(),
        [](const ModelEntry& a, const ModelEntry& b) {
            return a.version < b.version;
        });

    fprintf(stderr, "[AutoRig] Found %d model(s):", (int)model_versions_.size());
    for (auto& e : model_versions_)
        fprintf(stderr, " %s", e.label().c_str());
    fprintf(stderr, "\n");
}

int AutoRigPlugin::latestModelIndex() const {
    // "Latest" = the most recently TRAINED model, i.e. the newest file on disk.
    // We pick by file modification time rather than by name/version number so a
    // freshly trained model of ANY architecture is chosen even if an older
    // architecture happens to carry a higher version number.  Ties (identical
    // mtime) fall back to the higher version.
    int best = -1;
    for (int i = 0; i < (int)model_versions_.size(); ++i) {
        if (best < 0) { best = i; continue; }
        const auto& a = model_versions_[i];
        const auto& b = model_versions_[best];
        if (a.mtime > b.mtime ||
            (a.mtime == b.mtime && a.version > b.version))
            best = i;
    }
    return best;
}

bool AutoRigPlugin::loadModelByIndex(int idx) {
    // NOTE: caller must call scanModelVersions() before this.
    // We don't re-scan here because that would invalidate the idx.

    // -1 means latest
    int target_idx = idx;
    if (target_idx < 0) {
        if (model_versions_.empty()) {
            fprintf(stderr, "[AutoRig] No trained models found — using stub.\n");
            diffusion_model_ = std::make_unique<RigDiffusionModel>();
            diffusion_model_->load("", "cpu");
            model_loaded_idx_ = -1;
            return true;
        }
        target_idx = latestModelIndex();  // most recently trained (by mtime)
    }

    if (target_idx < 0 || target_idx >= (int)model_versions_.size()) {
        fprintf(stderr, "[AutoRig] Model index %d out of range.\n", target_idx);
        return false;
    }

    auto& me = model_versions_[target_idx];
    if (!std::filesystem::exists(me.path)) {
        fprintf(stderr, "[AutoRig] Model %s not found: %s\n",
                me.label().c_str(), me.path.c_str());
        return false;
    }

    fprintf(stderr, "[AutoRig] Loading model %s: %s\n",
            me.label().c_str(), me.path.c_str());
    diffusion_model_ = std::make_unique<RigDiffusionModel>();
    bool ok = diffusion_model_->load(me.path, "cpu");
    if (ok) {
        model_loaded_idx_ = target_idx;
        fprintf(stderr, "[AutoRig] Model %s loaded successfully.\n", me.label().c_str());
    } else {
        fprintf(stderr, "[AutoRig] Model %s LibTorch load failed — stub active.\n",
                me.label().c_str());
        model_loaded_idx_ = -1;
    }
    return ok;
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

    // ---- Normalize the source mesh into canonical Y-up, ~human scale ---------
    //  EVERYTHING downstream assumes the character stands along +Y at roughly
    //  human (metre) scale: the capture cameras frame a Y-up ~1.8 m figure, the
    //  mesh-adaptive joint placement HARD-CODES up_ax=1 (Y), the floor-removal
    //  pass below keys off bbox_min.y, and the predicted skeleton the engine
    //  animates is Y-up ~1.8 m.
    //
    //  Source assets imported from FBX are frequently Z-up and in centimetres.
    //  (e.g. this character: extent ~(6.5, 2.8, 19.2) — clearly Z-up, ~10x).
    //  Nothing previously reoriented them, so the mesh sat in Z-up/×10 space
    //  while the predicted joints sat in Y-up/metre space.  Every mesh vertex
    //  was then ~18 units from the entire (tiny) skeleton, so the nearest-bone
    //  weighting in computeSkinWeights() found ALL bones roughly equidistant
    //  and smeared each vertex across ~4 bones — including opposite-side limbs
    //  (left arm + right arm, both legs).  That cross-binding is what tears the
    //  mesh (e.g. the head splitting) the instant any bone moves.
    //
    //  Fix: rotate the dominant extent axis to +Y, scale to a canonical height,
    //  and centre X/Z with the feet at Y=0 — BEFORE captures, joint prediction,
    //  weighting and IBM export.  The SAME transform is folded into
    //  mesh_node_world_transform_ so the exported inverse-bind matrices (which
    //  still operate on the ORIGINAL, un-normalized vertex positions) keep
    //  reproducing the correct bind pose at runtime.
    {
        const glm::vec3 ext = mesh_.bbox_max - mesh_.bbox_min;

        // Detect the up axis as the largest extent, but KEEP Y when the two
        // largest extents are comparable: a Y-up T-pose has arm-span (X) ≈
        // height (Y), which must not be mistaken for a sideways up-axis.
        int a0 = 0;                       // largest-extent axis
        if (ext[1] > ext[a0]) a0 = 1;
        if (ext[2] > ext[a0]) a0 = 2;
        int a1 = (a0 == 0) ? 1 : 0;       // runner-up axis
        for (int k = 0; k < 3; ++k)
            if (k != a0 && ext[k] > ext[a1]) a1 = k;
        const int up = (ext[a0] > ext[a1] * 1.5f) ? a0 : 1;

        const float height = (ext[up] > 1e-4f) ? ext[up] : 1.7f;

        // Only act when something is actually off-convention: a non-Y up axis,
        // or a wildly non-metric height (cm-scale or tiny).  Leave already-clean
        // Y-up metre assets completely untouched.
        const bool needs_fix = (up != 1) || (height > 4.0f) || (height < 0.4f);
        if (needs_fix) {
            constexpr float kTargetHeight = 1.8f;
            const float s = kTargetHeight / height;

            // Rotation sending the detected up axis to +Y (identity if up==Y).
            glm::mat4 R(1.0f);
            if (up == 2) {
                // Z-up -> Y-up: rotate -90° about X  ((x,y,z) -> (x, z, -y)).
                R = glm::rotate(glm::mat4(1.0f), -1.57079632679f,
                                glm::vec3(1.0f, 0.0f, 0.0f));
            } else if (up == 0) {
                // X-up -> Y-up: rotate +90° about Z  ((x,y,z) -> (-y, x, z)).
                R = glm::rotate(glm::mat4(1.0f),  1.57079632679f,
                                glm::vec3(0.0f, 0.0f, 1.0f));
            }

            const glm::mat4 SR = glm::scale(glm::mat4(1.0f), glm::vec3(s)) * R;

            // Transform the 8 bbox corners to find the post-rotate/scale bounds,
            // then translate so the body is centred on X/Z with feet at Y=0.
            glm::vec3 tmin( 1e30f), tmax(-1e30f);
            for (int c = 0; c < 8; ++c) {
                const glm::vec3 corner(
                    (c & 1) ? mesh_.bbox_max.x : mesh_.bbox_min.x,
                    (c & 2) ? mesh_.bbox_max.y : mesh_.bbox_min.y,
                    (c & 4) ? mesh_.bbox_max.z : mesh_.bbox_min.z);
                const glm::vec3 tc = glm::vec3(SR * glm::vec4(corner, 1.0f));
                tmin = glm::min(tmin, tc);
                tmax = glm::max(tmax, tc);
            }
            const glm::vec3 t(
                -0.5f * (tmin.x + tmax.x),   // centre X
                -tmin.y,                     // feet to Y = 0
                -0.5f * (tmin.z + tmax.z));  // centre Z
            const glm::mat4 norm_xform =
                glm::translate(glm::mat4(1.0f), t) * SR;

            const glm::mat3 normal_mat =
                glm::transpose(glm::inverse(glm::mat3(norm_xform)));
            for (auto& p : mesh_.positions)
                p = glm::vec3(norm_xform * glm::vec4(p, 1.0f));
            for (auto& n : mesh_.normals)
                n = glm::normalize(normal_mat * n);

            // Keep the exported inverse-bind matrices consistent: the exported
            // primitives still carry the ORIGINAL vertex positions, so the
            // original->world matrix must now include this normalization.
            mesh_node_world_transform_ = norm_xform * mesh_node_world_transform_;

            mesh_.recomputeBounds();
            fprintf(stderr,
                "[AutoRig] normalized mesh: detected up_axis=%d src_height=%.3f "
                "scale=%.4f -> bbox (%.2f,%.2f,%.2f)..(%.2f,%.2f,%.2f)\n",
                up, height, s,
                mesh_.bbox_min.x, mesh_.bbox_min.y, mesh_.bbox_min.z,
                mesh_.bbox_max.x, mesh_.bbox_max.y, mesh_.bbox_max.z);
        }
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

    // Try to read camera_distance from training data so auto-rig captures
    // match the scale the model was trained on.
    if (training_camera_dist_ < 0) {
        // Collect candidate directories that might contain _meta.json files.
        std::vector<std::string> search_dirs;
        // Primary: rigs_training subdirectories (always exists after training).
        try {
            std::string rt_dir = model_dir_ + "/../rigs_training";
            if (std::filesystem::exists(rt_dir)) {
                for (auto& d : std::filesystem::directory_iterator(rt_dir)) {
                    if (d.is_directory()) search_dirs.push_back(d.path().string());
                }
            }
        } catch (...) {}
        // Secondary: mesh-specific training_data dir.
        try {
            auto mesh_path = std::filesystem::path(source_mesh_path_);
            std::string td = mesh_path.parent_path().string() + "/training_data/"
                           + mesh_path.stem().string();
            if (std::filesystem::exists(td))
                search_dirs.push_back(td);
        } catch (...) {}

        // Scan for the first _meta.json with a camera_distance field.
        for (auto& dir : search_dirs) {
            if (training_camera_dist_ > 0) break;
            try {
                for (auto& f : std::filesystem::directory_iterator(dir)) {
                    std::string fname = f.path().filename().string();
                    if (fname.find("_meta.json") == std::string::npos) continue;
                    FILE* fp = std::fopen(f.path().string().c_str(), "r");
                    if (!fp) continue;
                    char buf[4096];
                    size_t n = std::fread(buf, 1, sizeof(buf) - 1, fp);
                    std::fclose(fp);
                    buf[n] = '\0';
                    const char* cd = strstr(buf, "\"camera_distance\"");
                    if (cd) {
                        const char* colon = strchr(cd, ':');
                        if (colon) {
                            float val = 0;
                            if (sscanf(colon + 1, " %f", &val) == 1 && val > 0) {
                                training_camera_dist_ = val;
                                fprintf(stderr, "[AutoRig] Training camera_distance=%.4f "
                                        "(from %s)\n", val, f.path().string().c_str());
                                break;
                            }
                        }
                    }
                }
            } catch (...) {}
        }
        if (training_camera_dist_ < 0) {
            training_camera_dist_ = camera_distance_;
            fprintf(stderr, "[AutoRig] No training camera_distance found, "
                    "using current slider value %.4f\n", camera_distance_);
        }
    }

    float capture_dist = (use_training_scale_ && training_camera_dist_ > 0)
                         ? training_camera_dist_ : camera_distance_;
    // Elevation 0° matches the default rig-editor view (eye level).
    captures_ = rasterizer_->captureOrbit(mesh_, num_views, resolution,
                                          0.0f, capture_dist);
    fprintf(stderr, "[AutoRig] captured %d views @ %dx%d (capture_dist=%.4f, "
            "training_dist=%.4f, slider=%.4f, use_training=%d)\n",
        (int)captures_.size(), resolution, resolution, capture_dist,
        training_camera_dist_, camera_distance_, (int)use_training_scale_);

    // Dump auto-rig captures for comparison with training data.
    {
        std::string dump_dir = model_dir_ + "/../debug_autorig_captures";
        try { std::filesystem::create_directories(dump_dir); } catch (...) {}
        for (int i = 0; i < (int)captures_.size(); ++i) {
            const auto& cap = captures_[i];
            if (cap.color.empty()) continue;
            char buf[512];
            std::snprintf(buf, sizeof(buf), "%s/autorig_view%02d_color.png",
                          dump_dir.c_str(), i);
            stbi_write_png(buf, cap.width, cap.height, 3,
                           cap.color.data(), cap.width * 3);
        }
        fprintf(stderr, "[AutoRig] Dumped %d capture images to %s\n",
                (int)captures_.size(), dump_dir.c_str());
    }

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
        float ax = -kJointDefs[j].lr * ca + kJointDefs[j].fb * sa;  // negate lr: character's left = screen right
        float px = sil_cx + ax * sil_h;
        float py = sil_min_y + kJointDefs[j].ry * sil_h;
        state.joints[j].uv = glm::clamp(
            glm::vec2((px + 0.5f) / W, (py + 0.5f) / H),
            glm::vec2(0.0f), glm::vec2(1.0f));
        state.joints[j].edited = false;
    }
}

// ============================================================================
//  initEditableJointsFromModel – pre-fill joints with the trained 2D model's
//  prediction so editing starts from the ML guess (heuristic fallback).
// ============================================================================

void AutoRigPlugin::initEditableJointsFromModel(
    ViewEditState& state, const ViewCapture& cap)
{
    // Heuristic first: sets metadata + a fallback for low-confidence joints.
    initEditableJointsForView(state, cap);

    if (!diffusion_model_ || !diffusion_model_->isLoaded()) return;

    std::vector<ViewCapture> one{cap};
    std::vector<ViewJointPrediction> preds = diffusion_model_->predictBatch(one);
    if (preds.empty()) return;

    const ViewJointPrediction& vp = preds[0];
    for (int j = 0; j < kNumEditJoints && j < (int)vp.joints.size()
                 && j < (int)state.joints.size(); ++j) {
        if (vp.joints[j].confidence >= 0.05f) {
            state.joints[j].uv = glm::clamp(vp.joints[j].peak_uv,
                                            glm::vec2(0.0f), glm::vec2(1.0f));
            state.joints[j].edited = false;   // model guess — user tweaks from here
        }
    }
}

// ============================================================================
//  saveViewForTraining – write the current edited view as one training sample
//  (color + grayscale + depth + silhouette + view_proj + confirmed joints) to
//  assets/rigs_training/<mesh>/, ready for build_3d_labels.py / train_rig3d.py.
// ============================================================================

bool AutoRigPlugin::saveViewForTraining() {
    if (!edit_capture_valid_) { ui_status_ = "Capture a view first"; return false; }
    const ViewCapture& cap = edit_capture_;
    const int W = cap.width, H = cap.height;

    std::string stem = re_selected_path_.empty() ? "untitled"
        : std::filesystem::path(re_selected_path_).stem().string();
    std::string dir = "assets/rigs_training/" + stem;   // engine runs from realworld/
    std::error_code ec; std::filesystem::create_directories(dir, ec);

    int idx = 0;                                        // append after existing views
    if (std::filesystem::exists(dir))
        for (auto& e : std::filesystem::directory_iterator(dir))
            if (e.path().filename().string().find("_meta.json") != std::string::npos) ++idx;

    char path[1024];
    auto fn = [&](const char* suf) {
        std::snprintf(path, sizeof(path), "%s/%s_view%d_%s", dir.c_str(), stem.c_str(), idx, suf);
        return path;
    };

    if (!cap.color.empty())      stbi_write_png(fn("color.png"), W, H, 3, cap.color.data(), W * 3);
    if (!cap.silhouette.empty()) stbi_write_png(fn("silhouette.png"), W, H, 1, cap.silhouette.data(), W);
    if (!cap.depth.empty()) {
        std::vector<uint8_t> d8(cap.depth.size());
        for (size_t i = 0; i < cap.depth.size(); ++i)
            d8[i] = (uint8_t)std::clamp(cap.depth[i] * 255.0f, 0.0f, 255.0f);
        stbi_write_png(fn("depth.png"), W, H, 1, d8.data(), W);
    }
    if (!cap.color.empty()) {
        std::vector<uint8_t> g8(W * H);
        for (int i = 0; i < W * H; ++i) {
            const uint8_t* p = &cap.color[i * 3];
            g8[i] = (uint8_t)std::clamp(0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2], 0.0f, 255.0f);
        }
        stbi_write_png(fn("gray.png"), W, H, 1, g8.data(), W);
    }

    FILE* mf = std::fopen(fn("meta.json"), "w");
    if (!mf) { ui_status_ = "Save failed (meta)"; return false; }
    const auto& names   = getStandardJointNames();
    const auto& parents = getStandardJointParents();
    const float* vp = glm::value_ptr(cap.view_proj);
    fprintf(mf, "{\n  \"view_index\": %d,\n", idx);
    fprintf(mf, "  \"azimuth_deg\": %.4f,\n  \"elevation_deg\": %.4f,\n",
            cap.azimuth_deg, cap.elevation_deg);
    fprintf(mf, "  \"width\": %d,\n  \"height\": %d,\n", W, H);
    fprintf(mf, "  \"view_proj\": [");
    for (int k = 0; k < 16; ++k) fprintf(mf, "%s%.8f", k ? ", " : "", vp[k]);
    fprintf(mf, "],\n  \"joints\": [\n");
    for (int j = 0; j < kNumEditJoints; ++j) {
        glm::vec2 uv = (j < (int)edit_view_.joints.size()) ? edit_view_.joints[j].uv : glm::vec2(0.0f);
        fprintf(mf, "    { \"name\": \"%s\", \"uv\": [%.6f, %.6f], \"edited\": true }%s\n",
                names[j].c_str(), uv.x, uv.y, (j < kNumEditJoints - 1) ? "," : "");
    }
    fprintf(mf, "  ],\n  \"parents\": [");
    for (size_t j = 0; j < parents.size(); ++j) fprintf(mf, "%s%d", j ? ", " : "", parents[j]);
    fprintf(mf, "]\n}\n");
    std::fclose(mf);

    char msg[256];
    std::snprintf(msg, sizeof(msg), "Saved training view %d -> %s", idx, dir.c_str());
    ui_status_ = msg; fprintf(stderr, "[AutoRig] %s\n", msg);
    return true;
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

    std::string loaded_label = (model_loaded_idx_ >= 0 && model_loaded_idx_ < (int)model_versions_.size())
        ? model_versions_[model_loaded_idx_].label() : "stub";
    fprintf(stderr, "[AutoRig] predictJoints: %d captures, each %dx%d, model has %d joints, "
        "loaded=%s, model_dir=%s\n",
        (int)captures_.size(),
        captures_[0].width, captures_[0].height,
        diffusion_model_->numJoints(),
        loaded_label.c_str(),
        model_dir_.c_str());

    ui_status_ = "Running model " + loaded_label + "...";

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
//  Multi-view triangulation helpers (OpenCV-style: DLT + RANSAC)
//
//  Instead of reading a per-view depth buffer, we reconstruct each joint's
//  3D position the way cv2.triangulatePoints does: every view contributes its
//  camera matrix (view_proj) and the 2D image point where the joint was
//  detected, giving two linear constraints on the unknown 3D point.  Stacking
//  the constraints from >=2 views and solving the homogeneous least-squares
//  system (smallest eigenvector of AᵀA via Jacobi) yields the 3D point.
//  RANSAC over view pairs rejects views whose 2D detection is an outlier.
// ============================================================================
namespace {

struct TriObs {
    glm::mat4 P;       // view_proj (world -> clip)
    glm::vec2 ndc;     // observed point in normalised device coords
    float     conf;    // detection confidence (heatmap peak)
    int       view;    // source view index
};

// Eigen-decompose a symmetric 4x4 (cyclic Jacobi).  Columns of V are the
// eigenvectors; eval holds the corresponding eigenvalues.
void jacobiEigen4(double A[4][4], double V[4][4], double eval[4]) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) V[i][j] = (i == j) ? 1.0 : 0.0;

    for (int sweep = 0; sweep < 100; ++sweep) {
        int p = 0, q = 1; double off = 0.0;
        for (int i = 0; i < 4; ++i)
            for (int j = i + 1; j < 4; ++j)
                if (std::fabs(A[i][j]) > off) { off = std::fabs(A[i][j]); p = i; q = j; }
        if (off < 1e-20) break;

        double app = A[p][p], aqq = A[q][q], apq = A[p][q];
        double theta = 0.5 * std::atan2(2.0 * apq, aqq - app);
        double c = std::cos(theta), s = std::sin(theta);

        // A <- Jᵀ A J, applied as column rotation then row rotation.
        double T[4][4];
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) T[i][j] = A[i][j];
        for (int i = 0; i < 4; ++i) {
            T[i][p] = c * A[i][p] - s * A[i][q];
            T[i][q] = s * A[i][p] + c * A[i][q];
        }
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) A[i][j] = T[i][j];
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) T[i][j] = A[i][j];
        for (int j = 0; j < 4; ++j) {
            T[p][j] = c * A[p][j] - s * A[q][j];
            T[q][j] = s * A[p][j] + c * A[q][j];
        }
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) A[i][j] = T[i][j];

        double Vt[4][4];
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) Vt[i][j] = V[i][j];
        for (int i = 0; i < 4; ++i) {
            Vt[i][p] = c * V[i][p] - s * V[i][q];
            Vt[i][q] = s * V[i][p] + c * V[i][q];
        }
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) V[i][j] = Vt[i][j];
    }
    for (int i = 0; i < 4; ++i) eval[i] = A[i][i];
}

// Weighted DLT triangulation over the selected observations.
bool triangulateDLT(const std::vector<TriObs>& obs,
                    const std::vector<int>& sel, glm::vec3& out) {
    double ATA[4][4] = {{0}};
    int used = 0;
    for (int idx : sel) {
        const glm::mat4& P = obs[idx].P;
        // Rows of P (glm is column-major: P[col][row]).
        const double r0[4] = { P[0][0], P[1][0], P[2][0], P[3][0] };
        const double r1[4] = { P[0][1], P[1][1], P[2][1], P[3][1] };
        const double r3[4] = { P[0][3], P[1][3], P[2][3], P[3][3] };
        const double w = std::sqrt((double)std::max(obs[idx].conf, 1e-3f));
        double ex[4], ey[4];
        for (int k = 0; k < 4; ++k) {
            ex[k] = w * (obs[idx].ndc.x * r3[k] - r0[k]);
            ey[k] = w * (obs[idx].ndc.y * r3[k] - r1[k]);
        }
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) ATA[a][b] += ex[a] * ex[b] + ey[a] * ey[b];
        ++used;
    }
    if (used < 2) return false;

    double V[4][4], ev[4];
    jacobiEigen4(ATA, V, ev);
    int kmin = 0;
    for (int k = 1; k < 4; ++k) if (ev[k] < ev[kmin]) kmin = k;
    const double Xh[4] = { V[0][kmin], V[1][kmin], V[2][kmin], V[3][kmin] };
    if (std::fabs(Xh[3]) < 1e-12) return false;
    out = glm::vec3(Xh[0] / Xh[3], Xh[1] / Xh[3], Xh[2] / Xh[3]);
    return std::isfinite(out.x) && std::isfinite(out.y) && std::isfinite(out.z);
}

// NDC reprojection error of a candidate 3D point in one view.
float reprojNdcErr(const glm::mat4& P, const glm::vec3& X, const glm::vec2& ndc) {
    glm::vec4 c = P * glm::vec4(X, 1.0f);
    if (std::fabs(c.w) < 1e-9f) return 1e9f;
    return glm::length(glm::vec2(c.x / c.w, c.y / c.w) - ndc);
}

// RANSAC over view pairs, then refine a weighted DLT on the inliers.
bool triangulateRansac(const std::vector<TriObs>& obs, float tol, glm::vec3& out) {
    const int M = (int)obs.size();
    if (M < 2) return false;
    if (M == 2) return triangulateDLT(obs, { 0, 1 }, out);

    std::vector<int> best_inliers;
    for (int i = 0; i < M; ++i) {
        for (int j = i + 1; j < M; ++j) {
            glm::vec3 X;
            if (!triangulateDLT(obs, { i, j }, X)) continue;
            std::vector<int> inl;
            for (int k = 0; k < M; ++k)
                if (reprojNdcErr(obs[k].P, X, obs[k].ndc) < tol) inl.push_back(k);
            if (inl.size() > best_inliers.size()) best_inliers.swap(inl);
        }
    }
    if (best_inliers.size() < 2) return false;   // no consensus -> caller falls back
    return triangulateDLT(obs, best_inliers, out);
}

// Depth-buffer unprojection fallback (the original method), per joint/view.
bool depthUnproject(const ViewCapture& cap,
                    const JointHeatmap& jh, glm::vec3& out) {
    if (cap.depth.empty() || cap.width <= 0 || cap.height <= 0) return false;
    float ndc_x = jh.peak_uv.x * 2.0f - 1.0f;
    float ndc_y = jh.peak_uv.y * 2.0f - 1.0f;
    int px = std::clamp((int)(jh.peak_uv.x * cap.width),  0, cap.width  - 1);
    int py = std::clamp((int)(jh.peak_uv.y * cap.height), 0, cap.height - 1);
    float depth_val = cap.depth[py * cap.width + px];
    if (depth_val >= 0.999f) {
        float best = 1.0f; int r = cap.width / 8;
        for (int dy = -r; dy <= r; ++dy)
            for (int dx = -r; dx <= r; ++dx) {
                int sx = std::clamp(px + dx, 0, cap.width  - 1);
                int sy = std::clamp(py + dy, 0, cap.height - 1);
                float dv = cap.depth[sy * cap.width + sx];
                if (dv < best) { best = dv;
                    ndc_x = (sx + 0.5f) / cap.width  * 2.0f - 1.0f;
                    ndc_y = (sy + 0.5f) / cap.height * 2.0f - 1.0f; }
            }
        depth_val = best;
    }
    if (depth_val >= 0.999f) return false;
    float ndc_z = depth_val * 2.0f - 1.0f;
    glm::vec4 clip = glm::inverse(cap.view_proj) * glm::vec4(ndc_x, ndc_y, ndc_z, 1.0f);
    if (std::fabs(clip.w) < 1e-9f) return false;
    out = glm::vec3(clip) / clip.w;
    return true;
}

}  // namespace

// ============================================================================
//  fuseAndBuildSkeleton – triangulate 2D predictions into 3D, then fuse.
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

    // ── Multi-view triangulation (RANSAC + DLT), depth as fallback ────
    // Gather every confident 2D detection of each joint across views,
    // each carrying its camera (view_proj) and normalised image point.
    constexpr float kConfMin      = 0.05f;
    constexpr float kInlierNdcTol = 0.05f;   // NDC reprojection tolerance

    std::vector<std::vector<TriObs>> obs(J);
    for (auto& vp : view_predictions_) {
        if (vp.view_idx < 0 || vp.view_idx >= (int)captures_.size()) continue;
        const auto& cap = captures_[vp.view_idx];
        for (int j = 0; j < (int)vp.joints.size() && j < J; ++j) {
            const auto& jh = vp.joints[j];
            if (jh.confidence < kConfMin) continue;
            obs[j].push_back(TriObs{
                cap.view_proj,
                glm::vec2(jh.peak_uv.x * 2.0f - 1.0f, jh.peak_uv.y * 2.0f - 1.0f),
                jh.confidence, vp.view_idx });
        }
    }

    int n_tri = 0, n_depth = 0, n_none = 0;
    for (int j = 0; j < J; ++j) {
        glm::vec3 X;
        // 1) Multi-view triangulation when >=2 confident views agree.
        if (obs[j].size() >= 2 && triangulateRansac(obs[j], kInlierNdcTol, X)) {
            accum[j].pos_sum    = X;
            accum[j].weight_sum = 1.0f;
            ++n_tri;
            continue;
        }
        // 2) Fallback: depth-buffer unprojection, confidence-averaged.
        glm::vec3 psum(0.0f); float wsum = 0.0f;
        for (auto& vp : view_predictions_) {
            if (vp.view_idx < 0 || vp.view_idx >= (int)captures_.size()) continue;
            if (j >= (int)vp.joints.size()) continue;
            const auto& cap = captures_[vp.view_idx];
            const auto& jh  = vp.joints[j];
            if (jh.confidence < kConfMin) continue;
            glm::vec3 wp;
            if (depthUnproject(cap, jh, wp)) { psum += wp * jh.confidence; wsum += jh.confidence; }
        }
        if (wsum > 1e-6f) { accum[j].pos_sum = psum; accum[j].weight_sum = wsum; ++n_depth; }
        else ++n_none;   // 3) no data -> mesh-adaptive default fills this in below
    }
    fprintf(stderr, "[AutoRig] joint 3D recon: %d triangulated (multi-view DLT), "
            "%d depth-fallback, %d defaulted\n", n_tri, n_depth, n_none);

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

    constexpr int up_ax = 1;      // Y = up (mesh is Y-up after node transforms)
    // Auto-detect the LEFT/RIGHT (arm-span) axis.  Of the two horizontal axes,
    // the one with the larger extent across the UPPER body (shoulders/torso,
    // above the flared skirt) is left/right; the narrower is front/back depth.
    // The old code hard-coded right=X, but exported assets vary -- this rig's
    // arm-span is along Z, and placing limbs on the wrong axis throws the arms
    // and legs front/back through the body instead of out to the sides (the
    // "tangle" of crossing bones).
    int right_ax = 0, fwd_ax = 2;
    {
        float ylo = mesh_.bbox_min.y + 0.55f * (mesh_.bbox_max.y - mesh_.bbox_min.y);
        float xmn = 1e30f, xmx = -1e30f, zmn = 1e30f, zmx = -1e30f;
        for (const auto& p : mesh_.positions) {
            if (p.y < ylo) continue;
            xmn = std::min(xmn, p.x); xmx = std::max(xmx, p.x);
            zmn = std::min(zmn, p.z); zmx = std::max(zmx, p.z);
        }
        if ((zmx - zmn) > (xmx - xmn)) { right_ax = 2; fwd_ax = 0; }
        fprintf(stderr, "[AutoRig] arm-span axis = %s (upper-body extent X=%.3f Z=%.3f)\n",
            right_ax == 2 ? "Z" : "X", xmx - xmn, zmx - zmn);
    }

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

    // Arm joints, arms-DOWN aware.  Each joint is placed at a DESCENDING
    // height (shoulder -> upper_arm -> lower_arm -> hand walk down the body)
    // and pushed toward the silhouette edge along the left/right axis by
    // spread_frac.  For an arms-at-side pose the silhouette at hand height
    // already includes the hand at its outer edge, so sampling the LOCAL
    // extent at each height naturally tracks the arm down the side instead of
    // assuming a horizontal T-pose (which left every arm joint at shoulder
    // height, sticking the arms straight out).
    auto armPos = [&](float norm_h, float spread_frac, bool is_left) -> glm::vec3 {
        int si = hSlice(norm_h);
        float lo = sliceRightMin(si), hi = sliceRightMax(si);
        float mid = (lo + hi) * 0.5f, hw = (hi - lo) * 0.5f;
        hw = std::max(hw, shoulder_hw * 0.5f);   // guard against a pinched slice
        float pos_r = is_left ? mid - spread_frac * hw
                              : mid + spread_frac * hw;
        return makePos(up_base + height * norm_h, pos_r, sliceCenter(si, fwd_ax));
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
        /* 5  left_shoulder   */ armPos(0.80f, 0.45f, true),
        /* 6  left_upper_arm  */ armPos(0.70f, 0.75f, true),
        /* 7  left_lower_arm  */ armPos(0.60f, 0.90f, true),
        /* 8  left_hand       */ armPos(0.50f, 0.98f, true),
        /* 9  right_shoulder  */ armPos(0.80f, 0.45f, false),
        /* 10 right_upper_arm */ armPos(0.70f, 0.75f, false),
        /* 11 right_lower_arm */ armPos(0.60f, 0.90f, false),
        /* 12 right_hand      */ armPos(0.50f, 0.98f, false),
        /* 13 left_upper_leg  */ legPos(0.50f, true),
        /* 14 left_lower_leg  */ legPos(0.26f, true),
        /* 15 left_foot       */ legPos(0.03f, true),
        /* 16 right_upper_leg */ legPos(0.50f, false),
        /* 17 right_lower_leg */ legPos(0.26f, false),
        /* 18 right_foot      */ legPos(0.03f, false),
    };

    // Anchor EVERY joint to the silhouette-based geometric placement, which is
    // derived directly from the mesh and cannot be left/right-swapped.  The
    // multi-view 3D estimate is accepted only when it AGREES with the geometry
    // (the model's 2D limb detections are the unreliable part; even centerline
    // detections can drift).  This is what keeps the skeleton from tangling.
    std::string dump;
    {
        char hdr[160];
        snprintf(hdr, sizeof(hdr),
            "[autorig skeleton] arm_span_axis=%s  height=%.3f  up_base=%.3f\n",
            right_ax == 2 ? "Z" : "X", height, up_base);
        dump += hdr;
    }
    for (int j = 0; j < J; ++j) {
        Joint& jt = skeleton_.joints[j];
        jt.name   = (j < (int)names.size()) ? names[j] : ("joint_" + std::to_string(j));
        jt.parent = (j < (int)parents.size()) ? parents[j] : -1;

        const glm::vec3 geom = (j < 19) ? default_pos[j] : center;
        glm::vec3 mlp(0.0f);
        const float w = accum[j].weight_sum;
        bool used_ml = false;
        if (w > 1e-6f) {
            glm::vec3 p = accum[j].pos_sum / w;
            mlp = p;
            const glm::vec3 lo = mesh_.bbox_min - 0.15f * extent;
            const glm::vec3 hi = mesh_.bbox_max + 0.15f * extent;
            const bool inside =
                p.x >= lo.x && p.y >= lo.y && p.z >= lo.z &&
                p.x <= hi.x && p.y <= hi.y && p.z <= hi.z &&
                std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
            // LEFT/RIGHT SIDE CONSTRAINT (the actual fix for the tangle):
            // the model's 2D detections confuse left vs right in side/back
            // views, so a "left_*" joint sometimes triangulates onto the
            // body's RIGHT, crossing a bone through the chest.  Each joint's
            // anatomical side is encoded in its geometric placement, so we
            // reject any 3D estimate that lands on the opposite side of the
            // sagittal plane (centerline along the arm-span axis) and fall
            // back to geometry.  Centerline joints (<=4) have no side.
            const float c = default_pos[0][right_ax];   // hips = centerline ref
            const bool same_side = (j <= 4) ||
                ((p[right_ax] - c) * (geom[right_ax] - c) >= 0.0f);
            const float tol = (j <= 4 ? 0.18f : 0.30f) * height;
            const bool agrees = (j < 19) && glm::length(p - geom) < tol;
            if (inside && same_side && agrees) { jt.position = p; used_ml = true; }
            else                               { jt.position = geom; }
        } else {
            jt.position = geom;
        }

        char line[256];
        snprintf(line, sizeof(line),
            "  %2d %-16s pos=(%.3f,%.3f,%.3f) src=%-4s w=%.3f "
            "geom=(%.3f,%.3f,%.3f) ml=(%.3f,%.3f,%.3f)\n",
            j, jt.name.c_str(), jt.position.x, jt.position.y, jt.position.z,
            used_ml ? "ML" : "geom", w,
            geom.x, geom.y, geom.z, mlp.x, mlp.y, mlp.z);
        dump += line;
        fprintf(stderr, "[AutoRig] %s", line);
    }
    // ── Enforce left/right symmetry on the limb pairs ────────────────
    // The model's per-side noise leaves the arms/legs slightly mismatched
    // (one upper-arm lower than the other, a foot pulled toward centre).
    // For an upright character the limbs should mirror, so we average the
    // up/depth components of each pair and mirror the left/right offset
    // about the body centerline.  This removes the residual "slightly off"
    // asymmetry without disturbing the (centerline) spine.
    {
        static const int pr[7][2] = {
            {5,9},{6,10},{7,11},{8,12},{13,16},{14,17},{15,18} };
        const float cref = default_pos[0][right_ax];
        for (const auto& q : pr) {
            if (q[0] >= J || q[1] >= J) continue;
            glm::vec3& L = skeleton_.joints[q[0]].position;
            glm::vec3& R = skeleton_.joints[q[1]].position;
            const glm::vec3 avg = (L + R) * 0.5f;
            const float mag = 0.5f * (std::fabs(L[right_ax] - cref) +
                                      std::fabs(R[right_ax] - cref));
            L = avg; R = avg;
            L[right_ax] = cref - mag;   // left  = negative side
            R[right_ax] = cref + mag;   // right = positive side
        }
    }

    // Write a ground-truth dump I can inspect outside the engine (final
    // positions, after symmetry).  Path is relative to the working
    // directory (the engine runs from realworld/).
    {
        std::ofstream sf("assets/debug_autorig_skeleton.txt");
        if (sf) {
            sf << dump << "--- after symmetry ---\n";
            for (int j = 0; j < (int)skeleton_.joints.size(); ++j) {
                const auto& p = skeleton_.joints[j].position;
                char l2[160];
                snprintf(l2, sizeof(l2), "  %2d %-16s (%.3f,%.3f,%.3f)\n",
                    j, skeleton_.joints[j].name.c_str(), p.x, p.y, p.z);
                sf << l2;
            }
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
    // ── Pre-skinning: CENTER each joint on its limb's MEDIAL AXIS ──
    // Predicted joints sit off-centre, so the bones zig-zag (unnatural) and a
    // foot/ankle joint can drift toward the OTHER foot — its seed rays then
    // strike the wrong foot and the weight bleeds across.  For each joint we take
    // the limb axis (parent->joint->child), fire rays around the perpendicular
    // CROSS-SECTION, and slide the joint to the centre of that cross-section.
    // Centred joints lie on the limb's medial axis: straight natural bones, well
    // inside the body, and seed rays that can only reach the joint's OWN limb.
    {
        const int nj = (int)skeleton_.joints.size();
        const size_t ntri = mesh_.indices.size() / 3;
        std::vector<std::vector<int>> kids(nj);
        for (int j = 0; j < nj; ++j) {
            const int p = skeleton_.joints[j].parent;
            if (p >= 0 && p < nj) kids[p].push_back(j);
        }
        // Moller-Trumbore ray vs original-mesh triangle (nearest positive t).
        auto rayTriM = [&](const glm::vec3& o, const glm::vec3& d,
                           int ia, int ib, int ic, float& outT) -> bool {
            const glm::vec3& v0 = mesh_.positions[ia];
            const glm::vec3& v1 = mesh_.positions[ib];
            const glm::vec3& v2 = mesh_.positions[ic];
            const glm::vec3 e1 = v1 - v0, e2 = v2 - v0;
            const glm::vec3 pv = glm::cross(d, e2);
            const float det = glm::dot(e1, pv);
            if (std::fabs(det) < 1e-12f) return false;
            const float inv = 1.0f / det;
            const glm::vec3 tv = o - v0;
            const float u = glm::dot(tv, pv) * inv;
            if (u < -1e-5f || u > 1.0f + 1e-5f) return false;
            const glm::vec3 qv = glm::cross(tv, e1);
            const float vv = glm::dot(d, qv) * inv;
            if (vv < -1e-5f || u + vv > 1.0f + 1e-5f) return false;
            const float tt = glm::dot(e2, qv) * inv;
            if (tt <= 1e-5f) return false;
            outT = tt; return true;
        };
        const int kCenterPasses = 3;
        int centred = 0;
        for (int pass = 0; pass < kCenterPasses; ++pass) {
            for (int j = 0; j < nj; ++j) {
                const glm::vec3 p   = skeleton_.joints[j].position;
                const int       par = skeleton_.joints[j].parent;
                // limb axis at this joint
                glm::vec3 axis(0.0f);
                if (par >= 0) {
                    const glm::vec3 din = p - skeleton_.joints[par].position;
                    if (glm::length(din) > 1e-6f) axis += glm::normalize(din);
                }
                for (int c : kids[j]) {
                    const glm::vec3 dout = skeleton_.joints[c].position - p;
                    if (glm::length(dout) > 1e-6f) axis += glm::normalize(dout);
                }
                float al = glm::length(axis);
                if (al < 1e-6f) axis = glm::vec3(0.0f, 1.0f, 0.0f); else axis /= al;
                // orthonormal basis (u,v) spanning the cross-section plane
                const glm::vec3 ref = (std::fabs(axis.y) < 0.9f)
                    ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
                const glm::vec3 u = glm::normalize(glm::cross(axis, ref));
                const glm::vec3 v = glm::cross(axis, u);
                const int N = 16;
                glm::vec3 sum(0.0f); int hits = 0; float sumd = 0.0f;
                for (int s = 0; s < N; ++s) {
                    const float ang = 6.2831853f * (float)s / (float)N;
                    const glm::vec3 dir = u * std::cos(ang) + v * std::sin(ang);
                    float bestT = 1e30f; bool got = false;
                    for (size_t t = 0; t < ntri; ++t) {
                        float tt;
                        if (!rayTriM(p, dir, mesh_.indices[t*3], mesh_.indices[t*3+1],
                                     mesh_.indices[t*3+2], tt)) continue;
                        if (tt < bestT) { bestT = tt; got = true; }
                    }
                    if (got) { sum += p + dir * bestT; ++hits; sumd += bestT; }
                }
                if (hits >= N / 2) {
                    const glm::vec3 center = sum / (float)hits;
                    glm::vec3 off = center - p;
                    off -= axis * glm::dot(off, axis);             // in-plane only
                    const float maxMove = 0.5f * (sumd / (float)hits);
                    const float ol = glm::length(off);
                    if (ol > maxMove && ol > 1e-9f) off *= (maxMove / ol);
                    skeleton_.joints[j].position = p + off;
                    if (pass == 0) ++centred;
                }
            }
        }
        if (centred > 0) {
            char cb[160];
            std::snprintf(cb, sizeof(cb),
                "[AutoRig] centred %d joint(s) on the limb medial axis "
                "(straighter bones, no cross-limb seeding).", centred);
            std::cout << cb << std::endl;
        }
    }

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

    const int nv = static_cast<int>(mesh_.positions.size());
    const int nj = static_cast<int>(skeleton_.joints.size());
    skin_weights_ = SkinWeights{};
    skin_weights_.per_vertex.resize(nv);

    // ONE bone per joint.  A bone is the segment parent→joint; it is the
    // ray-cast origin line for seeding and the source of the tau falloff width
    // (ALL weighting distances are GEODESIC/on-surface — no euclidean
    // point-to-segment distance anywhere in this pass).
    //
    // LEAF joints (hands, head, toe tips) get the segment EXTENDED past the
    // joint along the incoming bone direction.  A leaf owns a whole terminal
    // extremity (the hand past the wrist, the skull past the neck) but with the
    // segment ending AT the joint, the seed ray fired down that extremity is
    // the longest of the bunch and the outlier cut below (cutoff = 2×median
    // hit) rejects it — so the extremity is never seeded and the geodesic never
    // reaches it.  It then fell back to the euclidean nearest-skin (the thigh,
    // with arms down) and inherited LEG weights.  Extending the segment marches
    // the seed samples into the extremity so the geodesic genuinely covers it.
    std::vector<int> childCount(nj, 0);
    for (int j = 0; j < nj; ++j) {
        const int p = skeleton_.joints[j].parent;
        if (p >= 0 && p < nj) ++childCount[p];
    }
    struct BoneSeg { int joint_idx; glm::vec3 a, b; };
    std::vector<BoneSeg> bones(nj);
    for (int j = 0; j < nj; ++j) {
        const int p = skeleton_.joints[j].parent;
        const glm::vec3 jp = skeleton_.joints[j].position;
        const glm::vec3 a  = (p >= 0) ? skeleton_.joints[p].position : jp;
        glm::vec3 b = jp;
        if (p >= 0 && childCount[j] == 0) {                // leaf → extend past
            const glm::vec3 dir = jp - skeleton_.joints[p].position;
            const float len = glm::length(dir);
            if (len > 1e-6f) b = jp + (dir / len) * (len * 0.75f);
        }
        bones[j] = { j, a, b };
    }

    const int nb = static_cast<int>(bones.size());
    if (nb == 0) return false;

    // ── Build surface connectivity by MANIFOLD EDGE matching ──
    // The two feet have NO real connection, yet the OLD spatial-cell weld fused
    // them: it merged any two vertices that shared a tiny grid cell as long as
    // their normals were not opposing.  Where the separate feet sit close
    // together their SAME-FACING surfaces (soles point down, tops point up, so
    // the normals AGREE) got welded into shared vertices — inventing a geodesic
    // shortcut and bleeding left/right foot weights.
    //
    // Instead, two vertices are merged ONLY when they are the endpoints of a
    // COINCIDENT triangle EDGE — the seam where two triangles of the SAME
    // surface meet.  Adjacent triangles of one surface always share such an edge
    // (identical endpoint positions), so every connected surface is rebuilt;
    // two pieces that merely TOUCH (the feet) share no tessellated edge and stay
    // SEPARATE islands, so the per-bone geodesic can never cross between them.
    const glm::vec3 _bext = mesh_.bbox_max - mesh_.bbox_min;
    const double weld_eps = std::max((double)glm::length(_bext) * 1e-4, 1e-9);
    auto cellKey = [weld_eps](const glm::vec3& p) -> uint64_t {
        uint64_t h = 1469598103934665603ull;
        for (int i = 0; i < 3; ++i) {
            const int64_t qi = (int64_t)std::llround((double)p[i] / weld_eps);
            h ^= (uint64_t)qi; h *= 1099511628211ull;
        }
        return h;
    };
    std::vector<uint64_t> vcell(nv);
    for (int i = 0; i < nv; ++i) vcell[i] = cellKey(mesh_.positions[i]);

    // Union-find over the original vertices.
    std::vector<int> uf(nv);
    for (int i = 0; i < nv; ++i) uf[i] = i;
    std::function<int(int)> ufFind = [&](int x) {
        while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; }
        return x;
    };
    auto ufUnion = [&](int a, int b) { uf[ufFind(a)] = ufFind(b); };

    // Order-independent key for an edge built from its two endpoint cell keys.
    auto edgeKey = [](uint64_t ka, uint64_t kb) -> uint64_t {
        const uint64_t lo = ka < kb ? ka : kb, hi = ka < kb ? kb : ka;
        uint64_t h = 1469598103934665603ull;
        h ^= lo; h *= 1099511628211ull;
        h ^= hi; h *= 1099511628211ull;
        return h;
    };
    // First triangle to claim an edge stores its endpoints; later triangles that
    // share the SAME spatial edge weld their matching endpoints to it.
    std::unordered_map<uint64_t, std::pair<int,int>> edgeRep;
    edgeRep.reserve(mesh_.indices.size());
    auto stitch = [&](int a, int b) {
        if (vcell[a] == vcell[b]) return;                 // degenerate edge
        const uint64_t ek = edgeKey(vcell[a], vcell[b]);
        auto it = edgeRep.find(ek);
        if (it == edgeRep.end()) { edgeRep.emplace(ek, std::make_pair(a, b)); return; }
        const int ra = it->second.first, rb = it->second.second;
        if (vcell[a] == vcell[ra]) { ufUnion(a, ra); ufUnion(b, rb); }
        else                       { ufUnion(a, rb); ufUnion(b, ra); }
    };
    for (size_t t = 0; t + 2 < mesh_.indices.size(); t += 3) {
        const int a = mesh_.indices[t], b = mesh_.indices[t + 1],
                  c = mesh_.indices[t + 2];
        stitch(a, b); stitch(b, c); stitch(c, a);
    }

    // Compact union-find roots into dense welded ids → wid / wpos.
    std::vector<int>       wid(nv);
    std::vector<glm::vec3> wpos;
    {
        std::unordered_map<int,int> root2id;
        root2id.reserve(static_cast<size_t>(nv));
        for (int i = 0; i < nv; ++i) {
            const int r = ufFind(i);
            auto it = root2id.find(r);
            if (it == root2id.end()) {
                const int id = static_cast<int>(wpos.size());
                root2id.emplace(r, id);
                wpos.push_back(mesh_.positions[i]);
                wid[i] = id;
            } else {
                wid[i] = it->second;
            }
        }
    }
    const int W = static_cast<int>(wpos.size());

    // Welded triangle list (shared by the layer split, the skin ray test and
    // the geodesic graph).
    std::vector<glm::ivec3> tris;
    tris.reserve(mesh_.indices.size() / 3);
    for (size_t t = 0; t + 2 < mesh_.indices.size(); t += 3)
        tris.push_back({ wid[mesh_.indices[t]], wid[mesh_.indices[t + 1]],
                         wid[mesh_.indices[t + 2]] });

    // ── LAYER SPLIT: connected components over the welded surface ──
    // A character often arrives as several stacked layers — skin, clothing,
    // hair — that we cannot tell apart from metadata.  Exact-position welding
    // keeps them as SEPARATE connected components (different layers rarely share
    // a vertex), so a union-find over the triangle edges recovers one component
    // per layer-piece.
    std::vector<int> comp(W);
    for (int i = 0; i < W; ++i) comp[i] = i;
    auto findc = [&](int x) {
        while (comp[x] != x) { comp[x] = comp[comp[x]]; x = comp[x]; }
        return x;
    };
    auto unite = [&](int a, int b) { comp[findc(a)] = findc(b); };
    for (const auto& t : tris) { unite(t.x, t.y); unite(t.y, t.z); }
    for (int i = 0; i < W; ++i) comp[i] = findc(i);

    // Ray vs welded triangle (Möller–Trumbore); fills hit distance + barycentric
    // (u,v) for the v1,v2 corners (v0 weight = 1-u-v).
    auto rayTri = [&](const glm::vec3& o, const glm::vec3& d,
                      const glm::ivec3& t, float& outT,
                      float& bu, float& bv) -> bool {
        const glm::vec3& v0 = wpos[t.x]; const glm::vec3& v1 = wpos[t.y];
        const glm::vec3& v2 = wpos[t.z];
        const glm::vec3 e1 = v1 - v0, e2 = v2 - v0;
        const glm::vec3 pv = glm::cross(d, e2);
        const float det = glm::dot(e1, pv);
        if (std::fabs(det) < 1e-12f) return false;
        const float inv = 1.0f / det;
        const glm::vec3 tvec = o - v0;
        const float u = glm::dot(tvec, pv) * inv;
        if (u < -1e-5f || u > 1.0f + 1e-5f) return false;
        const glm::vec3 qv = glm::cross(tvec, e1);
        const float v = glm::dot(d, qv) * inv;
        if (v < -1e-5f || u + v > 1.0f + 1e-5f) return false;
        const float tt = glm::dot(e2, qv) * inv;
        if (tt <= 1e-6f) return false;
        outT = tt; bu = u; bv = v;
        return true;
    };

    // ── SKIN CLASSIFICATION by joint ray casting ──
    // Skin is the INNERMOST layer: a ray fired outward from a joint (which sits
    // inside the body) pierces skin first, then any clothing, then hair.  So
    // over a sphere of directions from every joint, the NEAREST triangle hit
    // belongs to skin.  Tally per component how often it is that nearest hit
    // versus merely crossed; a component whose hits are usually the closest
    // (>50%) is skin.  Cloth/hair, always sitting behind skin, score low.  A
    // single-component mesh (no separate layers) scores 100% → all skin, i.e.
    // identical behaviour to before.
    const int kDirs = 128;
    std::vector<long> firstHits(W, 0), totalHits(W, 0);   // keyed by comp rep
    for (int j = 0; j < nj; ++j) {
        const glm::vec3 o = skeleton_.joints[j].position;
        for (int s = 0; s < kDirs; ++s) {
            const float k     = static_cast<float>(s) + 0.5f;
            const float phi   = std::acos(1.0f - 2.0f * k / kDirs);
            const float theta = 2.39996323f * static_cast<float>(s);  // golden
            const glm::vec3 d(std::sin(phi) * std::cos(theta),
                              std::sin(phi) * std::sin(theta),
                              std::cos(phi));
            float bestT = 1e30f; int bestComp = -1;
            for (const auto& t : tris) {
                float tt, bu, bv;
                if (!rayTri(o, d, t, tt, bu, bv)) continue;
                ++totalHits[comp[t.x]];
                if (tt < bestT) { bestT = tt; bestComp = comp[t.x]; }
            }
            if (bestComp >= 0) ++firstHits[bestComp];
        }
    }
    // Decide skin per component representative, then label every vertex (O(W)).
    // Never-hit pockets default to skin so they are still rigged directly; only
    // components actually struck by rays count as distinct layers in the log.
    std::vector<char> compSkin(W, 0);
    int n_layers = 0;
    for (int i = 0; i < W; ++i) {
        if (comp[i] != i) continue;                       // representatives
        if (totalHits[i] > 0) ++n_layers;
        compSkin[i] = (totalHits[i] == 0) ||
                      (static_cast<float>(firstHits[i]) >=
                       0.5f * static_cast<float>(totalHits[i])) ? 1 : 0;
    }
    std::vector<bool> isSkin(W, false);
    int skin_verts = 0;
    for (int w = 0; w < W; ++w)
        if (compSkin[comp[w]]) { isSkin[w] = true; ++skin_verts; }

    // Stash the classification (pre-override) per ORIGINAL vertex for the
    // "Skin layer only" debug draw: 1 = base/innermost skin, 0 = cloth/hair.
    base_skin_vert_.assign(nv, 1);
    for (int v = 0; v < nv; ++v)
        base_skin_vert_[v] = isSkin[wid[v]] ? 1 : 0;
    // Keep the REAL per-welded-vertex classification too: the base-skin layer
    // is weighted PURELY by geodesic distance (no euclidean fallback), so the
    // last-resort euclidean bind below must skip these verts.
    std::vector<char> realSkinW(isSkin.begin(), isSkin.end());

    // -- FORCE ALL-SKIN (layer split DISABLED) --
    // This asset is a single BASE-SKIN model with NO separate cloth/hair layer.
    // The ray classifier above can wrongly label a disconnected base-skin
    // submesh (e.g. a dress / lower body) as "cloth" and then PROJECT it, which
    // collapses its weights (the uniform-colour region).  Override it: mark
    // EVERY welded vertex SKIN so the geodesic field runs over the WHOLE surface
    // and the projection pass below is SKIPPED (skin_verts == W).  Pure geodesic
    // everywhere, no projection.
    isSkin.assign(W, true);
    skin_verts = W;

    // ── Geodesic graph from SKIN triangles only ──
    // The surface distance must run on the skin manifold; clothing/hair edges
    // would add shortcuts that corrupt it, so non-skin triangles are excluded.
    std::vector<std::vector<std::pair<int, float>>> adj(W);
    auto addEdge = [&](int a, int b) {
        if (a == b) return;
        const float w = glm::length(wpos[a] - wpos[b]);
        adj[a].push_back({b, w});
        adj[b].push_back({a, w});
    };
    for (const auto& t : tris) {
        if (!isSkin[t.x]) continue;                       // skin triangles only
        addEdge(t.x, t.y); addEdge(t.y, t.z); addEdge(t.z, t.x);
    }
    // ── Bridge DISCONNECTED components into the geodesic graph ──
    // The mesh arrives as several welded islands (the toe box separate from the
    // foot, a foot/shoe separate from the leg, an eye, a stray patch).  The
    // per-bone Dijkstra below cannot cross a gap, so an isolated island stays at
    // INFINITE geodesic distance (reads as 0 in the Dist view) and the foot
    // bone's field never reaches e.g. the toes.
    //
    // Connect each component to its NEAREST vertex in ANY OTHER component, with
    // a single short edge, but ONLY across a SMALL gap.  "Nearest other part"
    // keeps the chain on the same limb (the toe's nearest part is the foot, the
    // foot's nearest is its own ankle/leg — each closer than the OTHER foot), so
    // the geodesic flows toe→foot→leg→body.  The small-gap cap then blocks the
    // wide foot-to-foot gap, so no cross-limb shortcut / weight bleed returns.
    {
        std::unordered_map<int, int> csz;
        for (int w = 0; w < W; ++w) ++csz[comp[w]];
        int bigC = -1, bigN = -1;
        for (const auto& kv : csz) if (kv.second > bigN) { bigN = kv.second; bigC = kv.first; }
        if ((int)csz.size() > 1) {
            const glm::vec3 ext = mesh_.bbox_max - mesh_.bbox_min;
            const float diag = std::max(glm::length(ext), 1e-4f);
            // Max gap to bridge.  Seam cracks (toe↔foot) are sub-cm; the
            // foot↔other-foot gap is far larger, so this stays well between them.
            const float kMaxBridge  = diag * 0.03f;
            const float kMaxBridge2 = kMaxBridge * kMaxBridge;
            // Grid cell >= the cap so any vertex within the cap lies in one of
            // the 27 neighbouring cells.
            const float cell = std::max(kMaxBridge, diag / 64.0f);
            auto hsh = [](int64_t x, int64_t y, int64_t z) -> int64_t {
                return (x * 73856093LL) ^ (y * 19349663LL) ^ (z * 83492791LL); };
            auto cof = [cell](float v) -> int64_t {
                return (int64_t)std::floor((double)v / cell); };
            std::unordered_map<int64_t, std::vector<int>> grid;     // ALL verts
            for (int w = 0; w < W; ++w)
                grid[hsh(cof(wpos[w].x), cof(wpos[w].y), cof(wpos[w].z))].push_back(w);

            // Per component: its single closest cross-component link within cap.
            std::unordered_map<int, std::pair<int,int>> bridge;   // comp -> (w, other)
            std::unordered_map<int, float>              bridgeD;
            for (int w = 0; w < W; ++w) {
                const int cw = comp[w];
                if (cw == bigC) continue;          // biggest part needn't initiate
                const glm::vec3 p = wpos[w];
                const int64_t cx = cof(p.x), cy = cof(p.y), cz = cof(p.z);
                int   best = -1;
                float bd2  = kMaxBridge2;          // only consider within the cap
                for (int64_t dx = -1; dx <= 1; ++dx)
                for (int64_t dy = -1; dy <= 1; ++dy)
                for (int64_t dz = -1; dz <= 1; ++dz) {
                    auto it = grid.find(hsh(cx + dx, cy + dy, cz + dz));
                    if (it == grid.end()) continue;
                    for (int ow : it->second) {
                        if (comp[ow] == cw) continue;        // a DIFFERENT part only
                        const glm::vec3 dv = wpos[ow] - p;
                        const float d2 = glm::dot(dv, dv);
                        if (d2 < bd2) { bd2 = d2; best = ow; }
                    }
                }
                if (best < 0) continue;
                auto it = bridgeD.find(cw);
                if (it == bridgeD.end() || bd2 < it->second) {
                    bridgeD[cw] = bd2; bridge[cw] = { w, best };
                }
            }
            int nbridge = 0;
            for (const auto& kv : bridge) {
                addEdge(kv.second.first, kv.second.second); ++nbridge;
            }
            if (nbridge > 0) {
                char bb[200];
                std::snprintf(bb, sizeof(bb),
                    "[AutoRig] bridged %d component(s) to their nearest neighbour "
                    "part (toe→foot→leg); wide gaps left separate.", nbridge);
                std::cout << bb << std::endl;
            }
        }
    }
    // Outer-layer (cloth/hair) edge graph — kept OUT of the geodesic (outer
    // edges would shortcut the on-surface distance) but used by the final
    // weight smoothing, so projected cloth weights diffuse along the cloth
    // surface itself.  Layers are separate welded components, so this can
    // never bleed weights across the cloth/skin gap.
    std::vector<std::vector<std::pair<int, float>>> adjOuter(W);
    {
        auto addOuterEdge = [&](int a, int b) {
            if (a == b) return;
            const float w = glm::length(wpos[a] - wpos[b]);
            adjOuter[a].push_back({b, w});
            adjOuter[b].push_back({a, w});
        };
        for (const auto& t : tris) {
            if (isSkin[t.x]) continue;                    // outer triangles only
            addOuterEdge(t.x, t.y); addOuterEdge(t.y, t.z); addOuterEdge(t.z, t.x);
        }
    }

    // Per-bone blend RADIUS (falloff width): a bone must reach into — and so
    // COVER — its CONNECTED (skeleton-adjacent) bones, so vertices between two
    // connected bones are shared.  Radius = max length of this bone and its
    // neighbours at either joint.  Long limbs blend wide; short bones (neck /
    // shoulder) blend narrow but still reach their longer neighbours.
    // Falloff width is driven by the TRUE joint-to-joint distance (parent joint
    // → this joint), NOT the leaf-extended seed segment — so the blend reach is
    // a function of the skeleton's joint spacing.
    std::vector<float> boneLen(nb);
    for (int bi = 0; bi < nb; ++bi) {
        const int cj = bones[bi].joint_idx;
        const int pj = (cj >= 0 && cj < nj) ? skeleton_.joints[cj].parent : -1;
        boneLen[bi] = (pj >= 0)
            ? glm::length(skeleton_.joints[cj].position - skeleton_.joints[pj].position)
            : glm::length(bones[bi].b - bones[bi].a);   // root: fall back to segment
    }
    std::vector<std::vector<int>> jointBones(nj);
    for (int bi = 0; bi < nb; ++bi) {
        const int cj = bones[bi].joint_idx;
        const int pj = (cj >= 0 && cj < nj) ? skeleton_.joints[cj].parent : -1;
        if (cj >= 0 && cj < nj) jointBones[cj].push_back(bi);
        if (pj >= 0 && pj < nj) jointBones[pj].push_back(bi);
    }
    std::vector<float> tau(nb);
    for (int bi = 0; bi < nb; ++bi) {
        float r = boneLen[bi];
        const int cj = bones[bi].joint_idx;
        const int pj = (cj >= 0 && cj < nj) ? skeleton_.joints[cj].parent : -1;
        auto consider = [&](int jt) {
            if (jt < 0 || jt >= nj) return;
            for (int n : jointBones[jt]) r = std::max(r, boneLen[n]);
        };
        consider(cj); consider(pj);
        tau[bi] = std::max(r * 0.9f, 1e-4f);   // ~one neighbour-bone reach
    }

    // Stash tau per JOINT for the debug ring overlay (bone bi ends at its joint).
    debug_tau_.assign(nj, 0.0f);
    for (int bi = 0; bi < nb; ++bi) {
        const int j = bones[bi].joint_idx;
        if (j >= 0 && j < nj) debug_tau_[j] = tau[bi];
    }


    // ── Seed each bone from the surface that ENCLOSES it (ray casting) ──
    // Straight-line "nearest joint/bone" assignment LEAKS across gaps: with the
    // arm down, the elbow joint is euclidean-near the waist, so the waist would
    // wrongly seed the forearm and the geodesic would measure distance from the
    // wrong place.  Instead we fire rays from points ALONG each bone segment
    // over a sphere of directions; the NEAREST skin hit in each direction is the
    // limb surface that wraps the bone (the bone sits inside its own limb), so
    // seeds always land on the correct part.  Rays in ALL directions also seed
    // BOTH faces of thick parts (pelvis/torso), and the leaf/root bones are
    // handled by the same rule — no special cases.
    std::vector<std::vector<int>> seeds(nb);
    {
        const int kBoneDirs = 64;     // directions per sample point
        const int kSamples  = 4;      // segment samples (kSamples+1 points)
        std::vector<std::pair<float, int>> hits;   // (hit distance, triangle)
        for (int bi = 0; bi < nb; ++bi) {
            hits.clear();
            for (int sp = 0; sp <= kSamples; ++sp) {
                const float u = static_cast<float>(sp) / kSamples;
                const glm::vec3 o = bones[bi].a + (bones[bi].b - bones[bi].a) * u;
                for (int s = 0; s < kBoneDirs; ++s) {
                    const float k     = static_cast<float>(s) + 0.5f;
                    const float phi   = std::acos(1.0f - 2.0f * k / kBoneDirs);
                    const float theta = 2.39996323f * static_cast<float>(s);
                    const glm::vec3 d(std::sin(phi) * std::cos(theta),
                                      std::sin(phi) * std::sin(theta),
                                      std::cos(phi));
                    float bestT = 1e30f; int bestTri = -1;
                    for (size_t ti = 0; ti < tris.size(); ++ti) {
                        const glm::ivec3& tv = tris[ti];
                        if (!isSkin[tv.x]) continue;         // skin surface only
                        float tt, bu, bv;
                        if (!rayTri(o, d, tv, tt, bu, bv)) continue;
                        // Only the surface the bone EXITS through (its own
                        // enclosing limb) is a valid seed.  A foreign part across
                        // an air gap is ENTERED — its face normal points back at
                        // the ray (dot<=0) — so reject it; otherwise a hand next
                        // to the hip would seed the hip.
                        const glm::vec3 fn = glm::cross(wpos[tv.y] - wpos[tv.x],
                                                        wpos[tv.z] - wpos[tv.x]);
                        if (glm::dot(fn, d) <= 0.0f) continue;
                        if (tt < bestT) { bestT = tt; bestTri = static_cast<int>(ti); }
                    }
                    if (bestTri >= 0) hits.push_back({ bestT, bestTri });
                }
            }
            if (hits.empty()) continue;
            // Local limb radius = MEDIAN nearest-hit distance.  Reject far
            // outliers — rays that escaped along the bone axis into a NEIGHBOUR
            // (e.g. up from the shoulder into the head), which would otherwise
            // seed the wrong part.  Keeps each bone's own cross-section.
            std::vector<float> ds; ds.reserve(hits.size());
            for (const auto& h : hits) ds.push_back(h.first);
            std::nth_element(ds.begin(), ds.begin() + ds.size() / 2, ds.end());
            const float cutoff = ds[ds.size() / 2] * 2.0f + 1e-4f;
            for (const auto& h : hits) {
                if (h.first > cutoff) continue;
                const glm::ivec3& t = tris[h.second];
                seeds[bi].push_back(t.x);
                seeds[bi].push_back(t.y);
                seeds[bi].push_back(t.z);
            }

            // Also seed the surface vertex NEAREST the bone's end joint.  A
            // bone like the foot runs UP the shin, so its shaft seeds (above)
            // only cover the shin cross-section — the foot, which sits forward
            // of and perpendicular to the joint, is past the outlier cutoff and
            // never seeded, leaving its geodesic distance huge despite being
            // right at the joint.  The end joint sits inside its own extremity,
            // so its nearest skin vertex is ON that extremity (the foot/ankle):
            // seeding it gives the extremity a distance-0 origin at the joint.
            const int jidx = bones[bi].joint_idx;
            if (jidx >= 0 && jidx < nj) {
                const glm::vec3 jp = skeleton_.joints[jidx].position;
                int   bestW = -1;
                float bestD2 = 1e30f;
                for (int w = 0; w < W; ++w) {
                    if (!isSkin[w]) continue;
                    const glm::vec3 dv = wpos[w] - jp;
                    const float d2 = glm::dot(dv, dv);
                    if (d2 < bestD2) { bestD2 = d2; bestW = w; }
                }
                if (bestW >= 0) seeds[bi].push_back(bestW);
            }
        }
    }

    // ── Per-bone GEODESIC distance from its core (multi-source Dijkstra) ──
    // geo[w * nb + bi] = on-surface distance from welded vertex w to bone bi's
    // core.  A hand by the thigh is euclidean-near but a whole arm away across
    // the surface, so its bone never reaches the thigh's vertices.
    std::vector<float> geo(static_cast<size_t>(W) * nb, 1e30f);
    std::vector<float> dist(W);
    for (int bi = 0; bi < nb; ++bi) {
        std::fill(dist.begin(), dist.end(), 1e30f);
        std::priority_queue<std::pair<float, int>,
                            std::vector<std::pair<float, int>>,
                            std::greater<std::pair<float, int>>> pq;
        for (int s : seeds[bi])
            if (dist[s] > 0.0f) { dist[s] = 0.0f; pq.push({0.0f, s}); }
        while (!pq.empty()) {
            const std::pair<float, int> top = pq.top(); pq.pop();
            const float d = top.first; const int u = top.second;
            if (d > dist[u]) continue;
            for (const auto& e : adj[u]) {
                const float nd = d + e.second;
                if (nd < dist[e.first]) { dist[e.first] = nd; pq.push({nd, e.first}); }
            }
        }
        // ── Blur the geodesic distance over the surface graph ──
        // Dijkstra on an irregular triangle mesh gives a slightly zig-zag
        // distance (the path snaps along edges), which shows as banding/streaks
        // in the closeness field.  A few Laplacian passes (each vertex averaged
        // with its skin-edge neighbours) smooth it into clean, rounded bands.
        // Only REACHED vertices participate, so unreachable islands stay at inf.
        {
            const int kBlurPasses = 3;
            std::vector<float> tmp(W);
            for (int pass = 0; pass < kBlurPasses; ++pass) {
                for (int w = 0; w < W; ++w) {
                    if (dist[w] >= 1e29f) { tmp[w] = dist[w]; continue; }
                    float sum = dist[w]; int cnt = 1;
                    for (const auto& e : adj[w]) {
                        const float nd = dist[e.first];
                        if (nd < 1e29f) { sum += nd; ++cnt; }
                    }
                    tmp[w] = sum / (float)cnt;
                }
                dist.swap(tmp);
            }
        }
        // Disconnected islands (no edge path to any seed) stay at inf and are
        // handled by the nearest-skin inheritance below.
        for (int w = 0; w < W; ++w) geo[static_cast<size_t>(w) * nb + bi] = dist[w];
    }

    // ── DIAGNOSTIC: graph + per-bone distance stats, to compare against the
    //    preview's distance computation (which logs the same numbers).  If W /
    //    edges differ, the welded mesh itself differs; if seeds/reach differ for
    //    a bone, the seeding diverges; if geo-max differs, the scale/tau differ.
    {
        size_t edges = 0;
        for (const auto& a : adj) edges += a.size();
        glm::vec3 dmn = mesh_.bbox_min, dmx = mesh_.bbox_max;
        char b[256];
        // std::cout is routed to the editor's on-screen Output Log window
        // (main.cpp PhysicsRouteBuf → EditorLog); fprintf(stderr) is not.
        std::snprintf(b, sizeof(b),
            "[AutoRig][dist-diag] meshNv=%d weldedW=%d edges=%zu tris=%zu diag=%.4f",
            nv, W, edges / 2, tris.size(), glm::length(dmx - dmn));
        std::cout << b << std::endl;
        for (int bi = 0; bi < nb; ++bi) {
            int reach = 0; float gmax = 0.0f;
            for (int w = 0; w < W; ++w) {
                const float g = geo[static_cast<size_t>(w) * nb + bi];
                if (g < 1e29f) { ++reach; if (g > gmax) gmax = g; }
            }
            const int jidx = bones[bi].joint_idx;
            std::snprintf(b, sizeof(b),
                "[AutoRig][dist-diag]   bone %2d %-16s seeds=%zu reach=%d geoMax=%.4f",
                bi,
                (jidx >= 0 && jidx < (int)skeleton_.joints.size())
                    ? skeleton_.joints[jidx].name.c_str() : "?",
                seeds[bi].size(), reach, gmax);
            std::cout << b << std::endl;
        }
    }

    // ── Closeness from the geodesic DISTANCE — PER-BONE falloff ──
    //    Each bone's falloff WIDTH is tau[bi] (computed above ≈ one
    //    neighbour-bone reach), NOT a single global fraction of the whole-body
    //    scale.  A global width (0.25 * model_scale, further widened 1.25× by a
    //    kGeoDistScale=0.8) over-widened the SMALL bones: a foot/toe bone then
    //    reached ~1/6 of the entire body along the surface, so its influence
    //    bled across to the ADJACENT foot.  Scaling the width to the LOCAL bone
    //    size keeps each bone's reach proportional to its own limb, which stops
    //    the left/right foot bleed.
    //      c = exp(-geo / tau[bi]);  the falloff ends at the "yellow" band
    //    kYellow and is linearly remapped: closeness = (c - kYellow)/(1-kYellow),
    //    0 below it.
    //    Cf keeps this RAW closeness for baking (the Dist debug view displays
    //    it verbatim); Wf starts as a copy and then goes through the weight
    //    pipeline (fallbacks, Laplacian smoothing, top-4) below — that
    //    processing must never leak back into the baked closeness.
    // Geodesic falloff WITH an out-of-range cutoff.  weight = exp(-geo/tau);
    // a bone whose closeness drops to/below kYellow is OUT OF RANGE for that
    // vertex and contributes EXACTLY ZERO (e.g. the root has no reach to the
    // foot -> 0 there, so debugging joint 0 shows the foot BLACK).  In-range
    // closeness is remapped to (c-kYellow)/(1-kYellow).
    // Lower threshold = WIDER overlap between neighbouring bones = smoother
    // blend across a joint.  At 0.625 a vertex only blended two bones within a
    // very narrow band around the joint (the rest hard-assigned to one bone),
    // which creased/distorted when posed; 0.35 lets each bone's influence reach
    // well past the joint so adjacent bones share a wide gradient.
    const float kYellow = 0.35f;      // out-of-range threshold (zero beyond it)
    std::vector<std::vector<float>> Cf(W, std::vector<float>(nb, 0.0f));
    std::vector<std::vector<float>> Wf(W, std::vector<float>(nb, 0.0f));
    for (int bi = 0; bi < nb; ++bi) {
        const float tau_bi = std::max(tau[bi], 1e-4f);     // per-bone falloff width
        for (int w = 0; w < W; ++w) {
            const float g = geo[static_cast<size_t>(w) * nb + bi];
            if (g >= 1e29f) continue;                      // unreachable from this bone
            const float c = std::exp(-g / tau_bi);         // closeness, 1 at the bone
            if (c <= kYellow) continue;                    // out of range -> zero weight
            const float cl = (c - kYellow) / (1.0f - kYellow);
            Cf[w][bi] = cl;
            Wf[w][bi] = cl;
        }
    }

    // ── GEODESIC last-resort: in-surface vertices past every bone's range ──
    // A LEAF joint (foot/head/hand) has no child, so geometry extending past it
    // — the forefoot/toes beyond the foot joint, the skull above the head — can
    // fall outside the leaf bone's yellow range and get NO weight, then collapse
    // to the object origin on the GPU (the forward foot "spikes").  If such a
    // vertex is still GEODESICALLY reachable from some bone (connected to the
    // skeleton along the surface), bind it fully to its surface-nearest bone.
    // This is deliberately a GEODESIC fallback, not a euclidean one: a truly
    // disconnected island (a held bag) has infinite geo to every bone, so it is
    // NOT caught here — it stays unweighted and is reported as an error below.
    for (int w = 0; w < W; ++w) {
        if (!isSkin[w]) continue;
        float s = 0.0f;
        for (int bi = 0; bi < nb; ++bi) s += Wf[w][bi];
        if (s > 0.0f) continue;                            // already in range of a bone
        const float* g = &geo[static_cast<size_t>(w) * nb];
        float gmin = 1e29f; int bmin = -1;
        for (int bi = 0; bi < nb; ++bi)
            if (g[bi] < gmin) { gmin = g[bi]; bmin = bi; }
        if (bmin >= 0 && gmin < 1e29f) Wf[w][bmin] = 1.0f; // surface-nearest bone
    }

    // ── Project OUTER layers (cloth/hair) onto the SKIN beneath ──
    // Clothing rides the body underneath it, so each outer vertex INHERITS
    // the skin's weights at the surface point directly beneath it.  Two-stage
    // projection, per cloth COMPONENT:
    //   1. INWARD RAY (primary): march a ray from the cloth vertex along its
    //      ANTI-NORMAL and take the nearest skin hit whose front face looks
    //      back at the cloth.  That is the literal "skin beneath" — a skirt
    //      panel hanging NEXT TO a hand fires its ray inward at the leg, so
    //      it can never bind to the euclidean-closer hand.  The successful
    //      hits also VOTE an allowed-bone mask for the whole cloth piece
    //      (a dress votes legs/hips/torso — never hand or head bones).
    //   2. CLOSEST-POINT (fallback, for verts whose inward ray misses: hem
    //      rings, folds, flipped normals): exact closest point on the skin
    //      triangles, accelerated by a uniform spatial hash, preferring
    //      candidates that (a) FACE the same way as the cloth and (b) whose
    //      dominant bone is in the component's allowed-bone mask; tier (b)
    //      keeps even normal-broken vertices off the hands.
    // Weights AND baked closeness are barycentric-blended from the chosen
    // triangle; the outer-layer Laplacian smoothing below then diffuses any
    // residual seams (e.g. a hem ring crossing between the two legs).
    size_t projected_count = 0, project_fallback = 0;
    // projAnchor[w] = this outer vert got a reliable INWARD-RAY hit (real skin
    // directly beneath it).  Overhang verts (toe box past the toes, sole/heel
    // edges) miss that ray and only get an edge-snapped closest-point match,
    // which fans out when the toe bends.  After projection we hold the anchors
    // fixed and harmonically fill the non-anchors from them along the layer's
    // own surface, so the overhang inherits a coherent field (no fan-out).
    std::vector<char> projAnchor(W, 0);
    if (skin_verts > 0 && skin_verts < W) {
        // Skin triangle list + face normals + spatial hash over tri AABBs.
        std::vector<int>       skinTri;            // indices into tris
        skinTri.reserve(tris.size());
        for (size_t ti = 0; ti < tris.size(); ++ti)
            if (isSkin[tris[ti].x]) skinTri.push_back((int)ti);
        std::vector<glm::vec3> skinTriN(skinTri.size());
        for (size_t k = 0; k < skinTri.size(); ++k) {
            const glm::ivec3& t = tris[skinTri[k]];
            glm::vec3 fn = glm::cross(wpos[t.y] - wpos[t.x],
                                      wpos[t.z] - wpos[t.x]);
            const float l = glm::length(fn);
            skinTriN[k] = (l > 1e-12f) ? fn / l : glm::vec3(0.0f);
        }
        const glm::vec3 pext = mesh_.bbox_max - mesh_.bbox_min;
        const float pdiag = std::max(glm::length(pext), 1e-4f);
        const float pcell = std::max(pdiag / 64.0f, 1e-6f);
        auto pHash = [](int64_t x, int64_t y, int64_t z) -> int64_t {
            return (x * 73856093LL) ^ (y * 19349663LL) ^ (z * 83492791LL);
        };
        auto pCellOf = [pcell](float v) -> int64_t {
            return (int64_t)std::floor((double)v / pcell);
        };
        std::unordered_map<int64_t, std::vector<int>> pgrid;  // cell → skinTri idx
        pgrid.reserve(skinTri.size() * 2);
        for (size_t k = 0; k < skinTri.size(); ++k) {
            const glm::ivec3& t = tris[skinTri[k]];
            const glm::vec3 lo = glm::min(wpos[t.x], glm::min(wpos[t.y], wpos[t.z]));
            const glm::vec3 hi = glm::max(wpos[t.x], glm::max(wpos[t.y], wpos[t.z]));
            for (int64_t cx = pCellOf(lo.x); cx <= pCellOf(hi.x); ++cx)
            for (int64_t cy = pCellOf(lo.y); cy <= pCellOf(hi.y); ++cy)
            for (int64_t cz = pCellOf(lo.z); cz <= pCellOf(hi.z); ++cz)
                pgrid[pHash(cx, cy, cz)].push_back((int)k);
        }
        // Exact closest point on a triangle (Ericson, RTCD 5.1.5) →
        // distance² + barycentric (u toward v1, v toward v2).
        auto closestOnTri = [&](const glm::vec3& p, const glm::ivec3& t,
                                float& bu, float& bv) -> float {
            const glm::vec3& a = wpos[t.x];
            const glm::vec3& b = wpos[t.y];
            const glm::vec3& c = wpos[t.z];
            const glm::vec3 ab = b - a, ac = c - a, ap = p - a;
            const float d1 = glm::dot(ab, ap), d2 = glm::dot(ac, ap);
            if (d1 <= 0.0f && d2 <= 0.0f) { bu = 0; bv = 0;
                return glm::dot(p - a, p - a); }
            const glm::vec3 bp = p - b;
            const float d3 = glm::dot(ab, bp), d4 = glm::dot(ac, bp);
            if (d3 >= 0.0f && d4 <= d3) { bu = 1; bv = 0;
                return glm::dot(p - b, p - b); }
            const float vc = d1 * d4 - d3 * d2;
            if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
                bu = d1 / (d1 - d3); bv = 0;
                const glm::vec3 q = a + ab * bu;
                return glm::dot(p - q, p - q);
            }
            const glm::vec3 cp = p - c;
            const float d5 = glm::dot(ab, cp), d6 = glm::dot(ac, cp);
            if (d6 >= 0.0f && d5 <= d6) { bu = 0; bv = 1;
                return glm::dot(p - c, p - c); }
            const float vb = d5 * d2 - d1 * d6;
            if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
                bu = 0; bv = d2 / (d2 - d6);
                const glm::vec3 q = a + ac * bv;
                return glm::dot(p - q, p - q);
            }
            const float va = d3 * d6 - d5 * d4;
            if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
                const float w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                bu = 1.0f - w2; bv = w2;
                const glm::vec3 q = b + (c - b) * w2;
                return glm::dot(p - q, p - q);
            }
            const float denom = 1.0f / (va + vb + vc);
            bu = vb * denom; bv = vc * denom;
            const glm::vec3 q = a + ab * bu + ac * bv;
            return glm::dot(p - q, p - q);
        };
        // Shell walker (chebyshev radius r cell ring), as in the fallback below.
        auto pShell = [](int64_t r,
                         const std::function<void(int64_t,int64_t,int64_t)>& fn) {
            if (r == 0) { fn(0, 0, 0); return; }
            for (int64_t dx = -r; dx <= r; ++dx)
            for (int64_t dy = -r; dy <= r; ++dy) {
                if (std::llabs(dx) == r || std::llabs(dy) == r)
                    for (int64_t dz = -r; dz <= r; ++dz) fn(dx, dy, dz);
                else { fn(dx, dy, -r); fn(dx, dy, r); }
            }
        };

        // Grid ray-march (Amanatides–Woo DDA) over the skin-triangle hash:
        // nearest ENTERING hit (front face toward the ray origin) within tmax.
        std::vector<int> triStamp(skinTri.size(), -1);
        int stampId = 0;
        auto rayCastSkin = [&](const glm::vec3& o, const glm::vec3& d,
                               float tmax, int& outK,
                               float& bu, float& bv) -> bool {
            ++stampId;
            outK = -1;
            float bestT = tmax;
            int64_t cx = pCellOf(o.x), cy = pCellOf(o.y), cz = pCellOf(o.z);
            glm::vec3 tDelta(1e30f), tNext(1e30f);
            int64_t step[3] = {0, 0, 0};
            for (int i = 0; i < 3; ++i) {
                if (std::fabs(d[i]) < 1e-12f) continue;
                step[i] = d[i] > 0.0f ? 1 : -1;
                tDelta[i] = pcell / std::fabs(d[i]);
                const int64_t c = (i == 0) ? cx : (i == 1) ? cy : cz;
                const float boundary =
                    (float)((double)(c + (d[i] > 0.0f ? 1 : 0)) * pcell);
                tNext[i] = (boundary - o[i]) / d[i];
            }
            for (;;) {
                auto it = pgrid.find(pHash(cx, cy, cz));
                if (it != pgrid.end()) {
                    for (int k : it->second) {
                        if (triStamp[k] == stampId) continue;
                        triStamp[k] = stampId;
                        float tt, u, v;
                        if (!rayTri(o, d, tris[skinTri[k]], tt, u, v)) continue;
                        if (tt > bestT) continue;
                        // ENTERING hits only: the skin face must look back at
                        // the cloth, otherwise we are exiting through the far
                        // side of a limb.
                        if (glm::dot(skinTriN[k], d) >= 0.0f) continue;
                        bestT = tt; outK = k; bu = u; bv = v;
                    }
                }
                const float texit =
                    std::min(tNext.x, std::min(tNext.y, tNext.z));
                if (outK >= 0 && bestT <= texit) return true;  // settled
                if (texit > tmax) return outK >= 0;
                if (tNext.x <= tNext.y && tNext.x <= tNext.z) {
                    cx += step[0]; tNext.x += tDelta.x;
                } else if (tNext.y <= tNext.z) {
                    cy += step[1]; tNext.y += tDelta.y;
                } else {
                    cz += step[2]; tNext.z += tDelta.z;
                }
            }
        };

        auto applyHit = [&](int w, int k, float bu, float bv) {
            const glm::ivec3& t = tris[skinTri[k]];
            const float w0 = 1.0f - bu - bv;
            for (int bi = 0; bi < nb; ++bi) {
                Wf[w][bi] = w0 * Wf[t.x][bi] + bu * Wf[t.y][bi]
                          + bv * Wf[t.z][bi];
                Cf[w][bi] = w0 * Cf[t.x][bi] + bu * Cf[t.y][bi]
                          + bv * Cf[t.z][bi];
            }
        };

        // Group outer vertices by connected component (each layer piece).
        std::unordered_map<int, std::vector<int>> outerComps;
        for (int w = 0; w < W; ++w)
            if (!isSkin[w]) outerComps[comp[w]].push_back(w);

        // Per-layer-vertex projection: find the CLOSEST BONE, cast toward it,
        // intersect the BODY SKIN between vertex and bone, and inherit that skin
        // point's per-bone weights/closeness.  Aiming at the closest bone makes
        // the ray pass through the limb the layer actually wraps, so a shoe's
        // toe box / sole hits the foot skin near the toe bone and inherits
        // coherent foot/toe weights instead of edge-snapping.
        for (auto& oc : outerComps) {
            for (int w : oc.second) {
                // For EVERY bone (a bone = the line between two joints,
                // parent -> joint), project the layer vertex toward that bone
                // and take the FIRST body-skin intersection.  Select the bone
                // whose intersection is CLOSEST to the layer vertex — i.e. by the
                // layer->underlying-skin distance, NOT the raw distance to the
                // bone line.  That nearest intersection's skin weights/closeness
                // are inherited below.
                int   best_k = -1;
                float best_dist = 1e30f, best_bu = 0.0f, best_bv = 0.0f;
                for (int bi = 0; bi < nb; ++bi) {
                    const int jj = bones[bi].joint_idx;
                    const int pj = skeleton_.joints[jj].parent;
                    const glm::vec3 b = skeleton_.joints[jj].position;
                    const glm::vec3 a = (pj >= 0)
                        ? skeleton_.joints[pj].position : b;
                    const glm::vec3 ab = b - a;
                    const float len2 = glm::dot(ab, ab);
                    const float tt = (len2 > 1e-12f)
                        ? glm::clamp(glm::dot(wpos[w] - a, ab) / len2, 0.0f, 1.0f)
                        : 0.0f;
                    const glm::vec3 cp = a + ab * tt;
                    glm::vec3 dir = cp - wpos[w];
                    const float dl = glm::length(dir);
                    if (dl < 1e-9f) continue;
                    dir /= dl;
                    int k; float bu, bv;
                    if (!rayCastSkin(wpos[w], dir, dl + 1e-3f * pdiag, k, bu, bv))
                        continue;
                    const glm::ivec3& th = tris[skinTri[k]];
                    const glm::vec3 hp = (1.0f - bu - bv) * wpos[th.x]
                                       + bu * wpos[th.y] + bv * wpos[th.z];
                    const float hdist = glm::length(hp - wpos[w]);
                    if (hdist < best_dist) {
                        best_dist = hdist; best_k = k;
                        best_bu = bu; best_bv = bv;
                    }
                }
                if (best_k >= 0) {
                    applyHit(w, best_k, best_bu, best_bv);
                    projAnchor[w] = 1;
                    ++projected_count;
                    continue;
                }
                // 2) Fallback (ray missed: fold/edge) -> nearest skin point; the
                //    overhang harmonic fill below refines any residue.
                const glm::vec3 P = wpos[w];
                const int64_t cx = pCellOf(P.x), cy = pCellOf(P.y), cz = pCellOf(P.z);
                int best = -1; float bd2 = 1e30f, fu = 0.0f, fv = 0.0f;
                for (int64_t r = 0; r <= 1024; ++r) {
                    pShell(r, [&](int64_t dx, int64_t dy, int64_t dz) {
                        auto it = pgrid.find(pHash(cx + dx, cy + dy, cz + dz));
                        if (it == pgrid.end()) return;
                        for (int kk : it->second) {
                            float u, v;
                            const float d2 = closestOnTri(P, tris[skinTri[kk]], u, v);
                            if (d2 < bd2) { bd2 = d2; best = kk; fu = u; fv = v; }
                        }
                    });
                    const float scanned = (float)r * pcell;
                    if (best >= 0 && scanned * scanned >= bd2) break;
                }
                if (best >= 0) {
                    applyHit(w, best, fu, fv);
                    ++projected_count;
                    ++project_fallback;
                }
            }
        }
    }

    // ── Last-resort: never leave a vertex unweighted ──
    // The geodesic falloff is the ONLY thing that shapes the (blended) weights.
    // But a vertex that NO bone's geodesic reached bakes to ZERO total weight,
    // and the finalize then defaults it to joint 0 (HIPS) — so a detached foot
    // island whose geodesic never connected ends up rigged to the HIPS, which
    // drags the foot to the pelvis when posed.  Bind any such zero-weight vertex
    // to its EUCLIDEAN-nearest bone instead (the foot vert's nearest bone is the
    // foot, never the hips).  This is a degenerate-case rescue, not part of the
    // geodesic weight calc, so it can't bleed across the separated limbs.
    size_t skin_fallback = 0, detached_unweighted = 0;
    (void)realSkinW;
    for (int w = 0; w < W; ++w) {
        float s = 0.0f;
        for (int bi = 0; bi < nb; ++bi) s += Wf[w][bi];
        if (s > 0.0f) continue;
        const glm::vec3 p = wpos[w];
        float best2 = 1e30f; int bmin = -1;
        for (int bi = 0; bi < nb; ++bi) {
            const glm::vec3 a = bones[bi].a, b = bones[bi].b;
            const glm::vec3 ab = b - a;
            const float len2 = glm::dot(ab, ab);
            const float t = (len2 > 1e-12f)
                ? glm::clamp(glm::dot(p - a, ab) / len2, 0.0f, 1.0f) : 0.0f;
            const glm::vec3 cp = a + ab * t;
            const float d2 = glm::dot(p - cp, p - cp);
            if (d2 < best2) { best2 = d2; bmin = bi; }
        }
        if (bmin >= 0) { Wf[w][bmin] = 1.0f; ++skin_fallback; }
        else           { ++detached_unweighted; }
    }

    // ── Light smoothing of the baked DISTANCE map (closeness) ──
    // A couple of Laplacian passes over the vertex connectivity (skin edges
    // + cloth/hair edges — layers stay separate, so no cross-layer bleed)
    // remove the per-vertex speckle and hard projection seams visible in the
    // Dist debug view.  Deliberately gentle: the map should stay faithful to
    // the raw geodesic field, just without the high-frequency noise.
    {
        const int kClosenessSmoothPasses = 2;
        std::vector<std::vector<float>> tmp(W, std::vector<float>(nb, 0.0f));
        for (int pass = 0; pass < kClosenessSmoothPasses; ++pass) {
            for (int w = 0; w < W; ++w) {
                std::vector<float>& d = tmp[w];
                const std::vector<float>& s = Cf[w];
                for (int bi = 0; bi < nb; ++bi) d[bi] = s[bi];
                float cnt = 1.0f;
                for (const auto& e : adj[w]) {           // skin edges
                    const std::vector<float>& nc = Cf[e.first];
                    for (int bi = 0; bi < nb; ++bi) d[bi] += nc[bi];
                    cnt += 1.0f;
                }
                for (const auto& e : adjOuter[w]) {      // cloth/hair edges
                    const std::vector<float>& nc = Cf[e.first];
                    for (int bi = 0; bi < nb; ++bi) d[bi] += nc[bi];
                    cnt += 1.0f;
                }
                const float inv = 1.0f / cnt;
                for (int bi = 0; bi < nb; ++bi) d[bi] *= inv;
            }
            Cf.swap(tmp);
        }
    }

    // ── Smooth the base-skin WEIGHTS for a smooth joint transition ──
    // The geodesic falloff still SHAPES the weights, but the per-vertex field
    // can still step at the seed boundary, creasing/distorting the mesh when a
    // joint bends.  A few Laplacian passes over the SKIN graph (adj) average
    // each vertex's weights with its on-surface neighbours, turning hard
    // boundaries into wide gradients.  adj follows the real surface and never
    // crosses the geodesically-separated limbs, so this can't bleed e.g. one
    // foot into the other; it only blends bones that already meet on the limb.
    {
        const int kWeightSmoothPasses = 4;
        std::vector<std::vector<float>> tmp(W);
        for (int pass = 0; pass < kWeightSmoothPasses; ++pass) {
            for (int w = 0; w < W; ++w) {
                if (!isSkin[w]) { tmp[w] = Wf[w]; continue; }
                std::vector<float> d = Wf[w];
                float cnt = 1.0f;
                for (const auto& e : adj[w]) {
                    if (!isSkin[e.first]) continue;
                    const std::vector<float>& nw = Wf[e.first];
                    for (int bi = 0; bi < nb; ++bi) d[bi] += nw[bi];
                    cnt += 1.0f;
                }
                const float inv = 1.0f / cnt;
                for (int bi = 0; bi < nb; ++bi) d[bi] *= inv;
                tmp[w] = std::move(d);
            }
            Wf.swap(tmp);
        }
    }

    // ── Overhang repair: harmonically fill NON-anchor outer verts ──
    // Anchors (verts with a real inward-ray "skin beneath" hit) are held FIXED;
    // every other outer vert is iteratively replaced by the average of its
    // cloth-surface neighbours (Jacobi relaxation of Laplace, anchors = Dirichlet
    // boundary).  Overhang verts — a shoe's toe box past the toes, the sole/heel
    // edges — have no skin beneath, so they only got an edge-snapped closest-
    // point match that FANNED OUT when the toe bent.  Filling them from the part
    // of the shoe that hugs the foot gives a smooth, coherent field instead, so
    // the shoe deforms as one piece.  Free regions with no anchor in their own
    // cloth component keep their initial projection (harmless fallback).
    {
        const int kFillPasses = 48;
        std::vector<int> freeV;
        freeV.reserve(W);
        for (int w = 0; w < W; ++w)
            if (!isSkin[w] && !projAnchor[w] && !adjOuter[w].empty())
                freeV.push_back(w);
        if (!freeV.empty()) {
            std::vector<std::vector<float>> tmp = Wf;
            for (int pass = 0; pass < kFillPasses; ++pass) {
                for (int w : freeV) {
                    std::vector<float>& d = tmp[w];
                    for (int bi = 0; bi < nb; ++bi) d[bi] = 0.0f;
                    float cnt = 0.0f;
                    for (const auto& e : adjOuter[w]) {
                        const std::vector<float>& nw2 = Wf[e.first];
                        for (int bi = 0; bi < nb; ++bi) d[bi] += nw2[bi];
                        cnt += 1.0f;
                    }
                    if (cnt > 0.0f) {
                        const float inv = 1.0f / cnt;
                        for (int bi = 0; bi < nb; ++bi) d[bi] *= inv;
                    } else {
                        const std::vector<float>& s = Wf[w];
                        for (int bi = 0; bi < nb; ++bi) d[bi] = s[bi];
                    }
                }
                for (int w : freeV) Wf[w].swap(tmp[w]);
            }
        }
    }

    // ── Extra CLOTH-ONLY smoothing: relax residual projection seams ──
    // Projection inherits weights POINTWISE, so neighbouring cloth vertices
    // can land on different limbs (a hem ring crossing between the legs, a
    // panel edge at the arm gap) and STRETCH when those limbs separate.
    // Skin keeps its crisp 4-pass field above; cloth gets additional
    // Laplacian diffusion along its OWN surface only (adjOuter never crosses
    // the cloth/skin gap, and skin weights are read but never written here),
    // turning hard per-vertex jumps into wide gradual blends.
    {
        const int kClothSmoothPasses = 8;
        std::vector<int> outer;
        outer.reserve(W);
        for (int w = 0; w < W; ++w)
            if (!isSkin[w] && !adjOuter[w].empty()) outer.push_back(w);
        if (!outer.empty()) {
            std::vector<std::vector<float>> tmp = Wf;
            for (int pass = 0; pass < kClothSmoothPasses; ++pass) {
                for (int w : outer) {
                    std::vector<float>& d = tmp[w];
                    const std::vector<float>& s = Wf[w];
                    for (int bi = 0; bi < nb; ++bi) d[bi] = s[bi];
                    float cnt = 1.0f;
                    for (const auto& e : adjOuter[w]) {
                        const std::vector<float>& nw2 = Wf[e.first];
                        for (int bi = 0; bi < nb; ++bi) d[bi] += nw2[bi];
                        cnt += 1.0f;
                    }
                    const float inv = 1.0f / cnt;
                    for (int bi = 0; bi < nb; ++bi) d[bi] *= inv;
                }
                for (int w : outer) Wf[w].swap(tmp[w]);
            }
        }
    }

    // ── Finalize per ORIGINAL vertex: keep top-K, normalize ──
    // Wf holds the SMOOTHED per-bone weights (geodesic field + nearest-skin
    // inheritance, then Laplacian-smoothed).  Per vertex we keep the K
    // (= kMaxVertexInfluences, 8 for the 8-bone debug path) strongest bones
    // and normalize.  Anything still empty (no skin at all) is counted.
    constexpr int K = kMaxVertexInfluences;
    size_t unweighted_count = 0;
    for (int v = 0; v < nv; ++v) {
        const std::vector<float>& wv = Wf[wid[v]];
        auto& vsd = skin_weights_.per_vertex[v];
        int   idx[K]; float val[K];
        for (int i = 0; i < K; ++i) { idx[i] = -1; val[i] = 0.0f; }
        for (int bi = 0; bi < nb; ++bi) {
            const float wgt = wv[bi];
            if (wgt <= 0.0f) continue;                    // only bones that reached it
            for (int s = 0; s < K; ++s)                   // insertion into the top-K
                if (wgt > val[s]) {
                    for (int t = K - 1; t > s; --t) { val[t] = val[t-1]; idx[t] = idx[t-1]; }
                    val[s] = wgt; idx[s] = bi; break;
                }
        }
        // Relative cut: keep the dominant bone (val[0]) plus only those within
        // kKeepFrac of it; drop the far, exponentially-weaker bones.  This
        // LOCALIZES each vertex to its geodesically-closest bones (clean blend)
        // instead of a smear of many distant bones, while NEVER dropping the
        // dominant -> no all-zero weights, no joint-0 default.  Still purely
        // geodesic (the kept bones are the closest by geodesic distance).
        if (val[0] > 0.0f) {
            const float kKeepFrac = 0.20f;
            const float thr = kKeepFrac * val[0];
            for (int i = 1; i < K; ++i)
                if (val[i] < thr) { val[i] = 0.0f; idx[i] = -1; }
        }
        float total = 0.0f;
        for (int i = 0; i < K; ++i) {
            if (idx[i] < 0 || val[i] <= 0.0f) {
                vsd.joint_indices[i] = 0; vsd.weights[i] = 0.0f;
                vsd.closeness[i] = 0.0f; continue;
            }
            vsd.joint_indices[i] = bones[idx[i]].joint_idx;
            vsd.weights[i] = val[i];
            // RAW preview-mode closeness for this bone — NOT the smoothed /
            // truncated weight.  The Dist debug view displays this verbatim,
            // so it matches the preview's live distance computation.
            vsd.closeness[i] = Cf[wid[v]][idx[i]];
            total += val[i];
        }
        if (total > 1e-8f) {
            for (int i = 0; i < K; ++i) vsd.weights[i] /= total;   // partition of unity
        } else {
            ++unweighted_count;   // no skin at all to inherit from (degenerate)
        }
    }
    char sb[360];
    if (skin_fallback > 0 || detached_unweighted > 0) {
        std::snprintf(sb, sizeof(sb),
            "[AutoRig] zero-weight fallback: %zu unreached vert(s) bound to their "
            "EUCLIDEAN-nearest bone (prevents origin-collapse / stretch); "
            "%zu vert(s) had no bone at all.",
            skin_fallback, detached_unweighted);
        std::cout << sb << std::endl;   // → on-screen Output Log window
    }
    std::snprintf(sb, sizeof(sb),
        "[AutoRig] geodesic skinning: %d verts (%d welded), %d bones | "
        "%d layer(s), %d skin / %zu cloth-projected (%zu normal-mismatch) / "
        "%zu geo-fallback / %zu detached-unweighted | %zu still-unweighted",
        nv, W, nb, n_layers, skin_verts, projected_count, project_fallback,
        skin_fallback, detached_unweighted, unweighted_count);
    std::cout << sb << std::endl;
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
    // Name files after the Rig Editor's own selected mesh (this is a rig-editor
    // feature, independent of the auto-rig mesh).
    std::string base_name = std::filesystem::path(re_selected_path_).stem().string();
    if (base_name.empty())
        base_name = std::filesystem::path(re_src_path_).stem().string();
    if (base_name.empty())
        base_name = "untitled";
    std::string data_dir = output_dir + "/" + base_name;

    // Create directory structure
    try {
        std::filesystem::create_directories(data_dir);
    } catch (const std::exception& e) {
        fprintf(stderr, "[AutoRig] Failed to create training data directory: %s\n", e.what());
        return false;
    }

    int views_saved = 0;

    // ---- Scan existing files to find the next available view index ----------
    int next_view_idx = 0;
    try {
        for (auto& entry : std::filesystem::directory_iterator(data_dir)) {
            std::string fname = entry.path().filename().string();
            // Match pattern: <base>_view<N>_meta.json
            std::string prefix = base_name + "_view";
            std::string suffix = "_meta.json";
            if (fname.size() > prefix.size() + suffix.size() &&
                fname.compare(0, prefix.size(), prefix) == 0 &&
                fname.compare(fname.size() - suffix.size(), suffix.size(), suffix) == 0) {
                std::string num_str = fname.substr(
                    prefix.size(), fname.size() - prefix.size() - suffix.size());
                try {
                    int idx = std::stoi(num_str);
                    next_view_idx = std::max(next_view_idx, idx + 1);
                } catch (...) {}
            }
        }
    } catch (...) {
        // Directory may not exist yet — that's fine, start at 0.
    }
    fprintf(stderr, "[AutoRig] Existing training data: next view index = %d\n", next_view_idx);

    // Process each saved edit view
    for (int v = 0; v < (int)saved_edits_.size(); ++v) {
        const auto& cap = saved_edits_[v].capture;
        const auto& edit_state = saved_edits_[v].edit;
        int W = cap.width;
        int H = cap.height;

        int file_idx = next_view_idx + views_saved;  // continue from existing data
        char filename_buf[512];

        // ---- Save color image ----
        if (!cap.color.empty()) {
            std::snprintf(filename_buf, sizeof(filename_buf),
                "%s/%s_view%d_color.png", data_dir.c_str(), base_name.c_str(), file_idx);
            if (!stbi_write_png(filename_buf, W, H, 3,
                    cap.color.data(), W * 3)) {
                fprintf(stderr, "[AutoRig] Failed to write %s\n", filename_buf);
            }
        }

        // ---- Save silhouette ----
        if (!cap.silhouette.empty()) {
            std::snprintf(filename_buf, sizeof(filename_buf),
                "%s/%s_view%d_silhouette.png", data_dir.c_str(), base_name.c_str(), file_idx);
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
                "%s/%s_view%d_heatmap.png", data_dir.c_str(), base_name.c_str(), file_idx);
            if (!stbi_write_png(filename_buf, W, H, 1,
                    heatmap.data(), W)) {
                fprintf(stderr, "[AutoRig] Failed to write %s\n", filename_buf);
            }
        }

        // ---- Save metadata JSON ----
        std::snprintf(filename_buf, sizeof(filename_buf),
            "%s/%s_view%d_meta.json", data_dir.c_str(), base_name.c_str(), file_idx);
        FILE* meta_file = std::fopen(filename_buf, "w");
        if (!meta_file) {
            fprintf(stderr, "[AutoRig] Failed to open %s for writing\n", filename_buf);
            continue;
        }

        // Write JSON metadata
        fprintf(meta_file, "{\n");
        fprintf(meta_file, "  \"view_index\": %d,\n", file_idx);
        fprintf(meta_file, "  \"azimuth_deg\": %.1f,\n", cap.azimuth_deg);
        fprintf(meta_file, "  \"elevation_deg\": %.1f,\n", cap.elevation_deg);
        fprintf(meta_file, "  \"camera_distance\": %.4f,\n", camera_distance_);
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

    fprintf(stderr, "[AutoRig] Exported training data: %d new views (index %d..%d) to %s\n",
        views_saved, next_view_idx, next_view_idx + views_saved - 1, data_dir.c_str());

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

    // ---- Joint indices (JOINTS_0 + JOINTS_1: uvec4 as unsigned short) ------
    //  kMaxVertexInfluences (8) influences per vertex, split into two glTF
    //  skin sets of 4 (set 1 = influences 4..7) for the 8-bone debug path.
    static_assert(kMaxVertexInfluences == 8,
                  "glb export assumes exactly two vec4 skin sets");
    std::vector<uint16_t> joint_data(total_verts * 4, 0);
    std::vector<uint16_t> joint_data1(total_verts * 4, 0);
    for (size_t v = 0; v < total_verts; ++v) {
        for (int i = 0; i < 4; ++i) {
            joint_data[v * 4 + i] = static_cast<uint16_t>(
                skin_weights_.per_vertex[v].joint_indices[i]);
            joint_data1[v * 4 + i] = static_cast<uint16_t>(
                skin_weights_.per_vertex[v].joint_indices[4 + i]);
        }
    }
    size_t joints_offset = appendData(
        joint_data.data(), joint_data.size() * sizeof(uint16_t));
    size_t joints_size = joint_data.size() * sizeof(uint16_t);
    size_t joints1_offset = appendData(
        joint_data1.data(), joint_data1.size() * sizeof(uint16_t));
    size_t joints1_size = joint_data1.size() * sizeof(uint16_t);

    // ---- Weights (WEIGHTS_0 + WEIGHTS_1: vec4 float) -----------------------
    std::vector<float> weight_data(total_verts * 4, 0.0f);
    std::vector<float> weight_data1(total_verts * 4, 0.0f);
    for (size_t v = 0; v < total_verts; ++v) {
        for (int i = 0; i < 4; ++i) {
            weight_data[v * 4 + i] = skin_weights_.per_vertex[v].weights[i];
            weight_data1[v * 4 + i] =
                skin_weights_.per_vertex[v].weights[4 + i];
        }
    }
    size_t weights_offset = appendData(
        weight_data.data(), weight_data.size() * sizeof(float));
    size_t weights_size = weight_data.size() * sizeof(float);
    size_t weights1_offset = appendData(
        weight_data1.data(), weight_data1.size() * sizeof(float));
    size_t weights1_size = weight_data1.size() * sizeof(float);

    // ---- Closeness (_CLOSENESS_0/_CLOSENESS_1: vec4 float, custom) ---------
    //  Baked distance-derived closeness for the same joints (pre-normalize),
    //  so the debug display renders the auto-rig's own distance field instead
    //  of recomputing it.  Rides through as custom glTF vertex attributes.
    std::vector<float> close_data(total_verts * 4, 0.0f);
    std::vector<float> close_data1(total_verts * 4, 0.0f);
    for (size_t v = 0; v < total_verts; ++v) {
        for (int i = 0; i < 4; ++i) {
            close_data[v * 4 + i] = skin_weights_.per_vertex[v].closeness[i];
            close_data1[v * 4 + i] =
                skin_weights_.per_vertex[v].closeness[4 + i];
        }
    }
    size_t close_offset = appendData(
        close_data.data(), close_data.size() * sizeof(float));
    size_t close_size = close_data.size() * sizeof(float);
    size_t close1_offset = appendData(
        close_data1.data(), close_data1.size() * sizeof(float));
    size_t close1_size = close_data1.size() * sizeof(float);

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

    int bv_joints   = addBV(joints_offset,   joints_size,   34962);  // ARRAY_BUFFER
    int bv_joints1  = addBV(joints1_offset,  joints1_size,  34962);
    int bv_weights  = addBV(weights_offset,  weights_size,  34962);
    int bv_weights1 = addBV(weights1_offset, weights1_size, 34962);
    int bv_close    = addBV(close_offset,    close_size,    34962);
    int bv_close1   = addBV(close1_offset,   close1_size,   34962);
    int bv_ibm      = addBV(ibm_offset,      ibm_size);

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

                // Helper: per-primitive accessor over a vertex slice.
                auto addPrimAcc = [&](int bv, int comp_type, size_t elem_sz,
                                      const char* attr) {
                    tinygltf::Accessor acc;
                    acc.bufferView    = bv;
                    acc.byteOffset    = vertex_offset * 4 * elem_sz;
                    acc.componentType = comp_type;
                    acc.type          = TINYGLTF_TYPE_VEC4;
                    acc.count         = prim_verts;
                    int idx = static_cast<int>(out.accessors.size());
                    out.accessors.push_back(acc);
                    prim.attributes[attr] = idx;
                };

                // Skin set 0 (influences 0..3) + set 1 (influences 4..7).
                addPrimAcc(bv_joints, TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT,
                           sizeof(uint16_t), "JOINTS_0");
                addPrimAcc(bv_joints1, TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT,
                           sizeof(uint16_t), "JOINTS_1");
                addPrimAcc(bv_weights, TINYGLTF_COMPONENT_TYPE_FLOAT,
                           sizeof(float), "WEIGHTS_0");
                addPrimAcc(bv_weights1, TINYGLTF_COMPONENT_TYPE_FLOAT,
                           sizeof(float), "WEIGHTS_1");
                addPrimAcc(bv_close, TINYGLTF_COMPONENT_TYPE_FLOAT,
                           sizeof(float), "_CLOSENESS_0");
                addPrimAcc(bv_close1, TINYGLTF_COMPONENT_TYPE_FLOAT,
                           sizeof(float), "_CLOSENESS_1");

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

    {
        std::string ml = (model_loaded_idx_ >= 0 && model_loaded_idx_ < (int)model_versions_.size())
            ? model_versions_[model_loaded_idx_].label() : "stub/heuristic";
        reportProgress(3, kTotalSteps, "Running model " + ml + "...");
    }
    if (!predictJoints()) { state_ = PluginState::kError; return false; }

    reportProgress(4, kTotalSteps, "Fusing 3D skeleton...");
    if (!fuseAndBuildSkeleton()) { state_ = PluginState::kError; return false; }

    // Auto-rig also produces an editable skeleton — record it and auto-save the
    // generated joints so they can be reopened/edited in the manual workflow.
    joints_generated_ = true;
    joints_edited_    = false;
    saveEditedJoints(baseJointsPath());   // auto-write "<character>.joints"

    reportProgress(5, kTotalSteps, "Computing skin weights...");
    if (!computeSkinWeights()) { state_ = PluginState::kError; return false; }

    reportProgress(6, kTotalSteps, "Exporting glTF...");
    if (!exportGltf(output_gltf_path)) { state_ = PluginState::kError; return false; }

    state_ = PluginState::kFinished;
    reportProgress(kTotalSteps, kTotalSteps, "Done!");
    return true;
}

// ============================================================================
//  Manual 3-pass workflow
// ============================================================================

// Pass 1: build the skeleton (no skin weights).  Assumes the mesh has already
// been loaded by the file-selection UI.  Mirrors rigCharacter() steps 2-4.
bool AutoRigPlugin::generateJoints() {
    if (mesh_.empty()) {
        ui_status_ = "Select a mesh first.";
        return false;
    }

    state_ = PluginState::kRunning;
    joints_generated_ = false;
    joints_edited_    = false;
    weights_baked_    = false;
    skin_weights_     = SkinWeights{};  // drop any stale weights
    weight_view_mode_ = 0;
    edit3d_render_.width = 0;            // force editor re-render
    edit3d_drag_joint_  = -1;

    const int kTotalSteps = 3;

    reportProgress(1, kTotalSteps, "Capturing multi-view renders...");
    if (!captureViews(num_views_, capture_resolution_)) {
        state_ = PluginState::kError; return false;
    }

    initEditableJoints();

    {
        std::string ml = (model_loaded_idx_ >= 0 && model_loaded_idx_ < (int)model_versions_.size())
            ? model_versions_[model_loaded_idx_].label() : "stub/heuristic";
        reportProgress(2, kTotalSteps, "Running model " + ml + "...");
    }
    if (!predictJoints()) { state_ = PluginState::kError; return false; }

    reportProgress(3, kTotalSteps, "Fusing 3D skeleton...");
    if (!fuseAndBuildSkeleton()) { state_ = PluginState::kError; return false; }

    joints_generated_ = true;
    joints_edited_    = false;
    saveEditedJoints(baseJointsPath());   // auto-write "<character>.joints"

    state_ = PluginState::kFinished;
    reportProgress(kTotalSteps, kTotalSteps,
                   "Joints generated. Edit in 3D, then bake weights.");
    return true;
}

// Rebuild inverse-bind matrices from the current joint positions.  Joint edits
// in Pass 2 move joint.position, which invalidates the IBMs computed during
// fuseAndBuildSkeleton(); this restores the same relationship used there
// (IBM = inverse(translate(pos)) * meshNodeWorld) before baking/export.
void AutoRigPlugin::refreshInverseBindMatrices() {
    for (auto& j : skeleton_.joints) {
        glm::mat4 joint_world = glm::translate(glm::mat4(1.0f), j.position);
        j.inverse_bind_matrix = glm::inverse(joint_world) * mesh_node_world_transform_;
    }
}

// Sidecar paths for saved joints, next to the source mesh:
//   "<character>.joints"         (generated / unedited)
//   "<character>_edited.joints"  (hand-edited)
std::string AutoRigPlugin::baseJointsPath() const {
    if (source_mesh_path_.empty()) return "character.joints";
    std::filesystem::path p(source_mesh_path_);
    return (p.parent_path() / (p.stem().string() + ".joints")).string();
}

std::string AutoRigPlugin::editedJointsPath() const {
    if (source_mesh_path_.empty()) return "character_edited.joints";
    std::filesystem::path p(source_mesh_path_);
    return (p.parent_path() / (p.stem().string() + "_edited.joints")).string();
}

// Where a save should go: the "_edited" file once joints have been dragged,
// otherwise the plain generated-joints file.
std::string AutoRigPlugin::saveJointsPath() const {
    return joints_edited_ ? editedJointsPath() : baseJointsPath();
}

// Where a load should come from: prefer the edited file if it exists.
std::string AutoRigPlugin::loadJointsPath() const {
    return std::filesystem::exists(editedJointsPath())
        ? editedJointsPath() : baseJointsPath();
}

bool AutoRigPlugin::savedJointsExist() const {
    return std::filesystem::exists(editedJointsPath()) ||
           std::filesystem::exists(baseJointsPath());
}

// Save the current skeleton to a simple, human-readable, round-trippable text
// file.  One row per joint: "<idx> <parent> <px> <py> <pz> <name>".  Joint
// names never contain spaces, so the name can safely be the last field.
bool AutoRigPlugin::saveEditedJoints(const std::string& path) {
    if (skeleton_.empty()) {
        edit3d_save_status_ = "Nothing to save — generate joints first.";
        return false;
    }
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        edit3d_save_status_ = "Save failed: cannot open " + path;
        fprintf(stderr, "[AutoRig] saveEditedJoints: cannot open %s\n", path.c_str());
        return false;
    }
    std::fprintf(f, "# auto-rig edited joints v1\n");
    std::fprintf(f, "# fields: index parent pos_x pos_y pos_z name\n");
    std::fprintf(f, "count %d\n", (int)skeleton_.joints.size());
    std::fprintf(f, "root %d\n", skeleton_.root);
    for (int j = 0; j < (int)skeleton_.joints.size(); ++j) {
        const Joint& jt = skeleton_.joints[j];
        std::fprintf(f, "%d %d %.6f %.6f %.6f %s\n",
            j, jt.parent, jt.position.x, jt.position.y, jt.position.z,
            jt.name.c_str());
    }
    std::fclose(f);
    edit3d_save_status_ = "Saved " + std::to_string(skeleton_.joints.size()) +
        " joints to " + std::filesystem::path(path).filename().string();
    fprintf(stderr, "[AutoRig] saved %d joints to %s\n",
        (int)skeleton_.joints.size(), path.c_str());
    return true;
}

// Rebuild skeleton_ from a file written by saveEditedJoints().  After loading,
// the rig is ready to edit further or bake (joints_generated_ is set true).
bool AutoRigPlugin::loadEditedJoints(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) {
        edit3d_save_status_ = "Load failed: " +
            std::filesystem::path(path).filename().string() + " not found.";
        return false;
    }
    Skeleton loaded;
    int count = 0, root = -1;
    char line[512];
    while (std::fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        if (std::strncmp(line, "count", 5) == 0) { std::sscanf(line, "count %d", &count); continue; }
        if (std::strncmp(line, "root",  4) == 0) { std::sscanf(line, "root %d",  &root);  continue; }
        int idx = 0, parent = -1; float x = 0, y = 0, z = 0; char name[128] = {0};
        if (std::sscanf(line, "%d %d %f %f %f %127s", &idx, &parent, &x, &y, &z, name) == 6) {
            if (idx >= (int)loaded.joints.size()) loaded.joints.resize(idx + 1);
            Joint& jt = loaded.joints[idx];
            jt.name = name;
            jt.parent = parent;
            jt.position = glm::vec3(x, y, z);
            jt.rotation = glm::quat(1, 0, 0, 0);
            jt.scale = glm::vec3(1.0f);
        }
    }
    std::fclose(f);

    if (loaded.joints.empty()) {
        edit3d_save_status_ = "Load failed: no joints parsed.";
        return false;
    }
    loaded.root = (root >= 0) ? root : 0;
    skeleton_ = std::move(loaded);
    refreshInverseBindMatrices();
    joints_generated_   = true;
    weights_baked_      = false;
    // Loading the "_edited" file keeps the edited destination for re-saves.
    joints_edited_      = (path.find("_edited.joints") != std::string::npos);
    edit3d_drag_joint_  = -1;
    edit3d_render_.width = 0;  // force editor re-render
    edit3d_save_status_ = "Loaded " + std::to_string(skeleton_.joints.size()) +
        " joints from " + std::filesystem::path(path).filename().string();
    fprintf(stderr, "[AutoRig] loaded %d joints from %s\n",
        (int)skeleton_.joints.size(), path.c_str());
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
// Per-weight-slot colors (mode 1) and a per-bone palette (mode 2).
static ImU32 boneColor(int j) {
    // Distinct hue per joint via HSV->RGB (s=0.65, v=1.0).
    float h = (j * (360.0f / 19.0f)) / 60.0f;
    float s = 0.65f, v = 1.0f;
    int   i = (int)std::floor(h) % 6;
    float f = h - std::floor(h);
    float p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
    float r = 0, g = 0, b = 0;
    switch (i) {
        case 0: r=v; g=t; b=p; break;  case 1: r=q; g=v; b=p; break;
        case 2: r=p; g=v; b=t; break;  case 3: r=p; g=q; b=v; break;
        case 4: r=t; g=p; b=v; break;  default: r=v; g=p; b=q; break;
    }
    return IM_COL32((int)(r*255), (int)(g*255), (int)(b*255), 255);
}

// Color for one vertex from its skin weights, in the chosen mode.
static void weightVertexColor(const VertexSkinData& vsd, int mode,
                              float& r, float& g, float& b) {
    r = g = b = 0.0f;
    // Mode 1: eight fixed slot colors (R, G, B, yellow, magenta, cyan,
    // orange, white) — one per influence slot.
    static const float kSlot[kMaxVertexInfluences][3] = {
        {1,0.2f,0.2f}, {0.2f,1,0.2f}, {0.3f,0.5f,1}, {1,0.85f,0.15f},
        {1,0.2f,1},    {0.2f,1,1},    {1,0.55f,0.1f}, {0.9f,0.9f,0.9f} };
    for (int k = 0; k < kMaxVertexInfluences; ++k) {
        float w = vsd.weights[k];
        if (w <= 0.0f) continue;
        if (mode == 1) {
            r += w * kSlot[k][0]; g += w * kSlot[k][1]; b += w * kSlot[k][2];
        } else {  // mode 2: per-bone palette
            ImU32 c = boneColor(vsd.joint_indices[k]);
            r += w * ((c >> IM_COL32_R_SHIFT) & 0xFF) / 255.0f;
            g += w * ((c >> IM_COL32_G_SHIFT) & 0xFF) / 255.0f;
            b += w * ((c >> IM_COL32_B_SHIFT) & 0xFF) / 255.0f;
        }
    }
}

static bool drawModel3DPreview(
    const TriangleMesh& mesh, const Skeleton& skeleton,
    float canvas_size, ImVec2 canvas_pos, ImDrawList* dl,
    float& yaw, float& pitch, bool& dragging, ImVec2& drag_start,
    const SkinWeights* weights = nullptr, int weight_mode = 0,
    const std::vector<float>* joint_tau = nullptr, bool show_tau = false)
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
        const bool show_w = (weight_mode > 0 && weights &&
                             weights->per_vertex.size() == mesh.positions.size());
        for (size_t t = 0; t < tri_count; t += step) {
            uint32_t i0 = mesh.indices[t * 3 + 0];
            uint32_t i1 = mesh.indices[t * 3 + 1];
            uint32_t i2 = mesh.indices[t * 3 + 2];
            ImVec2 v0 = project(mesh.positions[i0]);
            ImVec2 v1 = project(mesh.positions[i1]);
            ImVec2 v2 = project(mesh.positions[i2]);
            if (show_w) {
                // Average the 3 vertices' weight-colors for a flat-shaded tri.
                float r = 0, g = 0, b = 0, rr, gg, bb;
                for (uint32_t idx : { i0, i1, i2 }) {
                    weightVertexColor(weights->per_vertex[idx], weight_mode, rr, gg, bb);
                    r += rr; g += gg; b += bb;
                }
                ImU32 col = IM_COL32(
                    (int)std::min(255.0f, r / 3.0f * 255.0f),
                    (int)std::min(255.0f, g / 3.0f * 255.0f),
                    (int)std::min(255.0f, b / 3.0f * 255.0f), 235);
                dl->AddTriangleFilled(v0, v1, v2, col);
            } else {
                dl->AddTriangleFilled(v0, v1, v2, IM_COL32(80, 130, 180, 76));
                dl->AddTriangle(v0, v1, v2, IM_COL32(100, 160, 220, 50), 1.0f);
            }
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
        // Tau (geodesic falloff width) per joint, drawn as a world-scaled ring.
        if (show_tau && joint_tau &&
            (int)joint_tau->size() == (int)skeleton.joints.size()) {
            for (int j = 0; j < (int)skeleton.joints.size(); ++j) {
                const float t = (*joint_tau)[j];
                if (t <= 0.0f) continue;
                const ImVec2 c = project(skeleton.joints[j].position);
                dl->AddCircle(c, t * proj_scale, IM_COL32(255, 235, 60, 150), 0, 1.5f);
            }
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
// ---------------------------------------------------------------------------
//  Pass 2: interactive 3D joint editor.
//
//  Draws the original character translucently (OIT) and overlays the skeleton
//  with draggable joint handles.  Dragging a joint moves it in the camera
//  view-plane (depth held constant in clip space); rotate the view (drag empty
//  space) to reach the third axis.  Mesh is only re-rendered when the camera
//  or opacity changes, so dragging stays cheap.
// ---------------------------------------------------------------------------
void AutoRigPlugin::drawJointEditor3D(float canvas_size) {
    if (mesh_.empty() || skeleton_.empty() || !rasterizer_) {
        ImGui::TextDisabled("Run 'Generate Joints' (Pass 1) first.");
        return;
    }

    ImGui::TextWrapped(
        "Drag a joint to move it in the view-plane. Drag empty space to "
        "rotate. Rotate the view to adjust depth, then drag again.");

    // ---- Controls ----
    float old_op = edit3d_opacity_;
    ImGui::SetNextItemWidth(160);
    ImGui::SliderFloat("Mesh Opacity", &edit3d_opacity_, 0.05f, 1.0f, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(160);
    ImGui::SliderFloat("Zoom##edit3d", &camera_distance_, 0.5f, 1.5f, "%.2f");

    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##edit3d_canvas", ImVec2(canvas_size, canvas_size));
    bool hovered = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    // ---- Camera (same construction as the Rig Editor capture) ----
    float az = -glm::degrees(preview_yaw_);
    float el = -glm::degrees(preview_pitch_);
    glm::vec3 centre = (mesh_.bbox_min + mesh_.bbox_max) * 0.5f;
    float ext    = glm::length(mesh_.bbox_max - mesh_.bbox_min);
    float radius = ext * camera_distance_;
    float fov    = glm::radians(45.0f);
    float z_near = radius * 0.01f, z_far = radius * 10.0f;
    glm::mat4 proj = glm::perspective(fov, 1.0f, z_near, z_far);
    proj[1][1] *= -1.0f;
    float az_rad = glm::radians(az), el_rad = glm::radians(el);
    glm::vec3 eye;
    eye.x = centre.x + radius * cosf(el_rad) * cosf(az_rad);
    eye.y = centre.y + radius * sinf(el_rad);
    eye.z = centre.z + radius * cosf(el_rad) * sinf(az_rad);
    glm::mat4 view_mat = glm::lookAt(eye, centre, glm::vec3(0, 1, 0));
    glm::mat4 vp     = proj * view_mat;
    glm::mat4 inv_vp = glm::inverse(vp);

    // ---- Re-render OIT mesh only when camera / opacity changed ----
    int res = std::max(64, (int)canvas_size);
    bool needs_render =
        edit3d_render_.width != res ||
        std::abs(preview_yaw_   - edit3d_render_yaw_)   > 0.001f ||
        std::abs(preview_pitch_ - edit3d_render_pitch_) > 0.001f ||
        std::abs(camera_distance_ - edit3d_render_dist_) > 0.0001f ||
        std::abs(edit3d_opacity_  - edit3d_render_op_)   > 0.001f;
    if (needs_render) {
        edit3d_render_ = rasterizer_->renderOIT(
            mesh_, res, res, view_mat, proj, edit3d_opacity_, az, el);
        edit3d_render_yaw_   = preview_yaw_;
        edit3d_render_pitch_ = preview_pitch_;
        edit3d_render_dist_  = camera_distance_;
        edit3d_render_op_    = edit3d_opacity_;
    }
    (void)old_op;

    // World joint -> canvas pixel, using the SAME mapping the rasterizer uses
    // (ndc.x/y in [-1,1]; image row 0 = top, matching drawViewPixels()).
    auto worldToScreen = [&](const glm::vec3& P, bool& ok) -> ImVec2 {
        glm::vec4 clip = vp * glm::vec4(P, 1.0f);
        if (clip.w <= 1e-6f) { ok = false; return ImVec2(0, 0); }
        ok = true;
        float u = (clip.x / clip.w) * 0.5f + 0.5f;
        float v = (clip.y / clip.w) * 0.5f + 0.5f;
        return ImVec2(canvas_pos.x + u * canvas_size,
                      canvas_pos.y + v * canvas_size);
    };

    // ---- Background + translucent mesh ----
    dl->PushClipRect(canvas_pos,
        ImVec2(canvas_pos.x + canvas_size, canvas_pos.y + canvas_size), true);
    dl->AddRectFilled(canvas_pos,
        ImVec2(canvas_pos.x + canvas_size, canvas_pos.y + canvas_size),
        IM_COL32(25, 25, 30, 255));
    if (edit3d_render_.width > 0)
        drawViewPixels(dl, canvas_pos, canvas_size, canvas_size, edit3d_render_);

    // ---- Per-joint projection + camera-depth coloring ----
    //  Hue encodes distance from the camera: near = warm (red/orange),
    //  far = cool (blue).  Both joints and bones use it so the 3D structure
    //  reads clearly through the translucent mesh.
    auto hsv2col = [](float h, float s, float v) -> ImU32 {
        float r = 0, g = 0, b = 0;
        int   i = (int)std::floor(h * 6.0f);
        float f = h * 6.0f - i;
        float p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
        switch (((i % 6) + 6) % 6) {
            case 0: r=v; g=t; b=p; break; case 1: r=q; g=v; b=p; break;
            case 2: r=p; g=v; b=t; break; case 3: r=p; g=q; b=v; break;
            case 4: r=t; g=p; b=v; break; default: r=v; g=p; b=q; break;
        }
        return IM_COL32((int)(r*255), (int)(g*255), (int)(b*255), 255);
    };

    const int nJoints = (int)skeleton_.joints.size();
    std::vector<ImVec2>        jscreen(nJoints);
    std::vector<unsigned char> jok(nJoints, 0);
    std::vector<float>         jdepth(nJoints, 0.0f);
    std::vector<ImU32>         jcol(nJoints, IM_COL32(200, 200, 200, 255));
    float dmin = 1e30f, dmax = -1e30f;
    for (int j = 0; j < nJoints; ++j) {
        bool ok; jscreen[j] = worldToScreen(skeleton_.joints[j].position, ok);
        jok[j] = ok ? 1 : 0;
        if (!ok) continue;
        float dcam = -(view_mat * glm::vec4(skeleton_.joints[j].position, 1.0f)).z;
        jdepth[j] = dcam;
        dmin = std::min(dmin, dcam);
        dmax = std::max(dmax, dcam);
    }
    const float drange = (dmax > dmin) ? (dmax - dmin) : 1.0f;
    for (int j = 0; j < nJoints; ++j) {
        if (!jok[j]) continue;
        float t = (jdepth[j] - dmin) / drange;        // 0 = nearest, 1 = farthest
        jcol[j] = hsv2col(t * 0.62f, 0.85f, 1.0f);    // red (near) -> blue (far)
    }

    // ---- Pick / drag detection ----
    ImVec2 mouse = ImGui::GetMousePos();
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        float best = 14.0f;  // pixel pick radius
        int   best_j = -1;
        for (int j = 0; j < nJoints; ++j) {
            if (!jok[j]) continue;
            const ImVec2& p = jscreen[j];
            float d = std::sqrt((mouse.x-p.x)*(mouse.x-p.x) +
                                (mouse.y-p.y)*(mouse.y-p.y));
            if (d < best) { best = d; best_j = j; }
        }
        if (best_j >= 0) {
            edit3d_drag_joint_ = best_j;     // grab a joint
            edit3d_rotating_   = false;
        } else {
            edit3d_rotating_   = true;       // empty space -> rotate
            preview_drag_start_ = mouse;
        }
    }

    // ---- Apply joint drag: keep clip-space depth (z,w) fixed, move in plane ----
    if (edit3d_drag_joint_ >= 0) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            Joint& jt = skeleton_.joints[edit3d_drag_joint_];
            glm::vec4 clip0 = vp * glm::vec4(jt.position, 1.0f);
            if (clip0.w > 1e-6f) {
                float ndx = ((mouse.x - canvas_pos.x) / canvas_size) * 2.0f - 1.0f;
                float ndy = ((mouse.y - canvas_pos.y) / canvas_size) * 2.0f - 1.0f;
                glm::vec4 clip_new(ndx * clip0.w, ndy * clip0.w, clip0.z, clip0.w);
                glm::vec4 world = inv_vp * clip_new;
                if (std::abs(world.w) > 1e-6f) {
                    jt.position = glm::vec3(world) / world.w;
                    joints_edited_ = true;   // saves now go to "<name>_edited.joints"
                }
            }
        } else {
            edit3d_drag_joint_ = -1;
        }
    }

    // ---- Handle view rotation (empty-space drag) ----
    if (edit3d_rotating_) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            if (!lock_azimuth_)
                preview_yaw_   += (mouse.x - preview_drag_start_.x) * 0.01f;
            if (!lock_elevation_) {
                preview_pitch_ += (mouse.y - preview_drag_start_.y) * 0.01f;
                preview_pitch_  = std::clamp(preview_pitch_, -1.5f, 1.5f);
            }
            preview_drag_start_ = mouse;
        } else {
            edit3d_rotating_ = false;
        }
    }

    // ---- Draw bones (depth gradient: each half tinted toward its joint) ----
    for (int j = 0; j < nJoints; ++j) {
        const Joint& jt = skeleton_.joints[j];
        int pj = jt.parent;
        if (pj < 0 || !jok[j] || !jok[pj]) continue;
        ImVec2 a = jscreen[pj];
        ImVec2 b = jscreen[j];
        ImVec2 mid((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f);
        dl->AddLine(a, b, IM_COL32(0, 0, 0, 150), 4.5f);   // dark outline
        dl->AddLine(a, mid, jcol[pj], 2.4f);               // parent half
        dl->AddLine(mid, b, jcol[j],  2.4f);               // child half
    }

    // ---- Draw joint handles (filled with depth color) ----
    for (int j = 0; j < nJoints; ++j) {
        if (!jok[j]) continue;
        const Joint& jt = skeleton_.joints[j];
        const ImVec2& p = jscreen[j];
        bool is_root = (jt.parent < 0);
        bool active  = (edit3d_drag_joint_ == j);
        bool hov = false;
        if (hovered) {
            float d = std::sqrt((mouse.x-p.x)*(mouse.x-p.x) +
                                (mouse.y-p.y)*(mouse.y-p.y));
            hov = (d < 14.0f);
        }
        float rad = (active || hov) ? 7.0f : (is_root ? 6.0f : 5.0f);
        dl->AddCircleFilled(p, rad, jcol[j]);              // depth color
        // Outline: white when active/hovered, extra ring for the root.
        dl->AddCircle(p, rad, IM_COL32(0, 0, 0, 230), 0, 1.5f);
        if (active || hov)
            dl->AddCircle(p, rad + 2.5f, IM_COL32(255, 255, 255, 235), 0, 2.0f);
        else if (is_root)
            dl->AddCircle(p, rad + 2.5f, IM_COL32(255, 255, 255, 180), 0, 1.5f);
        if (active || hov) {
            dl->AddText(ImVec2(p.x + rad + 3, p.y - 7),
                        IM_COL32(0, 0, 0, 200), jt.name.c_str());
            dl->AddText(ImVec2(p.x + rad + 2, p.y - 8),
                        IM_COL32(255, 255, 255, 240), jt.name.c_str());
        }
    }

    // ---- Near/far color legend (bottom-right) ----
    {
        const float lw = 90.0f, lh = 10.0f;
        float lx = canvas_pos.x + canvas_size - lw - 10.0f;
        float ly = canvas_pos.y + canvas_size - lh - 22.0f;
        const int steps = 24;
        for (int s = 0; s < steps; ++s) {
            float t0 = (float)s / steps;
            float x0 = lx + t0 * lw;
            float x1 = lx + (float)(s + 1) / steps * lw;
            dl->AddRectFilled(ImVec2(x0, ly), ImVec2(x1, ly + lh),
                              hsv2col(t0 * 0.62f, 0.85f, 1.0f));
        }
        dl->AddRect(ImVec2(lx, ly), ImVec2(lx + lw, ly + lh),
                    IM_COL32(0, 0, 0, 200));
        dl->AddText(ImVec2(lx - 2, ly + lh + 1),    IM_COL32(0,0,0,200), "near");
        dl->AddText(ImVec2(lx - 3, ly + lh),        IM_COL32(255,255,255,235), "near");
        dl->AddText(ImVec2(lx + lw - 22, ly + lh + 1), IM_COL32(0,0,0,200), "far");
        dl->AddText(ImVec2(lx + lw - 23, ly + lh),     IM_COL32(255,255,255,235), "far");
    }

    // ---- Border + angle tag ----
    dl->AddRect(canvas_pos,
        ImVec2(canvas_pos.x + canvas_size, canvas_pos.y + canvas_size),
        IM_COL32(120, 120, 120, 200));
    {
        char ang[64];
        std::snprintf(ang, sizeof(ang), "az:%.0f el:%.0f", az, el);
        dl->AddText(ImVec2(canvas_pos.x + 4, canvas_pos.y + 3),
                    IM_COL32(0, 0, 0, 200), ang);
        dl->AddText(ImVec2(canvas_pos.x + 3, canvas_pos.y + 2),
                    IM_COL32(255, 255, 0, 255), ang);
    }
    dl->PopClipRect();
}

// Load a mesh into a caller-provided buffer without disturbing the auto-rig
// mesh_.  loadMesh() writes mesh_ / source_mesh_path_ / mesh_node_world_transform_,
// so we save those, run it, capture the result, then restore the originals.
bool AutoRigPlugin::loadMeshInto(const std::string& path, TriangleMesh& out_mesh,
                                 std::string& out_src, glm::mat4& out_xform) {
    TriangleMesh keep_mesh  = std::move(mesh_);
    std::string  keep_src   = source_mesh_path_;
    glm::mat4    keep_xform  = mesh_node_world_transform_;

    bool ok = loadMesh(path);                 // populates mesh_ + the two members

    out_mesh  = std::move(mesh_);
    out_src   = source_mesh_path_;
    out_xform = mesh_node_world_transform_;

    mesh_                      = std::move(keep_mesh);
    source_mesh_path_          = keep_src;
    mesh_node_world_transform_ = keep_xform;
    return ok;
}

// Mesh file picker.  Writes the supplied selection state.  When auto_rig is
// true it loads/prepares the auto-rig mesh_; otherwise it loads the Rig
// Editor's independent re_mesh_, leaving the auto-rig mesh untouched.
void AutoRigPlugin::drawMeshSelector(int& sel_idx, std::string& sel_path,
                                     bool auto_rig) {
    if (!files_scanned_) { refreshMeshFileList(); files_scanned_ = true; }

    if (ImGui::Button("Refresh")) refreshMeshFileList();
    ImGui::SameLine();
    ImGui::Text("Mesh Files (%d found)", (int)mesh_file_list_.size());

    {
        float list_h = ImGui::GetTextLineHeightWithSpacing() * std::min((int)mesh_file_list_.size(), 5) + 4.0f;
        if (list_h < ImGui::GetTextLineHeightWithSpacing() * 2) list_h = ImGui::GetTextLineHeightWithSpacing() * 2;
        // Unique child id per context so the two lists scroll independently.
        const char* list_id = auto_rig ? "MeshFileList_ar" : "MeshFileList_re";
        if (ImGui::BeginChild(list_id, ImVec2(-1, list_h), true)) {
            for (int i = 0; i < (int)mesh_file_list_.size(); ++i) {
                const bool is_sel = (sel_idx == i);
                std::string fn = std::filesystem::path(mesh_file_list_[i]).filename().string();
                // Selected row: arrow marker + accent text, on top of the
                // Selectable's highlighted background.
                std::string dname = (is_sel ? "> " : "  ") + fn;
                if (is_sel)
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f, 1.0f, 0.55f, 1.0f));
                bool clicked = ImGui::Selectable(dname.c_str(), is_sel);
                if (is_sel) ImGui::PopStyleColor();
                if (clicked) {
                    sel_idx  = i;
                    sel_path = mesh_file_list_[i];
                    if (auto_rig) {
                        auto p = std::filesystem::path(sel_path);
                        auto out = p.parent_path() / (p.stem().string() + "-skinned" + p.extension().string());
                        std::snprintf(output_path_buf_, sizeof(output_path_buf_), "%s", out.string().c_str());
                        if (mesh_.empty() || source_mesh_path_ != sel_path) {
                            loadMesh(sel_path);
                            captures_ = rasterizer_->captureOrbit(
                                mesh_, num_views_, capture_resolution_, 0.0f, camera_distance_);
                            initEditableJoints();
                            edit_capture_valid_ = false;
                        }
                    } else {
                        // Rig Editor: load into its own mesh buffer.
                        if (re_mesh_.empty() || re_src_path_ != sel_path) {
                            loadMeshInto(sel_path, re_mesh_, re_src_path_, re_xform_);
                            edit_capture_valid_ = false;
                        }
                    }
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", mesh_file_list_[i].c_str());
            }
        }
        ImGui::EndChild();
    }

    if (sel_idx >= 0) {
        ImGui::Text("Source:");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.45f, 1.0f, 0.55f, 1.0f), "%s",
            std::filesystem::path(sel_path).filename().string().c_str());
    } else {
        ImGui::TextDisabled("No mesh selected");
    }
}

// Vector icon painter for the card buttons (no texture assets needed).
//   kind 0 = skeleton, 1 = 2D edit canvas, 2 = training network.
void AutoRigPlugin::drawButtonIcon(int kind, ImVec2 c, float r, ImU32 col) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float th = 2.4f;
    if (kind == 0) {
        ImVec2 head (c.x,            c.y - r);
        ImVec2 chest(c.x,            c.y - r * 0.25f);
        ImVec2 hips (c.x,            c.y + r * 0.55f);
        ImVec2 lh   (c.x - r*0.75f,  c.y + r * 0.15f);
        ImVec2 rh   (c.x + r*0.75f,  c.y + r * 0.15f);
        ImVec2 lf   (c.x - r*0.50f,  c.y + r * 1.05f);
        ImVec2 rf   (c.x + r*0.50f,  c.y + r * 1.05f);
        dl->AddLine(head, chest, col, th);
        dl->AddLine(chest, hips, col, th);
        dl->AddLine(chest, lh, col, th);
        dl->AddLine(chest, rh, col, th);
        dl->AddLine(hips, lf, col, th);
        dl->AddLine(hips, rf, col, th);
        for (ImVec2 j : { chest, hips, lh, rh, lf, rf })
            dl->AddCircleFilled(j, 2.7f, col);
        dl->AddCircleFilled(head, 4.4f, col);
    } else if (kind == 1) {
        ImVec2 a(c.x - r, c.y - r * 0.8f), b(c.x + r, c.y + r * 0.8f);
        dl->AddRect(a, b, IM_COL32(150,150,160,200), 3.0f, 0, 1.6f);
        ImVec2 n0(c.x - r*0.45f, c.y + r*0.35f);
        ImVec2 n1(c.x,           c.y - r*0.25f);
        ImVec2 n2(c.x + r*0.50f, c.y + r*0.10f);
        dl->AddLine(n0, n1, col, th);
        dl->AddLine(n1, n2, col, th);
        for (ImVec2 j : { n0, n1, n2 })
            dl->AddCircleFilled(j, 3.1f, col);
        dl->AddLine(ImVec2(n1.x-5,n1.y), ImVec2(n1.x+5,n1.y), IM_COL32(255,255,255,190), 1.0f);
        dl->AddLine(ImVec2(n1.x,n1.y-5), ImVec2(n1.x,n1.y+5), IM_COL32(255,255,255,190), 1.0f);
    } else {
        // Small neural network: two input nodes -> hidden -> two output nodes.
        ImVec2 a (c.x - r*0.75f, c.y - r*0.5f);
        ImVec2 b (c.x - r*0.75f, c.y + r*0.5f);
        ImVec2 m (c.x,           c.y);
        ImVec2 d (c.x + r*0.75f, c.y - r*0.5f);
        ImVec2 e (c.x + r*0.75f, c.y + r*0.5f);
        dl->AddLine(a, m, col, th); dl->AddLine(b, m, col, th);
        dl->AddLine(m, d, col, th); dl->AddLine(m, e, col, th);
        for (ImVec2 j : { a, b, d, e })
            dl->AddCircleFilled(j, 3.0f, col);
        dl->AddCircleFilled(m, 4.2f, col);
    }
}

// Lazy-load optional PNG icons for the launcher cards.  Must run inside a
// frame (ImGui's Vulkan backend initialised), so it's called from drawImGui().
// Best-effort: on any failure the cards fall back to the vector icons.
void AutoRigPlugin::loadUiTextures() {
    if (ui_textures_loaded_) return;
    ui_textures_loaded_ = true;          // attempt only once
    if (!device_) return;
    try {
        if (!ui_sampler_)
            ui_sampler_ = device_->createSampler(
                engine::renderer::Filter::LINEAR,
                engine::renderer::SamplerAddressMode::CLAMP_TO_EDGE,
                engine::renderer::SamplerMipmapMode::LINEAR, 1.0f,
                std::source_location::current());

        // Load one PNG from content/images/ (tries a few relative roots) into
        // an ImGui texture; returns 0 on any failure (card keeps vector icon).
        auto loadIcon = [&](const char* file,
                            std::shared_ptr<engine::renderer::TextureInfo>& info)
                            -> ImTextureID {
            const std::string roots[] = {
                "content/images/", "../content/images/",
                "../realworld/content/images/",
            };
            std::string path;
            for (const auto& r : roots)
                if (std::filesystem::exists(r + file)) { path = r + file; break; }
            if (path.empty()) return 0;
            info = std::make_shared<engine::renderer::TextureInfo>();
            engine::helper::createTextureImage(
                device_, path, engine::renderer::Format::R8G8B8A8_UNORM,
                true, *info, std::source_location::current());
            if (info->view && ui_sampler_)
                return engine::renderer::Helper::addImTextureID(ui_sampler_, info->view);
            return 0;
        };

        autorig_tex_id_ = loadIcon("auto-rig.png", autorig_tex_info_);
        rigedit_tex_id_ = loadIcon("rig-edit.png", rigedit_tex_info_);
        train_tex_id_   = loadIcon("ml-train.png", train_tex_info_);
    } catch (...) {
        // Leave whatever loaded; unset ids fall back to vector icons.
    }
}

// Card-style button: rounded box + accent border + (PNG or vector icon) + label.
bool AutoRigPlugin::drawIconButton(const char* id, const char* label, int kind,
                                   ImU32 accent, ImVec2 size, bool enabled,
                                   ImTextureID image) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (!enabled) ImGui::BeginDisabled();
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    bool clicked = ImGui::InvisibleButton(id, size);
    bool hov = enabled && ImGui::IsItemHovered();
    bool act = enabled && ImGui::IsItemActive();
    ImVec2 p1(p0.x + size.x, p0.y + size.y);
    ImU32 bg = act ? IM_COL32(58,58,70,255)
             : hov ? IM_COL32(48,48,60,255)
                   : IM_COL32(36,36,44,255);
    ImU32 border = hov ? accent : IM_COL32(90,90,100,255);
    ImU32 iconc  = enabled ? accent : IM_COL32(110,110,118,255);
    ImU32 textc  = enabled ? IM_COL32(228,228,233,255) : IM_COL32(140,140,146,255);
    dl->AddRectFilled(p0, p1, bg, 8.0f);
    dl->AddRect(p0, p1, border, 8.0f, 0, hov ? 2.5f : 1.0f);
    if (image) {
        // Draw the PNG centered in the area above the label.
        float side = std::min(size.x - 22.0f, size.y - 34.0f);
        ImVec2 ic(p0.x + size.x * 0.5f, p0.y + (size.y - 22.0f) * 0.5f);
        ImVec2 ia(ic.x - side * 0.5f, ic.y - side * 0.5f);
        ImVec2 ib(ic.x + side * 0.5f, ic.y + side * 0.5f);
        ImU32 tint = enabled ? IM_COL32(255,255,255,255) : IM_COL32(255,255,255,120);
        dl->AddImage(image, ia, ib, ImVec2(0,0), ImVec2(1,1), tint);
    } else {
        drawButtonIcon(kind, ImVec2(p0.x + size.x * 0.5f, p0.y + size.y * 0.4f), 20.0f, iconc);
    }
    ImVec2 ts = ImGui::CalcTextSize(label);
    dl->AddText(ImVec2(p0.x + (size.x - ts.x) * 0.5f, p1.y - 24.0f), textc, label);
    if (!enabled) ImGui::EndDisabled();
    return clicked;
}

// Model-info readout — shown in the main launcher window.  Long paths wrap so
// they don't stretch the auto-sized window.
void AutoRigPlugin::drawModelInfo() {
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 940.0f);

    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Model Info:");
    ImGui::Text("  Model dir: %s", model_dir_.c_str());
    ImGui::Text("  Versions found: %d", (int)model_versions_.size());
    if (model_loaded_idx_ >= 0 && model_loaded_idx_ < (int)model_versions_.size()) {
        auto& me = model_versions_[model_loaded_idx_];
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
            "  Loaded: %s  (%s)", me.label().c_str(), me.path.c_str());
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.2f, 1.0f),
            "  Loaded: STUB (heuristic placement)");
    }
    if (diffusion_model_) {
        ImGui::Text("  Model loaded: %s, joints: %d, module ptr: %s",
            diffusion_model_->isLoaded() ? "YES" : "NO",
            diffusion_model_->numJoints(),
            (model_loaded_idx_ >= 0) ? "LibTorch" : "stub");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f),
            "  diffusion_model_ is NULL!");
    }
    ImGui::Text("  CWD: %s", std::filesystem::current_path().string().c_str());

    ImGui::PopTextWrapPos();
}

void AutoRigPlugin::drawImGui() {
    // Track open/close transition (must run before early return).
    bool just_opened = show_window_ && !show_window_prev_;
    show_window_prev_ = show_window_;

    // Deferred Bake & Export: the button only sets bake_pending_ + clears the
    // progress bar (ui_progress_ = 0).  We run the (synchronous) bake here at
    // the TOP of the NEXT frame, so the cleared bar is shown for a frame first
    // instead of jumping straight to 100%.
    if (bake_pending_) {
        bake_pending_ = false;
        weights_baked_ = false;
        refreshInverseBindMatrices();   // edits moved joints
        bool ok = computeSkinWeights();
        if (ok) ok = exportGltf(output_path_buf_);
        if (ok) {
            weights_baked_ = true;
            state_ = PluginState::kFinished;
            ui_status_ = "Baked weights + exported: " +
                std::filesystem::path(output_path_buf_).filename().string();
            ui_progress_ = 1.0f;
        } else {
            state_ = PluginState::kError;
            ui_status_ = "Bake/export failed.";
        }
    }

    // The workflow / editor windows are independent: keep drawing them even
    // when the main launcher window itself is hidden.
    if (!show_window_) { drawAutoRigWorkflowWindow(); drawRigEditorWindow(); return; }

    ImGuiIO& io = ImGui::GetIO();

    // The launcher is a small hub — auto-size it to its content (no big empty
    // window), with a sensible width range so long paths wrap instead of
    // stretching it.  Centre on first appearance.
    ImGui::SetNextWindowPos(
        ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f),
        ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(720.0f, 0.0f),
                                        ImVec2(1000.0f, 100000.0f));

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
        ImGuiWindowFlags_NoDocking |
        ImGuiWindowFlags_AlwaysAutoResize;

    if (!ImGui::Begin("Auto Rig", &show_window_, flags)) {
        ImGui::End();
        return;
    }

    loadUiTextures();   // lazy-load PNG launcher icons (once, inside a frame)

    // ---- Train Model (root level, always visible) ----
    {
        // Model architecture selector — sized to the longest option so the
        // preview text never clips.
        {
            const ImGuiStyle& st = ImGui::GetStyle();
            float arch_w = ImGui::GetFrameHeight() + st.FramePadding.x * 2.0f;
            for (int i = 0; i < kNumModelArchs; ++i)
                arch_w = std::max(arch_w,
                    ImGui::CalcTextSize(kModelArchLabels[i]).x
                        + st.FramePadding.x * 2.0f + ImGui::GetFrameHeight() + 6.0f);
            ImGui::SetNextItemWidth(arch_w);
        }
        if (ImGui::BeginCombo("Architecture", kModelArchLabels[training_model_arch_])) {
            for (int i = 0; i < kNumModelArchs; ++i) {
                bool selected = (i == training_model_arch_);
                if (ImGui::Selectable(kModelArchLabels[i], selected))
                    training_model_arch_ = i;
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        // Snapshot the disabled state so it stays consistent even if the
        // button handler flips training_running_ mid-frame.
        const bool disable_train = training_running_;
        if (drawIconButton("##btn_train", "Train Model", 2,
                           IM_COL32(120,180,90,255), ImVec2(180.0f, 100.0f),
                           !disable_train, train_tex_id_)) {
            training_running_ = true;
            training_finished_ = false;
            training_epoch_ = 0;
            training_total_epochs_ = 0;
            training_loss_ = 0.0f;
            training_exit_code_ = -1;
            training_log_.clear();
            training_status_ = "Training started...";
            training_device_ = "?";
            training_device_detail_.clear();

            // Find the ml_training directory relative to the executable
            std::string exe_dir = std::filesystem::current_path().string();
            std::string ml_dir;

            for (const auto& candidate : {
                exe_dir + "/../ml_training",
                exe_dir + "/../../ml_training",
                exe_dir + "/ml_training",
            }) {
                if (std::filesystem::exists(candidate + "/train.py")) {
                    ml_dir = std::filesystem::canonical(candidate).string();
                    break;
                }
            }

            if (ml_dir.empty()) {
                training_status_ = "Error: cannot find ml_training/train.py";
                training_running_ = false;
            } else {
                // Find the training data directory.
                // The export writes to: <mesh_parent>/training_data/<stem>/
                // or the user may have data in assets/rigs_training/<stem>/
                std::string data_dir;

                // 1) If the Rig Editor has a mesh selected, check its sibling
                //    training_data folder (training data is exported per
                //    rig-editor mesh).
                if (!re_selected_path_.empty()) {
                    std::string stem = std::filesystem::path(re_selected_path_).stem().string();
                    std::string parent = std::filesystem::path(re_selected_path_)
                        .parent_path().string();
                    // Check training_data/<stem>
                    std::string cand = parent + "/training_data/" + stem;
                    if (std::filesystem::exists(cand)) data_dir = cand;
                }

                // 2) Scan assets/rigs_training/ for any subfolder with *_meta.json
                if (data_dir.empty()) {
                    for (const auto& rigs_dir : {
                        exe_dir + "/assets/rigs_training",
                        exe_dir + "/../realworld/assets/rigs_training",
                    }) {
                        if (!std::filesystem::exists(rigs_dir)) continue;
                        for (auto& sub : std::filesystem::directory_iterator(rigs_dir)) {
                            if (!sub.is_directory()) continue;
                            // Check if it has at least one *_meta.json
                            for (auto& f : std::filesystem::directory_iterator(sub.path())) {
                                if (f.path().string().find("_meta.json") != std::string::npos) {
                                    data_dir = sub.path().string();
                                    break;
                                }
                            }
                            if (!data_dir.empty()) break;
                        }
                        if (!data_dir.empty()) break;
                    }
                }

                // 3) Broader scan: any directory under assets/ containing *_meta.json
                if (data_dir.empty()) {
                    for (const auto& assets_root : {
                        exe_dir + "/assets",
                        exe_dir + "/../realworld/assets",
                    }) {
                        if (!std::filesystem::exists(assets_root)) continue;
                        try {
                            for (auto& entry : std::filesystem::recursive_directory_iterator(
                                     assets_root, std::filesystem::directory_options::skip_permission_denied)) {
                                if (entry.is_regular_file() &&
                                    entry.path().string().find("_meta.json") != std::string::npos) {
                                    data_dir = entry.path().parent_path().string();
                                    break;
                                }
                            }
                        } catch (...) {}
                        if (!data_dir.empty()) break;
                    }
                }

                if (data_dir.empty()) {
                    training_status_ = "Error: no training data found. Export data from Rig Editor first.";
                    fprintf(stderr, "[AutoRig] No training data found. Searched from: %s\n", exe_dir.c_str());
                    training_running_ = false;
                } else {
                    fprintf(stderr, "[AutoRig] Found training data: %s\n", data_dir.c_str());

                    // Determine the next version number for selected architecture
                    scanModelVersions();
                    std::string arch = kModelArchNames[training_model_arch_];
                    int next_ver = nextVersionForArch(arch);
                    std::string model_out = modelPathForArchVersion(arch, next_ver);

                    training_target_path_ = model_out;
                    std::filesystem::create_directories(ml_dir + "/datasets");
                    std::filesystem::create_directories(model_dir_);

                    std::string cmd =
                        "cd \"" + ml_dir + "\" && python train_from_captures.py"
                        " --data_dir \"" + data_dir + "\""
                        " --output_model \"" + model_out + "\""
                        " --model " + arch +
                        " --epochs 500"
                        " --device auto"
                        " 2>&1";

                    fprintf(stderr, "[AutoRig] Training command: %s\n", cmd.c_str());

                    std::thread([this, cmd]() {
#ifdef _WIN32
                        FILE* pipe = _popen(cmd.c_str(), "r");
#else
                        FILE* pipe = popen(cmd.c_str(), "r");
#endif
                        if (!pipe) {
                            training_status_ = "Error: failed to launch training process";
                            training_running_ = false;
                            return;
                        }

                        char buf[256];
                        while (fgets(buf, sizeof(buf), pipe)) {
                            training_log_ += buf;
                            std::string line(buf);
                            if (!line.empty() && line.back() == '\n')
                                line.pop_back();
                            if (!line.empty())
                                training_status_ = line;

                            // Parse epoch progress for progress bar.
                            // Formats:
                            //   "Epoch   5/500  loss=0.001234 ..."        (no val split)
                            //   "Epoch   5/500  train=0.001234  val=..."  (with val split)
                            {
                                const char* ep = strstr(buf, "Epoch");
                                if (ep) {
                                    int cur = 0, total = 0;
                                    float loss = 0.0f;
                                    if (sscanf(ep, "Epoch %d/%d loss=%f", &cur, &total, &loss) >= 2) {
                                        training_epoch_ = cur;
                                        training_total_epochs_ = total;
                                        training_loss_ = loss;
                                    } else if (sscanf(ep, "Epoch %d/%d train=%f", &cur, &total, &loss) >= 2) {
                                        training_epoch_ = cur;
                                        training_total_epochs_ = total;
                                        training_loss_ = loss;
                                        // Also grab val loss if present.
                                        const char* vp = strstr(ep, "val=");
                                        if (vp) {
                                            float vl = 0.0f;
                                            if (sscanf(vp, "val=%f", &vl) == 1)
                                                training_loss_ = vl;  // show val loss in UI
                                        }
                                    }
                                }
                            }

                            // Parse "[DEVICE] resolved=cuda  ..." and
                            // "[DEVICE] gpu[0]="NVIDIA ..."  compute_capability=8.6  cuda_runtime=12.1"
                            // so the UI can show which device training is on.
                            {
                                const char* dev_marker = strstr(buf, "[DEVICE]");
                                if (dev_marker) {
                                    // Find "resolved=<word>"
                                    const char* rs = strstr(dev_marker, "resolved=");
                                    if (rs) {
                                        rs += 9;  // skip "resolved="
                                        char tok[32] = {0};
                                        int ti = 0;
                                        while (*rs && *rs != ' ' && *rs != '\n' &&
                                               *rs != '\r' && ti < (int)sizeof(tok) - 1) {
                                            tok[ti++] = *rs++;
                                        }
                                        if (ti > 0) training_device_ = tok;
                                    }

                                    // Grab the GPU name line for the detail string.
                                    // Format: [DEVICE] gpu[0]="NAME"  compute_capability=X.Y  cuda_runtime=Z
                                    const char* gpu = strstr(dev_marker, "gpu[");
                                    if (gpu) {
                                        const char* q1 = strchr(gpu, '"');
                                        const char* q2 = q1 ? strchr(q1 + 1, '"') : nullptr;
                                        if (q1 && q2 && q2 > q1) {
                                            std::string name(q1 + 1, q2 - q1 - 1);
                                            std::string cap, cuda_rt;
                                            const char* cc = strstr(q2, "compute_capability=");
                                            if (cc) {
                                                cc += 19;
                                                while (*cc && *cc != ' ' && *cc != '\n') cap += *cc++;
                                            }
                                            const char* cr = strstr(q2, "cuda_runtime=");
                                            if (cr) {
                                                cr += 13;
                                                while (*cr && *cr != ' ' && *cr != '\n') cuda_rt += *cr++;
                                            }
                                            training_device_detail_ = name;
                                            if (!cap.empty())
                                                training_device_detail_ += " (cc " + cap;
                                            if (!cuda_rt.empty())
                                                training_device_detail_ += ", cuda " + cuda_rt + ")";
                                            else if (!cap.empty())
                                                training_device_detail_ += ")";
                                        }
                                    }

                                    // Capture the "reason_no_cuda=..." explanation
                                    const char* reason = strstr(dev_marker, "reason_no_cuda=");
                                    if (reason) {
                                        reason += 15;
                                        std::string r;
                                        while (*reason && *reason != '\n' && *reason != '\r')
                                            r += *reason++;
                                        training_device_detail_ = r;
                                    }
                                }
                            }
                        }

#ifdef _WIN32
                        training_exit_code_ = _pclose(pipe);
#else
                        training_exit_code_ = pclose(pipe);
#endif
                        training_running_ = false;
                        training_finished_ = true;

                        if (training_exit_code_ == 0) {
                            training_status_ = "Training complete! Model exported.";
                        } else {
                            training_status_ = "Training failed (exit code " +
                                std::to_string(training_exit_code_) + ")";
                        }
                    }).detach();
                }
            }
        }

        // Training status + progress bar
        if (training_running_) {
            if (training_total_epochs_ > 0) {
                float frac = (float)training_epoch_ / (float)training_total_epochs_;
                char overlay[96];
                std::snprintf(overlay, sizeof(overlay),
                    "[%s] Epoch %d/%d  loss=%.6f",
                    training_device_.c_str(),
                    training_epoch_, training_total_epochs_, training_loss_);
                ImGui::ProgressBar(frac, ImVec2(-1, 0), overlay);
            } else {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Training...");
            }

            // Second line: explicit device readout — green for cuda, yellow for
            // cpu, grey while unknown. This is the thing you actually want to
            // look at when training feels too slow.
            ImVec4 dcol;
            if (training_device_ == "cuda")     dcol = ImVec4(0.3f, 1.0f, 0.3f, 1.0f);
            else if (training_device_ == "cpu") dcol = ImVec4(1.0f, 0.8f, 0.2f, 1.0f);
            else                                dcol = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
            ImGui::TextColored(dcol, "Device: %s", training_device_.c_str());
            if (!training_device_detail_.empty()) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                    "(%s)", training_device_detail_.c_str());
            }
        }
        if (!training_status_.empty()) {
            ImVec4 col = training_finished_ && training_exit_code_ == 0
                ? ImVec4(0.2f, 1.0f, 0.2f, 1.0f)
                : ImVec4(0.9f, 0.9f, 0.9f, 1.0f);
            ImGui::TextColored(col, "%s", training_status_.c_str());
        }

        // Auto-reload the just-trained model after successful training
        if (training_finished_ && training_exit_code_ == 0) {
            scanModelVersions();
            // Find the exact model that was just trained by matching filename
            // (avoids Windows forward/backslash path mismatches).
            int target_idx = -1;
            std::string target_filename =
                std::filesystem::path(training_target_path_).filename().string();
            for (int i = 0; i < (int)model_versions_.size(); ++i) {
                std::string entry_filename =
                    std::filesystem::path(model_versions_[i].path).filename().string();
                if (entry_filename == target_filename) {
                    target_idx = i;
                    break;
                }
            }
            fprintf(stderr, "[AutoRig] Post-train reload: target=%s found_idx=%d\n",
                    target_filename.c_str(), target_idx);
            if (target_idx >= 0 && loadModelByIndex(target_idx)) {
                training_status_ = "Training complete! Loaded model " +
                    model_versions_[target_idx].label();
                model_selected_idx_ = target_idx;
            } else if (loadModelByIndex(-1)) {
                // Fallback to latest if exact match not found
                std::string lbl = (model_loaded_idx_ >= 0 && model_loaded_idx_ < (int)model_versions_.size())
                    ? model_versions_[model_loaded_idx_].label() : "?";
                training_status_ = "Training complete! Loaded model " + lbl +
                    " (expected " + target_filename + ")";
                model_selected_idx_ = -1;
            } else {
                training_status_ = "Training complete but failed to load new model.";
            }
            training_finished_ = false;
        }

        if (!training_log_.empty()) {
            if (ImGui::CollapsingHeader("Training Log")) {
                ImGui::BeginChild("##train_log", ImVec2(-1, 200), true);
                ImGui::TextUnformatted(training_log_.c_str());
                if (training_running_)
                    ImGui::SetScrollHereY(1.0f);
                ImGui::EndChild();
            }
        }

        // ---- Model version selector ----
        {
            // Show loaded model info
            if (model_loaded_idx_ >= 0 && model_loaded_idx_ < (int)model_versions_.size()) {
                auto& loaded = model_versions_[model_loaded_idx_];
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                    "Active model: %s", loaded.label().c_str());
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                    "Active model: stub (no trained model)");
            }

            if (!model_versions_.empty()) {
                // Build combo items: "Latest (arch_vNNN)", individual entries...
                // "Latest" = newest file on disk (latestModelIndex), so the
                // label matches what loadModelByIndex(-1) actually loads.
                const int latest_idx = latestModelIndex();
                const std::string latest_label =
                    (latest_idx >= 0) ? model_versions_[latest_idx].label() : "none";
                std::string current_label;
                if (model_selected_idx_ < 0)
                    current_label = "Latest (" + latest_label + ")";
                else
                    current_label = model_versions_[model_selected_idx_].label();

                // Size the combo to its label so it never clips.
                const ImGuiStyle& st = ImGui::GetStyle();
                ImGui::SetNextItemWidth(
                    ImGui::CalcTextSize(current_label.c_str()).x
                        + st.FramePadding.x * 2.0f + ImGui::GetFrameHeight() + 6.0f);

                if (ImGui::BeginCombo("##model_ver", current_label.c_str())) {
                    // "Latest" option
                    {
                        std::string lbl = "Latest (" + latest_label + ")";
                        bool selected = (model_selected_idx_ < 0);
                        if (ImGui::Selectable(lbl.c_str(), selected)) {
                            model_selected_idx_ = -1;
                            loadModelByIndex(-1);
                        }
                    }
                    // Individual models (newest first)
                    for (int i = (int)model_versions_.size() - 1; i >= 0; --i) {
                        auto& me = model_versions_[i];
                        bool selected = (model_selected_idx_ == i);
                        if (ImGui::Selectable(me.label().c_str(), selected)) {
                            model_selected_idx_ = i;
                            loadModelByIndex(i);
                        }
                    }
                    ImGui::EndCombo();
                }
            }
        }
    }

    ImGui::Separator();

    // ---- Launch buttons: custom icon cards (vector-drawn, no textures) ----
    {
        const ImVec2 btn_sz(180.0f, 100.0f);
        if (drawIconButton("##btn_autorig", "Auto Rig", 0,
                           IM_COL32(217,136,59,255), btn_sz, true, autorig_tex_id_))
            show_autorig_workflow_ = true;
        ImGui::SameLine();
        if (drawIconButton("##btn_rigeditor", "2D Rig Editor", 1,
                           IM_COL32(74,127,196,255), btn_sz, true, rigedit_tex_id_))
            show_rig_editor_ = true;
    }
    ImGui::Separator();

    // ---- Model Info (kept here in the main launcher window) ----
    drawModelInfo();

    ImGui::End();   // end "Auto Rig" launcher window

    // The two workflows are drawn as their own independent popup windows.
    drawAutoRigWorkflowWindow();
    drawRigEditorWindow();
}

// ---------------------------------------------------------------------------
//  Auto-Rig Workflow — separate popup window.  Holds the one-click Run
//  Auto-Rig, the manual 3-pass workflow, and the 3D preview / debug views.
//  Opened via the "Open Auto Rig" button in the main launcher window.
// ---------------------------------------------------------------------------
void AutoRigPlugin::drawAutoRigWorkflowWindow() {
    bool just_opened = show_autorig_workflow_ && !show_autorig_workflow_prev_;
    show_autorig_workflow_prev_ = show_autorig_workflow_;
    if (!show_autorig_workflow_) return;

    ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowSize(
        ImVec2(io.DisplaySize.x * 0.8f, io.DisplaySize.y * 0.85f),
        ImGuiCond_Appearing);
    ImGui::SetNextWindowPos(
        ImVec2(io.DisplaySize.x * 0.48f, io.DisplaySize.y * 0.48f),
        ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (just_opened) ImGui::SetNextWindowFocus();

    ImGui::SetNextWindowViewport(0);
    ImGuiWindowClass popup_class;
    popup_class.ViewportFlagsOverrideSet = ImGuiViewportFlags_NoAutoMerge;
    ImGui::SetNextWindowClass(&popup_class);

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoDocking;

    if (!ImGui::Begin("Auto-Rig Workflow", &show_autorig_workflow_, flags)) {
        ImGui::End();
        return;
    }

    // ---- Mesh file selection (Auto Rig's own mesh) ----
    drawMeshSelector(selected_mesh_idx_, selected_mesh_path_, /*auto_rig=*/true);
    ImGui::Separator();

    {
        ui_mode_ = 0;

            ImGui::SetNextItemWidth(150);
            ImGui::SliderInt("Views", &ui_num_views_, 4, 16);
            ImGui::SameLine(0, 24);
            ImGui::SetNextItemWidth(190);
            ImGui::SliderInt("Resolution", &ui_resolution_, 128, 2048);
            ImGui::Text("Output: %s", std::filesystem::path(output_path_buf_).filename().string().c_str());

            bool no_mesh = selected_mesh_path_.empty();

            // ---- Manual Workflow ----
            //  Generate joints -> (optionally) hand-edit them in 3D -> bake.
            //  "Run Combined All" (after step 3) runs the whole pipeline at once.
            ImGui::Separator();
            if (ImGui::CollapsingHeader("Manual Workflow (3 passes)",
                                        ImGuiTreeNodeFlags_DefaultOpen)) {
                const bool   running = (state_ == PluginState::kRunning);
                const ImVec4 kDone(0.40f, 0.90f, 0.45f, 1.0f);

                // Size all buttons to the widest label so text never clips.
                const ImGuiStyle& st = ImGui::GetStyle();
                const char* kAllBtnLabels[] = {
                    "1. Generate Joints", "2. Hide Editor", "3. Bake & Export",
                    "Reset to Generated", "Save Joints", "Load Saved Joints",
                    "Load Generated", "Load Edited" };
                float kBtnW = 0.0f;
                for (const char* L : kAllBtnLabels)
                    kBtnW = std::max(kBtnW, ImGui::CalcTextSize(L).x);
                kBtnW += st.FramePadding.x * 2.0f + 14.0f;

                ImGui::Spacing();

                // ===== Pipeline (horizontal): 1 Get Joints -> 2 Edit -> 3 Bake =====
                // Buttons only, laid left-to-right in pipeline order; hover a
                // button for a description of that step.
                ImGui::SeparatorText("Rigging Pipeline   (left -> right)");

                // Distinct hover / active colours so buttons clearly react.
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.95f, 0.62f, 0.18f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.80f, 0.50f, 0.12f, 1.0f));

                // -- 1. Get joints: Generate OR Load (adjacent = same step) --
                {
                    const bool dis = running || no_mesh;
                    if (dis) ImGui::BeginDisabled();
                    if (ImGui::Button("1. Generate Joints", ImVec2(kBtnW, 0))) {
                        num_views_          = ui_num_views_;
                        capture_resolution_ = ui_resolution_;
                        generateJoints();
                        edit3d_show_ = true;
                    }
                    if (dis) ImGui::EndDisabled();
                    if (!dis && ImGui::IsItemHovered())
                        ImGui::SetTooltip("Step 1: predict a skeleton from the mesh "
                                          "(renders multiple views -> joints).");
                }
                ImGui::SameLine();
                {
                    const bool dis = running || no_mesh;
                    if (dis) ImGui::BeginDisabled();
                    if (ImGui::Button("Load Joints", ImVec2(kBtnW, 0))) {
                        if (mesh_.empty() || source_mesh_path_ != selected_mesh_path_)
                            loadMesh(selected_mesh_path_);
                        if (savedJointsExist()) {
                            if (loadEditedJoints(loadJointsPath())) edit3d_show_ = true;
                        } else {
                            ui_status_ = "No saved .joints for this mesh - Generate first.";
                            edit3d_save_status_ = ui_status_;
                        }
                    }
                    if (dis) ImGui::EndDisabled();
                    if (!dis && ImGui::IsItemHovered())
                        ImGui::SetTooltip("Step 1 (alternative): load previously saved "
                                          ".joints for this mesh and skip prediction.");
                }
                ImGui::SameLine(0.0f, 28.0f);   // gap -> next pipeline step
                // -- 2. Edit joints --
                {
                    const bool dis = running || !joints_generated_;
                    if (dis) ImGui::BeginDisabled();
                    if (ImGui::Button(edit3d_show_ ? "2. Hide Editor" : "2. Edit Joints",
                                      ImVec2(kBtnW, 0)))
                        edit3d_show_ = !edit3d_show_;
                    if (dis) ImGui::EndDisabled();
                    if (!dis && ImGui::IsItemHovered())
                        ImGui::SetTooltip("Step 2: open the 3D editor and drag joints to "
                                          "refine their placement before baking.");
                }
                ImGui::SameLine(0.0f, 28.0f);
                // -- 3. Bake + export --
                {
                    const bool dis = running || !joints_generated_;
                    if (dis) ImGui::BeginDisabled();
                    if (ImGui::Button("3. Bake & Export", ImVec2(kBtnW, 0))) {
                        state_       = PluginState::kRunning;
                        ui_progress_ = 0.0f;          // clear the bar from the start
                        ui_status_   = "Baking...";
                        bake_pending_ = true;         // actual bake runs next frame
                    }
                    if (dis) ImGui::EndDisabled();
                    if (!dis && ImGui::IsItemHovered())
                        ImGui::SetTooltip("Step 3: compute skin weights from the joints "
                                          "and export the rigged glTF.");
                }
                ImGui::SameLine(0.0f, 28.0f);
                // -- Run all three steps at once --
                {
                    const bool dis = running || no_mesh;
                    if (dis) ImGui::BeginDisabled();
                    if (ImGui::Button("Run Combined All", ImVec2(kBtnW, 0))) {
                        num_views_          = ui_num_views_;
                        capture_resolution_ = ui_resolution_;
                        rigCharacter(selected_mesh_path_, output_path_buf_);
                    }
                    if (dis) ImGui::EndDisabled();
                    if (!dis && ImGui::IsItemHovered())
                        ImGui::SetTooltip("Run all three steps in one shot "
                                          "(generate -> bake, no manual editing).");
                }

                ImGui::PopStyleColor(2);

                // ---- Progress + status (below the buttons) ----
                if (ui_progress_ > 0.0f) ImGui::ProgressBar(ui_progress_, ImVec2(-1, 0));
                if (!ui_status_.empty())
                    ImGui::TextColored(kDone, "%s", ui_status_.c_str());

                // ---- Step 2 editor panel (shown when toggled on) ----
                if (joints_generated_ && edit3d_show_) {
                    ImGui::Spacing();
                    constexpr float kEditCanvas = 500.0f;
                    drawJointEditor3D(kEditCanvas);

                    if (ImGui::Button("Reset to Generated", ImVec2(kBtnW, 0))) {
                        fuseAndBuildSkeleton();      // re-fuse from predictions
                        joints_edited_     = false;
                        edit3d_drag_joint_ = -1;
                    }
                    ImGui::SameLine();
                    // Saves to "<name>.joints", or "<name>_edited.joints" once edited.
                    if (ImGui::Button("Save Joints", ImVec2(kBtnW, 0)))
                        saveEditedJoints(saveJointsPath());
                    ImGui::SameLine();
                    // Load: if both the generated and edited files exist, let the
                    // user pick which one; otherwise a single button loads the one
                    // that's there.
                    {
                        const bool hasBase   = std::filesystem::exists(baseJointsPath());
                        const bool hasEdited = std::filesystem::exists(editedJointsPath());
                        if (hasBase && hasEdited) {
                            if (ImGui::Button("Load Generated", ImVec2(kBtnW, 0)))
                                loadEditedJoints(baseJointsPath());
                            if (ImGui::IsItemHovered())
                                ImGui::SetTooltip("%s", std::filesystem::path(
                                    baseJointsPath()).filename().string().c_str());
                            ImGui::SameLine();
                            if (ImGui::Button("Load Edited", ImVec2(kBtnW, 0)))
                                loadEditedJoints(editedJointsPath());
                            if (ImGui::IsItemHovered())
                                ImGui::SetTooltip("%s", std::filesystem::path(
                                    editedJointsPath()).filename().string().c_str());
                        } else {
                            const bool dis = !(hasBase || hasEdited);
                            if (dis) ImGui::BeginDisabled();
                            if (ImGui::Button("Load Saved Joints", ImVec2(kBtnW, 0)))
                                loadEditedJoints(loadJointsPath());
                            if (!dis && ImGui::IsItemHovered())
                                ImGui::SetTooltip("%s", std::filesystem::path(
                                    loadJointsPath()).filename().string().c_str());
                            if (dis) ImGui::EndDisabled();
                        }
                    }
                    if (!edit3d_save_status_.empty())
                        ImGui::TextColored(kDone, "%s", edit3d_save_status_.c_str());
                }

                ImGui::Spacing();
            }

            // (Model Info is shown in the main Auto Rig launcher window.)
            ImGui::Separator();

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

                // Skin-weight overlay controls.
                if (!skin_weights_.empty()) {
                    ImGui::Separator();
                    ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1), "Skin weights:");
                    const char* modes[] = { "Off", "Per-slot (RGBA)", "Per-bone palette" };
                    ImGui::SetNextItemWidth(-1);
                    ImGui::Combo("##wmode", &weight_view_mode_, modes, IM_ARRAYSIZE(modes));
                    if (weight_view_mode_ == 1)
                        ImGui::TextWrapped("R/G/B/Yellow = weight slots 0-3 (concentration).");
                    else if (weight_view_mode_ == 2)
                        ImGui::TextWrapped("Each bone a distinct hue; blended by weight.");
                }
                // Debug: per-joint tau (geodesic falloff width) rings.
                if (!debug_tau_.empty()) {
                    ImGui::Checkbox("Show tau (falloff)", &show_tau_);
                    if (show_tau_)
                        ImGui::TextWrapped("Yellow ring = each joint's falloff radius (tau).");
                }
                ImGui::EndChild();
                ImGui::SameLine();

                // 3D canvas
                ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
                ImGui::InvisibleButton("##ar_preview", ImVec2(kCanvasSize, kCanvasSize));
                drawModel3DPreview(mesh_, skeleton_, kCanvasSize, canvas_pos,
                    ImGui::GetWindowDrawList(),
                    preview_yaw_, preview_pitch_, preview_dragging_, preview_drag_start_,
                    &skin_weights_, weight_view_mode_,
                    &debug_tau_, show_tau_);
            }

            // ---- Debug controls ----
            ImGui::Checkbox("Show Debug Views", &show_debug_views_);
            ImGui::SameLine();
            if (ImGui::Checkbox("Use Training Scale", &use_training_scale_)) {
                training_camera_dist_ = -1.0f;  // force re-read on next auto-rig
            }
            if (!use_training_scale_) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "(manual scale)");
                ImGui::SetNextItemWidth(200);
                ImGui::SliderFloat("Debug Scale", &camera_distance_, 0.3f, 2.0f, "%.2f");
            }
            if (show_debug_views_ && !captures_.empty()) {
                // Show ML predictions if available, otherwise heuristic
                bool has_ml = !view_predictions_.empty();
                if (has_ml) {
                    ImGui::TextColored(ImVec4(0.2f,1.0f,0.2f,1.0f),
                        "Showing: ML predictions (yellow)");
                } else if (!view_edits_.empty()) {
                    ImGui::TextColored(ImVec4(0.0f,0.9f,0.9f,1.0f),
                        "Showing: Heuristic (cyan)");
                }

                if (ImGui::CollapsingHeader("Debug: Per-View Captures")) {
                    static float dbg_scale = 0.35f;
                    ImGui::SliderFloat("Scale", &dbg_scale, 0.15f, 1.0f, "%.2f");

                    int num_views = (int)captures_.size();
                    for (int v = 0; v < num_views; ++v) {
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

                            auto uv2s = [&](const glm::vec2& uv) -> ImVec2 {
                                return ImVec2(cp.x + uv.x * tw, cp.y + uv.y * th);
                            };

                            if (has_ml && v < (int)view_predictions_.size()) {
                                // Draw ML predictions in YELLOW
                                const auto& vp = view_predictions_[v];
                                const auto& parents = getStandardJointParents();
                                for (int j = 0; j < (int)vp.joints.size(); ++j) {
                                    int pj = (j < (int)parents.size()) ? parents[j] : -1;
                                    if (pj >= 0 && pj < (int)vp.joints.size()) {
                                        vdl->AddLine(
                                            uv2s(vp.joints[pj].peak_uv),
                                            uv2s(vp.joints[j].peak_uv),
                                            IM_COL32(255,220,0,200), 2.0f);
                                    }
                                }
                                for (int j = 0; j < (int)vp.joints.size(); ++j) {
                                    vdl->AddCircleFilled(uv2s(vp.joints[j].peak_uv), 4.0f,
                                        IM_COL32(255,200,0,240));
                                }
                            } else if (!view_edits_.empty() && v < (int)view_edits_.size()) {
                                // Fallback: heuristic in CYAN
                                auto& es = view_edits_[v];
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
                            }

                            vdl->AddRect(cp, ImVec2(cp.x + tw, cp.y + th), IM_COL32(80,80,80,200));
                            vdl->PopClipRect();
                            ImGui::TreePop();
                        }
                        ImGui::PopID();
                    }
                }
            }

    }  // end auto-rig controls block

    ImGui::End();   // end "Auto-Rig Workflow" window
}

// ---------------------------------------------------------------------------
//  Rig Editor — separate popup window (2D joint editing / training export).
//  Opened via the "Open Rig Editor" button in the Auto Rig window.
// ---------------------------------------------------------------------------
void AutoRigPlugin::drawRigEditorWindow() {
    bool just_opened = show_rig_editor_ && !show_rig_editor_prev_;
    show_rig_editor_prev_ = show_rig_editor_;
    if (!show_rig_editor_) return;

    ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowSize(
        ImVec2(io.DisplaySize.x * 0.8f, io.DisplaySize.y * 0.85f),
        ImGuiCond_Appearing);
    ImGui::SetNextWindowPos(
        ImVec2(io.DisplaySize.x * 0.52f, io.DisplaySize.y * 0.52f),
        ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (just_opened) ImGui::SetNextWindowFocus();

    ImGui::SetNextWindowViewport(0);
    ImGuiWindowClass popup_class;
    popup_class.ViewportFlagsOverrideSet = ImGuiViewportFlags_NoAutoMerge;
    ImGui::SetNextWindowClass(&popup_class);

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoDocking;

    if (!ImGui::Begin("Rig Editor", &show_rig_editor_, flags)) {
        ImGui::End();
        return;
    }

    // ---- Mesh file selection (Rig Editor's own mesh) ----
    drawMeshSelector(re_selected_idx_, re_selected_path_, /*auto_rig=*/false);
    ImGui::Separator();

    {
        ui_mode_ = 1;

            if (re_mesh_.empty()) {
                ImGui::TextDisabled("Select a mesh to begin editing.");
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
                        if (!lock_azimuth_)
                            preview_yaw_   += (m.x - preview_drag_start_.x) * 0.01f;
                        if (!lock_elevation_) {
                            preview_pitch_ += (m.y - preview_drag_start_.y) * 0.01f;
                            preview_pitch_  = std::clamp(preview_pitch_, -1.5f, 1.5f);
                        }
                        preview_drag_start_ = m;
                    } else {
                        preview_dragging_ = false;
                    }
                }
            }

            // Re-render the rasterized preview when the angle or size changes.
            int render_res = std::max(64, (int)canvas_dim);
            if (!re_mesh_.empty() && rasterizer_) {
                bool angle_changed =
                    std::abs(preview_yaw_ - preview_render_yaw_) > 0.001f ||
                    std::abs(preview_pitch_ - preview_render_pitch_) > 0.001f;
                bool size_changed = (preview_render_.width != render_res);

                if (angle_changed || size_changed || preview_render_.width == 0) {
                    float az = -glm::degrees(preview_yaw_);
                    float el = -glm::degrees(preview_pitch_);

                    glm::vec3 centre = (re_mesh_.bbox_min + re_mesh_.bbox_max) * 0.5f;
                    float ext = glm::length(re_mesh_.bbox_max - re_mesh_.bbox_min);
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
                        re_mesh_, render_res, render_res,
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

            // Show current angle — click label to lock/unlock that axis.
            float az_deg = -glm::degrees(preview_yaw_);
            float el_deg = -glm::degrees(preview_pitch_);
            {
                ImVec4 lock_col(1.0f, 0.6f, 0.2f, 1.0f);   // orange when locked
                ImVec4 norm_col = ImGui::GetStyleColorVec4(ImGuiCol_Text);

                if (lock_azimuth_) ImGui::PushStyleColor(ImGuiCol_Text, lock_col);
                char az_lbl[64]; std::snprintf(az_lbl, sizeof(az_lbl),
                    lock_azimuth_ ? "Azimuth: %.0f [LOCKED]" : "Azimuth: %.0f", az_deg);
                if (ImGui::Selectable(az_lbl, lock_azimuth_,
                        ImGuiSelectableFlags_None, ImVec2(canvas_dim * 0.48f, 0)))
                    lock_azimuth_ = !lock_azimuth_;
                if (lock_azimuth_) ImGui::PopStyleColor();

                ImGui::SameLine();

                if (lock_elevation_) ImGui::PushStyleColor(ImGuiCol_Text, lock_col);
                char el_lbl[64]; std::snprintf(el_lbl, sizeof(el_lbl),
                    lock_elevation_ ? "Elevation: %.0f [LOCKED]" : "Elevation: %.0f", el_deg);
                if (ImGui::Selectable(el_lbl, lock_elevation_,
                        ImGuiSelectableFlags_None, ImVec2(canvas_dim * 0.48f, 0)))
                    lock_elevation_ = !lock_elevation_;
                if (lock_elevation_) ImGui::PopStyleColor();
            }

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

                glm::vec3 centre = (re_mesh_.bbox_min + re_mesh_.bbox_max) * 0.5f;
                float ext = glm::length(re_mesh_.bbox_max - re_mesh_.bbox_min);
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
                // Training data export uses the color PNG from this capture, which
                // is opaque; OIT only affects the interactive editing overlay.
                edit_capture_ = rasterizer_->renderOIT(
                    re_mesh_, capture_resolution_, capture_resolution_,
                    view_mat, proj, mesh_opacity_, az, el);
                initEditableJointsFromModel(edit_view_, edit_capture_);  // ML pre-fill
                edit_capture_valid_ = true;

                fprintf(stderr, "[AutoRig] Edit capture: radius=%.4f (ext=%.4f * cam_dist=%.4f) res=%d\n",
                        radius, ext, camera_distance_, capture_resolution_);
                drag_joint_ = -1;

                fprintf(stderr, "[AutoRig] Captured edit view: az=%.1f el=%.1f res=%d radius=%.3f\n",
                        az, el, capture_resolution_, radius);
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
                if (ImGui::Button("Save View")) {
                    SavedEditView sv;
                    sv.edit    = edit_view_;
                    sv.capture = edit_capture_;
                    saved_edits_.push_back(std::move(sv));
                    fprintf(stderr, "[AutoRig] Saved view %d (az=%.0f el=%.0f)\n",
                        (int)saved_edits_.size() - 1,
                        edit_capture_.azimuth_deg, edit_capture_.elevation_deg);
                }

                ImGui::SameLine();
                ImGui::Text("Saved: %d views", (int)saved_edits_.size());

                if (ImGui::Button("  Save for Training  ")) saveViewForTraining();
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Write this view (color+gray+depth+silhouette + view_proj + "
                                      "joints) to assets/rigs_training/<mesh> as one training sample.");
                ImGui::SameLine(); ImGui::TextDisabled("%s", ui_status_.c_str());
            }

            ImGui::EndChild();  // ##re_2d

            // ---- Saved views gallery ----
            if (!saved_edits_.empty()) {
                ImGui::Separator();
                ImGui::Text("Saved Views (%d):", (int)saved_edits_.size());
                ImGui::SameLine();
                if (ImGui::Button("Clear All##saved")) {
                    saved_edits_.clear();
                }
                float thumb_sz = 80.0f;
                for (int i = 0; i < (int)saved_edits_.size(); ++i) {
                    ImGui::PushID(i);
                    char lbl[32];
                    std::snprintf(lbl, sizeof(lbl), "View %d", i);
                    ImGui::Text("%s (az=%.0f)", lbl,
                        saved_edits_[i].capture.azimuth_deg);
                    if (i + 1 < (int)saved_edits_.size())
                        ImGui::SameLine();
                    ImGui::PopID();
                }
            }

            // ---- Export button ----
            ImGui::Separator();

            {
                bool no_saved = saved_edits_.empty();
                if (no_saved) ImGui::BeginDisabled();

                if (ImGui::Button("  Export Training Data  ")) {
                    // Export to assets/rigs_training/ — same place the Train button reads from.
                    std::string exe_path;
                    {
                        #ifdef _WIN32
                        char buf[MAX_PATH]; GetModuleFileNameA(nullptr, buf, MAX_PATH);
                        exe_path = std::filesystem::path(buf).parent_path().string();
                        #else
                        exe_path = std::filesystem::canonical("/proc/self/exe").parent_path().string();
                        #endif
                    }
                    std::string out_dir;
                    // Try: exe_dir/../realworld/assets/rigs_training (dev layout)
                    std::string dev_path = exe_path + "/../realworld/assets/rigs_training";
                    std::string prod_path = exe_path + "/assets/rigs_training";
                    if (std::filesystem::exists(dev_path) || !std::filesystem::exists(prod_path))
                        out_dir = dev_path;
                    else
                        out_dir = prod_path;
                    if (exportTrainingData(out_dir)) {
                        training_status_ = "Exported " + std::to_string(saved_edits_.size()) +
                            " views to " + out_dir;
                    } else {
                        training_status_ = "Export failed!";
                    }
                }

                if (no_saved) ImGui::EndDisabled();
            }

    }  // end Rig Editor controls block

    ImGui::End();   // end "Rig Editor" window
}

}  // namespace auto_rig
}  // namespace plugins
      