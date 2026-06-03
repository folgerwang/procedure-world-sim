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
    const std::shared_ptr<engine::renderer::Device>& /*device*/)
{
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
        target_idx = (int)model_versions_.size() - 1;  // latest
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

    std::string stem = selected_mesh_path_.empty() ? "untitled"
        : std::filesystem::path(selected_mesh_path_).stem().string();
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

        // Sort by distance, take up to the 4 closest.
        std::sort(dists.begin(), dists.end(),
            [](const BoneDist& a, const BoneDist& b) { return a.dist < b.dist; });

        auto& vsd = skin_weights_.per_vertex[v];
        float total_w = 0.0f;
        int count = std::min(4, (int)dists.size());

        // Relative-distance cutoff: only bind to bones that are reasonably
        // close to the NEAREST bone.  Without this, a torso vertex whose
        // nearest bone is the spine still grabs three more "top 4" bones —
        // often an arm AND a leg, or a left AND a right limb — which makes the
        // skin tear when those opposite limbs move apart (e.g. the head split
        // and the left hand following the right arm).  A bone 3x farther than
        // the closest contributes negligibly to a real deformation, so we drop
        // it.  Vertices near a single bone now bind almost entirely to it;
        // vertices in a genuine blend region (near two joints) still share.
        const float nearest = dists.empty() ? 0.0f : dists[0].dist;
        constexpr float kRelCut = 3.0f;   // keep bones within 3x the nearest

        int kept = 0;
        for (int i = 0; i < count; ++i) {
            if (i > 0 && dists[i].dist > (nearest + 0.001f) * kRelCut) break;
            vsd.joint_indices[kept] = dists[i].joint;
            // Inverse-distance weighting with a small epsilon.
            float w = 1.0f / (dists[i].dist + 0.001f);
            vsd.weights[kept] = w;
            total_w += w;
            ++kept;
        }
        // Zero any unused influence slots so stale data can't leak in.
        for (int i = kept; i < 4; ++i) {
            vsd.joint_indices[i] = 0;
            vsd.weights[i] = 0.0f;
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

    {
        std::string ml = (model_loaded_idx_ >= 0 && model_loaded_idx_ < (int)model_versions_.size())
            ? model_versions_[model_loaded_idx_].label() : "stub/heuristic";
        reportProgress(3, kTotalSteps, "Running model " + ml + "...");
    }
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

// Color for one vertex from its 4 skin weights, in the chosen mode.
static void weightVertexColor(const VertexSkinData& vsd, int mode,
                              float& r, float& g, float& b) {
    r = g = b = 0.0f;
    // Mode 1: four fixed slot colors (R, G, B, yellow).
    static const float kSlot[4][3] = {
        {1,0.2f,0.2f}, {0.2f,1,0.2f}, {0.3f,0.5f,1}, {1,0.85f,0.15f} };
    for (int k = 0; k < 4; ++k) {
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
    const SkinWeights* weights = nullptr, int weight_mode = 0)
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

    // ---- Train Model (root level, always visible) ----
    {
        // Model architecture selector.
        ImGui::SetNextItemWidth(220);
        if (ImGui::BeginCombo("Architecture", kModelArchLabels[training_model_arch_])) {
            for (int i = 0; i < kNumModelArchs; ++i) {
                bool selected = (i == training_model_arch_);
                if (ImGui::Selectable(kModelArchLabels[i], selected))
                    training_model_arch_ = i;
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();

        // Snapshot the disabled state so Begin/End stay balanced even if the
        // button handler flips training_running_ mid-frame.
        const bool disable_train = training_running_;
        if (disable_train) ImGui::BeginDisabled();
        if (ImGui::Button("  Train Model  ")) {
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

                // 1) If a mesh is selected, check its sibling training_data folder
                if (!source_mesh_path_.empty()) {
                    std::string stem = std::filesystem::path(source_mesh_path_).stem().string();
                    std::string parent = std::filesystem::path(source_mesh_path_)
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
        if (disable_train) ImGui::EndDisabled();

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
                ImGui::SameLine();
                ImGui::SetNextItemWidth(200);

                // Build combo items: "Latest (arch_vNNN)", individual entries...
                std::string current_label;
                if (model_selected_idx_ < 0)
                    current_label = "Latest (" + model_versions_.back().label() + ")";
                else
                    current_label = model_versions_[model_selected_idx_].label();

                if (ImGui::BeginCombo("##model_ver", current_label.c_str())) {
                    // "Latest" option
                    {
                        std::string lbl = "Latest (" + model_versions_.back().label() + ")";
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
                        captures_ = rasterizer_->captureOrbit(mesh_, num_views_, capture_resolution_,
                                                              0.0f, camera_distance_);
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
            // Snapshot the disabled state — rigCharacter() flips state_ to
            // kRunning, which would otherwise unbalance Begin/End.
            const bool disable_run = (is_running || no_mesh);
            if (disable_run) ImGui::BeginDisabled();
            if (ImGui::Button("  Run Auto-Rig  ")) {
                num_views_ = ui_num_views_;
                capture_resolution_ = ui_resolution_;
                rigCharacter(selected_mesh_path_, output_path_buf_);
            }
            if (disable_run) ImGui::EndDisabled();
            ImGui::SameLine(); ImGui::Text("%s", ui_status_.c_str());
            if (ui_progress_ > 0.0f) ImGui::ProgressBar(ui_progress_, ImVec2(-1, 0));

            // ---- Debug: model info ----
            ImGui::Separator();
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
                ImGui::EndChild();
                ImGui::SameLine();

                // 3D canvas
                ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
                ImGui::InvisibleButton("##ar_preview", ImVec2(kCanvasSize, kCanvasSize));
                drawModel3DPreview(mesh_, skeleton_, kCanvasSize, canvas_pos,
                    ImGui::GetWindowDrawList(),
                    preview_yaw_, preview_pitch_, preview_dragging_, preview_drag_start_,
                    &skin_weights_, weight_view_mode_);
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
                // Training data export uses the color PNG from this capture, which
                // is opaque; OIT only affects the interactive editing overlay.
                edit_capture_ = rasterizer_->renderOIT(
                    mesh_, capture_resolution_, capture_resolution_,
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

            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}

}  // namespace auto_rig
}  // namespace plugins
      