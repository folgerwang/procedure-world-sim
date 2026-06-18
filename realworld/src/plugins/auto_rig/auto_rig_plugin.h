#pragma once
#include "plugins/plugin_interface.h"
#include "plugins/auto_rig/simple_rasterizer.h"
#include "plugins/auto_rig/rig_diffusion_model.h"
#include "plugins/auto_rig/rig_types.h"
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <glm/glm.hpp>
#include "imgui.h"

namespace engine { namespace renderer { class Sampler; struct TextureInfo; } }

namespace plugins {
namespace auto_rig {

// ---------------------------------------------------------------------------
//  AutoRigPlugin
//
//  End-to-end auto-rigging pipeline for 3D character models:
//
//    1. Load mesh vertex data (positions, normals, triangles)
//    2. Render the character from N camera angles using a simple
//       software rasterizer (depth + normal + silhouette maps)
//    3. Feed each view into a diffusion model (LibTorch) that predicts
//       2D joint heatmaps + bone connectivity for that viewpoint
//    4. Back-project the per-view 2D predictions into 3D using the
//       known camera matrices
//    5. Fuse the multi-view 3D joint proposals via weighted averaging
//       and build a skeleton hierarchy
//    6. Compute skinning weights (nearest-bone heat diffusion)
//    7. Export as glTF skin/joints
// ---------------------------------------------------------------------------
class AutoRigPlugin : public IPlugin {
public:
    AutoRigPlugin();
    ~AutoRigPlugin() override;

    // -- IPlugin --
    const char* getName()    const override { return "AutoRig"; }
    const char* getVersion() const override { return "0.1.0"; }
    bool init(const std::shared_ptr<engine::renderer::Device>& device) override;
    void shutdown() override;
    PluginState getState() const override { return state_; }
    void drawImGui() override;
    void setProgressCallback(ProgressCallback cb) override { progress_cb_ = cb; }
    void setVisible(bool v) override { show_window_ = v; }
    bool isVisible() const override { return show_window_; }

    // The Rig Editor lives in its own window with its own menu entry, toggled
    // independently of the main Auto Rig window.
    void setRigEditorVisible(bool v) { show_rig_editor_ = v; }
    bool isRigEditorVisible() const  { return show_rig_editor_; }

    // -----------------------------------------------------------------------
    //  Public API
    // -----------------------------------------------------------------------

    // Full pipeline: mesh in -> skeleton + weights out.
    bool rigCharacter(
        const std::string& mesh_path,
        const std::string& output_gltf_path);

    // Per-stage access for debugging / advanced use.
    bool loadMesh(const std::string& path);
    bool captureViews(int num_views = 8, int resolution = 512);
    bool predictJoints();
    bool fuseAndBuildSkeleton();
    bool computeSkinWeights();
    bool exportGltf(const std::string& output_path);

    // -----------------------------------------------------------------------
    //  Manual 3-pass workflow (Generate -> Edit -> Bake).
    //  Pass 1: build the skeleton joints only (no skin weights).  The mesh
    //          must already be loaded (it is, once selected in the UI).
    //  Pass 2: interactive 3D joint editing — see drawJointEditor3D().
    //  Pass 3: computeSkinWeights() + exportGltf() (driven from the UI).
    // -----------------------------------------------------------------------
    bool generateJoints();

    // Accessors.
    const Skeleton&    getSkeleton()    const { return skeleton_; }
    const SkinWeights& getSkinWeights() const { return skin_weights_; }
    const std::vector<ViewCapture>& getCaptures() const { return captures_; }

private:
    void reportProgress(int step, int total, const std::string& msg);

    PluginState     state_ = PluginState::kUnloaded;
    ProgressCallback progress_cb_;

    // Model versioning — models stored as rig_diffusion_<arch>_v###.pt
    // Also supports legacy rig_diffusion_v###.pt (arch = "unet").
    struct ModelEntry {
        int         version = 0;
        std::string arch;        // "unet", "hourglass", "resnet", or "" for legacy
        std::string path;        // full filesystem path
        std::filesystem::file_time_type mtime{};  // file last-write time
        std::string label() const {
            if (arch.empty()) return "v" + std::to_string(version);
            char buf[80];
            std::snprintf(buf, sizeof(buf), "%s_v%03d", arch.c_str(), version);
            return buf;
        }
    };
    std::string     model_dir_;            // resolved absolute path to models directory
    std::vector<ModelEntry> model_versions_;  // discovered models (sorted by version)
    int             model_loaded_idx_ = -1;   // index into model_versions_ (-1 = stub)
    int             model_selected_idx_ = -1; // UI selection (-1 = latest)

    void scanModelVersions();              // refresh model_versions_ from disk
    std::string modelPathForVersion(int v) const;  // full path for version v (legacy compat)
    std::string modelPathForArchVersion(const std::string& arch, int v) const;
    int nextVersionForArch(const std::string& arch) const;
    bool loadModelByIndex(int idx);        // load by index into model_versions_ (-1 = latest)
    // Index of the most-recently-TRAINED model (newest file mtime), independent
    // of architecture name or per-arch version numbering.  -1 if none.
    int latestModelIndex() const;

    // Input mesh data (Auto Rig workflow).
    std::string     source_mesh_path_;   // original file — reloaded at export time
    TriangleMesh    mesh_;
    glm::mat4       mesh_node_world_transform_ = glm::mat4(1.0f);  // world xform of the mesh node
    int             detected_up_axis_ = 1;  // 0=X, 1=Y, 2=Z  (auto-detected from bbox)

    // Independent mesh for the 2D Rig Editor window — kept fully separate from
    // the auto-rig mesh_ so selecting/working in one window never affects the
    // other.
    TriangleMesh    re_mesh_;
    std::string     re_src_path_;
    glm::mat4       re_xform_ = glm::mat4(1.0f);

    // Multi-view capture.
    std::unique_ptr<SimpleRasterizer> rasterizer_;
    std::vector<ViewCapture>          captures_;
    int capture_resolution_ = 1024;
    int num_views_          = 8;

    // Per-view joint predictions from the diffusion model.
    std::unique_ptr<RigDiffusionModel> diffusion_model_;
    std::vector<ViewJointPrediction>   view_predictions_;

    // Final outputs.
    Skeleton    skeleton_;
    SkinWeights skin_weights_;

    // Scan asset directories for mesh files.
    void refreshMeshFileList();

    // ImGui state.
    bool   show_window_  = false;
    bool   show_window_prev_ = false;  // track open transition for focus

    // The Rig Editor lives in its own popup window, opened from the Auto Rig
    // window.  Auto-rig content and rig-editor content are now separate windows.
    bool   show_rig_editor_      = false;
    bool   show_rig_editor_prev_ = false;
    // The auto-rig workflow (Run Auto-Rig + manual 3-pass + 3D preview) also
    // lives in its own popup window now; the main window is just a launcher.
    bool   show_autorig_workflow_      = false;
    bool   show_autorig_workflow_prev_ = false;
    void   drawAutoRigWorkflowWindow();// the "Auto-Rig Workflow" popup window
    void   drawRigEditorWindow();      // the "Rig Editor" popup window
    // Mesh file picker.  Operates on caller-supplied selection state and, when
    // auto_rig is false, loads into the Rig Editor's independent re_mesh_.
    void   drawMeshSelector(int& sel_idx, std::string& sel_path, bool auto_rig);
    // Load a mesh file into a caller-provided buffer without disturbing the
    // auto-rig mesh_ (gives the Rig Editor its own mesh).
    bool   loadMeshInto(const std::string& path, TriangleMesh& out_mesh,
                        std::string& out_src, glm::mat4& out_xform);
    void   drawModelInfo();            // model-info block (main launcher window)

    // Custom card-style icon buttons (vector-drawn, no texture assets).
    //   kind 0 = skeleton, 1 = 2D edit canvas, 2 = training network.
    // If `image` is non-zero the PNG is drawn instead of the vector icon.
    void   drawButtonIcon(int kind, ImVec2 c, float r, ImU32 col);
    bool   drawIconButton(const char* id, const char* label, int kind,
                          ImU32 accent, ImVec2 size, bool enabled = true,
                          ImTextureID image = 0);

    // ---- Optional PNG icons for the launcher cards (lazy-loaded) ----
    std::shared_ptr<engine::renderer::Device>      device_;       // kept from init()
    std::shared_ptr<engine::renderer::Sampler>     ui_sampler_;
    std::shared_ptr<engine::renderer::TextureInfo> autorig_tex_info_;
    std::shared_ptr<engine::renderer::TextureInfo> rigedit_tex_info_;
    std::shared_ptr<engine::renderer::TextureInfo> train_tex_info_;
    ImTextureID autorig_tex_id_     = 0;
    ImTextureID rigedit_tex_id_     = 0;
    ImTextureID train_tex_id_       = 0;
    bool        ui_textures_loaded_ = false;   // attempted once
    void        loadUiTextures();              // needs ImGui initialised
    char   output_path_buf_[512] = "rigged_output.glb";
    int    ui_num_views_ = 8;
    int    ui_resolution_ = 1024;
    float  ui_progress_ = 0.0f;
    std::string ui_status_ = "Idle";

    // Mode: 0 = Auto Rig, 1 = Rig Editor
    int    ui_mode_ = 0;

    // File selection state.
    std::vector<std::string> mesh_file_list_;   // discovered .glb/.gltf/.obj/.fbx
    int    selected_mesh_idx_ = -1;
    std::string selected_mesh_path_;
    int    re_selected_idx_ = -1;          // Rig Editor's own selection
    std::string re_selected_path_;
    bool   files_scanned_ = false;

    // 3D preview rotation state (shared between both modes).
    float  preview_yaw_   = 0.0f;
    float  preview_pitch_ = 0.0f;
    bool   preview_dragging_ = false;
    ImVec2 preview_drag_start_ = {};
    bool   lock_azimuth_   = false;
    bool   lock_elevation_ = false;

    // Auto-rig debug multi-view toggle (off by default).
    bool   show_debug_views_ = false;

    // Skin-weight debug overlay on the 3D preview.
    //   0 = off (flat shaded), 1 = per-slot RGBA, 2 = per-bone palette.
    int    weight_view_mode_ = 0;

    // Live rasterized preview for Rig Editor (cached, re-rendered on angle change).
    ViewCapture preview_render_;
    float       preview_render_yaw_   = -999.0f;
    float       preview_render_pitch_ = -999.0f;
    int         preview_render_res_   = 384;  // lower res for interactive speed
    float       mesh_opacity_         = 0.6f; // OIT mesh opacity (0 = invisible, 1 = opaque)
    float       camera_distance_      = 0.9f; // camera orbit radius multiplier
    float       training_camera_dist_ = -1.0f; // camera distance from training data (-1 = not loaded)
    bool        use_training_scale_   = true;  // auto-rig uses training camera dist when true

    // ---- Rig editing state ----
    struct EditableJoint {
        glm::vec2 uv{0.0f};
        bool      edited = false;
    };
    struct ViewEditState {
        std::vector<EditableJoint> joints;
        bool any_edited = false;
        float azimuth_deg = 0.0f;
        float elevation_deg = 0.0f;
    };

    // The single edit view captured from current 3D preview angle.
    ViewEditState  edit_view_;
    ViewCapture    edit_capture_;       // the rendered 2D snapshot for editing
    bool           edit_capture_valid_ = false;  // true after "Capture View" click

    // All saved edit snapshots for export.
    struct SavedEditView {
        ViewEditState  edit;
        ViewCapture    capture;
    };
    std::vector<SavedEditView> saved_edits_;

    int   drag_joint_ = -1;

    // ---- Manual 3-pass workflow state ----
    bool   joints_generated_ = false;   // Pass 1 produced a skeleton
    bool   weights_baked_    = false;   // Pass 3 baked skin weights

    // ---- Pass 2: interactive 3D joint editor ----
    // Renders the original mesh translucently (OIT) and lets the user drag
    // skeleton joints in the camera view-plane.  Rotate the view to set depth.
    int    edit3d_drag_joint_   = -1;     // joint currently being dragged
    bool   edit3d_rotating_     = false;  // view rotation in progress
    bool   edit3d_show_         = false;  // Step-2 editor panel visible
    float  edit3d_opacity_      = 0.55f;  // OIT mesh opacity in the editor
    ViewCapture edit3d_render_;           // cached OIT mesh render
    float  edit3d_render_yaw_   = -999.0f;
    float  edit3d_render_pitch_ = -999.0f;
    float  edit3d_render_dist_  = -1.0f;
    float  edit3d_render_op_    = -1.0f;
    // Draw the Pass-2 3D joint editor canvas (mesh OIT + draggable joints).
    void   drawJointEditor3D(float canvas_size);
    // Rebuild inverse-bind matrices from current joint positions (after edits).
    void   refreshInverseBindMatrices();

    // ---- Save / load skeleton joints ----
    // Persist the current skeleton — names, parents and 3D positions — to a
    // small text file so edits survive across sessions and can be reloaded
    // without re-running joint generation.
    //   generated joints  ->  "<character>.joints"
    //   hand-edited joints ->  "<character>_edited.joints"
    bool        joints_edited_ = false;          // a joint was dragged this session
    std::string baseJointsPath()   const;        // "<character>.joints"
    std::string editedJointsPath() const;        // "<character>_edited.joints"
    std::string saveJointsPath()   const;        // edited path if edited, else base
    std::string loadJointsPath()   const;        // edited if it exists, else base
    bool        savedJointsExist() const;        // either file exists
    bool   saveEditedJoints(const std::string& path);
    bool   loadEditedJoints(const std::string& path);
    std::string edit3d_save_status_;             // last save/load message

    void initEditableJointsForView(ViewEditState& state, const ViewCapture& cap);
    void initEditableJointsFromModel(ViewEditState& state, const ViewCapture& cap);
    void initEditableJoints();  // for multi-view (auto-rig debug)
    bool exportTrainingData(const std::string& output_dir);
    bool saveViewForTraining();  // one-click: save current edited view as a training sample

    // Multi-view edits for debug panel.
    std::vector<ViewEditState> view_edits_;

    // ---- Model architecture selection ----
    // Maps to Python --model argument: "unet", "hourglass", "resnet"
    static constexpr const char* kModelArchNames[] = { "unet", "hourglass", "resnet" };
    static constexpr const char* kModelArchLabels[] = { "U-Net", "Stacked Hourglass", "SimpleBaseline (ResNet)" };
    static constexpr int kNumModelArchs = 3;
    int    training_model_arch_ = 1;   // index into kModelArchNames (default: hourglass)

    // ---- Training state ----
    std::string training_target_path_;  // path of model being trained (for reload)
    bool   training_running_ = false;
    bool   training_finished_ = false;
    int    training_exit_code_ = -1;
    std::string training_log_;
    std::string training_status_ = "Idle";
    int    training_epoch_ = 0;
    int    training_total_epochs_ = 0;
    float  training_loss_ = 0.0f;
    // Device info parsed from "[DEVICE] ..." lines emitted by
    // train_from_captures.py. Shown next to the training progress bar so
    // it's obvious whether CUDA or CPU is actually being used.
    std::string training_device_ = "?";  // e.g. "cuda" or "cpu"
    std::string training_device_detail_;  // "NVIDIA RTX 3090 (cc 8.6, cuda 12.1)"

};

}  // namespace auto_rig
}  // namespace plugins