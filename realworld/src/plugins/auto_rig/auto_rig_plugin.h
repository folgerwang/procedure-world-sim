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

    // Accessors.
    const Skeleton&    getSkeleton()    const { return skeleton_; }
    const SkinWeights& getSkinWeights() const { return skin_weights_; }
    const std::vector<ViewCapture>& getCaptures() const { return captures_; }

private:
    void reportProgress(int step, int total, const std::string& msg);

    PluginState     state_ = PluginState::kUnloaded;
    ProgressCallback progress_cb_;

    // Input mesh data.
    std::string     source_mesh_path_;   // original file — reloaded at export time
    TriangleMesh    mesh_;
    glm::mat4       mesh_node_world_transform_ = glm::mat4(1.0f);  // world xform of the mesh node
    int             detected_up_axis_ = 1;  // 0=X, 1=Y, 2=Z  (auto-detected from bbox)

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
    bool   files_scanned_ = false;

    // 3D preview rotation state (shared between both modes).
    float  preview_yaw_   = 0.0f;
    float  preview_pitch_ = 0.0f;
    bool   preview_dragging_ = false;
    ImVec2 preview_drag_start_ = {};

    // Auto-rig debug multi-view toggle (off by default).
    bool   show_debug_views_ = false;

    // Live rasterized preview for Rig Editor (cached, re-rendered on angle change).
    ViewCapture preview_render_;
    float       preview_render_yaw_   = -999.0f;
    float       preview_render_pitch_ = -999.0f;
    int         preview_render_res_   = 384;  // lower res for interactive speed
    float       mesh_opacity_         = 0.6f; // OIT mesh opacity (0 = invisible, 1 = opaque)
    float       camera_distance_      = 0.9f; // camera orbit radius multiplier

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

    void initEditableJointsForView(ViewEditState& state, const ViewCapture& cap);
    void initEditableJoints();  // for multi-view (auto-rig debug)
    bool exportTrainingData(const std::string& output_dir);

    // Multi-view editable joints (for auto-rig debug overlay).
    std::vector<ViewEditState> view_edits_;
};

} // namespace auto_rig
} // namespace plugins
