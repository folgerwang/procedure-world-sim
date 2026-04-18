#pragma once
#include "rig_types.h"
#include <string>
#include <memory>
#include <vector>

// Forward-declare LibTorch types to avoid pulling the header everywhere.
namespace torch { namespace jit { class Module; } }

namespace plugins {
namespace auto_rig {

// ---------------------------------------------------------------------------
//  RigDiffusionModel – wraps a LibTorch-scripted diffusion model that
//  predicts 2D joint heatmaps + bone connectivity from a single-view
//  rendering of a character.
//
//  Model architecture (expected TorchScript):
//    Input:  (B, 7, H, W)  — RGB (3) + Normal (3) + Silhouette (1)
//    Output: (B, J, H, W)  — per-joint heatmaps (J = num_joints)
//            (B, J, J)     — bone adjacency confidence matrix
//
//  The model file is a TorchScript archive (.pt) loaded via torch::jit::load.
// ---------------------------------------------------------------------------
class RigDiffusionModel {
public:
    RigDiffusionModel();
    ~RigDiffusionModel();

    // Load the TorchScript model from disk.
    // device_str: "cpu", "cuda:0", etc.
    bool load(const std::string& model_path, const std::string& device_str = "cpu");
    bool isLoaded() const { return loaded_; }

    // Run inference on a single view capture.
    // Returns the predicted joints and bones for that view.
    ViewJointPrediction predict(const ViewCapture& capture) const;

    // Run inference on all views in batch (if the model supports batching).
    std::vector<ViewJointPrediction> predictBatch(
        const std::vector<ViewCapture>& captures) const;

    // How many joints the model predicts.
    int numJoints() const { return num_joints_; }

private:
    // Build the (1, 7, H, W) input tensor from a ViewCapture.
    // Returns raw float data in CHW layout.
    std::vector<float> prepareInput(const ViewCapture& capture) const;

    // Decode the model output tensors into structured predictions.
    ViewJointPrediction decodeOutput(
        const std::vector<float>& heatmaps,  // (J, H, W) flattened
        const std::vector<float>& adjacency, // (J, J) flattened
        int view_idx,
        int width, int height) const;

    bool loaded_ = false;
    int  num_joints_ = 0;

    // Opaque pointer to the TorchScript module.
    // Using void* to keep the header lightweight; the .cpp casts to the real type.
    void* module_ = nullptr;
    void* device_ptr_ = nullptr;  // torch::Device*
};

} // namespace auto_rig
} // namespace plugins
