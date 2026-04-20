// ---------------------------------------------------------------------------
//  rig_diffusion_model.cpp – LibTorch inference wrapper.
//
//  This file has a hard dependency on LibTorch headers.  When LibTorch is
//  not available (HAS_LIBTORCH == 0) the model falls back to a deterministic
//  stub that places joints at heuristic positions derived from the silhouette.
// ---------------------------------------------------------------------------
#ifndef HAS_LIBTORCH
#  define HAS_LIBTORCH 0   // flip to 1 once LibTorch is linked
#endif

#include "rig_diffusion_model.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <numeric>

#if HAS_LIBTORCH
#  include <torch/script.h>
#  include <torch/torch.h>
#endif

namespace plugins {
namespace auto_rig {

// ============================================================================
//  Ctor / Dtor
// ============================================================================

RigDiffusionModel::RigDiffusionModel()  = default;

RigDiffusionModel::~RigDiffusionModel() {
#if HAS_LIBTORCH
    delete static_cast<torch::jit::Module*>(module_);
    delete static_cast<torch::Device*>(device_ptr_);
#endif
    module_     = nullptr;
    device_ptr_ = nullptr;
    loaded_     = false;
}

// ============================================================================
//  load
// ============================================================================

bool RigDiffusionModel::load(
    const std::string& model_path,
    const std::string& device_str)
{
    fprintf(stderr, "[RigDiffusionModel] load(\"%s\", \"%s\")\n",
        model_path.c_str(), device_str.c_str());
    fprintf(stderr, "[RigDiffusionModel] HAS_LIBTORCH=%d\n", HAS_LIBTORCH);

#if HAS_LIBTORCH
    if (!model_path.empty()) {
        try {
            fprintf(stderr, "[RigDiffusionModel] Creating device '%s'...\n", device_str.c_str());
            auto* dev = new torch::Device(device_str);

            fprintf(stderr, "[RigDiffusionModel] Loading TorchScript model...\n");
            auto* mod = new torch::jit::Module(torch::jit::load(model_path, *dev));
            mod->eval();
            fprintf(stderr, "[RigDiffusionModel] Model loaded, set to eval mode.\n");

            device_ptr_ = dev;
            module_     = mod;
            loaded_     = true;

            // Infer num_joints from the model by running a dummy forward pass.
            fprintf(stderr, "[RigDiffusionModel] Running dummy forward pass (1, 7, 64, 64)...\n");
            auto dummy = torch::zeros({1, 7, 64, 64}, dev->is_cuda()
                             ? torch::kCUDA : torch::kCPU);
            auto out = mod->forward({dummy});

            // The model may return a single tensor (heatmaps only) or a tuple
            // (heatmaps, adjacency).  Detect which format.
            if (out.isTensor()) {
                num_joints_ = static_cast<int>(out.toTensor().size(1));
                auto sizes = out.toTensor().sizes();
                fprintf(stderr, "[RigDiffusionModel] Output format: single tensor [%lld, %lld, %lld, %lld]\n",
                    (long long)sizes[0], (long long)sizes[1],
                    (long long)sizes[2], (long long)sizes[3]);
            } else {
                auto tup = out.toTuple();
                num_joints_ = static_cast<int>(tup->elements()[0].toTensor().size(1));
                fprintf(stderr, "[RigDiffusionModel] Output format: tuple (%d elements)\n",
                    (int)tup->elements().size());
            }

            fprintf(stderr, "[RigDiffusionModel] Loaded OK: %s on %s  (joints=%d)\n",
                model_path.c_str(), device_str.c_str(), num_joints_);
            return true;
        } catch (const std::exception& e) {
            fprintf(stderr, "[RigDiffusionModel] LOAD FAILED: %s\n", e.what());
            fprintf(stderr, "[RigDiffusionModel] Falling back to stub mode.\n");
        }
    }
#else
    (void)model_path;
    (void)device_str;
#endif

    // Stub mode: use standard humanoid joint set with heuristic placement.
    num_joints_ = static_cast<int>(getStandardJointNames().size());
    loaded_ = true;
    fprintf(stderr, "[RigDiffusionModel] running in STUB mode (%d joints)\n",
        num_joints_);
    return true;
}

// ============================================================================
//  prepareInput – build (1, 7, H, W) tensor data from a ViewCapture
// ============================================================================

std::vector<float> RigDiffusionModel::prepareInput(
    const ViewCapture& capture) const
{
    int w = capture.width;
    int h = capture.height;
    int npix = w * h;

    // Layout: CHW  (channels = RGB:3 + Normal:3 + Silhouette:1 = 7).
    std::vector<float> data(7 * npix, 0.0f);

    for (int i = 0; i < npix; ++i) {
        // RGB [0,1]
        float r = capture.color[i * 3 + 0] / 255.0f;
        float g = capture.color[i * 3 + 1] / 255.0f;
        float b = capture.color[i * 3 + 2] / 255.0f;
        data[0 * npix + i] = r;
        data[1 * npix + i] = g;
        data[2 * npix + i] = b;
        // Normals: use RGB as proxy to match training pipeline.
        // Training data uses color_np as normals (same as RGB, [0,1]).
        // Using actual normal_map here would create a mismatch.
        data[3 * npix + i] = r;
        data[4 * npix + i] = g;
        data[5 * npix + i] = b;
        // Silhouette [0,1]
        data[6 * npix + i] = capture.silhouette[i] / 255.0f;
    }
    return data;
}

// ============================================================================
//  decodeOutput – convert raw model output into ViewJointPrediction
// ============================================================================

ViewJointPrediction RigDiffusionModel::decodeOutput(
    const std::vector<float>& heatmaps,
    const std::vector<float>& adjacency,
    int view_idx,
    int width, int height) const
{
    ViewJointPrediction pred;
    pred.view_idx = view_idx;

    const auto& names = getStandardJointNames();
    int J = num_joints_;
    int npix = width * height;

    // -- Extract per-joint heatmaps and find peaks --
    for (int j = 0; j < J; ++j) {
        JointHeatmap jh;
        jh.name = (j < (int)names.size()) ? names[j] : ("joint_" + std::to_string(j));
        jh.heatmap.resize(npix);

        float best_val = -1.0f;
        int best_idx = 0;
        for (int i = 0; i < npix; ++i) {
            float v = heatmaps[j * npix + i];
            jh.heatmap[i] = v;
            if (v > best_val) { best_val = v; best_idx = i; }
        }

        int py = best_idx / width;
        int px = best_idx % width;
        jh.peak_uv = glm::vec2(
            (px + 0.5f) / width,
            (py + 0.5f) / height);
        jh.confidence = best_val;

        pred.joints.push_back(std::move(jh));
    }

    // -- Extract bone connectivity --
    const auto& parent_template = getStandardJointParents();
    for (int j = 0; j < J; ++j) {
        int parent = (j < (int)parent_template.size()) ? parent_template[j] : -1;
        if (parent < 0) continue;

        float conf = (parent < J && j < J)
            ? adjacency[parent * J + j]
            : 0.5f;

        BoneEdge edge;
        edge.parent_joint = parent;
        edge.child_joint  = j;
        edge.confidence   = conf;
        pred.bones.push_back(edge);
    }

    return pred;
}

// ============================================================================
//  predict (single view)
// ============================================================================

ViewJointPrediction RigDiffusionModel::predict(
    const ViewCapture& capture) const
{
    int w = capture.width;
    int h = capture.height;
    int npix = w * h;
    int J = num_joints_;

#if HAS_LIBTORCH
    if (module_) {
        auto* mod = static_cast<torch::jit::Module*>(module_);
        auto* dev = static_cast<torch::Device*>(device_ptr_);

        auto input_data = prepareInput(capture);
        auto tensor = torch::from_blob(input_data.data(), {1, 7, h, w},
                                       torch::kFloat32);

        // Resize to model training resolution.
        constexpr int kModelRes = 256;
        if (h != kModelRes || w != kModelRes) {
            namespace F = torch::nn::functional;
            tensor = F::interpolate(tensor,
                F::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{kModelRes, kModelRes})
                    .mode(torch::kBilinear)
                    .align_corners(false));
        }
        tensor = tensor.to(*dev);

        auto output = mod->forward({tensor});

        torch::Tensor heat_t;
        std::vector<float> adjacency;

        if (output.isTensor()) {
            // Single-tensor output: heatmaps only.
            heat_t = output.toTensor().cpu().contiguous();
            // Build adjacency from the standard parent template.
            adjacency.resize(J * J, 0.0f);
            const auto& parents = getStandardJointParents();
            for (int j = 0; j < J && j < (int)parents.size(); ++j) {
                if (parents[j] >= 0 && parents[j] < J) {
                    adjacency[parents[j] * J + j] = 1.0f;
                    adjacency[j * J + parents[j]] = 1.0f;
                }
            }
        } else {
            // Tuple output: (heatmaps, adjacency).
            auto tup = output.toTuple();
            heat_t = tup->elements()[0].toTensor().cpu().contiguous();
            auto adj_t = tup->elements()[1].toTensor().cpu().contiguous();
            adjacency.assign(adj_t.data_ptr<float>(),
                adj_t.data_ptr<float>() + adj_t.numel());
        }

        // Use model's output resolution, not capture resolution
        int out_h = static_cast<int>(heat_t.size(2));
        int out_w = static_cast<int>(heat_t.size(3));

        std::vector<float> heatmaps(heat_t.data_ptr<float>(),
            heat_t.data_ptr<float>() + heat_t.numel());

        return decodeOutput(heatmaps, adjacency, 0, out_w, out_h);
    }
#endif

    // ---- Stub: heuristic joint placement from silhouette -------------------

    fprintf(stderr, "[RigDiffusionModel] *** STUB MODE *** — heuristic placement\n");

    // Find silhouette bounding box.
    int sil_min_x = w, sil_max_x = 0, sil_min_y = h, sil_max_y = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (capture.silhouette[y * w + x]) {
                sil_min_x = std::min(sil_min_x, x);
                sil_max_x = std::max(sil_max_x, x);
                sil_min_y = std::min(sil_min_y, y);
                sil_max_y = std::max(sil_max_y, y);
            }
        }
    }

    float cx = (sil_min_x + sil_max_x) * 0.5f;
    float bh = (float)(sil_max_y - sil_min_y);
    if (bh < 1.0f) bh = 1.0f;

    // Humanoid proportional positions (0 = top of head, 1 = feet).
    struct JointPlacement { float rx; float ry; };
    const JointPlacement placements[] = {
        { 0.50f, 0.45f },  // hips
        { 0.50f, 0.38f },  // spine
        { 0.50f, 0.28f },  // chest
        { 0.50f, 0.18f },  // neck
        { 0.50f, 0.07f },  // head
        { 0.38f, 0.23f },  // left_shoulder
        { 0.28f, 0.33f },  // left_upper_arm
        { 0.20f, 0.43f },  // left_lower_arm
        { 0.15f, 0.52f },  // left_hand
        { 0.62f, 0.23f },  // right_shoulder
        { 0.72f, 0.33f },  // right_upper_arm
        { 0.80f, 0.43f },  // right_lower_arm
        { 0.85f, 0.52f },  // right_hand
        { 0.42f, 0.55f },  // left_upper_leg
        { 0.42f, 0.72f },  // left_lower_leg
        { 0.42f, 0.93f },  // left_foot
        { 0.58f, 0.55f },  // right_upper_leg
        { 0.58f, 0.72f },  // right_lower_leg
        { 0.58f, 0.93f },  // right_foot
    };

    // Build synthetic heatmaps (Gaussian blobs at the heuristic positions).
    std::vector<float> heatmaps(J * npix, 0.0f);
    float sigma = bh * 0.04f;
    float inv_2sigma2 = 1.0f / (2.0f * sigma * sigma);

    for (int j = 0; j < J && j < 19; ++j) {
        float jx = sil_min_x + placements[j].rx * (sil_max_x - sil_min_x);
        float jy = sil_min_y + placements[j].ry * bh;

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float dx = x - jx;
                float dy = y - jy;
                float g = std::exp(-(dx * dx + dy * dy) * inv_2sigma2);
                heatmaps[j * npix + y * w + x] = g;
            }
        }
    }

    // Build synthetic adjacency (standard parent-child connections = 1.0).
    std::vector<float> adjacency(J * J, 0.0f);
    const auto& parents = getStandardJointParents();
    for (int j = 0; j < J && j < (int)parents.size(); ++j) {
        if (parents[j] >= 0 && parents[j] < J) {
            adjacency[parents[j] * J + j] = 1.0f;
            adjacency[j * J + parents[j]] = 1.0f;
        }
    }

    return decodeOutput(heatmaps, adjacency, 0, w, h);
}

// ============================================================================
//  predictBatch
// ============================================================================

std::vector<ViewJointPrediction> RigDiffusionModel::predictBatch(
    const std::vector<ViewCapture>& captures) const
{
    fprintf(stderr, "[RigDiffusionModel] predictBatch: %d views, loaded=%d, "
        "num_joints=%d, module=%p\n",
        (int)captures.size(), (int)loaded_, num_joints_, module_);

#if HAS_LIBTORCH
    if (module_ && !captures.empty()) {
        auto* mod = static_cast<torch::jit::Module*>(module_);
        auto* dev = static_cast<torch::Device*>(device_ptr_);

        int B = static_cast<int>(captures.size());
        int h = captures[0].height;
        int w = captures[0].width;
        int npix = h * w;

        fprintf(stderr, "[RigDiffusionModel] LibTorch path: batch=%d, %dx%d, device=%s\n",
            B, w, h, dev->str().c_str());

        // Build batched input tensor (B, 7, H, W).
        fprintf(stderr, "[RigDiffusionModel] Building input tensor (%d, 7, %d, %d)...\n",
            B, h, w);
        std::vector<float> batch_data(B * 7 * npix);
        for (int i = 0; i < B; ++i) {
            auto single = prepareInput(captures[i]);
            std::copy(single.begin(), single.end(),
                      batch_data.begin() + i * 7 * npix);

            // Log input statistics for first view
            if (i == 0) {
                float rgb_min = 1e9f, rgb_max = -1e9f;
                float sil_sum = 0.0f;
                for (int p = 0; p < npix; ++p) {
                    float r = single[0 * npix + p];
                    if (r < rgb_min) rgb_min = r;
                    if (r > rgb_max) rgb_max = r;
                    sil_sum += single[6 * npix + p];
                }
                fprintf(stderr, "[RigDiffusionModel]   view 0: R range=[%.3f, %.3f], "
                    "silhouette coverage=%.1f%%\n",
                    rgb_min, rgb_max, 100.0f * sil_sum / npix);
            }
        }

        // Resize input to the model's training resolution (256x256).
        // The U-Net is fully convolutional, so it CAN accept any size, but
        // the learned features (Gaussian blobs, receptive fields, etc.) only
        // make sense at the resolution it was trained at.
        constexpr int kModelRes = 256;
        fprintf(stderr, "[RigDiffusionModel] Building raw tensor (%d, 7, %d, %d)...\n",
            B, h, w);
        auto tensor = torch::from_blob(batch_data.data(), {B, 7, h, w},
                                        torch::kFloat32);

        if (h != kModelRes || w != kModelRes) {
            fprintf(stderr, "[RigDiffusionModel] Resizing input %dx%d -> %dx%d "
                "(model training resolution)\n", w, h, kModelRes, kModelRes);
            namespace F = torch::nn::functional;
            tensor = F::interpolate(tensor,
                F::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{kModelRes, kModelRes})
                    .mode(torch::kBilinear)
                    .align_corners(false));
        }
        tensor = tensor.to(*dev);

        fprintf(stderr, "[RigDiffusionModel] Running forward pass (input: %lldx%lld)...\n",
            (long long)tensor.size(3), (long long)tensor.size(2));
        auto output = mod->forward({tensor});
        fprintf(stderr, "[RigDiffusionModel] Forward pass complete.\n");

        torch::Tensor heat_t;
        if (output.isTensor()) {
            heat_t = output.toTensor().cpu().contiguous();
            fprintf(stderr, "[RigDiffusionModel] Output: single tensor %s\n",
                heat_t.sizes().vec().empty() ? "[]" :
                ("[" + std::to_string(heat_t.size(0)) + ", " +
                 std::to_string(heat_t.size(1)) + ", " +
                 std::to_string(heat_t.size(2)) + ", " +
                 std::to_string(heat_t.size(3)) + "]").c_str());
        } else {
            heat_t = output.toTuple()->elements()[0].toTensor().cpu().contiguous();
            fprintf(stderr, "[RigDiffusionModel] Output: tuple, heatmaps %s\n",
                ("[" + std::to_string(heat_t.size(0)) + ", " +
                 std::to_string(heat_t.size(1)) + ", " +
                 std::to_string(heat_t.size(2)) + ", " +
                 std::to_string(heat_t.size(3)) + "]").c_str());
        }

        // Log heatmap statistics
        {
            float hm_min = heat_t.min().item<float>();
            float hm_max = heat_t.max().item<float>();
            float hm_mean = heat_t.mean().item<float>();
            fprintf(stderr, "[RigDiffusionModel] Heatmaps: min=%.4f, max=%.4f, mean=%.4f\n",
                hm_min, hm_max, hm_mean);
        }

        // The model output may be at a different resolution than the input captures.
        // e.g. captures are 1024x1024 but model outputs 256x256 heatmaps.
        int out_h = static_cast<int>(heat_t.size(2));
        int out_w = static_cast<int>(heat_t.size(3));
        int out_npix = out_h * out_w;
        fprintf(stderr, "[RigDiffusionModel] Input: %dx%d, Output heatmaps: %dx%d\n",
            w, h, out_w, out_h);

        // Standard adjacency.
        int J = num_joints_;
        std::vector<float> adjacency(J * J, 0.0f);
        const auto& parents = getStandardJointParents();
        for (int j = 0; j < J && j < (int)parents.size(); ++j) {
            if (parents[j] >= 0 && parents[j] < J) {
                adjacency[parents[j] * J + j] = 1.0f;
                adjacency[j * J + parents[j]] = 1.0f;
            }
        }

        // Decode each view using the MODEL's output resolution, not the capture's.
        // decodeOutput normalizes peaks to UV [0,1], so resolution differences are OK.
        std::vector<ViewJointPrediction> results;
        results.reserve(B);
        const float* hm_ptr = heat_t.data_ptr<float>();
        int per_view = J * out_npix;
        for (int i = 0; i < B; ++i) {
            std::vector<float> hm(hm_ptr + i * per_view,
                                  hm_ptr + (i + 1) * per_view);
            auto pred = decodeOutput(hm, adjacency, i, out_w, out_h);

            // Log first view's joint predictions
            if (i == 0) {
                fprintf(stderr, "[RigDiffusionModel] View 0 predictions:\n");
                for (int j = 0; j < (int)pred.joints.size(); ++j) {
                    fprintf(stderr, "  [%2d] %-18s uv=(%.3f, %.3f) conf=%.4f\n",
                        j, pred.joints[j].name.c_str(),
                        pred.joints[j].peak_uv.x, pred.joints[j].peak_uv.y,
                        pred.joints[j].confidence);
                }
            }

            results.push_back(std::move(pred));
        }
        fprintf(stderr, "[RigDiffusionModel] Decoded %d views via LibTorch (output %dx%d)\n",
            B, out_w, out_h);
        return results;
    }

    fprintf(stderr, "[RigDiffusionModel] LibTorch module is null — falling through to stub\n");
#else
    fprintf(stderr, "[RigDiffusionModel] HAS_LIBTORCH=0 — using stub mode\n");
#endif

    // Fallback: sequential per-view prediction (stub mode).
    fprintf(stderr, "[RigDiffusionModel] Stub mode: %d views, %d joints\n",
        (int)captures.size(), num_joints_);

    std::vector<ViewJointPrediction> results;
    results.reserve(captures.size());
    for (int i = 0; i < (int)captures.size(); ++i) {
        auto pred = predict(captures[i]);
        pred.view_idx = i;
        results.push_back(std::move(pred));
    }
    return results;
}

}  // namespace auto_rig
}  // namespace plugins