#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace plugins {
namespace auto_rig {

// ---------------------------------------------------------------------------
//  Basic mesh representation (CPU-side, for the software rasterizer).
// ---------------------------------------------------------------------------
struct SimpleTexture {
    int width  = 0;
    int height = 0;
    int channels = 0;                        // 3=RGB, 4=RGBA
    std::vector<uint8_t> pixels;             // row-major, top-to-bottom

    bool empty() const { return pixels.empty(); }

    // Sample at UV (bilinear, wrapping).
    glm::vec3 sample(const glm::vec2& uv) const;
};

struct TriangleMesh {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;        // per-vertex UV (TEXCOORD_0)
    std::vector<glm::vec3> vertex_colors;    // per-vertex RGB [0,1] (COLOR_0)
    std::vector<uint32_t>  indices;          // triangle list

    SimpleTexture base_color_texture;        // base color / albedo map

    glm::vec3 bbox_min{  1e30f };
    glm::vec3 bbox_max{ -1e30f };

    void recomputeBounds();
    void recomputeNormals();
    bool empty() const { return positions.empty(); }
    bool hasColor() const {
        return !base_color_texture.empty() || !vertex_colors.empty();
    }
};

// ---------------------------------------------------------------------------
//  Multi-view capture output.
// ---------------------------------------------------------------------------
struct ViewCapture {
    int             width        = 0;
    int             height       = 0;
    float           azimuth_deg  = 0.0f;     // camera orbit angle
    float           elevation_deg= 0.0f;
    glm::mat4       view         = glm::mat4(1.0f);
    glm::mat4       proj         = glm::mat4(1.0f);
    glm::mat4       view_proj    = glm::mat4(1.0f);

    std::vector<float>    depth;             // W*H, linear depth
    std::vector<float>    normal_map;        // W*H*3, world-space normals
    std::vector<uint8_t>  silhouette;        // W*H, 0 or 255
    std::vector<uint8_t>  color;             // W*H*3, simple shaded RGB (opaque)
    std::vector<uint8_t>  color_rgba;        // W*H*4, OIT-composited RGBA (if rendered with alpha < 1)
};

// ---------------------------------------------------------------------------
//  Per-view 2D joint prediction (output of the diffusion model).
// ---------------------------------------------------------------------------
struct JointHeatmap {
    std::string         name;                // e.g. "left_shoulder"
    std::vector<float>  heatmap;             // W*H, [0..1] confidence
    glm::vec2           peak_uv;             // argmax in normalised coords
    float               confidence;          // peak value
};

struct BoneEdge {
    int parent_joint;                        // index into joints vector
    int child_joint;
    float confidence;
};

struct ViewJointPrediction {
    int                         view_idx;
    std::vector<JointHeatmap>   joints;
    std::vector<BoneEdge>       bones;
};

// ---------------------------------------------------------------------------
//  3D Skeleton (the fused output).
// ---------------------------------------------------------------------------
struct Joint {
    std::string  name;
    int          parent = -1;                // -1 = root
    glm::vec3    position{0.0f};             // bind pose, model space
    glm::quat    rotation{1, 0, 0, 0};
    glm::vec3    scale{1.0f};
    glm::mat4    inverse_bind_matrix = glm::mat4(1.0f);
};

struct Skeleton {
    std::vector<Joint> joints;
    int                root = -1;

    bool empty() const { return joints.empty(); }

    // Build inverse bind matrices from the joint transforms.
    void computeInverseBindMatrices();
};

// ---------------------------------------------------------------------------
//  Skinning weights.
// ---------------------------------------------------------------------------
struct VertexSkinData {
    int   joint_indices[4] = {0, 0, 0, 0};
    float weights[4]       = {0.0f, 0.0f, 0.0f, 0.0f};
};

struct SkinWeights {
    std::vector<VertexSkinData> per_vertex;  // one per mesh vertex
    bool empty() const { return per_vertex.empty(); }
};

// ---------------------------------------------------------------------------
//  Standard humanoid joint names used by the diffusion model.
// ---------------------------------------------------------------------------
inline const std::vector<std::string>& getStandardJointNames() {
    static const std::vector<std::string> names = {
        "hips",
        "spine",
        "chest",
        "neck",
        "head",
        "left_shoulder",
        "left_upper_arm",
        "left_lower_arm",
        "left_hand",
        "right_shoulder",
        "right_upper_arm",
        "right_lower_arm",
        "right_hand",
        "left_upper_leg",
        "left_lower_leg",
        "left_foot",
        "right_upper_leg",
        "right_lower_leg",
        "right_foot"
    };
    return names;
}

// Standard parent indices (-1 = root).
inline const std::vector<int>& getStandardJointParents() {
    static const std::vector<int> parents = {
        -1,  // hips
         0,  // spine -> hips
         1,  // chest -> spine
         2,  // neck -> chest
         3,  // head -> neck
         2,  // left_shoulder -> chest
         5,  // left_upper_arm -> left_shoulder
         6,  // left_lower_arm -> left_upper_arm
         7,  // left_hand -> left_lower_arm
         2,  // right_shoulder -> chest
         9,  // right_upper_arm -> right_shoulder
        10,  // right_lower_arm -> right_upper_arm
        11,  // right_hand -> right_lower_arm
         0,  // left_upper_leg -> hips
        13,  // left_lower_leg -> left_upper_leg
        14,  // left_foot -> left_lower_leg
         0,  // right_upper_leg -> hips
        16,  // right_lower_leg -> right_upper_leg
        17   // right_foot -> right_lower_leg
    };
    return parents;
}

} // namespace auto_rig
} // namespace plugins
