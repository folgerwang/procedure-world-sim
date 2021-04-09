#pragma once
#include "renderer.h"

namespace work {
namespace renderer {

struct MaterialInfo {
    int32_t                base_color_idx_ = -1;
    int32_t                normal_idx_ = -1;
    int32_t                metallic_roughness_idx_ = -1;
    int32_t                emissive_idx_ = -1;
    int32_t                occlusion_idx_ = -1;

    BufferInfo             uniform_buffer_;
    std::shared_ptr<DescriptorSet>  desc_set_;
};

struct BufferView {
    uint32_t                buffer_idx;
    uint64_t                stride;
    uint64_t                offset;
    uint64_t                range;
};

struct PrimitiveInfo {
    std::vector<uint32_t>   binding_list_;
    int32_t                 material_idx_;
    bool                    has_normal_ = false;
    bool                    has_tangent_ = false;
    bool                    has_texcoord_0_ = false;
    bool                    has_skin_set_0_ = false;
    glm::vec3               bbox_min_ = glm::vec3(std::numeric_limits<float>::max());
    glm::vec3               bbox_max_ = glm::vec3(std::numeric_limits<float>::min());
    IndexInputBindingDescription  index_desc_;
    PipelineInputAssemblyStateCreateInfo topology_info_;
    std::vector<VertexInputBindingDescription> binding_descs_;
    std::vector<VertexInputAttributeDescription> attribute_descs_;
};

struct MeshInfo {
    std::vector<PrimitiveInfo>  primitives_;
    glm::vec3                   bbox_min_ = glm::vec3(std::numeric_limits<float>::max());
    glm::vec3                   bbox_max_ = glm::vec3(std::numeric_limits<float>::min());
};

struct NodeInfo {
    std::vector<int32_t>        child_idx;
    int32_t                     mesh_idx = -1;
    std::shared_ptr<glm::mat4>  matrix;
};

struct SceneInfo {
    std::vector<int32_t>        nodes_;
    glm::vec3                   bbox_min_ = glm::vec3(std::numeric_limits<float>::max());
    glm::vec3                   bbox_max_ = glm::vec3(std::numeric_limits<float>::min());
};

struct ObjectData {
    int32_t                     default_scene_;
    std::vector<SceneInfo>      scenes_;
    std::vector<NodeInfo>       nodes_;
    std::vector<MeshInfo>       meshes_;
    std::vector<BufferInfo>     buffers_;
    std::vector<BufferView>     buffer_views_;

    std::vector<TextureInfo>    textures_;
    std::vector<MaterialInfo>   materials_;

public:
    void destroy(const std::shared_ptr<Device>& device);
};

std::shared_ptr<renderer::ObjectData> loadGltfModel(
    const renderer::DeviceInfo& device_info,
    const std::string& input_filename);

void drawNodes(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const std::shared_ptr<renderer::ObjectData>& gltf_object,
    const std::shared_ptr<renderer::PipelineLayout>& gltf_pipeline_layout,
    const std::shared_ptr<renderer::DescriptorSet>& global_tex_desc_set,
    std::vector<std::shared_ptr<renderer::DescriptorSet>>& desc_sets,
    int32_t node_idx,
    const uint32_t image_index,
    const glm::mat4& parent_matrix);

std::vector<renderer::TextureDescriptor> addGltfTextures(
    const std::shared_ptr<renderer::ObjectData>& gltf_object,
    const renderer::MaterialInfo& material,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& thin_film_lut_tex);

} // renderer
} // work