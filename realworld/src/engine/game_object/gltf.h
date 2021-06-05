#pragma once
#include <unordered_map>
#include "engine/renderer/renderer.h"

namespace engine {
namespace game_object {

struct MaterialInfo {
    int32_t                base_color_idx_ = -1;
    int32_t                normal_idx_ = -1;
    int32_t                metallic_roughness_idx_ = -1;
    int32_t                emissive_idx_ = -1;
    int32_t                occlusion_idx_ = -1;

    renderer::BufferInfo   uniform_buffer_;
    std::shared_ptr<renderer::DescriptorSet>  desc_set_;
};

struct BufferView {
    uint32_t                buffer_idx;
    uint64_t                stride;
    uint64_t                offset;
    uint64_t                range;
};

union PrimitiveHashTag {
    uint32_t                data = 0;
    struct {
        uint32_t                has_normal : 1;
        uint32_t                has_tangent : 1;
        uint32_t                has_texcoord_0 : 1;
        uint32_t                has_skin_set_0 : 1;
        uint32_t                restart_enable : 1;
        uint32_t                topology : 16;
    };
};

struct PrimitiveInfo {
private:
    size_t hash_ = 0;
public:
    std::vector<uint32_t>   binding_list_;
    int32_t                 material_idx_;
    int32_t                 indirect_draw_cmd_ofs_;
    PrimitiveHashTag        tag_;
    glm::vec3               bbox_min_ = glm::vec3(std::numeric_limits<float>::max());
    glm::vec3               bbox_max_ = glm::vec3(std::numeric_limits<float>::min());
    renderer::IndexInputBindingDescription  index_desc_;
    std::vector<renderer::VertexInputBindingDescription> binding_descs_;
    std::vector<renderer::VertexInputAttributeDescription> attribute_descs_;

    void generateHash();
    size_t getHash() const { return hash_; }
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
    const std::shared_ptr<renderer::Device>& device_;
    int32_t                     default_scene_;
    std::vector<SceneInfo>      scenes_;
    std::vector<NodeInfo>       nodes_;
    std::vector<MeshInfo>       meshes_;
    std::vector<renderer::BufferInfo>     buffers_;
    std::vector<BufferView>     buffer_views_;

    std::vector<renderer::TextureInfo>    textures_;
    std::vector<MaterialInfo>   materials_;

    uint32_t                    num_prims_ = 0;
    renderer::BufferInfo        indirect_draw_cmd_;
    renderer::BufferInfo        instance_buffer_;

    std::shared_ptr<renderer::DescriptorSet> indirect_draw_cmd_buffer_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> update_instance_buffer_desc_set_;

public:
    ObjectData(const std::shared_ptr<renderer::Device>& device) : device_(device) {}
    ~ObjectData() { destroy(); }

    void generateSharedDescriptorSet(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::DescriptorSetLayout>& gltf_indirect_draw_desc_set_layout,
        const std::shared_ptr<renderer::DescriptorSetLayout>& update_instance_buffer_desc_set_layout,
        const std::shared_ptr<renderer::BufferInfo>& game_objects_buffer);

    void destroy();
};

class GltfObject {
    enum {
        kMaxNumObjects = 10240
    };
    const renderer::DeviceInfo& device_info_;
    std::shared_ptr<ObjectData> object_;
    glm::mat4                   location_;

    // static members.
    static uint32_t max_alloc_game_objects_in_buffer;

    static std::shared_ptr<renderer::DescriptorSetLayout> material_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout> gltf_pipeline_layout_;
    static std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>> gltf_pipeline_list_;
    static std::unordered_map<std::string, std::shared_ptr<ObjectData>> object_list_;
    static std::shared_ptr<renderer::DescriptorSetLayout> gltf_indirect_draw_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout> gltf_indirect_draw_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline> gltf_indirect_draw_pipeline_;
    static std::shared_ptr<renderer::DescriptorSet> update_game_objects_buffer_desc_set_;
    static std::shared_ptr<renderer::DescriptorSetLayout> update_game_objects_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout> update_game_objects_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline> update_game_objects_pipeline_;
    static std::shared_ptr<renderer::DescriptorSetLayout> update_instance_buffer_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout> update_instance_buffer_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline> update_instance_buffer_pipeline_;
    static std::shared_ptr<renderer::BufferInfo> game_objects_buffer_;


public:
    GltfObject() = delete;
    GltfObject(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const renderer::TextureInfo& thin_film_lut_tex,
        const std::string& file_name,
        const glm::uvec2& display_size,
        glm::mat4 location = glm::mat4(1.0f));

    void updateInstanceBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    void updateIndirectDrawBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    void draw(const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_set_list);

    static void initStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts);

    static void createStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts);

    static void recreateStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const glm::uvec2& display_size);

    static void generateDescriptorSet(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const renderer::TextureInfo& thin_film_lut_tex);

    static void destoryStaticMembers(
        const std::shared_ptr<renderer::Device>& device);

    static void updateGameObjectsBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        int update_frame_count);
};

} // namespace game_object
} // namespace engine