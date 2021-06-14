#pragma once
#include "engine/renderer/renderer.h"

namespace engine {
namespace game_object {

class TileObject {
    const renderer::DeviceInfo& device_info_;

    glm::uvec2 segment_count_;
    glm::vec2 min_;
    glm::vec2 max_;

    renderer::BufferInfo vertex_buffer_;
    renderer::BufferInfo index_buffer_;

    std::shared_ptr<renderer::DescriptorSet> buffer_desc_set_;

    static std::shared_ptr<renderer::DescriptorSetLayout> tile_creator_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout> tile_creator_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline> tile_creator_pipeline_;
    static std::shared_ptr<renderer::PipelineLayout> tile_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline> tile_pipeline_;
    static std::shared_ptr<renderer::Pipeline> tile_water_pipeline_;

public:
    TileObject() = delete;
    TileObject(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool> descriptor_pool,
        const glm::uvec2& segment_count,
        const glm::vec2& min,
        const glm::vec2& max);

    ~TileObject() {
        destory();
    }

    void destory();

    static void initStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const glm::uvec2& display_size);

    static void createStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const glm::uvec2& display_size);

    static void recreateStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const glm::uvec2& display_size);

    static void destoryStaticMembers(
        const std::shared_ptr<renderer::Device>& device);

    void generateTileBuffers(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    void createMeshBuffers();
        
    void generateDescriptorSet(
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool);

    void draw(const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        bool is_base_pass);
};

} // namespace game_object
} // namespace engine