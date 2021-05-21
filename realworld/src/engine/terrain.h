#pragma once
#include "renderer/renderer.h"

namespace engine {
namespace renderer {

class TileMesh {
    const DeviceInfo& device_info_;

    glm::uvec2 segment_count_;
    glm::vec2 min_;
    glm::vec2 max_;

    uint32_t vertex_buffer_size_ = 0;
    uint32_t index_buffer_size_ = 0;

    BufferInfo vertex_buffer_;
    BufferInfo index_buffer_;

    std::shared_ptr<DescriptorSet> buffer_desc_set_;

    static std::shared_ptr<DescriptorSetLayout> tile_creator_desc_set_layout_;
    static std::shared_ptr<PipelineLayout> tile_creator_pipeline_layout_;
    static std::shared_ptr<Pipeline> tile_creator_pipeline_;
    static std::shared_ptr<PipelineLayout> tile_pipeline_layout_;
    static std::shared_ptr<Pipeline> tile_pipeline_;

public:
    TileMesh(
        const DeviceInfo& device_info,
        const std::shared_ptr<RenderPass>& render_pass,
        const GraphicPipelineInfo& graphic_pipeline_info,
        const std::shared_ptr<DescriptorPool> descriptor_pool,
        const DescriptorSetLayoutList& global_desc_set_layouts,
        const glm::uvec2& segment_count,
        const glm::vec2& min,
        const glm::vec2& max,
        const glm::uvec2& display_size);

    void destory();

    static void recreateStaticMembers(
        const std::shared_ptr<Device>& device,
        const std::shared_ptr<RenderPass>& render_pass,
        const GraphicPipelineInfo& graphic_pipeline_info,
        const DescriptorSetLayoutList& global_desc_set_layouts,
        const glm::uvec2& display_size);

    static void destoryStaticMembers(const std::shared_ptr<Device>& device);

    void generateTileBuffers(const std::shared_ptr<CommandBuffer>& cmd_buf);

    void createMeshBuffers();
        
    void generateDescriptorSet(
        const std::shared_ptr<DescriptorPool>& descriptor_pool);

    void draw(const std::shared_ptr<CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_set_list);
};

} // namespace renderer
} // namespace engine