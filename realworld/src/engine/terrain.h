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

public:
    TileMesh(
        const DeviceInfo& device_info,
        const std::shared_ptr<DescriptorPool> descriptor_pool,
        const std::shared_ptr<DescriptorSetLayout> desc_set_layout,
        const glm::uvec2& segment_count,
        const glm::vec2& min,
        const glm::vec2& max) :
        device_info_(device_info),
        segment_count_(segment_count),
        min_(min),
        max_(max) {
        createMeshBuffers();
        generateDescriptorSet(descriptor_pool, desc_set_layout);
    }

    void destory();

    void generateTileBuffers(
        const std::shared_ptr<CommandBuffer>& cmd_buf,
        const std::shared_ptr<Pipeline>& pipeline,
        const std::shared_ptr<PipelineLayout>& pipeline_layout);

    void createMeshBuffers();
        
    void generateDescriptorSet(
        const std::shared_ptr<DescriptorPool>& descriptor_pool,
        const std::shared_ptr<DescriptorSetLayout>& desc_set_layout);

    void draw(const std::shared_ptr<CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
        const renderer::DescriptorSetList& desc_set_list);
};

} // namespace renderer
} // namespace engine