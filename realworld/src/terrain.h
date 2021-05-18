#pragma once
#include "renderer/renderer.h"

namespace work {
namespace renderer {

class TileMesh {
    const DeviceInfo& device_info_;

    glm::uvec2 segment_count_;
    glm::vec2 min_;
    glm::vec2 max_;

    BufferInfo vertex_buffer_;
    BufferInfo index_buffer_;

public:
    TileMesh(const DeviceInfo& device_info, const glm::uvec2& segment_count, const glm::vec2& min, const glm::vec2& max) :
        device_info_(device_info),
        segment_count_(segment_count),
        min_(min),
        max_(max) {
        generateMesh();
    }

    void destory();

    void generateMesh();
    void draw(const std::shared_ptr<CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
        const renderer::DescriptorSetList& desc_set_list);
};

} // renderer
} // work