#pragma once
#include "engine/renderer/renderer.h"

namespace engine {
namespace scene_rendering {

class VolumeCloud {
    renderer::BufferInfo vertex_buffer_;
    renderer::BufferInfo index_buffer_;

    std::shared_ptr<renderer::DescriptorSetLayout> cloud_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet> cloud_tex_desc_set_;
    std::shared_ptr<renderer::PipelineLayout> cloud_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> cloud_pipeline_;
    std::shared_ptr<renderer::DescriptorSetLayout> draw_volume_moist_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet> draw_volume_moist_desc_set_[2];
    std::shared_ptr<renderer::PipelineLayout> draw_volume_moist_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> draw_volume_moist_pipeline_;

public:
    VolumeCloud(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
        const std::shared_ptr<renderer::DescriptorSetLayout>& ibl_desc_set_layout,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::ImageView>& src_depth,
        const std::vector<std::shared_ptr<renderer::ImageView>>& temp_moisture_texes,
        const std::shared_ptr<renderer::ImageView>& cloud_lighting_tex,
        const glm::uvec2& display_size);

    void recreate(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::ImageView>& src_depth,
        const std::vector<std::shared_ptr<renderer::ImageView>>& temp_moisture_texes,
        const std::shared_ptr<renderer::ImageView>& cloud_lighting_tex,
        const glm::uvec2& display_size);

    void draw(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::DescriptorSet>& frame_desc_set);

    void drawVolumeMoisture(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::DescriptorSet>& frame_desc_set,
        const glm::uvec2& display_size,
        int dbuf_idx,
        float current_time);

    void update();

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

}// namespace scene_rendering
}// namespace engine
