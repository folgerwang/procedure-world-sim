#pragma once
#include "engine/renderer/renderer.h"

namespace engine {
namespace scene_rendering {

class Skydome {
    glm::vec3 sun_dir_ = glm::vec3(0);

    renderer::BufferInfo vertex_buffer_;
    renderer::BufferInfo index_buffer_;

    std::shared_ptr<renderer::DescriptorSetLayout> skybox_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet> skybox_tex_desc_set_;
    std::shared_ptr<renderer::PipelineLayout> skybox_pipeline_layout_;
    std::shared_ptr<renderer::PipelineLayout> cube_skybox_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> skybox_pipeline_;
    std::shared_ptr<renderer::Pipeline> cube_skybox_pipeline_;

public:
    Skydome(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
        const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
        const std::shared_ptr<renderer::DescriptorSetLayout>& ibl_desc_set_layout,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::GraphicPipelineInfo& cube_graphic_pipeline_info,
        const renderer::TextureInfo& rt_envmap_tex,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const glm::uvec2& display_size,
        const uint32_t& cube_size);

    void recreate(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::TextureInfo& rt_envmap_tex,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const glm::uvec2& display_size);

    void draw(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::DescriptorSet>& frame_desc_set);

    void drawCubeSkyBox(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const std::shared_ptr<renderer::DescriptorSet>& envmap_tex_desc_set,
        const renderer::TextureInfo& rt_envmap_tex,
        const std::vector<renderer::ClearValue>& clear_values,
        const uint32_t& cube_size);

    void update(float latitude, float longtitude, int d, int h, int m, int s);

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

}// namespace scene_rendering
}// namespace engine