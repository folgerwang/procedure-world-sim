#pragma once
#include "engine/renderer/renderer.h"

namespace engine {
namespace scene_rendering {

class VolumeNoise {
#if 0
    renderer::BufferInfo vertex_buffer_;
    renderer::BufferInfo index_buffer_;
#endif

    renderer::TextureInfo perlin_noise_tex_;
    renderer::TextureInfo worley_noise_tex_;

#if 0
    std::shared_ptr<renderer::DescriptorSetLayout> cloud_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet> cloud_tex_desc_set_;
    std::shared_ptr<renderer::PipelineLayout> cloud_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> cloud_pipeline_;
#endif

    std::shared_ptr<renderer::DescriptorSetLayout> perlin_noise_init_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet> perlin_noise_init_tex_desc_set_;
    std::shared_ptr<renderer::PipelineLayout> perlin_noise_init_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> perlin_noise_init_pipeline_;

public:
    VolumeNoise(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
        const std::shared_ptr<renderer::DescriptorSetLayout>& ibl_desc_set_layout,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const glm::uvec2& display_size);

    void recreate(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const glm::uvec2& display_size);

    void initPerlinNoiseTexture(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    void draw(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::DescriptorSet>& frame_desc_set);

    void update();

    void destroy(const std::shared_ptr<renderer::Device>& device);

    const renderer::TextureInfo& getPerlinNoiseTexture() {
        return perlin_noise_tex_;
    }
};

}// namespace scene_rendering
}// namespace engine
