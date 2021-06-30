#pragma once
#include "engine/renderer/renderer.h"

namespace engine {
namespace scene_rendering {

class NaturalSystem {
    enum class NaturalSystemConst {
        kAirflowWidth = 1024,
        kAirflowHeight = 32
    };

    renderer::TextureInfo airflow_volume_;

    std::shared_ptr<renderer::DescriptorSetLayout> airflow_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet> airflow_tex_desc_set_;
    std::shared_ptr<renderer::PipelineLayout> airflow_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> airflow_pipeline_;

public:
    NaturalSystem(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler);

    inline std::shared_ptr<renderer::ImageView> getAirflowTex() {
        return airflow_volume_.view;
    }

    void recreate(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler);

    void updateAirflowBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    void update();

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

}// namespace scene_rendering
}// namespace engine
