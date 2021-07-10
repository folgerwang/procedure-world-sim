#pragma once
#include "engine/renderer/renderer.h"

namespace engine {
namespace scene_rendering {

class WeatherSystem {
    enum class WeatherSystemConst {
        kAirflowBufferWidth = 256,
        kAirflowBufferHeight = 128,
    };

    renderer::TextureInfo temp_moisture_volume_[2];
    renderer::TextureInfo airflow_volume_;

    std::shared_ptr<renderer::DescriptorSet> temperature_init_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSetLayout> temperature_init_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> temperature_init_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> temperature_init_pipeline_;
    std::shared_ptr<renderer::DescriptorSet> airflow_tex_desc_set_[2];
    std::shared_ptr<renderer::DescriptorSetLayout> airflow_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> airflow_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> airflow_pipeline_;

public:
    WeatherSystem(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::ImageView>& rock_layer_tex,
        const std::vector<std::shared_ptr<renderer::ImageView>>& soil_water_layer_tex);

    inline std::shared_ptr<renderer::ImageView> getTempMoistureTex(int idx) {
        return temp_moisture_volume_[idx].view;
    }

    inline std::vector<std::shared_ptr<renderer::ImageView>> getTempMoistureTexes() {
        return { temp_moisture_volume_[0].view, temp_moisture_volume_[1].view };
    }

    inline std::shared_ptr<renderer::ImageView> getAirflowTex() {
        return airflow_volume_.view;
    }

    void recreate(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::ImageView>& rock_layer_tex,
        const std::vector<std::shared_ptr<renderer::ImageView>>& soil_water_layer_tex);

    void initTemperatureBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    void updateAirflowBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        int dbuf_idx);

    void update();

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

}// namespace scene_rendering
}// namespace engine
