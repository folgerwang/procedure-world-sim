#pragma once
#include "engine/renderer/renderer.h"
#include "engine/scene_rendering/skydome.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace scene_rendering {

class WeatherSystem {
    renderer::TextureInfo temp_moisture_volume_[2];
    renderer::TextureInfo airflow_volume_;
    renderer::TextureInfo cloud_lighting_volume_;
    renderer::TextureInfo cloud_shadow_volume_;

    std::shared_ptr<renderer::DescriptorSet> temperature_init_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSetLayout> temperature_init_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> temperature_init_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> temperature_init_pipeline_;
    std::shared_ptr<renderer::DescriptorSet> airflow_tex_desc_set_[2];
    std::shared_ptr<renderer::DescriptorSetLayout> airflow_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> airflow_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> airflow_pipeline_;
    std::shared_ptr<renderer::DescriptorSet> cloud_lighting_tex_desc_set_[2];
    std::shared_ptr<renderer::DescriptorSetLayout> cloud_lighting_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> cloud_lighting_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> cloud_lighting_pipeline_;
    std::shared_ptr<renderer::DescriptorSet> cloud_shadow_tex_desc_set_[2];
    std::shared_ptr<renderer::DescriptorSetLayout> cloud_shadow_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> cloud_shadow_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> cloud_shadow_pipeline_;
    std::shared_ptr<renderer::Pipeline> cloud_shadow_init_pipeline_;
    std::shared_ptr<renderer::DescriptorSet> cloud_shadow_merge_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSetLayout> cloud_shadow_merge_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> cloud_shadow_merge_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> cloud_shadow_merge_pipeline_;

public:
    WeatherSystem(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
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

    inline std::shared_ptr<renderer::ImageView> getCloudLightingTex() {
        return cloud_lighting_volume_.view;
    }

    void recreate(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::ImageView>& rock_layer_tex,
        const std::vector<std::shared_ptr<renderer::ImageView>>& soil_water_layer_tex);

    void initTemperatureBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    void updateAirflowBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const glsl::WeatherControl& weather_controls,
        int dbuf_idx,
        float current_time);

    void updateCloudLighting(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        const std::shared_ptr<scene_rendering::Skydome>& skydome,
        int dbuf_idx,
        float current_time);

    void updateCloudShadow(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const glm::vec3& sun_dir,
        const float& light_ext_factor,
        const int& dbuf_idx,
        const float& current_time);

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

}// namespace scene_rendering
}// namespace engine
