#pragma once
#include "engine/renderer/renderer.h"
#include "engine/scene_rendering/skydome.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace scene_rendering {

class Menu {
    std::vector<std::string> gltf_file_names_;
    std::vector<std::string> to_load_gltf_names_;
    bool turn_off_water_pass_ = false;
    bool turn_off_volume_moist_ = false;
    bool turn_on_airflow_ = false;
    uint32_t debug_draw_type_ = 0;
    float air_flow_strength_ = 50.0f;
    float water_flow_strength_ = 1.0f;
    float light_ext_factor_ = 0.975f;
    float view_ext_factor_ = 0.405f;
    float view_ext_exponent_ = 0.923f;
    float cloud_ambient_intensity_ = 1.0f;
    float cloud_phase_intensity_ = 0.5f;
    float global_flow_dir_ = 85.0f;
    float global_flow_speed_ = 0.05f;

    glsl::WeatherControl weather_controls_;

public:
    Menu(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::Instance>& instance,
        GLFWwindow* window,
        const renderer::QueueFamilyIndices& queue_family_indices,
        const renderer::SwapChainInfo& swap_chain_info,
        std::shared_ptr<renderer::Queue> graphics_queue,
        std::shared_ptr<renderer::DescriptorPool> descriptor_pool,
        std::shared_ptr<renderer::RenderPass> render_pass,
        std::shared_ptr<renderer::CommandBuffer> command_buffer);

    std::vector<std::string> getToLoadGltfNamesAndClear() {
        auto result = to_load_gltf_names_;
        to_load_gltf_names_.clear();
        return result;
    }

    inline bool isWaterPassTurnOff() {
        return turn_off_water_pass_; 
    }

    inline bool isVolumeMoistTurnOff() {
        return turn_off_volume_moist_;
    }

    inline bool isAirfowOn() {
        return turn_on_airflow_;
    }

    inline uint32_t getDebugDrawType() {
        return debug_draw_type_;
    }

    inline const glsl::WeatherControl& getWeatherControls() {
        return weather_controls_;
    }

    inline const float getLightExtFactor() const {
        return light_ext_factor_ / 1000.0f;
    }

    inline const float getViewExtFactor() const {
        return view_ext_factor_ / 1000.0f;
    }

    inline const float getViewExtExponent() const {
        return view_ext_exponent_;
    }

    inline const float getCloudAmbientIntensity() const {
        return cloud_ambient_intensity_;
    }

    inline const float getCloudPhaseIntensity() const {
        return cloud_phase_intensity_;
    }

    inline float getAirFlowStrength() {
        return air_flow_strength_;
    }

    inline float getWaterFlowStrength() {
        return water_flow_strength_;
    }

    inline float getGloalFlowDir() {
        return glm::radians(global_flow_dir_);
    }

    inline float getGlobalFlowSpeed() {
        return global_flow_speed_;
    }

    void init(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::Instance>& instance,
        GLFWwindow* window,
        const renderer::QueueFamilyIndices& queue_family_indices,
        const renderer::SwapChainInfo& swap_chain_info,
        std::shared_ptr<renderer::Queue> graphics_queue,
        std::shared_ptr<renderer::DescriptorPool> descriptor_pool,
        std::shared_ptr<renderer::RenderPass> render_pass,
        std::shared_ptr<renderer::CommandBuffer> command_buffer);

    bool draw(
        std::shared_ptr<renderer::CommandBuffer> command_buffer,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const renderer::SwapChainInfo& swap_chain_info,
        const glm::uvec2& screen_size,
        const std::shared_ptr<scene_rendering::Skydome>& skydome,
        uint32_t image_index);

    void destroy();
};

}// namespace scene_rendering
}// namespace engine
