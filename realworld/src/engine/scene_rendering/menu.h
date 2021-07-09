#pragma once
#include "engine/renderer/renderer.h"

namespace engine {
namespace scene_rendering {

class Menu {
    std::vector<std::string> gltf_file_names_;
    std::vector<std::string> to_load_gltf_names_;
    bool turn_off_water_pass_ = false;
    bool turn_on_airflow_ = false;
    bool turn_on_debug_draw_volume_ = false;

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

    inline bool isAirfowOn() {
        return turn_on_airflow_;
    }

    inline bool isDebugDrawVolumeTurnOn(){
        return turn_on_debug_draw_volume_;
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
        uint32_t image_index);

    void destroy();
};

}// namespace scene_rendering
}// namespace engine
