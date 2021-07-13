#pragma once
#include "engine/renderer/renderer.h"
#include "shaders/global_definition.glsl.h"
#include "engine/game_object/gltf.h"
#include "engine/game_object/terrain.h"
#include "engine/game_object/debug_draw.h"
#include "engine/scene_rendering/skydome.h"
#include "engine/scene_rendering/weather_system.h"
#include "engine/scene_rendering/ibl_creator.h"
#include "engine/scene_rendering/menu.h"
#include "engine/scene_rendering/volume_cloud.h"

namespace er = engine::renderer;
namespace ego = engine::game_object;
namespace es = engine::scene_rendering;

struct GLFWwindow;
namespace work {
namespace app {

const int kMaxFramesInFlight = 2;
const int kCubemapSize = 512;
const int kDifuseCubemapSize = 256;

class RealWorldApplication {
public:
    void run();
    void setFrameBufferResized(bool resized) { framebuffer_resized_ = resized; }

private:
    void initWindow();
    void initVulkan();
    void createImageViews();
    void createDescriptorSetLayout();
    void createRenderPasses();
    void createCubemapRenderPass();
    void createFramebuffers(const glm::uvec2& display_size);
    void createCommandPool();
    void createDepthResources(const glm::uvec2& display_size);
    void createHdrColorBuffer(const glm::uvec2& display_size);
    void createColorBufferCopy(const glm::uvec2& display_size);
    void recreateRenderBuffer(const glm::uvec2& display_size);
    void createTextureSampler();
    void createUniformBuffers();
    void createDescriptorSets();
    void createCommandBuffers();
    void createSyncObjects();
    void updateViewConstBuffer(uint32_t current_image, float near_z = 0.5f);
    std::vector<er::TextureDescriptor> addGlobalTextures(
        const std::shared_ptr<er::DescriptorSet>& description_set);
    void mainLoop();
    void drawScene(
        std::shared_ptr<er::CommandBuffer> command_buffer,
        const er::SwapChainInfo& swap_chain_info,
        const std::vector<std::shared_ptr<er::DescriptorSet>>& frame_desc_sets,
        const glm::uvec2& screen_size,
        uint32_t image_index,
        float delta_t,
        float current_time);
    void drawFrame();
    void cleanup();
    void cleanupSwapChain();
    void recreateSwapChain();

private:
    GLFWwindow* window_ = nullptr;

    er::DeviceInfo device_info_;
    er::QueueFamilyIndices queue_indices_;
    er::GraphicPipelineInfo graphic_pipeline_info_;
    er::GraphicPipelineInfo graphic_fs_pipeline_info_;
    er::GraphicPipelineInfo graphic_cubemap_pipeline_info_;

    std::shared_ptr<er::Instance> instance_;
    er::PhysicalDeviceList physical_devices_;
    std::shared_ptr<er::PhysicalDevice> physical_device_;
    std::shared_ptr<er::Device> device_;
    std::shared_ptr<er::Queue> graphics_queue_;
    std::shared_ptr<er::Queue> present_queue_;
    std::shared_ptr<er::Surface> surface_;
    er::SwapChainInfo swap_chain_info_;
    std::shared_ptr<er::DescriptorPool> descriptor_pool_;
    std::vector<std::shared_ptr<er::DescriptorSet>> desc_sets_;
    std::shared_ptr<er::DescriptorSetLayout> view_desc_set_layout_;
    std::shared_ptr<er::DescriptorSetLayout> global_tex_desc_set_layout_;
    std::shared_ptr<er::DescriptorSet> global_tex_desc_set_;
    er::TextureInfo ibl_diffuse_tex_;
    er::TextureInfo ibl_specular_tex_;
    er::TextureInfo ibl_sheen_tex_;
    std::shared_ptr<er::RenderPass> final_render_pass_;
    std::shared_ptr<er::RenderPass> cubemap_render_pass_;
    std::shared_ptr<er::CommandPool> command_pool_;

    er::Format hdr_format_ = er::Format::B10G11R11_UFLOAT_PACK32;// E5B9G9R9_UFLOAT_PACK32; this format not supported here.
    er::Format depth_format_ = er::Format::D24_UNORM_S8_UINT;
    er::TextureInfo hdr_color_buffer_;
    er::TextureInfo hdr_color_buffer_copy_;
    er::TextureInfo depth_buffer_;
    er::TextureInfo depth_buffer_copy_;
    std::shared_ptr<er::Framebuffer> hdr_frame_buffer_;
    std::shared_ptr<er::Framebuffer> hdr_water_frame_buffer_;
    std::shared_ptr<er::RenderPass> hdr_render_pass_;
    std::shared_ptr<er::RenderPass> hdr_water_render_pass_;

    er::TextureInfo sample_tex_;
    er::TextureInfo ggx_lut_tex_;
    er::TextureInfo brdf_lut_tex_;
    er::TextureInfo charlie_lut_tex_;
    er::TextureInfo thin_film_lut_tex_;
    std::vector<uint32_t> binding_list_;
    std::shared_ptr<er::Sampler> texture_sampler_;
    std::vector<er::BufferInfo> view_const_buffers_;
    std::vector<std::shared_ptr<er::CommandBuffer>> command_buffers_;
    std::vector<std::shared_ptr<er::Semaphore>> image_available_semaphores_;
    std::vector<std::shared_ptr<er::Semaphore>> render_finished_semaphores_;
    std::vector<std::shared_ptr<er::Fence>> in_flight_fences_;
    std::vector<std::shared_ptr<er::Fence>> images_in_flight_;

    std::vector<std::shared_ptr<ego::GltfObject>> gltf_objects_;
    std::shared_ptr<es::Skydome> skydome_;
    std::shared_ptr<es::WeatherSystem> weather_system_;
    std::shared_ptr<es::VolumeCloud> volume_cloud_;
    std::shared_ptr<es::IblCreator> ibl_creator_;
    std::shared_ptr<es::Menu> menu_;

    std::vector<er::ClearValue> clear_values_;

    glsl::ViewParams view_params_{};
    std::chrono::high_resolution_clock::time_point last_frame_time_point_;
    float current_time_ = 0;
    float delta_t_ = 0;

    // menu
    bool show_gltf_selection_ = false;

    uint64_t current_frame_ = 0;
    bool framebuffer_resized_ = false;
};

}// namespace app
}// namespace work
