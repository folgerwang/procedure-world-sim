#pragma once
#include "engine/renderer/renderer.h"
#include "shaders/global_definition.glsl.h"
#include "engine/gltf.h"
#include "engine/terrain.h"

struct GLFWwindow;
namespace work {
namespace app {

const int kMaxFramesInFlight = 2;
const int kCubemapSize = 1024;
const int kDifuseCubemapSize = 256;

class RealWorldApplication {
public:
    void run();
    void setFrameBufferResized(bool resized) { framebuffer_resized_ = resized; }

private:
    void initWindow();
    void initVulkan();
    void initImgui();
    void createImageViews();
    void createCubemapFramebuffers();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createComputePipeline();
    void createGltfPipelineLayout();
    void createTileMeshPipelineLayout();
    void createSkyboxPipelineLayout();
    void createCubemapPipelineLayout();
    void createCubeSkyboxPipelineLayout();
    void createCubemapComputePipelineLayout();
    void createRenderPass();
    void createCubemapRenderPass();
    void createFramebuffers();
    void createCommandPool();
    void createDepthResources();
    void createTextureImage(
        const std::string& file_name,
        engine::renderer::Format format,
        engine::renderer::TextureInfo& texture);
    void createTextureSampler();
    void createVertexBuffer();
    void createIndexBuffer();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void createCommandBuffers();
    void createSyncObjects();
    void updateViewConstBuffer(uint32_t current_image, float radius = 2.0f);
    // todo remove vk interface here.
    std::vector<engine::renderer::TextureDescriptor> addGlobalTextures(
        const std::shared_ptr<engine::renderer::DescriptorSet>& description_set);
    std::vector<engine::renderer::TextureDescriptor> addSkyboxTextures(
        const std::shared_ptr<engine::renderer::DescriptorSet>& description_set);
    std::vector<engine::renderer::TextureDescriptor> addPanoramaTextures(
        const std::shared_ptr<engine::renderer::DescriptorSet>& description_set);
    std::vector<engine::renderer::TextureDescriptor> addIblTextures(
        const std::shared_ptr<engine::renderer::DescriptorSet>& description_set);
    std::vector<engine::renderer::TextureDescriptor> addIblComputeTextures(
        const std::shared_ptr<engine::renderer::DescriptorSet>& description_set, 
        const engine::renderer::TextureInfo& src_tex,
        const engine::renderer::TextureInfo& dst_tex);
    void mainLoop();
    void drawScene(
        std::shared_ptr<engine::renderer::CommandBuffer> command_buffer,
        std::shared_ptr<engine::renderer::Framebuffer> frame_buffer,
        std::shared_ptr<engine::renderer::DescriptorSet> frame_desc_set,
        const glm::uvec2& screen_size);
    void drawFrame();
    void cleanup();
    void cleanupSwapChain();
    void recreateSwapChain();

    void loadMtx2Texture(
        const std::string& input_filename,
        engine::renderer::TextureInfo& texture);

private:
    GLFWwindow* window_ = nullptr;

    engine::renderer::DeviceInfo device_info_;
    engine::renderer::QueueFamilyIndices queue_indices_;

    std::shared_ptr<engine::renderer::Instance> instance_;
    engine::renderer::PhysicalDeviceList physical_devices_;
    std::shared_ptr<engine::renderer::PhysicalDevice> physical_device_;
    std::shared_ptr<engine::renderer::Device> device_;
    std::shared_ptr<engine::renderer::Queue> graphics_queue_;
    std::shared_ptr<engine::renderer::Queue> present_queue_;
    std::shared_ptr<engine::renderer::Surface> surface_;
    engine::renderer::SwapChainInfo swap_chain_info_;
    std::shared_ptr<engine::renderer::DescriptorPool> descriptor_pool_;
    std::shared_ptr<engine::renderer::DescriptorSetLayout> desc_set_layout_;
    std::vector<std::shared_ptr<engine::renderer::DescriptorSet>> desc_sets_;
    std::shared_ptr<engine::renderer::DescriptorSetLayout> global_tex_desc_set_layout_;
    std::shared_ptr<engine::renderer::DescriptorSetLayout> material_tex_desc_set_layout_;
    std::shared_ptr<engine::renderer::DescriptorSetLayout> skybox_desc_set_layout_;
    std::shared_ptr<engine::renderer::DescriptorSetLayout> ibl_desc_set_layout_;
    std::shared_ptr<engine::renderer::DescriptorSetLayout> ibl_comp_desc_set_layout_;
    std::shared_ptr<engine::renderer::DescriptorSet> global_tex_desc_set_;
    std::shared_ptr<engine::renderer::DescriptorSet> skybox_tex_desc_set_;
    std::shared_ptr<engine::renderer::DescriptorSet> envmap_tex_desc_set_;
    std::shared_ptr<engine::renderer::DescriptorSet> ibl_tex_desc_set_;
    std::shared_ptr<engine::renderer::DescriptorSet> ibl_diffuse_tex_desc_set_;
    std::shared_ptr<engine::renderer::DescriptorSet> ibl_specular_tex_desc_set_;
    std::shared_ptr<engine::renderer::DescriptorSet> ibl_sheen_tex_desc_set_;
    std::shared_ptr<engine::renderer::RenderPass> render_pass_;
    std::shared_ptr<engine::renderer::RenderPass> cubemap_render_pass_;
    std::shared_ptr<engine::renderer::PipelineLayout> gltf_pipeline_layout_;
    std::shared_ptr<engine::renderer::PipelineLayout> tile_pipeline_layout_;
    std::shared_ptr<engine::renderer::PipelineLayout> skybox_pipeline_layout_;
    std::shared_ptr<engine::renderer::PipelineLayout> ibl_pipeline_layout_;
    std::shared_ptr<engine::renderer::PipelineLayout> cube_skybox_pipeline_layout_;
    std::shared_ptr<engine::renderer::PipelineLayout> ibl_comp_pipeline_layout_;
    std::shared_ptr<engine::renderer::Pipeline> gltf_pipeline_;
    std::shared_ptr<engine::renderer::Pipeline> tile_pipeline_;
    std::shared_ptr<engine::renderer::Pipeline> skybox_pipeline_;
    std::shared_ptr<engine::renderer::Pipeline> envmap_pipeline_;
    std::shared_ptr<engine::renderer::Pipeline> cube_skybox_pipeline_;
    std::shared_ptr<engine::renderer::Pipeline> lambertian_pipeline_;
    std::shared_ptr<engine::renderer::Pipeline> ggx_pipeline_;
    std::shared_ptr<engine::renderer::Pipeline> charlie_pipeline_;
    std::shared_ptr<engine::renderer::Pipeline> blur_comp_pipeline_;
    std::shared_ptr<engine::renderer::CommandPool> command_pool_;
    engine::renderer::BufferInfo vertex_buffer_;
    engine::renderer::BufferInfo index_buffer_;
    engine::renderer::TextureInfo sample_tex_;
    engine::renderer::TextureInfo depth_buffer_;
    engine::renderer::TextureInfo ggx_lut_tex_;
    engine::renderer::TextureInfo brdf_lut_tex_;
    engine::renderer::TextureInfo charlie_lut_tex_;
    engine::renderer::TextureInfo thin_film_lut_tex_;
    engine::renderer::TextureInfo panorama_tex_;
    engine::renderer::TextureInfo ibl_diffuse_tex_;
    engine::renderer::TextureInfo ibl_specular_tex_;
    engine::renderer::TextureInfo ibl_sheen_tex_;
    engine::renderer::TextureInfo rt_envmap_tex_;
    engine::renderer::TextureInfo tmp_ibl_diffuse_tex_;
    engine::renderer::TextureInfo tmp_ibl_specular_tex_;
    engine::renderer::TextureInfo tmp_ibl_sheen_tex_;
    engine::renderer::TextureInfo rt_ibl_diffuse_tex_;
    engine::renderer::TextureInfo rt_ibl_specular_tex_;
    engine::renderer::TextureInfo rt_ibl_sheen_tex_;
    std::vector<uint32_t> binding_list_;
    std::shared_ptr<engine::renderer::Sampler> texture_sampler_;
    std::vector<engine::renderer::BufferInfo> view_const_buffers_;
    std::vector<std::shared_ptr<engine::renderer::CommandBuffer>> command_buffers_;
    std::vector<std::shared_ptr<engine::renderer::Semaphore>> image_available_semaphores_;
    std::vector<std::shared_ptr<engine::renderer::Semaphore>> render_finished_semaphores_;
    std::vector<std::shared_ptr<engine::renderer::Fence>> in_flight_fences_;
    std::vector<std::shared_ptr<engine::renderer::Fence>> images_in_flight_;

    std::shared_ptr<engine::renderer::ObjectData> gltf_object_;
    std::shared_ptr<engine::renderer::TileMesh> tile_mesh_;

    uint64_t current_frame_ = 0;
    bool framebuffer_resized_ = false;
};

}// namespace app
}// namespace work

extern std::vector<glm::vec2> generateTileMeshVertex(const glm::vec3 corners[4], const glm::uvec2& segment_count);
extern std::vector<uint16_t> generateTileMeshIndex(const glm::uvec2& segment_count);
