#pragma once
#include "engine/renderer/renderer.h"
#include "shaders/global_definition.glsl.h"
#include "engine/gltf.h"
#include "engine/terrain.h"

namespace er = engine::renderer;

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
    void createImageViews();
    void createCubemapFramebuffers();
    void createDescriptorSetLayout();
    void createGraphicsPipeline(const glm::uvec2& display_size);
    void createCubeGraphicsPipeline();
    void createComputePipeline();
    void createGraphicPipelineLayout();
    void createCubemapPipelineLayout();
    void createCubeSkyboxPipelineLayout();
    void createCubemapComputePipelineLayout();
    void createRenderPass();
    void createCubemapRenderPass();
    void createFramebuffers(const glm::uvec2& display_size);
    void createCommandPool();
    void createDepthResources(const glm::uvec2& display_size);
    void createTextureImage(
        const std::string& file_name,
        er::Format format,
        er::TextureInfo& texture);
    void createTextureSampler();
    void createVertexBuffer();
    void createIndexBuffer();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void createCommandBuffers();
    void createSyncObjects();
    void destroyGraphicsPipeline();
    void updateViewConstBuffer(uint32_t current_image, float radius = 2.0f);
    // todo remove vk interface here.
    std::vector<er::TextureDescriptor> addGlobalTextures(
        const std::shared_ptr<er::DescriptorSet>& description_set);
    std::vector<er::TextureDescriptor> addSkyboxTextures(
        const std::shared_ptr<er::DescriptorSet>& description_set);
    std::vector<er::TextureDescriptor> addPanoramaTextures(
        const std::shared_ptr<er::DescriptorSet>& description_set);
    std::vector<er::TextureDescriptor> addIblTextures(
        const std::shared_ptr<er::DescriptorSet>& description_set);
    std::vector<er::TextureDescriptor> addIblComputeTextures(
        const std::shared_ptr<er::DescriptorSet>& description_set, 
        const er::TextureInfo& src_tex,
        const er::TextureInfo& dst_tex);
    std::vector<er::BufferDescriptor> addTileCreatorBuffers(
        const std::shared_ptr<er::DescriptorSet>& description_set,
        const er::BufferInfo& src_buffer,
        const er::BufferInfo& dst_buffer);
    void mainLoop();
    void drawScene(
        std::shared_ptr<er::CommandBuffer> command_buffer,
        std::shared_ptr<er::Framebuffer> frame_buffer,
        std::shared_ptr<er::DescriptorSet> frame_desc_set,
        const glm::uvec2& screen_size);
    void drawFrame();
    void cleanup();
    void cleanupSwapChain();
    void recreateSwapChain();

    void loadMtx2Texture(
        const std::string& input_filename,
        er::TextureInfo& texture);

private:
    GLFWwindow* window_ = nullptr;

    er::DeviceInfo device_info_;
    er::QueueFamilyIndices queue_indices_;
    er::GraphicPipelineInfo graphic_pipeline_info_;
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
    std::shared_ptr<er::DescriptorSetLayout> material_tex_desc_set_layout_;
    std::shared_ptr<er::DescriptorSetLayout> skybox_desc_set_layout_;
    std::shared_ptr<er::DescriptorSetLayout> ibl_desc_set_layout_;
    std::shared_ptr<er::DescriptorSetLayout> ibl_comp_desc_set_layout_;
    std::shared_ptr<er::DescriptorSet> global_tex_desc_set_;
    std::shared_ptr<er::DescriptorSet> skybox_tex_desc_set_;
    std::shared_ptr<er::DescriptorSet> envmap_tex_desc_set_;
    std::shared_ptr<er::DescriptorSet> ibl_tex_desc_set_;
    std::shared_ptr<er::DescriptorSet> ibl_diffuse_tex_desc_set_;
    std::shared_ptr<er::DescriptorSet> ibl_specular_tex_desc_set_;
    std::shared_ptr<er::DescriptorSet> ibl_sheen_tex_desc_set_;
    std::shared_ptr<er::RenderPass> render_pass_;
    std::shared_ptr<er::RenderPass> cubemap_render_pass_;
    std::shared_ptr<er::PipelineLayout> gltf_pipeline_layout_;
    std::shared_ptr<er::PipelineLayout> skybox_pipeline_layout_;
    std::shared_ptr<er::PipelineLayout> ibl_pipeline_layout_;
    std::shared_ptr<er::PipelineLayout> cube_skybox_pipeline_layout_;
    std::shared_ptr<er::PipelineLayout> ibl_comp_pipeline_layout_;
    std::shared_ptr<er::Pipeline> gltf_pipeline_;
    std::shared_ptr<er::Pipeline> skybox_pipeline_;
    std::shared_ptr<er::Pipeline> envmap_pipeline_;
    std::shared_ptr<er::Pipeline> cube_skybox_pipeline_;
    std::shared_ptr<er::Pipeline> lambertian_pipeline_;
    std::shared_ptr<er::Pipeline> ggx_pipeline_;
    std::shared_ptr<er::Pipeline> charlie_pipeline_;
    std::shared_ptr<er::Pipeline> blur_comp_pipeline_;
    std::shared_ptr<er::CommandPool> command_pool_;
    er::BufferInfo vertex_buffer_;
    er::BufferInfo index_buffer_;
    er::TextureInfo sample_tex_;
    er::TextureInfo depth_buffer_;
    er::TextureInfo ggx_lut_tex_;
    er::TextureInfo brdf_lut_tex_;
    er::TextureInfo charlie_lut_tex_;
    er::TextureInfo thin_film_lut_tex_;
    er::TextureInfo panorama_tex_;
    er::TextureInfo ibl_diffuse_tex_;
    er::TextureInfo ibl_specular_tex_;
    er::TextureInfo ibl_sheen_tex_;
    er::TextureInfo rt_envmap_tex_;
    er::TextureInfo tmp_ibl_diffuse_tex_;
    er::TextureInfo tmp_ibl_specular_tex_;
    er::TextureInfo tmp_ibl_sheen_tex_;
    er::TextureInfo rt_ibl_diffuse_tex_;
    er::TextureInfo rt_ibl_specular_tex_;
    er::TextureInfo rt_ibl_sheen_tex_;
    std::vector<uint32_t> binding_list_;
    std::shared_ptr<er::Sampler> texture_sampler_;
    std::vector<er::BufferInfo> view_const_buffers_;
    std::vector<std::shared_ptr<er::CommandBuffer>> command_buffers_;
    std::vector<std::shared_ptr<er::Semaphore>> image_available_semaphores_;
    std::vector<std::shared_ptr<er::Semaphore>> render_finished_semaphores_;
    std::vector<std::shared_ptr<er::Fence>> in_flight_fences_;
    std::vector<std::shared_ptr<er::Fence>> images_in_flight_;

    std::shared_ptr<er::ObjectData> gltf_object_;
    std::shared_ptr<er::TileMesh> tile_mesh_;

    uint64_t current_frame_ = 0;
    bool framebuffer_resized_ = false;
};

}// namespace app
}// namespace work

extern std::vector<glm::vec2> generateTileMeshVertex(const glm::vec3 corners[4], const glm::uvec2& segment_count);
extern std::vector<uint16_t> generateTileMeshIndex(const glm::uvec2& segment_count);