#pragma once
#include "renderer.h"
#include "src/shaders/global_definition.glsl.h"
#include "gltf.h"
#include "terrain.h"

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
    void createTextureImage(const std::string& file_name, renderer::Format format, renderer::TextureInfo& texture);
    void createTextureSampler();
    void createVertexBuffer();
    void createIndexBuffer();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void createCommandBuffers();
    void createSyncObjects();
    void updateViewConstBuffer(uint32_t current_image, const glm::vec3& center, float radius);
    // todo remove vk interface here.
    std::vector<renderer::TextureDescriptor> addGlobalTextures(const std::shared_ptr<renderer::DescriptorSet>& description_set);
    std::vector<renderer::TextureDescriptor> addSkyboxTextures(const std::shared_ptr<renderer::DescriptorSet>& description_set);
    std::vector<renderer::TextureDescriptor> addPanoramaTextures(const std::shared_ptr<renderer::DescriptorSet>& description_set);
    std::vector<renderer::TextureDescriptor> addIblTextures(const std::shared_ptr<renderer::DescriptorSet>& description_set);
    std::vector<renderer::TextureDescriptor> addIblComputeTextures(
        const std::shared_ptr<renderer::DescriptorSet>& description_set, 
        const renderer::TextureInfo& src_tex,
        const renderer::TextureInfo& dst_tex);
    void mainLoop();
    void drawMesh(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const renderer::MeshInfo& mesh_info,
        const ModelParams& model_params,
        const uint32_t image_index);
    void drawNodes(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        int32_t node_idx,
        const uint32_t image_index,
        const glm::mat4& parent_matrix);
    void drawFrame();
    void cleanup();
    void cleanupSwapChain();
    void recreateSwapChain();

    void loadMtx2Texture(const std::string& input_filename, renderer::TextureInfo& texture);

private:
    GLFWwindow* window_ = nullptr;

    renderer::DeviceInfo device_info_;
    renderer::QueueFamilyIndices queue_indices_;

    std::shared_ptr<renderer::Instance> instance_;
    renderer::PhysicalDeviceList physical_devices_;
    std::shared_ptr<renderer::PhysicalDevice> physical_device_;
    std::shared_ptr<renderer::Device> device_;
    std::shared_ptr<renderer::Queue> graphics_queue_;
    std::shared_ptr<renderer::Queue> present_queue_;
    std::shared_ptr<renderer::Surface> surface_;
    renderer::SwapChainInfo swap_chain_info_;
    std::shared_ptr<renderer::DescriptorPool> descriptor_pool_;
    std::shared_ptr<renderer::DescriptorSetLayout> desc_set_layout_;
    std::vector<std::shared_ptr<renderer::DescriptorSet>> desc_sets_;
    std::shared_ptr<renderer::DescriptorSetLayout> global_tex_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSetLayout> material_tex_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSetLayout> skybox_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSetLayout> ibl_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSetLayout> ibl_comp_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet> global_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> skybox_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> envmap_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> ibl_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> ibl_diffuse_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> ibl_specular_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> ibl_sheen_tex_desc_set_;
    std::shared_ptr<renderer::RenderPass> render_pass_;
    std::shared_ptr<renderer::RenderPass> cubemap_render_pass_;
    std::shared_ptr<renderer::PipelineLayout> gltf_pipeline_layout_;
    std::shared_ptr<renderer::PipelineLayout> tile_pipeline_layout_;
    std::shared_ptr<renderer::PipelineLayout> skybox_pipeline_layout_;
    std::shared_ptr<renderer::PipelineLayout> ibl_pipeline_layout_;
    std::shared_ptr<renderer::PipelineLayout> cube_skybox_pipeline_layout_;
    std::shared_ptr<renderer::PipelineLayout> ibl_comp_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> gltf_pipeline_;
    std::shared_ptr<renderer::Pipeline> tile_pipeline_;
    std::shared_ptr<renderer::Pipeline> skybox_pipeline_;
    std::shared_ptr<renderer::Pipeline> envmap_pipeline_;
    std::shared_ptr<renderer::Pipeline> cube_skybox_pipeline_;
    std::shared_ptr<renderer::Pipeline> lambertian_pipeline_;
    std::shared_ptr<renderer::Pipeline> ggx_pipeline_;
    std::shared_ptr<renderer::Pipeline> charlie_pipeline_;
    std::shared_ptr<renderer::Pipeline> blur_comp_pipeline_;
    std::shared_ptr<renderer::CommandPool> command_pool_;
    renderer::BufferInfo vertex_buffer_;
    renderer::BufferInfo index_buffer_;
    renderer::TextureInfo sample_tex_;
    renderer::TextureInfo depth_buffer_;
    renderer::TextureInfo ggx_lut_tex_;
    renderer::TextureInfo brdf_lut_tex_;
    renderer::TextureInfo charlie_lut_tex_;
    renderer::TextureInfo thin_film_lut_tex_;
    renderer::TextureInfo panorama_tex_;
    renderer::TextureInfo ibl_diffuse_tex_;
    renderer::TextureInfo ibl_specular_tex_;
    renderer::TextureInfo ibl_sheen_tex_;
    renderer::TextureInfo rt_envmap_tex_;
    renderer::TextureInfo tmp_ibl_diffuse_tex_;
    renderer::TextureInfo tmp_ibl_specular_tex_;
    renderer::TextureInfo tmp_ibl_sheen_tex_;
    renderer::TextureInfo rt_ibl_diffuse_tex_;
    renderer::TextureInfo rt_ibl_specular_tex_;
    renderer::TextureInfo rt_ibl_sheen_tex_;
    std::vector<uint32_t> binding_list_;
    std::shared_ptr<renderer::Sampler> texture_sampler_;
    std::vector<renderer::BufferInfo> view_const_buffers_;
    std::vector<std::shared_ptr<renderer::CommandBuffer>> command_buffers_;
    std::vector<std::shared_ptr<renderer::Semaphore>> image_available_semaphores_;
    std::vector<std::shared_ptr<renderer::Semaphore>> render_finished_semaphores_;
    std::vector<std::shared_ptr<renderer::Fence>> in_flight_fences_;
    std::vector<std::shared_ptr<renderer::Fence>> images_in_flight_;

    std::shared_ptr<renderer::ObjectData> gltf_object_;
    std::shared_ptr<renderer::TileMesh> tile_mesh_;

    uint64_t current_frame_ = 0;
    bool framebuffer_resized_ = false;
};

}//app
}//work

extern std::vector<glm::vec2> generateTileMeshVertex(const glm::vec3 corners[4], const glm::uvec2& segment_count);
extern std::vector<uint16_t> generateTileMeshIndex(const glm::uvec2& segment_count);
