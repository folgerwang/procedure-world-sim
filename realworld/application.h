#pragma once
#include "renderer.h"
#include "src/shaders/global_definition.glsl.h"

struct GLFWwindow;
namespace work {
namespace app {

const int kMaxFramesInFlight = 2;
const int kCubemapSize = 1024;

class RealWorldApplication {
public:
    void run();
    void setFrameBufferResized(bool resized) { framebuffer_resized_ = resized; }

private:
    void initWindow();
    void initVulkan();
    void createInstance();
    void setupDebugMessenger();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSurface();
    void createSwapChain();
    void createImageViews();
    void createCubemapFramebuffers();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createGltfPipelineLayout();
    void createSkyboxPipelineLayout();
    void createCubemapPipelineLayout();
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
    std::vector<VkWriteDescriptorSet> addGlobalTextures(const std::shared_ptr<renderer::DescriptorSet>& description_set);
    std::vector<VkWriteDescriptorSet> addGltfTextures(const std::shared_ptr<renderer::DescriptorSet>& description_set, const renderer::MaterialInfo& material);
    std::vector<VkWriteDescriptorSet> addSkyboxTextures(const std::shared_ptr<renderer::DescriptorSet>& description_set);
    std::vector<VkWriteDescriptorSet> addPanoramaTextures(const std::shared_ptr<renderer::DescriptorSet>& description_set);
    std::vector<VkWriteDescriptorSet> addIblTextures(const std::shared_ptr<renderer::DescriptorSet>& description_set);
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
    void loadGltfModel(const std::string& input_filename);
    void createCubemapTexture(
        uint32_t width, 
        uint32_t height, 
        uint32_t mip_count, 
        renderer::Format format, 
        const std::vector<work::renderer::BufferImageCopyInfo>& copy_regions, 
        renderer::TextureInfo& texture,
        uint64_t buffer_size = 0,
        void* data = nullptr);
    void loadMtx2Texture(const std::string& input_filename, renderer::TextureInfo& texture);

private:
    GLFWwindow* window_;
    VkInstance vk_instance_;
    VkDebugUtilsMessengerEXT debug_messenger_;
    std::shared_ptr<VkPhysicalDevice> vk_physical_device_;
    std::shared_ptr<renderer::Device> device_;
    std::shared_ptr<renderer::Queue> graphics_queue_;
    std::shared_ptr<renderer::Queue> present_queue_;
    std::shared_ptr<renderer::Surface> surface_;
    std::shared_ptr<renderer::Swapchain> swap_chain_;
    std::vector<std::shared_ptr<renderer::Image>> swap_chain_images_;
    std::vector<std::shared_ptr<renderer::ImageView>> swap_chain_image_views_;
    std::vector<std::shared_ptr<renderer::Framebuffer>> swap_chain_framebuffers_;
    renderer::Format swap_chain_image_format_;
    glm::uvec2 swap_chain_extent_;
    std::shared_ptr<renderer::DescriptorPool> descriptor_pool_;
    std::shared_ptr<renderer::DescriptorSetLayout> desc_set_layout_;
    std::vector<std::shared_ptr<renderer::DescriptorSet>> desc_sets_;
    std::shared_ptr<renderer::DescriptorSetLayout> global_tex_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSetLayout> material_tex_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSetLayout> skybox_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSetLayout> ibl_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet> global_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> skybox_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> envmap_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> ibl_tex_desc_set_;
    std::shared_ptr<renderer::RenderPass> render_pass_;
    std::shared_ptr<renderer::RenderPass> cubemap_render_pass_;
    std::shared_ptr<renderer::PipelineLayout> gltf_pipeline_layout_;
    std::shared_ptr<renderer::PipelineLayout> skybox_pipeline_layout_;
    std::shared_ptr<renderer::PipelineLayout> ibl_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> gltf_pipeline_;
    std::shared_ptr<renderer::Pipeline> skybox_pipeline_;
    std::shared_ptr<renderer::Pipeline> envmap_pipeline_;
    std::shared_ptr<renderer::Pipeline> lambertian_pipeline_;
    std::shared_ptr<renderer::Pipeline> ggx_pipeline_;
    std::shared_ptr<renderer::Pipeline> charlie_pipeline_;
    std::shared_ptr<renderer::CommandPool> command_pool_;
    renderer::BufferInfo vertex_buffer_;
    renderer::BufferInfo index_buffer_;
    renderer::TextureInfo sample_tex_;
    renderer::TextureInfo depth_buffer_;
    renderer::TextureInfo black_tex_;
    renderer::TextureInfo white_tex_;
    renderer::TextureInfo ggx_lut_tex_;
    renderer::TextureInfo brdf_lut_tex_;
    renderer::TextureInfo charlie_lut_tex_;
    renderer::TextureInfo thin_film_lut_tex_;
    renderer::TextureInfo panorama_tex_;
    renderer::TextureInfo ibl_diffuse_tex_;
    renderer::TextureInfo ibl_specular_tex_;
    renderer::TextureInfo ibl_sheen_tex_;
    renderer::TextureInfo rt_envmap_tex_;
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

    uint64_t current_frame_ = 0;
    bool framebuffer_resized_ = false;
};

}//app
}//work
