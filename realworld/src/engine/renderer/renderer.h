#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include <optional>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "device.h"
#include "command_buffer.h"

namespace engine {
namespace renderer {

class Queue {
public:
    virtual void submit(const std::vector<std::shared_ptr<CommandBuffer>>& command_buffers) = 0;
    virtual void waitIdle() = 0;
};

class CommandPool {
};

class Instance {
public:
    virtual void destroy() = 0;
    virtual void destroySurface(const std::shared_ptr<Surface>& surface) = 0;
};

class PhysicalDevice {
};

class Surface {
};

class Swapchain {
};

class DescriptorPool {
};

class Pipeline {
};

class PipelineLayout {
};

class RenderPass {
};

class Framebuffer {
};

class ImageView {
};

class Sampler {
};

class Image {
};

class Buffer {
public:
    virtual uint32_t getSize() = 0;
};

class Semaphore {
};

class Fence {
};

class DeviceMemory {
};

class DescriptorSetLayout {
};

class DescriptorSet {
};

class ShaderModule {
};

namespace vk {
class VulkanInstance : public Instance {
    VkInstance    instance_;
    VkDebugUtilsMessengerEXT debug_messenger_;

public:
    VkInstance get() { return instance_; }
    VkDebugUtilsMessengerEXT getDebugMessenger() { return debug_messenger_; }
    void set(const VkInstance& instance) { instance_ = instance; }
    void setDebugMessenger(const VkDebugUtilsMessengerEXT debug_messenger) {
        debug_messenger_ = debug_messenger;
    }

    virtual void destroy() final;
    virtual void destroySurface(const std::shared_ptr<Surface>& surface) final;
};

class VulkanPhysicalDevice : public PhysicalDevice {
    VkPhysicalDevice physical_device_;

public:
    VkPhysicalDevice get() { return physical_device_; }
    void set(const VkPhysicalDevice& physical_device) { physical_device_ = physical_device; }
};

class VulkanSurface : public Surface {
    VkSurfaceKHR    surface_;
public:
    VkSurfaceKHR get() { return surface_; }
    void set(const VkSurfaceKHR& surface) { surface_ = surface; }
};

class VulkanQueue : public Queue {
    VkQueue         queue_;
public:
    VkQueue get() { return queue_; }
    void set(const VkQueue& queue) { queue_ = queue; }

    virtual void submit(const std::vector<std::shared_ptr<CommandBuffer>>& command_buffers) final;
    virtual void waitIdle() final;
};

class VulkanCommandPool : public CommandPool {
    VkCommandPool    cmd_pool_;
public:
    VkCommandPool get() { return cmd_pool_; }
    void set(const VkCommandPool& cmd_pool) { cmd_pool_ = cmd_pool; }
};

class VulkanSwapchain : public Swapchain {
    VkSwapchainKHR    swap_chain_;
public:
    VkSwapchainKHR get() { return swap_chain_; }
    void set(VkSwapchainKHR swap_chain) { swap_chain_ = swap_chain; }
};

class VulkanDescriptorPool : public DescriptorPool {
    VkDescriptorPool    descriptor_pool_;
public:
    VkDescriptorPool get() { return descriptor_pool_; }
    void set(const VkDescriptorPool& descriptor_pool) { descriptor_pool_ = descriptor_pool; }
};

class VulkanPipeline : public Pipeline {
    VkPipeline    pipeline_;
public:
    VkPipeline get() { return pipeline_; }
    void set(const VkPipeline& pipeline) { pipeline_ = pipeline; }
};

class VulkanPipelineLayout : public PipelineLayout {
    VkPipelineLayout    pipeline_layout_;
public:
    VkPipelineLayout get() { return pipeline_layout_; }
    void set(const VkPipelineLayout& layout) { pipeline_layout_ = layout; }
};

class VulkanRenderPass : public RenderPass {
    VkRenderPass    render_pass_;
public:
    VkRenderPass get() { return render_pass_; }
    void set(const VkRenderPass& render_pass) { render_pass_ = render_pass; }
};

class VulkanFramebuffer : public Framebuffer {
    VkFramebuffer    frame_buffer_;
public:
    VkFramebuffer get() { return frame_buffer_; }
    void set(const VkFramebuffer frame_buffer) { frame_buffer_ = frame_buffer; }
};

class VulkanImageView : public ImageView {
    VkImageView     image_view_;
public:
    VkImageView get() { return image_view_; }
    void set(const VkImageView& image_view) { image_view_ = image_view; }
};

class VulkanSampler : public Sampler {
    VkSampler       sampler_;
public:
    VkSampler get() { return sampler_; }
    const VkSampler* getPtr()const { return &sampler_; }
    void set(const VkSampler& sampler) { sampler_ = sampler; }
};

class VulkanImage : public Image {
    VkImage         image_;
    ImageLayout     layout_ = ImageLayout::UNDEFINED;
public:
    VkImage get() { return image_; }
    ImageLayout getImageLayout() { return layout_; }
    void set(VkImage image) { image_ = image; }
    void setImageLayout(ImageLayout layout) { layout_ = layout; }
};

class VulkanBuffer : public Buffer {
    uint32_t         size_;
    VkBuffer         buffer_;
public:
    VkBuffer get() { return buffer_; }
    void set(const VkBuffer& buffer, uint32_t size) { buffer_ = buffer; size_ = size; }
    virtual uint32_t getSize() final {
        return size_;
    }
};

class VulkanSemaphore : public Semaphore {
    VkSemaphore      semaphore_;
public:
    VkSemaphore get() { return semaphore_; }
    void set(const VkSemaphore& semaphore) { semaphore_ = semaphore; }
};

class VulkanFence : public Fence {
    VkFence          fence_;
public:
    VkFence get() { return fence_; }
    void set(const VkFence& fence) { fence_ = fence; }
};

class VulkanDeviceMemory : public DeviceMemory {
    VkDeviceMemory  memory_;
public:
    VkDeviceMemory get() { return memory_; }
    void set(const VkDeviceMemory& memory) { memory_ = memory; }
};

class VulkanDescriptorSetLayout : public DescriptorSetLayout {
    VkDescriptorSetLayout  layout_;
public:
    VkDescriptorSetLayout get() { return layout_; }
    void set(const VkDescriptorSetLayout layout) { layout_ = layout; }

    void destroy(const std::shared_ptr<Device>& device);
};

class VulkanDescriptorSet : public DescriptorSet {
    VkDescriptorSet  desc_set_;
public:
    VkDescriptorSet get() { return desc_set_; }
    void set(const VkDescriptorSet desc_set) { desc_set_ = desc_set; }

    void destroy(const std::shared_ptr<Device>& device);
};
    

class VulkanShaderModule : public ShaderModule {
    VkShaderModule shader_module_;
public:
    VkShaderModule get() { return shader_module_; }
    void set(const VkShaderModule& shader_module) { shader_module_ = shader_module; }
};
} // namespace vk

struct SwapChainInfo {
    renderer::Format format;
    glm::uvec2 extent;
    std::shared_ptr<renderer::Swapchain> swap_chain;
    std::vector<std::shared_ptr<renderer::Image>> images;
    std::vector<std::shared_ptr<renderer::ImageView>> image_views;
    std::vector<std::shared_ptr<renderer::Framebuffer>> framebuffers;
};

class Helper {
public:
    static void init(const DeviceInfo& device_info);

    static void destroy(const std::shared_ptr<Device>& device);

    static Format findDepthFormat(const std::shared_ptr<Device>& device);

    static ImageResourceInfo getImageAsSource() { return image_source_info_; }
    static ImageResourceInfo getImageAsColorAttachment() { return image_as_color_attachement_; }
    static ImageResourceInfo getImageAsStore() { return image_as_store_; }
    static ImageResourceInfo getImageAsShaderSampler() { return image_as_shader_sampler_; }

    static std::shared_ptr<Instance> createInstance();

    static std::shared_ptr<Surface> createSurface(
        const std::shared_ptr<Instance>& instance,
        GLFWwindow* window);

    static PhysicalDeviceList collectPhysicalDevices(
        const std::shared_ptr<Instance>& instance);

    static std::shared_ptr<PhysicalDevice> pickPhysicalDevice(
        const PhysicalDeviceList& physical_devices,
        const std::shared_ptr<Surface>& surface);

    static QueueFamilyIndices findQueueFamilies(
        const std::shared_ptr<PhysicalDevice>& physical_device,
        const std::shared_ptr<Surface>& surface);

    static std::shared_ptr<Device> createLogicalDevice(
        const std::shared_ptr<PhysicalDevice>& physical_device,
        const std::shared_ptr<Surface>& surface,
        const QueueFamilyIndices& indices);

    static void createSwapChain(
        GLFWwindow* window,
        const std::shared_ptr<Device>& device,
        const std::shared_ptr<Surface>& surface,
        const QueueFamilyIndices& indices,
        SwapChainInfo& swap_chain_info,
        const ImageUsageFlags& usage);

    static void addOneTexture(
        std::vector<TextureDescriptor>& descriptor_writes,
        uint32_t binding,
        const std::shared_ptr<Sampler>& sampler,
        const std::shared_ptr<ImageView>& texture,
        const std::shared_ptr<DescriptorSet>& desc_set,
        DescriptorType desc_type = DescriptorType::COMBINED_IMAGE_SAMPLER,
        ImageLayout image_layout = ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    static void addOneBuffer(
        std::vector<BufferDescriptor>& descriptor_writes,
        uint32_t binding,
        const std::shared_ptr<Buffer>& buffer,
        const std::shared_ptr<DescriptorSet>& desc_set,
        DescriptorType desc_type,
        uint32_t range,
        uint32_t offset = 0);

    static void generateMipmapLevels(
        const std::shared_ptr<CommandBuffer>& cmd_buf,
        const std::shared_ptr<Image>& image,
        uint32_t mip_count,
        uint32_t width,
        uint32_t height,
        const ImageLayout& cur_image_layout);

    static void create2DTextureImage(
        const DeviceInfo& device_info,
        Format format,
        int tex_width,
        int tex_height,
        int tex_channels,
        const void* pixels,
        std::shared_ptr<Image>& texture_image,
        std::shared_ptr<DeviceMemory>& texture_image_memory);

    static void create2DTextureImage(
        const DeviceInfo& device_info,
        Format depth_format,
        glm::uvec2 size,
        TextureInfo& depth_buffer,
        const renderer::ImageUsageFlags& usage,
        const renderer::ImageLayout& image_layout);

    static void createDepthResources(
        const DeviceInfo& device_info,
        Format format,
        glm::uvec2 size,
        TextureInfo& texture_2d);

    static void createCubemapTexture(
        const DeviceInfo& device_info,
        const std::shared_ptr<RenderPass>& render_pass,
        uint32_t width,
        uint32_t height,
        uint32_t mip_count,
        Format format,
        const std::vector<BufferImageCopyInfo>& copy_regions,
        TextureInfo& texture,
        uint64_t buffer_size = 0,
        void* data = nullptr);

    static void createBufferWithSrcData(
        const DeviceInfo& device_info,
        const BufferUsageFlags& usage,
        const uint64_t& buffer_size,
        const void* src_data,
        std::shared_ptr<Buffer>& buffer,
        std::shared_ptr<DeviceMemory>& buffer_memory);

    static void transitionImageLayout(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::Image>& image,
        const renderer::Format& format,
        const renderer::ImageLayout& old_layout,
        const renderer::ImageLayout& new_layout,
        uint32_t base_mip_idx = 0,
        uint32_t mip_count = 1,
        uint32_t base_layer = 0,
        uint32_t layer_count = 1);

    static void blitImage(
        const std::shared_ptr<CommandBuffer>& cmd_buf,
        const std::shared_ptr<Image>& src_image,
        const std::shared_ptr<Image>& dst_image,
        const ImageResourceInfo& src_old_info,
        const ImageResourceInfo& src_new_info,
        const ImageResourceInfo& dst_old_info,
        const ImageResourceInfo& dst_new_info,
        const glm::ivec3& buffer_size);

    static bool acquireNextImage(
        const std::shared_ptr<Device>& device,
        const std::shared_ptr<renderer::Swapchain>& swap_chain,
        const std::shared_ptr<engine::renderer::Semaphore>& semaphore,
        uint32_t& image_index);

    static void addImGuiToCommandBuffer(
        const std::shared_ptr<CommandBuffer>& cmd_buf);

    static void submitQueue(
        const std::shared_ptr<Queue>& present_queue,
        const std::shared_ptr<Fence>& in_flight_fence,
        const std::vector<std::shared_ptr<Semaphore>>& wait_semaphores,
        const std::vector<std::shared_ptr<CommandBuffer>>& command_buffers,
        const std::vector<std::shared_ptr<Semaphore>>& signal_semaphores);

    static bool presentQueue(
        const std::shared_ptr<Queue>& present_queue,
        const std::vector<std::shared_ptr<Swapchain>>& swap_chains,
        const std::vector<std::shared_ptr<Semaphore>>& wait_semaphores,
        const uint32_t& image_index,
        bool& frame_buffer_resized);

    static void initImgui(
        const DeviceInfo& device_info,
        const std::shared_ptr<Instance>& instance,
        GLFWwindow* window,
        const QueueFamilyIndices& queue_family_indices,
        const SwapChainInfo& swap_chain_info,
        std::shared_ptr<Queue> graphics_queue,
        std::shared_ptr<DescriptorPool> descriptor_pool,
        std::shared_ptr<RenderPass> render_pass,
        std::shared_ptr<CommandBuffer> command_buffer);

    static TextureInfo& getBlackTexture() { return black_tex_; }
    static TextureInfo& getWhiteTexture() { return white_tex_; }

public:
    static TextureInfo black_tex_;
    static TextureInfo white_tex_;
    static ImageResourceInfo image_source_info_;
    static ImageResourceInfo image_as_color_attachement_;
    static ImageResourceInfo image_as_store_;
    static ImageResourceInfo image_as_shader_sampler_;
};

} // namespace renderer
} // namespace engine