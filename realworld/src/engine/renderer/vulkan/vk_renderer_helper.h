#pragma once
#include <vulkan/vulkan.h>
#include "..\renderer_structs.h"

struct GLFWwindow;

namespace engine {
namespace renderer {
namespace vk {
namespace helper {

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities_;
    std::vector<VkSurfaceFormatKHR> formats_;
    std::vector<VkPresentModeKHR> present_modes_;
};

bool hasEnabledValidationLayers();

// convert to vulkan data type. 
VkFormat toVkFormat(renderer::Format format);
VkBufferUsageFlags toVkBufferUsageFlags(renderer::BufferUsageFlags flags);
VkImageUsageFlags toVkImageUsageFlags(renderer::ImageUsageFlags flags);
VkImageCreateFlags toVkImageCreateFlags(renderer::ImageCreateFlags flags);
VkImageTiling toVkImageTiling(renderer::ImageTiling tiling);
VkImageLayout toVkImageLayout(renderer::ImageLayout layout);
VkShaderStageFlagBits toVkShaderStageFlagBits(renderer::ShaderStageFlagBits stage);
VkMemoryPropertyFlags toVkMemoryPropertyFlags(renderer::MemoryPropertyFlags flags);
VkCommandBufferUsageFlags toCommandBufferUsageFlags(renderer::CommandBufferUsageFlags flags);
VkImageType toVkImageType(renderer::ImageType view_type);
VkImageViewType toVkImageViewType(renderer::ImageViewType view_type);
VkImageAspectFlags toVkImageAspectFlags(renderer::ImageAspectFlags flags);
VkAccessFlags toVkAccessFlags(renderer::AccessFlags flags);
VkPipelineStageFlags toVkPipelineStageFlags(renderer::PipelineStageFlags flags);
VkShaderStageFlags toVkShaderStageFlags(renderer::ShaderStageFlags flags);
VkPipelineBindPoint toVkPipelineBindPoint(renderer::PipelineBindPoint bind);
VkCommandPoolCreateFlags toVkCommandPoolCreateFlags(renderer::CommandPoolCreateFlags flags);
VkFilter toVkFilter(renderer::Filter filter);
VkDescriptorType toVkDescriptorType(renderer::DescriptorType type);
VkSamplerAddressMode toVkSamplerAddressMode(renderer::SamplerAddressMode mode);
VkSamplerMipmapMode toVkSamplerMipmapMode(renderer::SamplerMipmapMode mode);
VkColorSpaceKHR toVkColorSpace(renderer::ColorSpace color_space);
VkPresentModeKHR toVkPresentMode(renderer::PresentMode mode);
VkSurfaceTransformFlagBitsKHR toVkSurfaceTransformFlags(renderer::SurfaceTransformFlagBits flag);
VkColorComponentFlags toVkColorComponentFlags(renderer::ColorComponentFlags flags);
VkPrimitiveTopology toVkPrimitiveTopology(renderer::PrimitiveTopology primitive);
VkBlendFactor toVkBlendFactor(renderer::BlendFactor blend_factor);
VkBlendOp toVkBlendOp(renderer::BlendOp blend_op);
VkIndexType toVkIndexType(renderer::IndexType index_type);
VkVertexInputRate toVkVertexInputRate(renderer::VertexInputRate input_rate);
VkLogicOp toVkLogicOp(renderer::LogicOp logic_op);
VkPolygonMode toVkPolygonMode(renderer::PolygonMode polygon_mode);
VkCullModeFlags toVkCullModeFlags(renderer::CullModeFlags flags);
VkFrontFace toVkFrontFace(renderer::FrontFace front_face);
VkSampleCountFlags toVkSampleCountFlags(renderer::SampleCountFlags flags);
VkCompareOp toVkCompareOp(renderer::CompareOp compare_op);
VkStencilOp toVkStencilOp(renderer::StencilOp stencil_op);
VkAttachmentLoadOp toVkAttachmentLoadOp(renderer::AttachmentLoadOp load_op);
VkAttachmentStoreOp toVkAttachmentStoreOp(renderer::AttachmentStoreOp store_op);
VkAttachmentDescriptionFlags toVkAttachmentDescriptionFlags(renderer::AttachmentDescriptionFlags flags);
VkDependencyFlags toVkDependencyFlags(renderer::DependencyFlags flags);
VkSubpassDescriptionFlags toVkSubpassDescriptionFlags(renderer::SubpassDescriptionFlags flags);
std::vector<VkVertexInputBindingDescription> toVkVertexInputBindingDescription(
    const std::vector<renderer::VertexInputBindingDescription>& description);
std::vector<VkVertexInputAttributeDescription> toVkVertexInputAttributeDescription(
    const std::vector<renderer::VertexInputAttributeDescription>& description);

// convert from vulkan data type.
renderer::Format fromVkFormat(VkFormat format);
renderer::ColorSpace fromVkColorSpace(VkColorSpaceKHR color_space);
renderer::SurfaceTransformFlagBits fromVkSurfaceTransformFlags(VkSurfaceTransformFlagBitsKHR flag);

// helper functions.
std::shared_ptr<renderer::Instance> createInstance();

std::shared_ptr<renderer::Surface> createSurface(
    const std::shared_ptr<renderer::Instance>& instance,
    GLFWwindow* window);

void DestroyDebugUtilsMessengerEXT(const VkInstance& instance,
    VkDebugUtilsMessengerEXT debug_messenger,
    const VkAllocationCallbacks* pAllocator);

renderer::PhysicalDeviceList collectPhysicalDevices(
    const std::shared_ptr<renderer::Instance>& instance);

renderer::QueueFamilyIndices findQueueFamilies(
    const std::shared_ptr<renderer::PhysicalDevice>& physical_device,
    const std::shared_ptr<renderer::Surface>& surface);

SwapChainSupportDetails querySwapChainSupport(
    const std::shared_ptr<renderer::PhysicalDevice>& physical_device,
    const std::shared_ptr<renderer::Surface>& surface);

std::shared_ptr<renderer::PhysicalDevice> pickPhysicalDevice(
    const renderer::PhysicalDeviceList& physical_devices,
    const std::shared_ptr<renderer::Surface>& surface);

std::shared_ptr<renderer::Device> createLogicalDevice(
    const std::shared_ptr<renderer::PhysicalDevice>& physical_device,
    const std::shared_ptr<renderer::Surface>& surface,
    const renderer::QueueFamilyIndices& indices);

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available_formats);

renderer::PresentMode chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);

VkExtent2D chooseSwapExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities);

renderer::Format findDepthFormat(const std::shared_ptr<renderer::Device>& device);

std::vector<VkPipelineShaderStageCreateInfo> getShaderStages(
    const std::vector<std::shared_ptr<renderer::ShaderModule>>& shader_modules);

uint32_t findMemoryType(
    const std::shared_ptr<renderer::PhysicalDevice>& physical_device,
    uint32_t type_filter,
    VkMemoryPropertyFlags properties);

VkWriteDescriptorSet addDescriptWrite(
    const VkDescriptorSet& description_set,
    const VkDescriptorImageInfo& image_info,
    uint32_t binding,
    const VkDescriptorType& desc_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

void create2DImage(
    const std::shared_ptr<renderer::Device>& device,
    glm::vec2 tex_size,
    const renderer::Format& format,
    const renderer::ImageTiling& tiling,
    const renderer::ImageUsageFlags& usage,
    const renderer::MemoryPropertyFlags& properties,
    std::shared_ptr<renderer::Image>& image,
    std::shared_ptr<renderer::DeviceMemory>& image_memory);

void copyBuffer(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::Buffer>& src_buffer,
    const std::shared_ptr<renderer::Buffer>& dst_buffer,
    uint64_t buffer_size);

void transitionImageLayout(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::Image>& image,
    const renderer::Format& format,
    const renderer::ImageLayout& old_layout,
    const renderer::ImageLayout& new_layout,
    uint32_t base_mip_idx = 0,
    uint32_t mip_count = 1,
    uint32_t base_layer = 0,
    uint32_t layer_count = 1);

void copyBufferToImageWithMips(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::Buffer>& buffer,
    const std::shared_ptr<renderer::Image>& image,
    const std::vector<renderer::BufferImageCopyInfo>& copy_regions);

void copyBufferToImage(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::Buffer>& buffer,
    const std::shared_ptr<renderer::Image>& image,
    const glm::uvec2& tex_size);

void generateMipmapLevels(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::Image>& image,
    uint32_t mip_count,
    uint32_t width,
    uint32_t height,
    const renderer::ImageLayout& cur_image_layout);

void create2x2Texture(
    const renderer::DeviceInfo& device_info,
    uint32_t color,
    renderer::TextureInfo& texture);

std::vector<VkPipelineShaderStageCreateInfo> getComputeShaderStages(
    const std::vector<std::shared_ptr<renderer::ShaderModule>>& shader_modules);

VkPipelineVertexInputStateCreateInfo fillVkPipelineVertexInputStateCreateInfo(
    const std::vector<VkVertexInputBindingDescription>& binding_descs,
    const std::vector<VkVertexInputAttributeDescription>& attribute_descs);

VkPipelineInputAssemblyStateCreateInfo fillVkPipelineInputAssemblyStateCreateInfo(
    const renderer::PipelineInputAssemblyStateCreateInfo& topology_info);

std::vector<VkPipelineColorBlendAttachmentState> fillVkPipelineColorBlendAttachments(
    const renderer::PipelineColorBlendStateCreateInfo& blend_info);

VkPipelineColorBlendStateCreateInfo fillVkPipelineColorBlendStateCreateInfo(
    const renderer::PipelineColorBlendStateCreateInfo& blend_info,
    const std::vector<VkPipelineColorBlendAttachmentState>& attachments);

VkPipelineRasterizationStateCreateInfo fillVkPipelineRasterizationStateCreateInfo(
    const renderer::PipelineRasterizationStateCreateInfo& rasterization_info);

VkPipelineMultisampleStateCreateInfo fillVkPipelineMultisampleStateCreateInfo(
    const renderer::PipelineMultisampleStateCreateInfo& ms_info);

VkViewport fillViewport(const glm::uvec2 size);

VkRect2D fillScissor(const glm::uvec2 size);

VkPipelineViewportStateCreateInfo fillVkPipelineViewportStateCreateInfo(
    const VkViewport* viewport,
    const VkRect2D* scissor);

std::vector<VkPipelineColorBlendAttachmentState> fillVkPipelineColorBlendAttachmentState(
    uint32_t num_attachments);

VkStencilOpState fillVkStencilOpState(const renderer::StencilOpState& stencil_op);

VkPipelineDepthStencilStateCreateInfo fillVkPipelineDepthStencilStateCreateInfo(
    const renderer::PipelineDepthStencilStateCreateInfo& depth_stencil_info);

VkAttachmentDescription FillVkAttachmentDescription(
    const renderer::AttachmentDescription& attachment_desc);

std::vector<VkAttachmentReference> FillVkAttachmentReference(
    const std::vector<renderer::AttachmentReference>& attachment_references);

struct SubpassAttachments {
    std::vector<VkAttachmentReference> input_attachments;
    std::vector<VkAttachmentReference> color_attachments;
    std::vector<VkAttachmentReference> resolve_attachments;
    std::vector<VkAttachmentReference> depth_stencil_attachment;
};
VkSubpassDescription FillVkSubpassDescription(
    const renderer::SubpassDescription& subpass,
    const SubpassAttachments& attachments);

VkSubpassDependency FillVkSubpassDependency(
    const renderer::SubpassDependency& dependency);

} // namespace helper
} // namespace vk
} // namespace renderer
} // namespace engine