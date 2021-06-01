#include <iostream>

#include "../renderer.h"
#include "vk_device.h"
#include "vk_command_buffer.h"
#include "vk_renderer_helper.h"

namespace engine {
namespace renderer {
namespace vk {

std::shared_ptr<Buffer> VulkanDevice::createBuffer(uint64_t buf_size, BufferUsageFlags usage, bool sharing/* = false*/) {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = buf_size;
    buffer_info.usage = helper::toVkBufferUsageFlags(usage);
    buffer_info.sharingMode = sharing ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    auto result = std::make_shared<VulkanBuffer>();
    result->set(buffer, buf_size);

    return result;
}

void VulkanDevice::createBuffer(
    const uint64_t& buffer_size,
    const BufferUsageFlags& usage,
    const MemoryPropertyFlags& properties,
    std::shared_ptr<Buffer>& buffer,
    std::shared_ptr<DeviceMemory>& buffer_memory) {
    buffer = createBuffer(buffer_size, usage);
    auto mem_requirements = getBufferMemoryRequirements(buffer);
    buffer_memory = allocateMemory(mem_requirements.size,
        mem_requirements.memory_type_bits,
        helper::toVkMemoryPropertyFlags(properties));
    bindBufferMemory(buffer, buffer_memory);
}

std::shared_ptr<Image> VulkanDevice::createImage(
    ImageType image_type,
    glm::uvec3 image_size,
    Format format,
    ImageUsageFlags usage,
    ImageTiling tiling,
    ImageLayout layout,
    ImageCreateFlags flags/* = 0*/,
    bool sharing/* = false*/,
    uint32_t num_samples/* = 1*/,
    uint32_t num_mips/* = 1*/,
    uint32_t num_layers/* = 1*/) {
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = helper::toVkImageType(image_type);
    image_info.extent.width = image_size.x;
    image_info.extent.height = image_size.y;
    image_info.extent.depth = image_size.z;
    image_info.mipLevels = num_mips;
    image_info.arrayLayers = num_layers;
    image_info.format = helper::toVkFormat(format);
    image_info.tiling = helper::toVkImageTiling(tiling);
    image_info.initialLayout = helper::toVkImageLayout(layout);
    image_info.usage = helper::toVkImageUsageFlags(usage);
    image_info.sharingMode = sharing ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    image_info.samples = static_cast<VkSampleCountFlagBits>(num_samples);
    image_info.flags = helper::toVkImageCreateFlags(flags);

    VkImage image;
    if (vkCreateImage(device_, &image_info, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    auto result = std::make_shared<VulkanImage>();
    result->set(image);
    return result;
}

std::shared_ptr<ImageView> VulkanDevice::createImageView(
    std::shared_ptr<Image> image,
    ImageViewType view_type,
    Format format,
    ImageAspectFlags aspect_flags,
    uint32_t base_mip/* = 0*/,
    uint32_t mip_count/* = 1*/,
    uint32_t base_layer/* = 0*/,
    uint32_t layer_count/* = 1*/) {
    auto vk_image = RENDER_TYPE_CAST(Image, image);

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = vk_image->get();
    view_info.viewType = helper::toVkImageViewType(view_type);
    view_info.format = helper::toVkFormat(format);
    view_info.subresourceRange.aspectMask = helper::toVkImageAspectFlags(aspect_flags);
    view_info.subresourceRange.baseMipLevel = base_mip;
    view_info.subresourceRange.levelCount = mip_count;
    view_info.subresourceRange.baseArrayLayer = base_layer;
    view_info.subresourceRange.layerCount = layer_count;

    VkImageView image_view;
    if (vkCreateImageView(device_, &view_info, nullptr, &image_view) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }

    auto result = std::make_shared<VulkanImageView>();
    result->set(image_view);

    return result;
}

std::shared_ptr<Sampler> VulkanDevice::createSampler(Filter filter, SamplerAddressMode address_mode, SamplerMipmapMode mipmap_mode, float anisotropy) {
    auto vk_filter = helper::toVkFilter(filter);
    auto vk_address_mode = helper::toVkSamplerAddressMode(address_mode);
    auto vk_mipmap_mode = helper::toVkSamplerMipmapMode(mipmap_mode);

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = vk_filter;
    sampler_info.minFilter = vk_filter;
    sampler_info.addressModeU = vk_address_mode;
    sampler_info.addressModeV = vk_address_mode;
    sampler_info.addressModeW = vk_address_mode;
    sampler_info.anisotropyEnable = anisotropy > 0 ? VK_TRUE : VK_FALSE;
    sampler_info.maxAnisotropy = anisotropy;
    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
    sampler_info.mipmapMode = vk_mipmap_mode;
    sampler_info.mipLodBias = 0.0f;
    sampler_info.minLod = 0.0f;
    sampler_info.maxLod = 1024.0f;

    VkSampler tex_sampler;
    if (vkCreateSampler(device_, &sampler_info, nullptr, &tex_sampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }

    auto vk_tex_sampler = std::make_shared<VulkanSampler>();
    vk_tex_sampler->set(tex_sampler);
    return vk_tex_sampler;
}

std::shared_ptr<Semaphore> VulkanDevice::createSemaphore() {
    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkSemaphore semaphore;
    if (vkCreateSemaphore(device_, &semaphore_info, nullptr, &semaphore) != VK_SUCCESS) {
        throw std::runtime_error("failed to create semaphore!");
    }

    auto vk_semaphore = std::make_shared<VulkanSemaphore>();
    vk_semaphore->set(semaphore);
    return vk_semaphore;
}

std::shared_ptr<Fence> VulkanDevice::createFence() {
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkFence fence;
    if (vkCreateFence(device_, &fence_info, nullptr, &fence) != VK_SUCCESS) {
        throw std::runtime_error("failed to create fence!");
    }

    auto vk_fence = std::make_shared<VulkanFence>();
    vk_fence->set(fence);
    return vk_fence;
}

std::shared_ptr<ShaderModule> VulkanDevice::createShaderModule(uint64_t size, void* data) {
    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = size;
    create_info.pCode = reinterpret_cast<const uint32_t*>(data);

    VkShaderModule shader_module;
    if (vkCreateShaderModule(device_, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }

    auto vk_shader_module = std::make_shared<VulkanShaderModule>();
    vk_shader_module->set(shader_module);

    return vk_shader_module;
}

std::shared_ptr<CommandPool> VulkanDevice::createCommandPool(uint32_t queue_family_index, CommandPoolCreateFlags flags) {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family_index;
    pool_info.flags = helper::toVkCommandPoolCreateFlags(flags);

    VkCommandPool cmd_pool;
    if (vkCreateCommandPool(device_, &pool_info, nullptr, &cmd_pool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }

    auto vk_cmd_pool = std::make_shared<VulkanCommandPool>();
    vk_cmd_pool->set(cmd_pool);
    return vk_cmd_pool;
}

std::shared_ptr<Queue> VulkanDevice::getDeviceQueue(uint32_t queue_family_index, uint32_t queue_index/* = 0*/) {
    VkQueue queue;
    vkGetDeviceQueue(device_, queue_family_index, queue_index, &queue);
    auto vk_queue = std::make_shared<VulkanQueue>();
    vk_queue->set(queue);
    return vk_queue;
}

std::shared_ptr<DescriptorSetLayout> VulkanDevice::createDescriptorSetLayout(
    const std::vector<DescriptorSetLayoutBinding>& bindings) {
    std::vector<VkDescriptorSetLayoutBinding> vk_bindings(bindings.size());
    for (auto i = 0; i < bindings.size(); i++) {
        const auto& binding = bindings[i];
        auto& vk_binding = vk_bindings[i];
        vk_binding.binding = binding.binding;
        vk_binding.descriptorType = helper::toVkDescriptorType(binding.descriptor_type);
        vk_binding.descriptorCount = binding.descriptor_count;
        vk_binding.stageFlags = helper::toVkShaderStageFlags(binding.stage_flags);
        auto vk_samplers = RENDER_TYPE_CAST(Sampler, binding.immutable_samplers);
        vk_binding.pImmutableSamplers = vk_samplers ? vk_samplers->getPtr() : nullptr;
    }

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(vk_bindings.size());
    layout_info.pBindings = vk_bindings.data();

    VkDescriptorSetLayout descriptor_set_layout;
    if (vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    auto vk_set_layout = std::make_shared<VulkanDescriptorSetLayout>();
    vk_set_layout->set(descriptor_set_layout);
    return vk_set_layout;
}

std::shared_ptr<RenderPass> VulkanDevice::createRenderPass(
    const std::vector<AttachmentDescription>& attachments,
    const std::vector<SubpassDescription>& subpasses,
    const std::vector<SubpassDependency>& dependencies) {

    std::vector<VkAttachmentDescription> vk_attachments(attachments.size());
    for (int i = 0; i < attachments.size(); i++) {
        vk_attachments[i] = helper::FillVkAttachmentDescription(attachments[i]);
    }

    std::vector<helper::SubpassAttachments> subpass_attachments(subpasses.size());
    for (uint32_t i = 0; i < subpasses.size(); i++) {
        subpass_attachments[i].input_attachments = helper::FillVkAttachmentReference(subpasses[i].input_attachments);
        subpass_attachments[i].color_attachments = helper::FillVkAttachmentReference(subpasses[i].color_attachments);
        subpass_attachments[i].resolve_attachments = helper::FillVkAttachmentReference(subpasses[i].resolve_attachments);
        subpass_attachments[i].depth_stencil_attachment = helper::FillVkAttachmentReference(subpasses[i].depth_stencil_attachment);
    }

    std::vector<VkSubpassDescription> vk_subpasses(subpasses.size());
    for (int i = 0; i < subpasses.size(); i++) {
        vk_subpasses[i] = helper::FillVkSubpassDescription(subpasses[i], subpass_attachments[i]);
    }

    std::vector<VkSubpassDependency> vk_dependencies(dependencies.size());
    for (int i = 0; i < dependencies.size(); i++) {
        vk_dependencies[i] = helper::FillVkSubpassDependency(dependencies[i]);
    }

    VkRenderPassCreateInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = static_cast<uint32_t>(vk_attachments.size());
    render_pass_info.pAttachments = vk_attachments.data();
    render_pass_info.subpassCount = static_cast<uint32_t>(vk_subpasses.size());
    render_pass_info.pSubpasses = vk_subpasses.data();
    render_pass_info.dependencyCount = static_cast<uint32_t>(vk_dependencies.size());
    render_pass_info.pDependencies = vk_dependencies.data();

    VkRenderPass render_pass;
    if (vkCreateRenderPass(device_, &render_pass_info, nullptr, &render_pass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }

    auto vk_render_pass = std::make_shared<VulkanRenderPass>();
    vk_render_pass->set(render_pass);
    return vk_render_pass;
}

DescriptorSetList VulkanDevice::createDescriptorSets(
    std::shared_ptr<DescriptorPool> descriptor_pool,
    std::shared_ptr<DescriptorSetLayout> descriptor_set_layout,
    uint64_t buffer_count) {
    auto vk_descriptor_pool = RENDER_TYPE_CAST(DescriptorPool, descriptor_pool);
    auto vk_descriptor_set_layout = RENDER_TYPE_CAST(DescriptorSetLayout, descriptor_set_layout);
    std::vector<VkDescriptorSetLayout> layouts(buffer_count, vk_descriptor_set_layout->get());
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = vk_descriptor_pool->get();
    alloc_info.descriptorSetCount = static_cast<uint32_t>(buffer_count);
    alloc_info.pSetLayouts = layouts.data();

    std::vector<VkDescriptorSet> vk_desc_sets;
    vk_desc_sets.resize(buffer_count);
    if (vkAllocateDescriptorSets(device_, &alloc_info, vk_desc_sets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    DescriptorSetList desc_sets(vk_desc_sets.size());
    for (uint32_t i = 0; i < buffer_count; i++) {
        auto vk_desc_set = std::make_shared<VulkanDescriptorSet>();
        vk_desc_set->set(vk_desc_sets[i]);
        desc_sets[i] = vk_desc_set;
    }

    return std::move(desc_sets);
}

std::shared_ptr<PipelineLayout> VulkanDevice::createPipelineLayout(
    const DescriptorSetLayoutList& desc_set_layouts,
    const std::vector<PushConstantRange>& push_const_ranges) {

    std::vector<VkPushConstantRange> vk_push_const_ranges;
    vk_push_const_ranges.reserve(push_const_ranges.size());
    for (auto& push_const_range : push_const_ranges) {
        VkPushConstantRange vk_push_const_range{};
        vk_push_const_range.stageFlags = helper::toVkShaderStageFlags(push_const_range.stage_flags);
        vk_push_const_range.offset = push_const_range.offset;
        vk_push_const_range.size = push_const_range.size;
        vk_push_const_ranges.push_back(vk_push_const_range);
    }

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    // todo.
    std::vector<VkDescriptorSetLayout> vk_layouts;
    vk_layouts.reserve(desc_set_layouts.size());
    for (auto& desc_set_layout : desc_set_layouts) {
        vk_layouts.push_back(RENDER_TYPE_CAST(DescriptorSetLayout, desc_set_layout)->get());
    }

    pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(vk_layouts.size());
    pipeline_layout_info.pSetLayouts = vk_layouts.data();
    pipeline_layout_info.pushConstantRangeCount = static_cast<uint32_t>(vk_push_const_ranges.size());
    pipeline_layout_info.pPushConstantRanges = vk_push_const_ranges.data();

    VkPipelineLayout pipeline_layout;
    if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }
    auto vk_pipeline_layout = std::make_shared<VulkanPipelineLayout>();
    vk_pipeline_layout->set(pipeline_layout);

    return vk_pipeline_layout;
}

std::shared_ptr<Pipeline> VulkanDevice::createPipeline(
    const std::shared_ptr<RenderPass>& render_pass,
    const std::shared_ptr<PipelineLayout>& pipeline_layout,
    const std::vector<VertexInputBindingDescription>& binding_descs,
    const std::vector<VertexInputAttributeDescription>& attribute_descs,
    const PipelineInputAssemblyStateCreateInfo& topology_info,
    const GraphicPipelineInfo& graphic_pipeline_info,
    const ShaderModuleList& shader_modules,
    const glm::uvec2& extent) {

    VkGraphicsPipelineCreateInfo pipeline_info{};

    auto viewport = helper::fillViewport(extent);
    auto scissor = helper::fillScissor(extent);

    auto vk_blend_attachments = helper::fillVkPipelineColorBlendAttachments(*graphic_pipeline_info.blend_state_info);
    auto vk_color_blending = helper::fillVkPipelineColorBlendStateCreateInfo(*graphic_pipeline_info.blend_state_info, vk_blend_attachments);
    auto vk_rasterizer = helper::fillVkPipelineRasterizationStateCreateInfo(*graphic_pipeline_info.rasterization_info);
    auto vk_multisampling = helper::fillVkPipelineMultisampleStateCreateInfo(*graphic_pipeline_info.ms_info);
    auto vk_depth_stencil = helper::fillVkPipelineDepthStencilStateCreateInfo(*graphic_pipeline_info.depth_stencil_info);

    auto viewport_state = helper::fillVkPipelineViewportStateCreateInfo(&viewport, &scissor);

    auto vk_binding_descs = helper::toVkVertexInputBindingDescription(binding_descs);
    auto vk_attribute_descs = helper::toVkVertexInputAttributeDescription(attribute_descs);
    auto vk_vertex_input_info = helper::fillVkPipelineVertexInputStateCreateInfo(vk_binding_descs, vk_attribute_descs);
    auto vk_input_assembly = helper::fillVkPipelineInputAssemblyStateCreateInfo(topology_info);
    auto shader_stages = helper::getShaderStages(shader_modules);

    auto vk_pipeline_layout = RENDER_TYPE_CAST(PipelineLayout, pipeline_layout);
    assert(vk_pipeline_layout);

    auto vk_render_pass = RENDER_TYPE_CAST(RenderPass, render_pass);
    assert(vk_render_pass);

    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = static_cast<uint32_t>(shader_stages.size());
    pipeline_info.pStages = shader_stages.data();
    pipeline_info.pVertexInputState = &vk_vertex_input_info;
    pipeline_info.pInputAssemblyState = &vk_input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &vk_rasterizer;
    pipeline_info.pMultisampleState = &vk_multisampling;
    pipeline_info.pDepthStencilState = &vk_depth_stencil;
    pipeline_info.pColorBlendState = &vk_color_blending;
    //    pipeline_info.pDynamicState = nullptr; // Optional
    pipeline_info.layout = vk_pipeline_layout->get();
    pipeline_info.renderPass = vk_render_pass->get();
    pipeline_info.subpass = 0;
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE; // Optional
    pipeline_info.basePipelineIndex = -1; // Optional

    VkPipeline graphics_pipeline;
    auto result = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &graphics_pipeline);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }
    auto vk_pipeline = std::make_shared<VulkanPipeline>();
    vk_pipeline->set(graphics_pipeline);
    return vk_pipeline;
}

std::shared_ptr<Pipeline> VulkanDevice::createPipeline(
    const std::shared_ptr<PipelineLayout>& pipeline_layout,
    const std::shared_ptr<ShaderModule>& shader_module) {
    auto ibl_compute_shader_stages = helper::getComputeShaderStages({ shader_module });

    auto vk_ibl_comp_pipeline_layout = RENDER_TYPE_CAST(PipelineLayout, pipeline_layout);
    assert(vk_ibl_comp_pipeline_layout);

    // flags = 0, - e.g. disable optimization
    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = ibl_compute_shader_stages[0];
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
    pipeline_info.basePipelineIndex = -1;
    pipeline_info.layout = vk_ibl_comp_pipeline_layout->get();

    VkPipeline compute_pipeline;
    if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &compute_pipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create compute pipeline!");
    }

    auto vk_pipeline = std::make_shared<VulkanPipeline>();
    vk_pipeline->set(compute_pipeline);

    return vk_pipeline;
}

std::shared_ptr<Swapchain> VulkanDevice::createSwapchain(
    std::shared_ptr<Surface> surface,
    uint32_t image_count,
    Format format,
    glm::uvec2 buf_size,
    ColorSpace color_space,
    SurfaceTransformFlagBits transform,
    PresentMode present_mode,
    std::vector<uint32_t> queue_index) {

    auto vk_surface = RENDER_TYPE_CAST(Surface, surface);

    VkSwapchainCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = vk_surface->get();
    create_info.minImageCount = image_count;
    create_info.imageFormat = helper::toVkFormat(format);
    create_info.imageColorSpace = helper::toVkColorSpace(color_space);
    create_info.imageExtent = { buf_size.x, buf_size.y };
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; //VK_IMAGE_USAGE_TRANSFER_DST_BIT

    create_info.imageSharingMode = queue_index.size() > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    create_info.queueFamilyIndexCount = static_cast<uint32_t>(queue_index.size() <= 1 ? 0 : queue_index.size());
    create_info.pQueueFamilyIndices = queue_index.size() <= 1 ? nullptr : queue_index.data();

    create_info.preTransform = helper::toVkSurfaceTransformFlags(transform);
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = helper::toVkPresentMode(present_mode);
    create_info.clipped = VK_TRUE;
    create_info.oldSwapchain = VK_NULL_HANDLE; //need to be handled when resize.

    VkSwapchainKHR swap_chain;
    if (vkCreateSwapchainKHR(device_, &create_info, nullptr, &swap_chain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }

    auto vk_swap_chain = std::make_shared<VulkanSwapchain>();
    vk_swap_chain->set(swap_chain);
    return vk_swap_chain;
}

std::shared_ptr<Framebuffer> VulkanDevice::createFrameBuffer(
    const std::shared_ptr<RenderPass>& render_pass,
    const std::vector<std::shared_ptr<ImageView>>& attachments,
    const glm::uvec2& extent) {

    std::vector<VkImageView> image_views(attachments.size());
    for (int i = 0; i < attachments.size(); i++) {
        auto vk_image_view = RENDER_TYPE_CAST(ImageView, attachments[i]);
        image_views[i] = vk_image_view->get();
    }

    auto vk_render_pass = RENDER_TYPE_CAST(RenderPass, render_pass);
    VkFramebufferCreateInfo framebuffer_info{};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = vk_render_pass->get();
    framebuffer_info.attachmentCount = static_cast<uint32_t>(image_views.size());
    framebuffer_info.pAttachments = image_views.data();
    framebuffer_info.width = extent.x;
    framebuffer_info.height = extent.y;
    framebuffer_info.layers = 1;

    VkFramebuffer frame_buffer;
    if (vkCreateFramebuffer(device_, &framebuffer_info, nullptr, &frame_buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer!");
    }

    auto vk_frame_buffer = std::make_shared<VulkanFramebuffer>();
    vk_frame_buffer->set(frame_buffer);
    return vk_frame_buffer;
}

std::shared_ptr<DescriptorPool> VulkanDevice::createDescriptorPool() {
    VkDescriptorPoolSize pool_sizes[] =
    {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 256 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 256 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 256 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 256 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 256 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 256 }
    };
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 256 * IM_ARRAYSIZE(pool_sizes);
    pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;
    VkDescriptorPool descriptor_pool;
    if (vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }

    auto vk_descriptor_pool = std::make_shared<VulkanDescriptorPool>();
    vk_descriptor_pool->set(descriptor_pool);
    return vk_descriptor_pool;
}

void VulkanDevice::updateDescriptorSets(
    const std::vector<TextureDescriptor>& texture_list,
    const std::vector<BufferDescriptor>& buffer_list) {

    std::vector<VkWriteDescriptorSet> descriptor_writes;
    std::vector<VkDescriptorImageInfo> desc_image_infos(texture_list.size());
    descriptor_writes.reserve(10);
    for (auto i = 0; i < texture_list.size(); i++) {
        const auto& tex = texture_list[i];
        auto vk_texture = RENDER_TYPE_CAST(ImageView, tex.texture);
        auto vk_sampler = RENDER_TYPE_CAST(Sampler, tex.sampler);
        auto vk_desc_set = RENDER_TYPE_CAST(DescriptorSet, tex.desc_set);
        desc_image_infos[i].imageLayout = helper::toVkImageLayout(tex.image_layout);
        desc_image_infos[i].imageView = vk_texture->get();
        desc_image_infos[i].sampler = vk_sampler->get();
        descriptor_writes.push_back(
            helper::addDescriptWrite(
                vk_desc_set->get(),
                desc_image_infos[i],
                tex.binding,
                helper::toVkDescriptorType(tex.desc_type)));
    }

    std::vector<VkDescriptorBufferInfo> desc_buffer_infos(buffer_list.size());
    for (auto i = 0; i < buffer_list.size(); i++) {
        const auto& buf = buffer_list[i];
        auto vk_buffer = RENDER_TYPE_CAST(Buffer, buf.buffer);
        auto vk_desc_set = RENDER_TYPE_CAST(DescriptorSet, buf.desc_set);
        desc_buffer_infos[i].buffer = vk_buffer->get();
        desc_buffer_infos[i].offset = buf.offset;
        desc_buffer_infos[i].range = buf.range;

        VkWriteDescriptorSet descriptor_write = {};
        descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_write.dstSet = vk_desc_set->get();
        descriptor_write.dstBinding = buf.binding;
        descriptor_write.dstArrayElement = 0;

        descriptor_write.descriptorType = helper::toVkDescriptorType(buf.desc_type);
        descriptor_write.descriptorCount = 1;
        descriptor_write.pBufferInfo = &desc_buffer_infos[i];
        descriptor_writes.push_back(descriptor_write);
    }

    vkUpdateDescriptorSets(device_,
        static_cast<uint32_t>(descriptor_writes.size()),
        descriptor_writes.data(),
        0,
        nullptr);
}

void VulkanDevice::updateBufferMemory(
    const std::shared_ptr<DeviceMemory>& memory,
    uint64_t size,
    const void* src_data,
    uint64_t offset/* = 0*/) {
    if (memory) {
        void* dst_data = mapMemory(memory, size, offset);
        assert(dst_data);
        memcpy(dst_data, src_data, size);
        unmapMemory(memory);
    }
}

std::vector<std::shared_ptr<Image>> VulkanDevice::getSwapchainImages(std::shared_ptr<Swapchain> swap_chain) {
    auto vk_swap_chain = RENDER_TYPE_CAST(Swapchain, swap_chain);
    uint32_t image_count;
    std::vector<VkImage> swap_chain_images;
    vkGetSwapchainImagesKHR(device_, vk_swap_chain->get(), &image_count, nullptr);
    swap_chain_images.resize(image_count);
    vkGetSwapchainImagesKHR(device_, vk_swap_chain->get(), &image_count, swap_chain_images.data());

    std::vector<std::shared_ptr<Image>> vk_swap_chain_images(swap_chain_images.size());
    for (int i = 0; i < swap_chain_images.size(); i++) {
        auto vk_image = std::make_shared<VulkanImage>();
        vk_image->set(swap_chain_images[i]);
        vk_swap_chain_images[i] = vk_image;
    }
    return vk_swap_chain_images;
}

std::shared_ptr<DeviceMemory> VulkanDevice::allocateMemory(uint64_t buf_size, uint32_t memory_type_bits, MemoryPropertyFlags properties) {
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = buf_size;
    alloc_info.memoryTypeIndex = helper::findMemoryType(getPhysicalDevice(), memory_type_bits, properties);

    VkDeviceMemory memory;
    if (vkAllocateMemory(device_, &alloc_info, nullptr, &memory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    auto result = std::make_shared<VulkanDeviceMemory>();
    result->set(memory);

    return result;
}

MemoryRequirements VulkanDevice::getBufferMemoryRequirements(std::shared_ptr<Buffer> buffer) {
    auto vk_buffer = RENDER_TYPE_CAST(Buffer, buffer);

    MemoryRequirements mem_requirements = {};
    if (vk_buffer) {
        VkMemoryRequirements vk_mem_requirements;
        vkGetBufferMemoryRequirements(device_, vk_buffer->get(), &vk_mem_requirements);
        mem_requirements.size = vk_mem_requirements.size;
        mem_requirements.alignment = vk_mem_requirements.alignment;
        mem_requirements.memory_type_bits = vk_mem_requirements.memoryTypeBits;
    }

    return mem_requirements;
}

MemoryRequirements VulkanDevice::getImageMemoryRequirements(std::shared_ptr<Image> image) {
    auto vk_image = RENDER_TYPE_CAST(Image, image);

    MemoryRequirements mem_requirements = {};
    if (vk_image) {
        VkMemoryRequirements vk_mem_requirements;
        vkGetImageMemoryRequirements(device_, vk_image->get(), &vk_mem_requirements);
        mem_requirements.size = vk_mem_requirements.size;
        mem_requirements.alignment = vk_mem_requirements.alignment;
        mem_requirements.memory_type_bits = vk_mem_requirements.memoryTypeBits;
    }

    return mem_requirements;
}

void VulkanDevice::bindBufferMemory(std::shared_ptr<Buffer> buffer, std::shared_ptr<DeviceMemory> buffer_memory, uint64_t offset/* = 0*/) {
    auto vk_buffer = RENDER_TYPE_CAST(Buffer, buffer);
    auto vk_buffer_memory = RENDER_TYPE_CAST(DeviceMemory, buffer_memory);

    if (vk_buffer && vk_buffer_memory) {
        vkBindBufferMemory(device_, vk_buffer->get(), vk_buffer_memory->get(), offset);
    }
}

void VulkanDevice::bindImageMemory(std::shared_ptr<Image> image, std::shared_ptr<DeviceMemory> image_memory, uint64_t offset/* = 0*/) {
    auto vk_image = RENDER_TYPE_CAST(Image, image);
    auto vk_image_memory = RENDER_TYPE_CAST(DeviceMemory, image_memory);

    if (vk_image && vk_image_memory) {
        vkBindImageMemory(device_, vk_image->get(), vk_image_memory->get(), offset);
    }
}

std::vector<std::shared_ptr<CommandBuffer>> VulkanDevice::allocateCommandBuffers(
    std::shared_ptr<CommandPool> cmd_pool, 
    uint32_t num_buffers, 
    bool is_primary/* = true*/) {
    auto vk_cmd_pool = RENDER_TYPE_CAST(CommandPool, cmd_pool);
    std::vector<VkCommandBuffer> cmd_bufs(num_buffers);

    if (vk_cmd_pool) {
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = vk_cmd_pool->get();
        alloc_info.level = is_primary ? VK_COMMAND_BUFFER_LEVEL_PRIMARY : VK_COMMAND_BUFFER_LEVEL_SECONDARY;
        alloc_info.commandBufferCount = num_buffers;

        if (vkAllocateCommandBuffers(device_, &alloc_info, cmd_bufs.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    std::vector<std::shared_ptr<CommandBuffer>> result(num_buffers);
    for (uint32_t i = 0; i < num_buffers; i++) {
        auto cmd_buf = std::make_shared<VulkanCommandBuffer>();
        cmd_buf->set(cmd_bufs[i]);
        result[i] = cmd_buf;
    }

    return result;
}

void* VulkanDevice::mapMemory(std::shared_ptr<DeviceMemory> memory, uint64_t size, uint64_t offset /*=0*/) {
    void* data = nullptr;
    auto vk_memory = RENDER_TYPE_CAST(DeviceMemory, memory);
    if (vk_memory) {
        vkMapMemory(device_, vk_memory->get(), offset, size, 0/*reserved*/, &data);
    }

    return data;
}

void VulkanDevice::unmapMemory(std::shared_ptr<DeviceMemory> memory) {
    auto vk_memory = RENDER_TYPE_CAST(DeviceMemory, memory);
    if (vk_memory) {
        vkUnmapMemory(device_, vk_memory->get());
    }
}

void VulkanDevice::destroyCommandPool(std::shared_ptr<CommandPool> cmd_pool) {
    auto vk_cmd_pool = RENDER_TYPE_CAST(CommandPool, cmd_pool);
    if (vk_cmd_pool) {
        vkDestroyCommandPool(device_, vk_cmd_pool->get(), nullptr);
    }
}

void VulkanDevice::destroySwapchain(std::shared_ptr<Swapchain> swapchain) {
    auto vk_swapchain = RENDER_TYPE_CAST(Swapchain, swapchain);
    if (vk_swapchain) {
        vkDestroySwapchainKHR(device_, vk_swapchain->get(), nullptr);
    }
}

void VulkanDevice::destroyDescriptorPool(std::shared_ptr<DescriptorPool> descriptor_pool) {
    auto vk_descriptor_pool = RENDER_TYPE_CAST(DescriptorPool, descriptor_pool);
    if (vk_descriptor_pool) {
        vkDestroyDescriptorPool(device_, vk_descriptor_pool->get(), nullptr);
    }
}

void VulkanDevice::destroyPipeline(std::shared_ptr<Pipeline> pipeline) {
    auto vk_pipeline = RENDER_TYPE_CAST(Pipeline, pipeline);
    if (vk_pipeline) {
        vkDestroyPipeline(device_, vk_pipeline->get(), nullptr);
    }
}

void VulkanDevice::destroyPipelineLayout(std::shared_ptr<PipelineLayout> pipeline_layout) {
    auto vk_pipeline_layout = RENDER_TYPE_CAST(PipelineLayout, pipeline_layout);
    if (vk_pipeline_layout) {
        vkDestroyPipelineLayout(device_, vk_pipeline_layout->get(), nullptr);
    }
}

void VulkanDevice::destroyRenderPass(std::shared_ptr<RenderPass> render_pass) {
    auto vk_render_pass = RENDER_TYPE_CAST(RenderPass, render_pass);
    if (vk_render_pass) {
        vkDestroyRenderPass(device_, vk_render_pass->get(), nullptr);
    }
}

void VulkanDevice::destroyFramebuffer(std::shared_ptr<Framebuffer> frame_buffer) {
    auto vk_framebuffer = RENDER_TYPE_CAST(Framebuffer, frame_buffer);
    if (vk_framebuffer) {
        vkDestroyFramebuffer(device_, vk_framebuffer->get(), nullptr);
    }
}

void VulkanDevice::destroyImageView(std::shared_ptr<ImageView> image_view) {
    auto vk_image_view = RENDER_TYPE_CAST(ImageView, image_view);
    if (vk_image_view) {
        vkDestroyImageView(device_, vk_image_view->get(), nullptr);
    }
}

void VulkanDevice::destroySampler(std::shared_ptr<Sampler> sampler) {
    auto vk_sampler = RENDER_TYPE_CAST(Sampler, sampler);
    if (vk_sampler) {
        vkDestroySampler(device_, vk_sampler->get(), nullptr);
    }
}

void VulkanDevice::destroyImage(std::shared_ptr<Image> image) {
    auto vk_image = RENDER_TYPE_CAST(Image, image);
    if (vk_image) {
        vkDestroyImage(device_, vk_image->get(), nullptr);
    }
}

void VulkanDevice::destroyBuffer(std::shared_ptr<Buffer> buffer) {
    auto vk_buffer = RENDER_TYPE_CAST(Buffer, buffer);
    if (vk_buffer) {
        vkDestroyBuffer(device_, vk_buffer->get(), nullptr);
    }
}

void VulkanDevice::destroySemaphore(std::shared_ptr<Semaphore> semaphore) {
    auto vk_semaphore = RENDER_TYPE_CAST(Semaphore, semaphore);
    if (vk_semaphore) {
        vkDestroySemaphore(device_, vk_semaphore->get(), nullptr);
    }
}

void VulkanDevice::destroyFence(std::shared_ptr<Fence> fence) {
    auto vk_fence = RENDER_TYPE_CAST(Fence, fence);
    if (vk_fence) {
        vkDestroyFence(device_, vk_fence->get(), nullptr);
    }
}

void VulkanDevice::destroyDescriptorSetLayout(std::shared_ptr<DescriptorSetLayout> layout) {
    auto vk_layout = RENDER_TYPE_CAST(DescriptorSetLayout, layout);
    if (vk_layout) {
        vkDestroyDescriptorSetLayout(device_, vk_layout->get(), nullptr);
    }
}

void VulkanDevice::destroyShaderModule(std::shared_ptr<ShaderModule> shader_module) {
    auto vk_shader_module = RENDER_TYPE_CAST(ShaderModule, shader_module);
    if (vk_shader_module) {
        vkDestroyShaderModule(device_, vk_shader_module->get(), nullptr);
    }
}

void VulkanDevice::destroy() {
    vkDestroyDevice(device_, nullptr);
}

void VulkanDevice::freeMemory(std::shared_ptr<DeviceMemory> memory) {
    auto vk_memory = RENDER_TYPE_CAST(DeviceMemory, memory);
    if (vk_memory) {
        vkFreeMemory(device_, vk_memory->get(), nullptr);
    }
}

void VulkanDevice::freeCommandBuffers(std::shared_ptr<CommandPool> cmd_pool, const std::vector<std::shared_ptr<CommandBuffer>>& cmd_bufs) {
    auto vk_cmd_pool = RENDER_TYPE_CAST(CommandPool, cmd_pool);
    if (vk_cmd_pool) {
        std::vector<VkCommandBuffer> vk_cmd_bufs(cmd_bufs.size());
        for (uint32_t i = 0; i < cmd_bufs.size(); i++) {
            auto vk_cmd_buf = RENDER_TYPE_CAST(CommandBuffer, cmd_bufs[i]);
            vk_cmd_bufs[i] = vk_cmd_buf->get();
        }

        vkFreeCommandBuffers(device_, vk_cmd_pool->get(), static_cast<uint32_t>(vk_cmd_bufs.size()), vk_cmd_bufs.data());
    }
}

void VulkanDevice::resetFences(const std::vector<std::shared_ptr<Fence>>& fences) {
    std::vector<VkFence> vk_fences(fences.size());
    for (int i = 0; i < fences.size(); i++) {
        auto vk_fence = RENDER_TYPE_CAST(Fence, fences[i]);
        vk_fences[i] = vk_fence->get();
    }
    vkResetFences(device_, static_cast<uint32_t>(vk_fences.size()), vk_fences.data());
}

void VulkanDevice::waitForFences(const std::vector<std::shared_ptr<Fence>>& fences) {
    std::vector<VkFence> vk_fences(fences.size());
    for (int i = 0; i < fences.size(); i++) {
        auto vk_fence = RENDER_TYPE_CAST(Fence, fences[i]);
        vk_fences[i] = vk_fence->get();
    }
    vkWaitForFences(device_, static_cast<uint32_t>(vk_fences.size()), vk_fences.data(), VK_TRUE, UINT64_MAX);
}

void VulkanDevice::waitIdle() {
    vkDeviceWaitIdle(device_);
}

} // vk
} // renderer
} // engine