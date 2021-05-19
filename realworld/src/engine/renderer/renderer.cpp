#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <array>

#include "renderer.h"
#include "vulkan/vk_device.h"
#include "vulkan/vk_command_buffer.h"
#include "vulkan/vk_renderer_helper.h"

namespace engine {
namespace renderer {

namespace {
static void check_vk_result(VkResult err)
{
    if (err == 0)
        return;
    fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
    if (err < 0)
        abort();
}
}

namespace vk {
void VulkanInstance::destroySurface(const std::shared_ptr<Surface>& surface) {
    vkDestroySurfaceKHR(instance_, RENDER_TYPE_CAST(Surface, surface)->get(), nullptr);
}

void VulkanInstance::destroy() {
    if (helper::hasEnabledValidationLayers()) {
        helper::DestroyDebugUtilsMessengerEXT(instance_, debug_messenger_, nullptr);
    }
    vkDestroyInstance(instance_, nullptr);
}

void VulkanQueue::submit(const std::vector<std::shared_ptr<CommandBuffer>>& command_buffers) {
    std::vector<VkCommandBuffer> vk_cmd_bufs(command_buffers.size());
    for (int i = 0; i < command_buffers.size(); i++) {
        auto vk_cmd_buf = RENDER_TYPE_CAST(CommandBuffer, command_buffers[i]);
        vk_cmd_bufs[i] = vk_cmd_buf->get();
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = static_cast<uint32_t>(command_buffers.size());
    submit_info.pCommandBuffers = vk_cmd_bufs.data();

    vkQueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE);
}

void VulkanQueue::waitIdle() {
    vkQueueWaitIdle(queue_);
}
} // namespace vk

ImageResourceInfo Helper::image_source_info_;
ImageResourceInfo Helper::image_as_color_attachement_;
ImageResourceInfo Helper::image_as_store_;
ImageResourceInfo Helper::image_as_shader_sampler_;
TextureInfo Helper::white_tex_;
TextureInfo Helper::black_tex_;

void Helper::init(const DeviceInfo& device_info) {
    vk::helper::create2x2Texture(device_info, 0xffffffff, white_tex_);
    vk::helper::create2x2Texture(device_info, 0xff000000, black_tex_);

    image_source_info_ = {
        ImageLayout::UNDEFINED,
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };

    image_as_color_attachement_ = {
        ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT) };

    image_as_store_ = {
        ImageLayout::GENERAL,
        SET_FLAG_BIT(Access, SHADER_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };

    image_as_shader_sampler_ = {
        ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) |
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
}

void Helper::destroy(const std::shared_ptr<Device>& device) {
    black_tex_.destroy(device);
    white_tex_.destroy(device);
}

std::shared_ptr<Instance> Helper::createInstance() {
    return vk::helper::createInstance();
}

std::shared_ptr<Surface> Helper::createSurface(
    const std::shared_ptr<Instance>& instance,
    GLFWwindow* window) {
    return vk::helper::createSurface(instance, window);
}

PhysicalDeviceList Helper::collectPhysicalDevices(
    const std::shared_ptr<Instance>& instance) {
    return vk::helper::collectPhysicalDevices(instance);
}

std::shared_ptr<PhysicalDevice> Helper::pickPhysicalDevice(
    const PhysicalDeviceList& physical_devices,
    const std::shared_ptr<Surface>& surface) {
    return vk::helper::pickPhysicalDevice(physical_devices, surface);
}

QueueFamilyIndices Helper::findQueueFamilies(
    const std::shared_ptr<PhysicalDevice>& physical_device,
    const std::shared_ptr<Surface>& surface) {
    return vk::helper::findQueueFamilies(physical_device, surface);
}

std::shared_ptr<Device> Helper::createLogicalDevice(
    const std::shared_ptr<PhysicalDevice>& physical_device,
    const std::shared_ptr<Surface>& surface,
    const QueueFamilyIndices& indices) {
    return vk::helper::createLogicalDevice(physical_device, surface, indices);
}

void Helper::createSwapChain(
    GLFWwindow* window,
    const std::shared_ptr<Device>& device,
    const std::shared_ptr<Surface>& surface,
    const QueueFamilyIndices& indices,
    SwapChainInfo& swap_chain_info) {
    const auto& vk_device = RENDER_TYPE_CAST(Device, device);
    const auto& vk_physical_device = vk_device->getPhysicalDevice();
    vk::helper::SwapChainSupportDetails swap_chain_support = vk::helper::querySwapChainSupport(vk_physical_device, surface);

    VkSurfaceFormatKHR surface_format = vk::helper::chooseSwapSurfaceFormat(swap_chain_support.formats_);
    auto present_mode = vk::helper::chooseSwapPresentMode(swap_chain_support.present_modes_);
    VkExtent2D extent = vk::helper::chooseSwapExtent(window, swap_chain_support.capabilities_);

    uint32_t image_count = swap_chain_support.capabilities_.minImageCount + 1;
    if (swap_chain_support.capabilities_.maxImageCount > 0 && image_count > swap_chain_support.capabilities_.maxImageCount) {
        image_count = swap_chain_support.capabilities_.maxImageCount;
    }

    std::vector<uint32_t> queue_index(2);
    queue_index[0] = indices.graphics_family_.value();
    queue_index[1] = indices.present_family_.value();
    std::sort(queue_index.begin(), queue_index.end());
    auto last = std::unique(queue_index.begin(), queue_index.end());
    queue_index.erase(last, queue_index.end());

    swap_chain_info.format = vk::helper::fromVkFormat(surface_format.format);
    swap_chain_info.extent = glm::uvec2(extent.width, extent.height);

    swap_chain_info.swap_chain = device->createSwapchain(surface,
        image_count,
        swap_chain_info.format,
        swap_chain_info.extent,
        vk::helper::fromVkColorSpace(surface_format.colorSpace),
        vk::helper::fromVkSurfaceTransformFlags(swap_chain_support.capabilities_.currentTransform),
        present_mode,
        queue_index);

    swap_chain_info.images = device->getSwapchainImages(swap_chain_info.swap_chain);
}

Format Helper::findDepthFormat(const std::shared_ptr<Device>& device) {
    return vk::helper::findDepthFormat(device);
}

void Helper::addOneTexture(
    std::vector<TextureDescriptor>& descriptor_writes,
    uint32_t binding,
    const std::shared_ptr<Sampler>& sampler,
    const std::shared_ptr<ImageView>& texture,
    const std::shared_ptr<DescriptorSet>& desc_set,
    DescriptorType desc_type/* = DescriptorType::COMBINED_IMAGE_SAMPLER*/,
    ImageLayout image_layout/* = ImageLayout::SHADER_READ_ONLY_OPTIMAL*/) {

    TextureDescriptor tex_desc = {
        binding,
        sampler,
        texture,
        desc_set,
        desc_type,
        image_layout };

    descriptor_writes.push_back(tex_desc);
}

void Helper::addOneBuffer(
    std::vector<BufferDescriptor>& descriptor_writes,
    uint32_t binding,
    const std::shared_ptr<Buffer>& buffer,
    const std::shared_ptr<DescriptorSet>& desc_set,
    DescriptorType desc_type,
    uint32_t range,
    uint32_t offset/* = 0*/) {

    BufferDescriptor buffer_desc = {
        binding,
        offset,
        range,
        buffer,
        desc_set,
        desc_type };

    descriptor_writes.push_back(buffer_desc);
}

void Helper::createBufferWithSrcData(
    const DeviceInfo& device_info,
    const BufferUsageFlags& usage,
    const uint64_t& buffer_size,
    const void* src_data,
    std::shared_ptr<Buffer>& buffer,
    std::shared_ptr<DeviceMemory>& buffer_memory) {
    const auto& device = device_info.device;

    std::shared_ptr<Buffer> staging_buffer;
    std::shared_ptr<DeviceMemory> staging_buffer_memory;
    device->createBuffer(
        buffer_size,
        SET_FLAG_BIT(BufferUsage, TRANSFER_SRC_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        staging_buffer,
        staging_buffer_memory);

    device->updateBufferMemory(staging_buffer_memory, buffer_size, src_data);

    device->createBuffer(
        buffer_size,
        SET_FLAG_BIT(BufferUsage, TRANSFER_DST_BIT) | usage,
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        buffer,
        buffer_memory);

    vk::helper::copyBuffer(device_info, staging_buffer, buffer, buffer_size);

    device->destroyBuffer(staging_buffer);
    device->freeMemory(staging_buffer_memory);
}

void Helper::generateMipmapLevels(
    const std::shared_ptr<CommandBuffer>& cmd_buf,
    const std::shared_ptr<Image>& image,
    uint32_t mip_count,
    uint32_t width,
    uint32_t height,
    const ImageLayout& cur_image_layout) {
    vk::helper::generateMipmapLevels(cmd_buf, image, mip_count, width, height, cur_image_layout);
}

void Helper::create2DTextureImage(
    const DeviceInfo& device_info,
    Format format,
    int tex_width,
    int tex_height,
    int tex_channels,
    const void* pixels,
    std::shared_ptr<Image>& texture_image,
    std::shared_ptr<DeviceMemory>& texture_image_memory) {

    const auto& device = device_info.device;
    VkDeviceSize image_size = static_cast<VkDeviceSize>(tex_width * tex_height * 4);

    std::shared_ptr<Buffer> staging_buffer;
    std::shared_ptr<DeviceMemory> staging_buffer_memory;
    device->createBuffer(
        image_size,
        SET_FLAG_BIT(BufferUsage, TRANSFER_SRC_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        staging_buffer,
        staging_buffer_memory);

    device->updateBufferMemory(staging_buffer_memory, image_size, pixels);

    vk::helper::create2DImage(
        device,
        glm::vec2(tex_width, tex_height),
        format,
        ImageTiling::OPTIMAL,
        SET_FLAG_BIT(ImageUsage, TRANSFER_DST_BIT) |
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        texture_image,
        texture_image_memory);

    vk::helper::transitionImageLayout(device_info, texture_image, format, ImageLayout::UNDEFINED, ImageLayout::TRANSFER_DST_OPTIMAL);
    vk::helper::copyBufferToImage(device_info, staging_buffer, texture_image, glm::uvec2(tex_width, tex_height));
    vk::helper::transitionImageLayout(device_info, texture_image, format, ImageLayout::TRANSFER_DST_OPTIMAL, ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    device->destroyBuffer(staging_buffer);
    device->freeMemory(staging_buffer_memory);
}

void Helper::createDepthResources(
    const DeviceInfo& device_info,
    Format depth_format,
    glm::uvec2 size,
    TextureInfo& depth_buffer) {
    const auto& device = device_info.device;
    vk::helper::create2DImage(
        device,
        size,
        depth_format,
        ImageTiling::OPTIMAL,
        SET_FLAG_BIT(ImageUsage, DEPTH_STENCIL_ATTACHMENT_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        depth_buffer.image,
        depth_buffer.memory);

    depth_buffer.view =
        device->createImageView(
            depth_buffer.image,
            ImageViewType::VIEW_2D,
            depth_format,
            SET_FLAG_BIT(ImageAspect, DEPTH_BIT));

    vk::helper::transitionImageLayout(
        device_info,
        depth_buffer.image,
        depth_format,
        ImageLayout::UNDEFINED,
        ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
}

void Helper::createCubemapTexture(
    const DeviceInfo& device_info,
    const std::shared_ptr<RenderPass>& render_pass,
    uint32_t width,
    uint32_t height,
    uint32_t mip_count,
    Format format,
    const std::vector<BufferImageCopyInfo>& copy_regions,
    TextureInfo& texture,
    uint64_t buffer_size /*= 0*/,
    void* data /*= nullptr*/)
{
    const auto& device = device_info.device;
    bool use_as_framebuffer = data == nullptr;
    VkDeviceSize image_size = static_cast<VkDeviceSize>(buffer_size);

    std::shared_ptr<Buffer> staging_buffer;
    std::shared_ptr<DeviceMemory> staging_buffer_memory;
    if (data) {
        device->createBuffer(
            image_size,
            SET_FLAG_BIT(BufferUsage, TRANSFER_SRC_BIT),
            SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
            SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
            staging_buffer,
            staging_buffer_memory);

        device->updateBufferMemory(staging_buffer_memory, buffer_size, data);
    }

    auto image_usage_flags =
        SET_FLAG_BIT(ImageUsage, TRANSFER_DST_BIT) |
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT);

    if (use_as_framebuffer) {
        image_usage_flags |=
            SET_FLAG_BIT(ImageUsage, COLOR_ATTACHMENT_BIT) |
            SET_FLAG_BIT(ImageUsage, TRANSFER_SRC_BIT);
    }

    texture.image = device->createImage(
        ImageType::TYPE_2D,
        glm::uvec3(width, height, 1),
        format,
        image_usage_flags,
        ImageTiling::OPTIMAL,
        ImageLayout::UNDEFINED,
        SET_FLAG_BIT(ImageCreate, CUBE_COMPATIBLE_BIT),
        false,
        1,
        mip_count,
        6u);

    auto mem_requirements = device->getImageMemoryRequirements(texture.image);
    texture.memory = device->allocateMemory(
        mem_requirements.size,
        mem_requirements.memory_type_bits,
        vk::helper::toVkMemoryPropertyFlags(SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT)));
    device->bindImageMemory(texture.image, texture.memory);

    if (data) {
        vk::helper::transitionImageLayout(device_info, texture.image, format, ImageLayout::UNDEFINED, ImageLayout::TRANSFER_DST_OPTIMAL, 0, mip_count, 0, 6);
        vk::helper::copyBufferToImageWithMips(device_info, staging_buffer, texture.image, copy_regions);
        vk::helper::transitionImageLayout(device_info, texture.image, format, ImageLayout::TRANSFER_DST_OPTIMAL, ImageLayout::SHADER_READ_ONLY_OPTIMAL, 0, mip_count, 0, 6);
    }

    texture.view = device->createImageView(
        texture.image,
        ImageViewType::VIEW_CUBE,
        format,
        SET_FLAG_BIT(ImageAspect, COLOR_BIT),
        0,
        mip_count,
        0,
        6);

    assert(render_pass);

    if (use_as_framebuffer) {
        texture.surface_views.resize(mip_count);
        texture.framebuffers.resize(mip_count);
        auto w = width;
        auto h = height;

        for (uint32_t i = 0; i < mip_count; ++i)
        {
            texture.surface_views[i].resize(6, VK_NULL_HANDLE); //sides of the cube

            for (uint32_t j = 0; j < 6; j++)
            {
                texture.surface_views[i][j] =
                    device->createImageView(
                        texture.image,
                        ImageViewType::VIEW_2D,
                        format,
                        SET_FLAG_BIT(ImageAspect, COLOR_BIT),
                        i,
                        1,
                        j,
                        1);
            }

            texture.framebuffers[i] = device->createFrameBuffer(render_pass, texture.surface_views[i], glm::uvec2(w, h));
            w = std::max(w >> 1, 1u);
            h = std::max(h >> 1, 1u);
        }
    }

    if (data) {
        device->destroyBuffer(staging_buffer);
        device->freeMemory(staging_buffer_memory);
    }
}

bool Helper::acquireNextImage(
    const std::shared_ptr<Device>& device,
    const std::shared_ptr<renderer::Swapchain>& swap_chain,
    const std::shared_ptr<engine::renderer::Semaphore>& semaphore,
    uint32_t& image_index) {
    auto vk_device = RENDER_TYPE_CAST(Device, device);
    auto vk_swap_chain = RENDER_TYPE_CAST(Swapchain, swap_chain);
    auto vk_img_available_semphores = RENDER_TYPE_CAST(Semaphore, semaphore);
    assert(vk_device);

    auto result = vkAcquireNextImageKHR(vk_device->get(), vk_swap_chain->get(), UINT64_MAX, vk_img_available_semphores->get(), VK_NULL_HANDLE, &image_index);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        return true;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }
    return false;
}

void Helper::addImGuiToCommandBuffer(
    const std::shared_ptr<CommandBuffer>& cmd_buf) {
    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();
    ImGui_ImplVulkan_RenderDrawData(draw_data, RENDER_TYPE_CAST(CommandBuffer, cmd_buf)->get());
}

void Helper::submitQueue(
    const std::shared_ptr<Queue>& graphic_queue,
    const std::shared_ptr<Fence>& in_flight_fence,
    const std::vector<std::shared_ptr<Semaphore>>& wait_semaphores,
    const std::vector<std::shared_ptr<CommandBuffer>>& command_buffers,
    const std::vector<std::shared_ptr<Semaphore>>& signal_semaphores) {

    std::vector<VkSemaphore> vk_wait_semaphores(wait_semaphores.size());
    for (auto i = 0; i < wait_semaphores.size(); i++) {
        vk_wait_semaphores[i] = RENDER_TYPE_CAST(Semaphore, wait_semaphores[i])->get();
    }

    std::vector<VkCommandBuffer> vk_cmd_bufs(command_buffers.size());
    for (auto i = 0; i < command_buffers.size(); i++) {
        vk_cmd_bufs[i] = RENDER_TYPE_CAST(CommandBuffer, command_buffers[i])->get();
    }

    std::vector<VkSemaphore> vk_signal_semaphores(signal_semaphores.size());
    for (auto i = 0; i < signal_semaphores.size(); i++) {
        vk_signal_semaphores[i] = RENDER_TYPE_CAST(Semaphore, signal_semaphores[i])->get();
    }
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submit_info.waitSemaphoreCount = static_cast<uint32_t>(vk_wait_semaphores.size());
    submit_info.pWaitSemaphores = vk_wait_semaphores.data();
    submit_info.pWaitDstStageMask = wait_stages;
    submit_info.commandBufferCount = static_cast<uint32_t>(vk_cmd_bufs.size());
    submit_info.pCommandBuffers = vk_cmd_bufs.data();
    submit_info.signalSemaphoreCount = static_cast<uint32_t>(vk_signal_semaphores.size());
    submit_info.pSignalSemaphores = vk_signal_semaphores.data();

    auto vk_graphic_queue = RENDER_TYPE_CAST(Queue, graphic_queue);
    auto vk_in_flight_fence = RENDER_TYPE_CAST(Fence, in_flight_fence);
    if (vkQueueSubmit(vk_graphic_queue->get(), 1, &submit_info, vk_in_flight_fence->get()) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }
}

bool Helper::presentQueue(
    const std::shared_ptr<Queue>& present_queue,
    const std::vector<std::shared_ptr<Swapchain>>& swap_chains,
    const std::vector<std::shared_ptr<Semaphore>>& wait_semaphores,
    const uint32_t& image_index,
    bool& frame_buffer_resized) {
  
    std::vector<VkSemaphore> vk_wait_semaphores(wait_semaphores.size());
    for (auto i = 0; i < wait_semaphores.size(); i++) {
        vk_wait_semaphores[i] = RENDER_TYPE_CAST(Semaphore, wait_semaphores[i])->get();
    }

    std::vector<VkSwapchainKHR> vk_swap_chains(swap_chains.size());
    for (auto i = 0; i < swap_chains.size(); i++) {
        vk_swap_chains[i] = RENDER_TYPE_CAST(Swapchain, swap_chains[i])->get();
    }

    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = static_cast<uint32_t>(vk_wait_semaphores.size());
    present_info.pWaitSemaphores = vk_wait_semaphores.data();
    present_info.swapchainCount = static_cast<uint32_t>(vk_swap_chains.size());
    present_info.pSwapchains = vk_swap_chains.data();
    present_info.pImageIndices = &image_index;
    present_info.pResults = nullptr; // Optional

    auto vk_present_queue = RENDER_TYPE_CAST(Queue, present_queue);
    auto result = vkQueuePresentKHR(vk_present_queue->get(), &present_info);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || frame_buffer_resized) {
        frame_buffer_resized = false;
        return true;
    }
    else if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image!");
    }

    return false;
}

void Helper::initImgui(
    const DeviceInfo& device_info,
    const std::shared_ptr<Instance>& instance,
    GLFWwindow* window,
    const QueueFamilyIndices& queue_family_indices,
    const SwapChainInfo& swap_chain_info,
    std::shared_ptr<Queue> graphics_queue,
    std::shared_ptr<DescriptorPool> descriptor_pool,
    std::shared_ptr<RenderPass> render_pass,
    std::shared_ptr<CommandBuffer> command_buffer) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    auto logic_device = RENDER_TYPE_CAST(Device, device_info.device);
    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = RENDER_TYPE_CAST(Instance, instance)->get();
    init_info.PhysicalDevice = RENDER_TYPE_CAST(PhysicalDevice, logic_device->getPhysicalDevice())->get();
    init_info.Device = logic_device->get();
    init_info.QueueFamily = queue_family_indices.graphics_family_.value();
    init_info.Queue = RENDER_TYPE_CAST(Queue, graphics_queue)->get();
    init_info.PipelineCache = nullptr;// g_PipelineCache;
    init_info.DescriptorPool = RENDER_TYPE_CAST(DescriptorPool, descriptor_pool)->get();
    init_info.Allocator = nullptr; // g_Allocator;
    init_info.MinImageCount = static_cast<uint32_t>(swap_chain_info.framebuffers.size());
    init_info.ImageCount = static_cast<uint32_t>(swap_chain_info.framebuffers.size());
    init_info.CheckVkResultFn = check_vk_result;
    ImGui_ImplVulkan_Init(&init_info, RENDER_TYPE_CAST(RenderPass, render_pass)->get());

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Read 'docs/FONTS.md' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    //io.Fonts->AddFontDefault();
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
    //ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
    //IM_ASSERT(font != NULL);

    // Upload Fonts
    {
        // Use any command queue
        //VkCommandPool command_pool = wd->Frames[wd->FrameIndex].CommandPool;
        auto cmd_buf = RENDER_TYPE_CAST(CommandBuffer, command_buffer)->get();

        //auto err = vkResetCommandPool(init_info.Device, command_pool, 0);
        //check_vk_result(err);
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        auto err = vkBeginCommandBuffer(cmd_buf, &begin_info);
        check_vk_result(err);

        ImGui_ImplVulkan_CreateFontsTexture(cmd_buf);

        VkSubmitInfo end_info = {};
        end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        end_info.commandBufferCount = 1;
        end_info.pCommandBuffers = &cmd_buf;
        err = vkEndCommandBuffer(cmd_buf);
        check_vk_result(err);
        err = vkQueueSubmit(init_info.Queue, 1, &end_info, VK_NULL_HANDLE);
        check_vk_result(err);

        err = vkDeviceWaitIdle(init_info.Device);
        check_vk_result(err);
        ImGui_ImplVulkan_DestroyFontUploadObjects();
    }
}

void TextureInfo::destroy(const std::shared_ptr<Device>& device) {
    device->destroyImage(image);
    device->destroyImageView(view);
    device->freeMemory(memory);

    for (auto& s_views : surface_views) {
        for (auto& s_view : s_views) {
            device->destroyImageView(s_view);
        }
    }

    for (auto& framebuffer : framebuffers) {
        device->destroyFramebuffer(framebuffer);
    }
}

void BufferInfo::destroy(const std::shared_ptr<Device>& device) {
    device->destroyBuffer(buffer);
    device->freeMemory(memory);
}

} // namespace renderer
} // namespace engine