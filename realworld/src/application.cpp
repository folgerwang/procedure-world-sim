#include <iostream>
#include <vector>
#include <map>
#include <limits>
#include <chrono>
#include <filesystem>

#include "engine/renderer/renderer.h"
#include "engine/renderer/renderer_helper.h"
#include "engine/engine_helper.h"
#include "application.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"
#include "engine/tiny_mtx2.h"

namespace er = engine::renderer;
namespace ego = engine::game_object;

namespace {
constexpr int kWindowSizeX = 1920;
constexpr int kWindowSizeY = 1080;
static float s_sun_angle = 0.0f;
static int s_update_frame_count = -1;

struct SkyBoxVertex {
    glm::vec3 pos;

    static std::vector<er::VertexInputBindingDescription> getBindingDescription() {
        std::vector<er::VertexInputBindingDescription> binding_description(1);
        binding_description[0].binding = 0;
        binding_description[0].stride = sizeof(SkyBoxVertex);
        binding_description[0].input_rate = er::VertexInputRate::VERTEX;
        return binding_description;
    }

    static std::vector<er::VertexInputAttributeDescription> getAttributeDescriptions() {
        std::vector<er::VertexInputAttributeDescription> attribute_descriptions(1);
        attribute_descriptions[0].binding = 0;
        attribute_descriptions[0].location = 0;
        attribute_descriptions[0].format = er::Format::R32G32B32_SFLOAT;
        attribute_descriptions[0].offset = offsetof(SkyBoxVertex, pos);
        return attribute_descriptions;
    }
};

er::AttachmentDescription FillAttachmentDescription(
    er::Format format,
    er::SampleCountFlagBits samples,
    er::ImageLayout initial_layout,
    er::ImageLayout final_layout,
    er::AttachmentLoadOp load_op = er::AttachmentLoadOp::CLEAR,
    er::AttachmentStoreOp store_op = er::AttachmentStoreOp::STORE,
    er::AttachmentLoadOp stencil_load_op = er::AttachmentLoadOp::DONT_CARE,
    er::AttachmentStoreOp stencil_store_op = er::AttachmentStoreOp::DONT_CARE) {

    er::AttachmentDescription attachment{};
    attachment.format = format;
    attachment.samples = samples;
    attachment.initial_layout = initial_layout;
    attachment.final_layout = final_layout;
    attachment.load_op = load_op;
    attachment.store_op = store_op;
    attachment.stencil_load_op = stencil_load_op;
    attachment.stencil_store_op = stencil_store_op;

    return attachment;
}

er::SubpassDescription FillSubpassDescription(
    er::PipelineBindPoint pipeline_bind_point,
    const std::vector<er::AttachmentReference>& color_attachments,
    const er::AttachmentReference* depth_stencil_attachment,
    er::SubpassDescriptionFlags flags = static_cast<er::SubpassDescriptionFlags>(0),
    const std::vector<er::AttachmentReference>& input_attachments = {},
    const std::vector<er::AttachmentReference>& resolve_attachments = {}) {
    er::SubpassDescription desc{};
    desc.flags = flags;
    desc.input_attachments = input_attachments;
    desc.color_attachments = color_attachments;
    desc.resolve_attachments = resolve_attachments;
    if (depth_stencil_attachment) {
        desc.depth_stencil_attachment.resize(1);
        desc.depth_stencil_attachment[0] = *depth_stencil_attachment;
    }
    desc.preserve_attachment_count = 0;
    desc.preserve_attachments = nullptr;

    return desc;
}

er::SubpassDependency FillSubpassDependency(
    uint32_t src_subpass,
    uint32_t dst_subpass,
    er::PipelineStageFlags src_stage_mask,
    er::PipelineStageFlags dst_stage_mask,
    er::AccessFlags src_access_mask,
    er::AccessFlags dst_access_mask,
    er::DependencyFlags dependency_flags = 0) {
    er::SubpassDependency dependency{};
    dependency.src_subpass = src_subpass;
    dependency.dst_subpass = dst_subpass;
    dependency.src_stage_mask = src_stage_mask;
    dependency.dst_stage_mask = dst_stage_mask;
    dependency.src_access_mask = src_access_mask;
    dependency.dst_access_mask = dst_access_mask;
    dependency.dependency_flags = dependency_flags;
    return dependency;
}

std::shared_ptr<er::RenderPass> createRenderPass(
    std::shared_ptr<er::Device> device,
    er::Format format,
    er::Format depth_format,
    er::SampleCountFlagBits sample_count = er::SampleCountFlagBits::SC_1_BIT,
    er::ImageLayout color_image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) {
    auto color_attachment = FillAttachmentDescription(
        format,
        sample_count,
        er::ImageLayout::UNDEFINED,
        color_image_layout);

    er::AttachmentReference color_attachment_ref(0, er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    auto depth_attachment = FillAttachmentDescription(
        depth_format,
        sample_count,
        er::ImageLayout::UNDEFINED,
        er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    er::AttachmentReference depth_attachment_ref(1, er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    auto subpass = FillSubpassDescription(
        er::PipelineBindPoint::GRAPHICS,
        { color_attachment_ref },
        &depth_attachment_ref);

    auto depency = FillSubpassDependency(~0U, 0,
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT),
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT),
        0,
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT) |
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_READ_BIT));

    std::vector<er::AttachmentDescription> attachments(2);
    attachments[0] = color_attachment;
    attachments[1] = depth_attachment;

    return device->createRenderPass(attachments, { subpass }, { depency });
}

std::shared_ptr<er::RenderPass> createWaterRenderPass(
    std::shared_ptr<er::Device> device,
    er::Format format,
    er::Format depth_format,
    er::SampleCountFlagBits sample_count = er::SampleCountFlagBits::SC_1_BIT,
    er::ImageLayout color_image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) {
    auto color_attachment = FillAttachmentDescription(
        format,
        sample_count,
        color_image_layout,
        color_image_layout,
        er::AttachmentLoadOp::LOAD);

    er::AttachmentReference color_attachment_ref(0, er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    auto depth_attachment = FillAttachmentDescription(
        depth_format,
        sample_count,
        er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        er::AttachmentLoadOp::LOAD);

    er::AttachmentReference depth_attachment_ref(1, er::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL);

    auto subpass = FillSubpassDescription(
        er::PipelineBindPoint::GRAPHICS,
        { color_attachment_ref },
        &depth_attachment_ref,
        static_cast<er::SubpassDescriptionFlags>(0));

    auto depency = FillSubpassDependency(~0U, 0,
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT),
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT),
        0,
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT) |
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_READ_BIT));

    std::vector<er::AttachmentDescription> attachments(2);
    attachments[0] = color_attachment;
    attachments[1] = depth_attachment;

    return device->createRenderPass(attachments, { subpass }, { depency });
}
}

namespace work {
namespace app {

void RealWorldApplication::run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<work::app::RealWorldApplication*>(glfwGetWindowUserPointer(window));
    app->setFrameBufferResized(true);
}

static bool s_exit_game = false;
static bool s_game_paused = false;
static bool s_camera_paused = false;
static bool s_mouse_init = false;
static bool s_mouse_right_button_pressed = false;
static glm::vec2 s_last_mouse_pos;
static float s_yaw = 0.0f;
static float s_pitch = 0.0f;
const float s_camera_speed = 0.2f;
static glm::vec3 s_camera_pos = glm::vec3(0, -100.0f, 0);
static glm::vec3 s_camera_dir = glm::normalize(glm::vec3(1.0f, 0.0f, 0));
static glm::vec3 s_camera_up = glm::vec3(0, 1, 0);

static void keyInputCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    auto up_vector = abs(s_camera_dir[1]) < glm::min(abs(s_camera_dir[0]), abs(s_camera_dir[2])) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    auto camera_right = glm::normalize(glm::cross(s_camera_dir, s_camera_up));
    if (action != GLFW_RELEASE && !s_camera_paused) {
        if (key == GLFW_KEY_W)
            s_camera_pos += s_camera_speed * s_camera_dir;
        if (key == GLFW_KEY_S)
            s_camera_pos -= s_camera_speed * s_camera_dir;
        if (key == GLFW_KEY_A)
            s_camera_pos -= s_camera_speed * camera_right;
        if (key == GLFW_KEY_D)
            s_camera_pos += s_camera_speed * camera_right;
    }

    if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE) {
        s_exit_game = true;
    }

    if (action == GLFW_PRESS && key == GLFW_KEY_SPACE) {
        s_camera_paused = !s_camera_paused;
    }
}

void mouseInputCallback(GLFWwindow* window, double xpos, double ypos)
{
    glm::vec2 cur_mouse_pos = glm::vec2(xpos, ypos);
    if (!s_mouse_init)
    {
        s_last_mouse_pos = cur_mouse_pos;
        s_mouse_init = true;
    }

    auto mouse_offset = cur_mouse_pos - s_last_mouse_pos;
    s_last_mouse_pos = cur_mouse_pos;

    float sensitivity = 0.2f;
    mouse_offset *= sensitivity;

    if (!s_camera_paused && s_mouse_right_button_pressed) {
        s_yaw += mouse_offset.x;
        s_pitch += mouse_offset.y;

        if (s_pitch > 89.0f)
            s_pitch = 89.0f;
        if (s_pitch < -89.0f)
            s_pitch = -89.0f;

        glm::vec3 direction;
        direction.x = cos(glm::radians(-s_yaw)) * cos(glm::radians(s_pitch));
        direction.y = sin(glm::radians(s_pitch));
        direction.z = sin(glm::radians(-s_yaw)) * cos(glm::radians(s_pitch));
        s_camera_dir = glm::normalize(direction);
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        if (action == GLFW_PRESS) {
            s_mouse_right_button_pressed = true;
        }
        else if (action == GLFW_RELEASE) {
            s_mouse_right_button_pressed = false;
        }
    }
}

void RealWorldApplication::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(kWindowSizeX, kWindowSizeY, "Real World", nullptr, nullptr);
    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
    glfwSetKeyCallback(window_, keyInputCallback);
    glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetCursorPosCallback(window_, mouseInputCallback);
    glfwSetMouseButtonCallback(window_, mouseButtonCallback);
}

void RealWorldApplication::createDepthResources(const glm::uvec2& display_size) {
    auto depth_format = er::Helper::findDepthFormat(device_info_.device);
    er::Helper::createDepthResources(
        device_info_,
        depth_format,
        display_size,
        depth_buffer_);

    er::Helper::create2DTextureImage(
        device_info_,
        er::Format::D32_SFLOAT,
        display_size,
        depth_buffer_copy_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, DEPTH_STENCIL_ATTACHMENT_BIT) |
        SET_FLAG_BIT(ImageUsage, TRANSFER_DST_BIT),
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
}

void RealWorldApplication::createHdrColorBuffer(const glm::uvec2& display_size) {
    er::Helper::create2DTextureImage(
        device_info_,
        hdr_format_,
        display_size,
        hdr_color_buffer_,
        SET_FLAG_BIT(ImageUsage, COLOR_ATTACHMENT_BIT) |
        SET_FLAG_BIT(ImageUsage, TRANSFER_SRC_BIT),
        er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
}

void RealWorldApplication::createColorBufferCopy(const glm::uvec2& display_size) {
    er::Helper::create2DTextureImage(
        device_info_,
        hdr_format_,
        display_size,
        hdr_color_buffer_copy_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, TRANSFER_DST_BIT),
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
}

void RealWorldApplication::recreateRenderBuffer(const glm::uvec2& display_size) {
    createDepthResources(display_size);
    createHdrColorBuffer(display_size);
    createColorBufferCopy(display_size);
    createFramebuffers(display_size);
}

void RealWorldApplication::createRenderPasses() {
    assert(device_);
    render_pass_ = createRenderPass(
        device_,
        swap_chain_info_.format,
        depth_format_,
        er::SampleCountFlagBits::SC_1_BIT,
        er::ImageLayout::PRESENT_SRC_KHR);
    hdr_render_pass_ = createRenderPass(device_, hdr_format_, depth_format_);
    hdr_water_render_pass_ = createWaterRenderPass(device_, hdr_format_, depth_format_);
}

void RealWorldApplication::initVulkan() {
    static auto color_blend_attachment = er::helper::fillPipelineColorBlendAttachmentState();
    static std::vector<er::PipelineColorBlendAttachmentState> color_blend_attachments(1, color_blend_attachment);
    static std::vector<er::PipelineColorBlendAttachmentState> cube_color_blend_attachments(6, color_blend_attachment);

    graphic_pipeline_info_.blend_state_info = 
        std::make_shared<er::PipelineColorBlendStateCreateInfo>(
            er::helper::fillPipelineColorBlendStateCreateInfo(color_blend_attachments));
    graphic_pipeline_info_.rasterization_info = 
        std::make_shared<er::PipelineRasterizationStateCreateInfo>(
            er::helper::fillPipelineRasterizationStateCreateInfo());
    graphic_pipeline_info_.ms_info =
        std::make_shared<er::PipelineMultisampleStateCreateInfo>(
            er::helper::fillPipelineMultisampleStateCreateInfo());
    graphic_pipeline_info_.depth_stencil_info =
        std::make_shared<er::PipelineDepthStencilStateCreateInfo>(
            er::helper::fillPipelineDepthStencilStateCreateInfo());

    graphic_cubemap_pipeline_info_.blend_state_info =
        std::make_shared<er::PipelineColorBlendStateCreateInfo>(
            er::helper::fillPipelineColorBlendStateCreateInfo(cube_color_blend_attachments));
    graphic_cubemap_pipeline_info_.rasterization_info =
        std::make_shared<er::PipelineRasterizationStateCreateInfo>(
            er::helper::fillPipelineRasterizationStateCreateInfo(
                false, false, er::PolygonMode::FILL,
                SET_FLAG_BIT(CullMode, NONE)));
    graphic_cubemap_pipeline_info_.ms_info =
        std::make_shared<er::PipelineMultisampleStateCreateInfo>(
            er::helper::fillPipelineMultisampleStateCreateInfo());
    graphic_cubemap_pipeline_info_.depth_stencil_info =
        std::make_shared<er::PipelineDepthStencilStateCreateInfo>(
            er::helper::fillPipelineDepthStencilStateCreateInfo(
                false, false, er::CompareOp::ALWAYS, false));

    // the initialization order has to be strict.
    instance_ = er::Helper::createInstance();
    physical_devices_ = er::Helper::collectPhysicalDevices(instance_);
    surface_ = er::Helper::createSurface(instance_, window_);
    physical_device_ = er::Helper::pickPhysicalDevice(physical_devices_, surface_);
    queue_indices_ = er::Helper::findQueueFamilies(physical_device_, surface_);
    device_ = er::Helper::createLogicalDevice(physical_device_, surface_, queue_indices_);
    assert(device_);
    depth_format_ = er::Helper::findDepthFormat(device_);
    device_info_.device = device_;
    graphics_queue_ = device_->getDeviceQueue(queue_indices_.graphics_family_.value());
    assert(graphics_queue_);
    device_info_.cmd_queue = graphics_queue_;
    present_queue_ = device_->getDeviceQueue(queue_indices_.present_family_.value());
    er::Helper::createSwapChain(
        window_,
        device_,
        surface_,
        queue_indices_,
        swap_chain_info_,
        SET_FLAG_BIT(ImageUsage, COLOR_ATTACHMENT_BIT)|
        SET_FLAG_BIT(ImageUsage, TRANSFER_DST_BIT));
    createRenderPasses();
    createImageViews();
    createCubemapRenderPass();
    createCubemapFramebuffers();
    createDescriptorSetLayout();
    createCommandPool();
    assert(command_pool_);
    device_info_.cmd_pool = command_pool_;
    er::Helper::init(device_info_);

    loadMtx2Texture("assets/environments/doge2/lambertian/diffuse.ktx2", ibl_diffuse_tex_);
    loadMtx2Texture("assets/environments/doge2/ggx/specular.ktx2", ibl_specular_tex_);
    loadMtx2Texture("assets/environments/doge2/charlie/sheen.ktx2", ibl_sheen_tex_);
    createGraphicPipelineLayout();
    createCubemapPipelineLayout();
    createCubeSkyboxPipelineLayout();
    createCubemapComputePipelineLayout();
    createGraphicsPipeline(swap_chain_info_.extent);
    createCubeGraphicsPipeline();
    createComputePipeline();
    recreateRenderBuffer(swap_chain_info_.extent);
    auto format = er::Format::R8G8B8A8_UNORM;
    createTextureImage("assets/statue.jpg", format, sample_tex_);
    createTextureImage("assets/brdfLUT.png", format, brdf_lut_tex_);
    createTextureImage("assets/lut_ggx.png", format, ggx_lut_tex_);
    createTextureImage("assets/lut_charlie.png", format, charlie_lut_tex_);
    createTextureImage("assets/lut_thin_film.png", format, thin_film_lut_tex_);
    createTextureImage("assets/environments/doge2.hdr", format, panorama_tex_);
    createTextureSampler();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    descriptor_pool_ = device_->createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();

    for (auto& image : swap_chain_info_.images) {
        er::Helper::transitionImageLayout(
            device_info_,
            image,
            swap_chain_info_.format,
            er::ImageLayout::UNDEFINED,
            er::ImageLayout::PRESENT_SRC_KHR);
    }

    auto desc_set_layouts = { global_tex_desc_set_layout_, view_desc_set_layout_ };
    ego::TileObject::initStaticMembers(
        device_,
        hdr_render_pass_,
        hdr_water_render_pass_,
        graphic_pipeline_info_,
        desc_set_layouts,
        swap_chain_info_.extent);
    ego::GltfObject::initStaticMembers(
        device_,
        descriptor_pool_,
        desc_set_layouts);
    
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            auto tile_obj = std::make_shared<ego::TileObject>(
                device_info_,
                descriptor_pool_,
                glm::uvec2(256, 256),
                glm::vec2(-256.0f + x * 512, -256.0f + y * 512),
                glm::vec2(256.0f + x * 512, 256.0f + y * 512));

            tile_objects_.push_back(tile_obj);
        }
    }

    ego::TileObject::updateStaticDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        hdr_color_buffer_copy_.view,
        depth_buffer_copy_.view);

    std::string path = "assets";
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        auto path_string = entry.path();
        auto ext_string = std::filesystem::path(path_string).extension();
        if (ext_string == ".glb" || ext_string == ".gltf") {
            gltf_file_names_.push_back(path_string.filename().string());
        }
    }

    er::Helper::initImgui(
        device_info_,
        instance_,
        window_, 
        queue_indices_,
        swap_chain_info_,
        graphics_queue_,
        descriptor_pool_,
        render_pass_,
        command_buffers_[0]);
}

void RealWorldApplication::recreateSwapChain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window_, &width, &height);
        glfwWaitEvents();
    }

    device_->waitIdle();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    cleanupSwapChain();

    er::Helper::createSwapChain(
        window_,
        device_,
        surface_,
        queue_indices_,
        swap_chain_info_,
        SET_FLAG_BIT(ImageUsage, COLOR_ATTACHMENT_BIT) |
        SET_FLAG_BIT(ImageUsage, TRANSFER_DST_BIT));

    createRenderPasses();
    createImageViews();
    createGraphicPipelineLayout();
    createCubemapPipelineLayout();
    createCubemapComputePipelineLayout();
    auto desc_set_layouts = { global_tex_desc_set_layout_, view_desc_set_layout_ };
    ego::TileObject::recreateStaticMembers(
        device_,
        hdr_render_pass_,
        hdr_water_render_pass_,
        graphic_pipeline_info_,
        desc_set_layouts,
        swap_chain_info_.extent);
    ego::GltfObject::recreateStaticMembers(
        device_,
        hdr_render_pass_,
        graphic_pipeline_info_,
        desc_set_layouts,
        swap_chain_info_.extent);
    createCubeSkyboxPipelineLayout();
    createGraphicsPipeline(swap_chain_info_.extent);
    createComputePipeline();
    recreateRenderBuffer(swap_chain_info_.extent);
    createUniformBuffers();
    descriptor_pool_ = device_->createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();

    for (auto& image : swap_chain_info_.images) {
        er::Helper::transitionImageLayout(
            device_info_,
            image,
            swap_chain_info_.format,
            er::ImageLayout::UNDEFINED,
            er::ImageLayout::PRESENT_SRC_KHR);
    }

    for (auto& tile_obj : tile_objects_) {
        tile_obj->generateDescriptorSet(descriptor_pool_);
    }

    ego::GltfObject::generateDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        thin_film_lut_tex_);

    ego::TileObject::updateStaticDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        hdr_color_buffer_copy_.view,
        depth_buffer_copy_.view);

    er::Helper::initImgui(
        device_info_,
        instance_,
        window_,
        queue_indices_,
        swap_chain_info_,
        graphics_queue_,
        descriptor_pool_,
        render_pass_,
        command_buffers_[0]);
}

void RealWorldApplication::createImageViews() {
    swap_chain_info_.image_views.resize(swap_chain_info_.images.size());
    for (uint64_t i_img = 0; i_img < swap_chain_info_.images.size(); i_img++) {
        swap_chain_info_.image_views[i_img] = device_->createImageView(
            swap_chain_info_.images[i_img],
            er::ImageViewType::VIEW_2D,
            swap_chain_info_.format,
            SET_FLAG_BIT(ImageAspect, COLOR_BIT));
    }
}

void RealWorldApplication::createCubemapFramebuffers() {
    uint32_t num_mips = static_cast<uint32_t>(std::log2(kCubemapSize) + 1);
    std::vector<er::BufferImageCopyInfo> dump_copies;

    er::Helper::createCubemapTexture(
        device_info_,
        cubemap_render_pass_,
        kCubemapSize,
        kCubemapSize,
        num_mips,
        er::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_envmap_tex_);

    er::Helper::createCubemapTexture(
        device_info_,
        cubemap_render_pass_,
        kCubemapSize,
        kCubemapSize,
        1,
        er::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        tmp_ibl_diffuse_tex_);

    er::Helper::createCubemapTexture(
        device_info_,
        cubemap_render_pass_,
        kCubemapSize,
        kCubemapSize,
        num_mips,
        er::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        tmp_ibl_specular_tex_);

    er::Helper::createCubemapTexture(
        device_info_,
        cubemap_render_pass_,
        kCubemapSize,
        kCubemapSize,
        num_mips,
        er::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        tmp_ibl_sheen_tex_);

    er::Helper::createCubemapTexture(
        device_info_,
        cubemap_render_pass_,
        kCubemapSize,
        kCubemapSize,
        1,
        er::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_ibl_diffuse_tex_);

    er::Helper::createCubemapTexture(
        device_info_,
        cubemap_render_pass_,
        kCubemapSize,
        kCubemapSize,
        num_mips,
        er::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_ibl_specular_tex_);

    er::Helper::createCubemapTexture(
        device_info_,
        cubemap_render_pass_,
        kCubemapSize,
        kCubemapSize,
        num_mips,
        er::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_ibl_sheen_tex_);
}

er::ShaderModuleList getSkyboxShaderModules(
    std::shared_ptr<er::Device> device)
{
    uint64_t vert_code_size, frag_code_size;
    er::ShaderModuleList shader_modules(2);
    auto vert_shader_code = engine::helper::readFile("lib/shaders/skybox_vert.spv", vert_code_size);
    auto frag_shader_code = engine::helper::readFile("lib/shaders/skybox_frag.spv", frag_code_size);

    shader_modules[0] = device->createShaderModule(vert_code_size, vert_shader_code.data());
    shader_modules[1] = device->createShaderModule(frag_code_size, frag_shader_code.data());

    return shader_modules;
}

er::ShaderModuleList getIblShaderModules(
    std::shared_ptr<er::Device> device)
{
    uint64_t vert_code_size, frag_code_size;
    er::ShaderModuleList shader_modules;
    shader_modules.reserve(7);
    auto vert_shader_code = engine::helper::readFile("lib/shaders/ibl_vert.spv", vert_code_size);
    shader_modules.push_back(device->createShaderModule(vert_code_size, vert_shader_code.data()));
    auto frag_shader_code = engine::helper::readFile("lib/shaders/panorama_to_cubemap_frag.spv", frag_code_size);
    shader_modules.push_back(device->createShaderModule(frag_code_size, frag_shader_code.data()));
    auto labertian_frag_shader_code = engine::helper::readFile("lib/shaders/ibl_labertian_frag.spv", frag_code_size);
    shader_modules.push_back(device->createShaderModule(frag_code_size, labertian_frag_shader_code.data()));
    auto ggx_frag_shader_code = engine::helper::readFile("lib/shaders/ibl_ggx_frag.spv", frag_code_size);
    shader_modules.push_back(device->createShaderModule(frag_code_size, ggx_frag_shader_code.data()));
    auto charlie_frag_shader_code = engine::helper::readFile("lib/shaders/ibl_charlie_frag.spv", frag_code_size);
    shader_modules.push_back(device->createShaderModule(frag_code_size, charlie_frag_shader_code.data()));
    auto cube_skybox_shader_code = engine::helper::readFile("lib/shaders/cube_skybox.spv", frag_code_size);
    shader_modules.push_back(device->createShaderModule(frag_code_size, cube_skybox_shader_code.data()));

    return shader_modules;
}

er::ShaderModuleList getIblComputeShaderModules(
    std::shared_ptr<er::Device> device)
{
    uint64_t compute_code_size;
    er::ShaderModuleList shader_modules;
    shader_modules.reserve(1);
    auto compute_shader_code = engine::helper::readFile("lib/shaders/ibl_smooth_comp.spv", compute_code_size);
    shader_modules.push_back(device->createShaderModule(compute_code_size, compute_shader_code.data()));

    return shader_modules;
}

void RealWorldApplication::createGraphicPipelineLayout()
{
    {
        er::PushConstantRange push_const_range{};
        push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
        push_const_range.offset = 0;
        push_const_range.size = sizeof(glsl::SunSkyParams);

        er::DescriptorSetLayoutList desc_set_layouts;
        desc_set_layouts.reserve(2);
        desc_set_layouts.push_back(skybox_desc_set_layout_);
        desc_set_layouts.push_back(view_desc_set_layout_);

        skybox_pipeline_layout_ = device_->createPipelineLayout(desc_set_layouts, { push_const_range });
    }
}

void RealWorldApplication::createCubemapPipelineLayout()
{
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::IblParams);

    er::DescriptorSetLayoutList desc_set_layouts(1);
    desc_set_layouts[0] = ibl_desc_set_layout_;

    ibl_pipeline_layout_ = device_->createPipelineLayout(desc_set_layouts, { push_const_range });
}

void RealWorldApplication::createCubeSkyboxPipelineLayout()
{
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::SunSkyParams);

    er::DescriptorSetLayoutList desc_set_layouts(1);
    desc_set_layouts[0] = ibl_desc_set_layout_;

    cube_skybox_pipeline_layout_ = device_->createPipelineLayout(desc_set_layouts, { push_const_range });
}

void RealWorldApplication::createCubemapComputePipelineLayout()
{
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::IblComputeParams);

    er::DescriptorSetLayoutList desc_set_layouts(1);
    desc_set_layouts[0] = ibl_comp_desc_set_layout_;

    ibl_comp_pipeline_layout_ = device_->createPipelineLayout(desc_set_layouts, { push_const_range });
}

void RealWorldApplication::createGraphicsPipeline(const glm::uvec2& display_size) {
#if 0
    VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_LINE_WIDTH
    };

    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = 2;
    dynamic_state.pDynamicStates = dynamic_states;
#endif
    er::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = er::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;
    {
        auto shader_modules = getSkyboxShaderModules(device_);
        skybox_pipeline_ = device_->createPipeline(
            hdr_render_pass_,
            skybox_pipeline_layout_,
            SkyBoxVertex::getBindingDescription(),
            SkyBoxVertex::getAttributeDescriptions(),
            input_assembly,
            graphic_pipeline_info_,
            shader_modules,
            display_size);

        for (auto& shader_module : shader_modules) {
            device_->destroyShaderModule(shader_module);
        }
    }
}

void RealWorldApplication::createCubeGraphicsPipeline() {
    er::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = er::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;
    {
        auto ibl_shader_modules = getIblShaderModules(device_);
        cube_skybox_pipeline_ = device_->createPipeline(
            cubemap_render_pass_,
            cube_skybox_pipeline_layout_,
            {}, {},
            input_assembly,
            graphic_cubemap_pipeline_info_,
            { ibl_shader_modules[0], ibl_shader_modules[5] },
            glm::uvec2(kCubemapSize, kCubemapSize));

        envmap_pipeline_ = device_->createPipeline(
            cubemap_render_pass_,
            ibl_pipeline_layout_,
            {}, {},
            input_assembly,
            graphic_cubemap_pipeline_info_,
            { ibl_shader_modules[0], ibl_shader_modules[1] },
            glm::uvec2(kCubemapSize, kCubemapSize));

        lambertian_pipeline_ = device_->createPipeline(
            cubemap_render_pass_,
            ibl_pipeline_layout_,
            {}, {},
            input_assembly,
            graphic_cubemap_pipeline_info_,
            { ibl_shader_modules[0], ibl_shader_modules[2] },
            glm::uvec2(kCubemapSize, kCubemapSize));

        ggx_pipeline_ = device_->createPipeline(
            cubemap_render_pass_,
            ibl_pipeline_layout_,
            {}, {},
            input_assembly,
            graphic_cubemap_pipeline_info_,
            { ibl_shader_modules[0], ibl_shader_modules[3] },
            glm::uvec2(kCubemapSize, kCubemapSize));

        charlie_pipeline_ = device_->createPipeline(
            cubemap_render_pass_,
            ibl_pipeline_layout_,
            {}, {},
            input_assembly,
            graphic_cubemap_pipeline_info_,
            { ibl_shader_modules[0], ibl_shader_modules[4] },
            glm::uvec2(kCubemapSize, kCubemapSize));

        for (auto& shader_module : ibl_shader_modules) {
            device_->destroyShaderModule(shader_module);
        }
    }
}

void RealWorldApplication::createComputePipeline()
{
    auto ibl_compute_shader_modules = getIblComputeShaderModules(device_);
    assert(ibl_compute_shader_modules.size() == 1);

    blur_comp_pipeline_ = device_->createPipeline(
        ibl_comp_pipeline_layout_,
        ibl_compute_shader_modules[0]);

    for (auto& shader_module : ibl_compute_shader_modules) {
        device_->destroyShaderModule(shader_module);
    }
}

void RealWorldApplication::createCubemapRenderPass() {
    auto color_attachment = FillAttachmentDescription(
        er::Format::R16G16B16A16_SFLOAT,
        er::SampleCountFlagBits::SC_1_BIT,
        er::ImageLayout::UNDEFINED,
        er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    std::vector<er::AttachmentReference> color_attachment_refs(6);
    for (uint32_t i = 0; i < 6; i++) {
        color_attachment_refs[i].attachment_ = i;
        color_attachment_refs[i].layout_ = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    }

    auto subpass = FillSubpassDescription(
        er::PipelineBindPoint::GRAPHICS,
        color_attachment_refs,
        nullptr);

    auto depency = FillSubpassDependency(~0U, 0,
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT),
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT),
        0,
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT));

    std::vector<er::AttachmentDescription> attachments = {6, color_attachment};

    cubemap_render_pass_ = device_->createRenderPass(attachments, { subpass }, { depency });
}

void RealWorldApplication::createFramebuffers(const glm::uvec2& display_size) {
    swap_chain_info_.framebuffers.resize(swap_chain_info_.image_views.size());
    assert(depth_buffer_.view);
    assert(render_pass_);
    for (uint64_t i = 0; i < swap_chain_info_.image_views.size(); i++) {
        assert(swap_chain_info_.image_views[i]);
        std::vector<std::shared_ptr<er::ImageView>> attachments(2);
        attachments[0] = swap_chain_info_.image_views[i];
        attachments[1] = depth_buffer_.view;

        swap_chain_info_.framebuffers[i] =
            device_->createFrameBuffer(render_pass_, attachments, display_size);
    }

    assert(hdr_render_pass_);
    assert(hdr_color_buffer_.view);
    std::vector<std::shared_ptr<er::ImageView>> attachments(2);
    attachments[0] = hdr_color_buffer_.view;
    attachments[1] = depth_buffer_.view;

    hdr_frame_buffer_ =
        device_->createFrameBuffer(hdr_render_pass_, attachments, display_size);

    assert(hdr_water_render_pass_);
    hdr_water_frame_buffer_ =
        device_->createFrameBuffer(hdr_water_render_pass_, attachments, display_size);
}

void RealWorldApplication::createCommandPool() {
    command_pool_ = device_->createCommandPool(
        queue_indices_.graphics_family_.value(),
        SET_FLAG_BIT(CommandPoolCreate, RESET_COMMAND_BUFFER_BIT));
}

void RealWorldApplication::createCommandBuffers() {
    command_buffers_ = device_->allocateCommandBuffers(
        command_pool_,
        static_cast<uint32_t>(swap_chain_info_.framebuffers.size()));

/*    std::vector<renderer::ClearValue> clear_values;
    clear_values.resize(2);
    clear_values[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
    clear_values[1].depth_stencil = { 1.0f, 0 };

    for (uint64_t i = 0; i < command_buffers_.size(); i++) {
        auto& cmd_buf = command_buffers_[i];

        cmd_buf->beginCommandBuffer(0);
        cmd_buf->beginRenderPass(render_pass_, swap_chain_info_.framebuffers[i], swap_chain_info_.extent, clear_values);
        cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, gltf_pipeline_);

        for (const auto& mesh : gltf_object_->meshes_) {
            for (const auto& prim : mesh.primitives_) {
                const auto& attrib_list = prim.attribute_descs_;

                const auto& material = gltf_object_->materials_[prim.material_idx_];

                std::vector<std::shared_ptr<renderer::Buffer>> buffers(attrib_list.size());
                std::vector<uint64_t> offsets(attrib_list.size());
                for (int i_attrib = 0; i_attrib < attrib_list.size(); i_attrib++) {
                    const auto& buffer_view = gltf_object_->buffer_views_[attrib_list[i_attrib].buffer_view];
                    buffers[i_attrib] = gltf_object_->buffers_[buffer_view.buffer_idx].buffer;
                    offsets[i_attrib] = attrib_list[i_attrib].buffer_offset;
                }
                cmd_buf->bindVertexBuffers(0, buffers, offsets);
                const auto& index_buffer_view = gltf_object_->buffer_views_[prim.index_desc_.binding];
                cmd_buf->bindIndexBuffer(gltf_object_->buffers_[index_buffer_view.buffer_idx].buffer,
                    prim.index_desc_.offset + index_buffer_view.offset,
                    prim.index_desc_.index_type);

                // todo.
                auto vk_cmd_buf = RENDER_TYPE_CAST(CommandBuffer, cmd_buf);
                auto vk_pipeline_layout = RENDER_TYPE_CAST(PipelineLayout, pipeline_layout_);
                VkDescriptorSet desc_sets[] = {
                    RENDER_TYPE_CAST(DescriptorSet, global_tex_desc_set_)->get(),
                    RENDER_TYPE_CAST(DescriptorSet, desc_sets_[i])->get(),
                    RENDER_TYPE_CAST(DescriptorSet, material.desc_set_)->get() };
                vkCmdBindDescriptorSets(vk_cmd_buf->get(), VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout->get(), 0, 3, desc_sets, 0, nullptr);

                cmd_buf->drawIndexed(static_cast<uint32_t>(prim.index_desc_.index_count));
            }
        }
        cmd_buf->endRenderPass();
        cmd_buf->endCommandBuffer();
    }*/
}

void RealWorldApplication::createSyncObjects() {
    image_available_semaphores_.resize(kMaxFramesInFlight);
    render_finished_semaphores_.resize(kMaxFramesInFlight);
    in_flight_fences_.resize(kMaxFramesInFlight);
    images_in_flight_.resize(swap_chain_info_.images.size(), VK_NULL_HANDLE);

    assert(device_);
    for (uint64_t i = 0; i < kMaxFramesInFlight; i++) {
        image_available_semaphores_[i] = device_->createSemaphore();
        render_finished_semaphores_[i] = device_->createSemaphore();
        in_flight_fences_[i] = device_->createFence();
    }
}

void RealWorldApplication::createVertexBuffer() {
    const std::vector<SkyBoxVertex> vertices = {
        {{-1.0f, -1.0f, -1.0f}},
        {{1.0f, -1.0f, -1.0f}},
        {{-1.0f, 1.0f, -1.0f}},
        {{1.0f, 1.0f, -1.0f}},
        {{-1.0f, -1.0f, 1.0f}},
        {{1.0f, -1.0f, 1.0f}},
        {{-1.0f, 1.0f, 1.0f}},
        {{1.0f, 1.0f, 1.0f}},
    };

    uint64_t buffer_size = sizeof(vertices[0]) * vertices.size();

    er::Helper::createBufferWithSrcData(
        device_info_,
        SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
        buffer_size,
        vertices.data(),
        vertex_buffer_.buffer,
        vertex_buffer_.memory);
}

void RealWorldApplication::createIndexBuffer() {
    const std::vector<uint16_t> indices = {
        0, 1, 2, 2, 1, 3,
        4, 6, 5, 5, 6, 7,
        0, 4, 1, 1, 4, 5,
        2, 3, 6, 6, 3, 7,
        1, 5, 3, 3, 5, 7,
        0, 2, 4, 4, 2, 6 };

    uint64_t buffer_size =
        sizeof(indices[0]) * indices.size();

    er::Helper::createBufferWithSrcData(
        device_info_,
        SET_FLAG_BIT(BufferUsage, INDEX_BUFFER_BIT),
        buffer_size,
        indices.data(),
        index_buffer_.buffer,
        index_buffer_.memory);
}

void RealWorldApplication::createDescriptorSetLayout() {
    // global texture descriptor set layout.
    {
        std::vector<er::DescriptorSetLayoutBinding> bindings;
        bindings.reserve(5);

        bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(GGX_LUT_INDEX));
        bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(CHARLIE_LUT_INDEX));
        bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(LAMBERTIAN_ENV_TEX_INDEX));
        bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(GGX_ENV_TEX_INDEX));
        bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(CHARLIE_ENV_TEX_INDEX));

        global_tex_desc_set_layout_ = device_->createDescriptorSetLayout(bindings);
    }

    {
        std::vector<er::DescriptorSetLayoutBinding> bindings(1);

        er::DescriptorSetLayoutBinding ubo_layout_binding{};
        ubo_layout_binding.binding = VIEW_CONSTANT_INDEX;
        ubo_layout_binding.descriptor_count = 1;
        ubo_layout_binding.descriptor_type = er::DescriptorType::UNIFORM_BUFFER;
        ubo_layout_binding.stage_flags = 
            SET_FLAG_BIT(ShaderStage, VERTEX_BIT) |
            SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
        ubo_layout_binding.immutable_samplers = nullptr; // Optional
        bindings[0] = ubo_layout_binding;

        view_desc_set_layout_ = device_->createDescriptorSetLayout(bindings);
    }

    {
        std::vector<er::DescriptorSetLayoutBinding> bindings(1);
        bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(BASE_COLOR_TEX_INDEX);

        skybox_desc_set_layout_ = device_->createDescriptorSetLayout(bindings);
    }

    // ibl texture descriptor set layout.
    {
        std::vector<er::DescriptorSetLayoutBinding> bindings(1);
        bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(PANORAMA_TEX_INDEX);
        //bindings[1] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(ENVMAP_TEX_INDEX);

        ibl_desc_set_layout_ = device_->createDescriptorSetLayout(bindings);
    }

    // ibl compute texture descriptor set layout.
    {
        std::vector<er::DescriptorSetLayoutBinding> bindings(2);
        bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(SRC_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);
        bindings[1] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(DST_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);

        ibl_comp_desc_set_layout_ = device_->createDescriptorSetLayout(bindings);
    }
}

void RealWorldApplication::createUniformBuffers() {
    uint64_t buffer_size = sizeof(glsl::ViewParams);
    const auto& images_count = swap_chain_info_.images.size();

    view_const_buffers_.resize(images_count);
    for (uint64_t i = 0; i < images_count; i++) {
        device_->createBuffer(
            buffer_size,
            SET_FLAG_BIT(BufferUsage, UNIFORM_BUFFER_BIT),
            SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
            SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
            view_const_buffers_[i].buffer,
            view_const_buffers_[i].memory);
    }
}

void RealWorldApplication::updateViewConstBuffer(uint32_t current_image, float near_z) {
    auto aspect = swap_chain_info_.extent.x / (float)swap_chain_info_.extent.y;

    view_params_.camera_pos = glm::vec4(s_camera_pos, 0);
    view_params_.view = glm::lookAt(s_camera_pos, s_camera_pos + s_camera_dir, s_camera_up);
    view_params_.proj = glm::perspective(glm::radians(45.0f), aspect, near_z, 2000.0f);
    view_params_.proj[1][1] *= -1;
    view_params_.input_features = glm::uvec4(0, 0, 0, 0);
    view_params_.depth_params = glm::vec4(
        view_params_.proj[2][2],
        view_params_.proj[3][2],
        1.0f / view_params_.proj[0][0],
        1.0f / view_params_.proj[1][1]);

    device_->updateBufferMemory(
        view_const_buffers_[current_image].memory,
        sizeof(view_params_),
        &view_params_);
}

std::vector<er::TextureDescriptor> RealWorldApplication::addGlobalTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set)
{
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(5);
    er::Helper::addOneTexture(descriptor_writes, GGX_LUT_INDEX, texture_sampler_, ggx_lut_tex_.view, description_set);
    er::Helper::addOneTexture(descriptor_writes, CHARLIE_LUT_INDEX, texture_sampler_, charlie_lut_tex_.view, description_set);
    er::Helper::addOneTexture(descriptor_writes, LAMBERTIAN_ENV_TEX_INDEX, texture_sampler_, rt_ibl_diffuse_tex_.view, description_set);
    er::Helper::addOneTexture(descriptor_writes, GGX_ENV_TEX_INDEX, texture_sampler_, rt_ibl_specular_tex_.view, description_set);
    er::Helper::addOneTexture(descriptor_writes, CHARLIE_ENV_TEX_INDEX, texture_sampler_, rt_ibl_sheen_tex_.view, description_set);

    return descriptor_writes;
}

std::vector<er::TextureDescriptor> RealWorldApplication::addSkyboxTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set)
{
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(1);

    // envmap texture.
    er::Helper::addOneTexture(descriptor_writes, BASE_COLOR_TEX_INDEX, texture_sampler_, rt_envmap_tex_.view, description_set);

    return descriptor_writes;
}

std::vector<er::TextureDescriptor> RealWorldApplication::addPanoramaTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set)
{
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(1);

    // envmap texture.
    er::Helper::addOneTexture(descriptor_writes, PANORAMA_TEX_INDEX, texture_sampler_, panorama_tex_.view, description_set);

    return descriptor_writes;
}

std::vector<er::TextureDescriptor> RealWorldApplication::addIblTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set)
{
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(1);

    er::Helper::addOneTexture(descriptor_writes, ENVMAP_TEX_INDEX, texture_sampler_, rt_envmap_tex_.view, description_set);

    return descriptor_writes;
}

std::vector<er::TextureDescriptor> RealWorldApplication::addIblComputeTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const er::TextureInfo& src_tex,
    const er::TextureInfo& dst_tex)
{
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(2);

    er::Helper::addOneTexture(
        descriptor_writes,
        SRC_TEX_INDEX,
        texture_sampler_,
        src_tex.view,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        er::ImageLayout::GENERAL);
    er::Helper::addOneTexture(
        descriptor_writes,
        DST_TEX_INDEX,
        texture_sampler_,
        dst_tex.view,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

void RealWorldApplication::createDescriptorSets() {
    auto buffer_count = swap_chain_info_.images.size();

    {
        global_tex_desc_set_ = device_->createDescriptorSets(descriptor_pool_, global_tex_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto global_texture_descs = addGlobalTextures(global_tex_desc_set_);
        device_->updateDescriptorSets(global_texture_descs, {});
    }

    {
        desc_sets_ = device_->createDescriptorSets(descriptor_pool_, view_desc_set_layout_, buffer_count);
        for (uint64_t i = 0; i < buffer_count; i++) {
            std::vector<er::BufferDescriptor> buffer_descs;
            er::Helper::addOneBuffer(
                buffer_descs,
                VIEW_CONSTANT_INDEX,
                view_const_buffers_[i].buffer,
                desc_sets_[i],
                er::DescriptorType::UNIFORM_BUFFER,
                sizeof(glsl::ViewParams));

            device_->updateDescriptorSets({}, buffer_descs);
        }
    }

    // skybox
    {
        skybox_tex_desc_set_ = device_->createDescriptorSets(descriptor_pool_, skybox_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto skybox_texture_descs = addSkyboxTextures(skybox_tex_desc_set_);
        device_->updateDescriptorSets(skybox_texture_descs, {});
    }

    // envmap
    {
        // only one descriptor layout.
        envmap_tex_desc_set_ = device_->createDescriptorSets(descriptor_pool_, ibl_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs = addPanoramaTextures(envmap_tex_desc_set_);
        device_->updateDescriptorSets(ibl_texture_descs, {});
    }

    // ibl
    {
        // only one descriptor layout.
        ibl_tex_desc_set_ = device_->createDescriptorSets(descriptor_pool_, ibl_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs = addIblTextures(ibl_tex_desc_set_);
        device_->updateDescriptorSets(ibl_texture_descs, {});
    }

    // ibl diffuse compute
    {
        // only one descriptor layout.
        ibl_diffuse_tex_desc_set_ = device_->createDescriptorSets(descriptor_pool_, ibl_comp_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs = addIblComputeTextures(ibl_diffuse_tex_desc_set_, tmp_ibl_diffuse_tex_, rt_ibl_diffuse_tex_);
        device_->updateDescriptorSets(ibl_texture_descs, {});
    }

    // ibl specular compute
    {
        // only one descriptor layout.
        ibl_specular_tex_desc_set_ = device_->createDescriptorSets(descriptor_pool_, ibl_comp_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs = addIblComputeTextures(ibl_specular_tex_desc_set_, tmp_ibl_specular_tex_, rt_ibl_specular_tex_);
        device_->updateDescriptorSets(ibl_texture_descs, {});
    }

    // ibl sheen compute
    {
        // only one descriptor layout.
        ibl_sheen_tex_desc_set_ = device_->createDescriptorSets(descriptor_pool_, ibl_comp_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs = addIblComputeTextures(ibl_sheen_tex_desc_set_, tmp_ibl_sheen_tex_, rt_ibl_sheen_tex_);
        device_->updateDescriptorSets(ibl_texture_descs, {});
    }
}

void RealWorldApplication::createTextureImage(
    const std::string& file_name,
    er::Format format,
    er::TextureInfo& texture) {
    int tex_width, tex_height, tex_channels;
    stbi_uc* pixels = stbi_load(file_name.c_str(), &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);

    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }
    er::Helper::create2DTextureImage(
        device_info_,
        format,
        tex_width,
        tex_height,
        tex_channels,
        pixels,
        texture.image,
        texture.memory);

    stbi_image_free(pixels);

    texture.view = device_->createImageView(
        texture.image,
        er::ImageViewType::VIEW_2D,
        format,
        SET_FLAG_BIT(ImageAspect, COLOR_BIT));
}

void RealWorldApplication::createTextureSampler() {
    texture_sampler_ = device_->createSampler(
        er::Filter::LINEAR,
        er::SamplerAddressMode::REPEAT,
        er::SamplerMipmapMode::LINEAR, 16.0f);
}

void RealWorldApplication::loadMtx2Texture(
    const std::string& input_filename,
    er::TextureInfo& texture) {
    uint64_t buffer_size;
    auto mtx2_data = engine::helper::readFile(input_filename, buffer_size);
    auto src_data = (char*)mtx2_data.data();

    // header block
    Mtx2HeaderBlock* header_block = reinterpret_cast<Mtx2HeaderBlock*>(src_data);
    src_data += sizeof(Mtx2HeaderBlock);

    assert(header_block->format == er::Format::R16G16B16A16_SFLOAT);

    // index block
    Mtx2IndexBlock* index_block = reinterpret_cast<Mtx2IndexBlock*>(src_data);
    src_data += sizeof(Mtx2IndexBlock);

    uint32_t width = header_block->pixel_width;
    uint32_t height = header_block->pixel_height;
    // level index block.
    uint32_t num_level_blocks = std::max(1u, header_block->level_count);
    std::vector<er::BufferImageCopyInfo> copy_regions(num_level_blocks);
    for (uint32_t i_level = 0; i_level < num_level_blocks; i_level++) {
        Mtx2LevelIndexBlock* level_block = reinterpret_cast<Mtx2LevelIndexBlock*>(src_data);

        auto& region = copy_regions[i_level];
        region.buffer_offset = level_block->byte_offset;
        region.buffer_row_length = 0;
        region.buffer_image_height = 0;

        region.image_subresource.aspect_mask = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
        region.image_subresource.mip_level = i_level;
        region.image_subresource.base_array_layer = 0;
        region.image_subresource.layer_count = 6;

        region.image_offset = glm::ivec3(0, 0, 0);
        region.image_extent = glm::uvec3(width, height, 1);
        width = std::max(1u, width / 2);
        height = std::max(1u, height / 2);

        src_data += sizeof(Mtx2LevelIndexBlock);
    }

    char* dfd_data_start = (char*)mtx2_data.data() + index_block->dfd_byte_offset;
    uint32_t dfd_total_size = *reinterpret_cast<uint32_t*>(dfd_data_start);
    src_data += sizeof(uint32_t);

    char* kvd_data_start = (char*)mtx2_data.data() + index_block->kvd_byte_offset;
    uint32_t key_value_byte_length = *reinterpret_cast<uint32_t*>(kvd_data_start);
    uint8_t* key_value = reinterpret_cast<uint8_t*>(kvd_data_start + 4);
    for (uint32_t i = 0; i < key_value_byte_length; i++) {
        auto result = key_value[i];
        int hit = 1;
    }

    char* sgd_data_start = nullptr;
    if (index_block->sgd_byte_length > 0) {
        sgd_data_start = (char*)mtx2_data.data() + index_block->sgd_byte_offset;
    }

    er::Helper::createCubemapTexture(
        device_info_,
        cubemap_render_pass_,
        header_block->pixel_width,
        header_block->pixel_height,
        num_level_blocks,
        header_block->format,
        copy_regions,
        texture,
        buffer_size,
        mtx2_data.data());
}

void RealWorldApplication::mainLoop() {
    while (!glfwWindowShouldClose(window_) && !s_exit_game) {
        glfwPollEvents();
        drawFrame();
    }

    device_->waitIdle();
}

void RealWorldApplication::drawScene(
    std::shared_ptr<er::CommandBuffer> command_buffer,
    const er::SwapChainInfo& swap_chain_info,
    const std::vector<std::shared_ptr<er::DescriptorSet>>& frame_desc_sets,
    const glm::uvec2& screen_size,
    uint32_t image_index) {

    auto frame_buffer = swap_chain_info.framebuffers[image_index];
    auto src_color = swap_chain_info.image_views[image_index];
    auto frame_desc_set = frame_desc_sets[image_index];

    std::vector<er::ClearValue> clear_values(2);
    clear_values[0].color = { 50.0f / 255.0f, 50.0f / 255.0f, 50.0f / 255.0f, 1.0f };
    clear_values[1].depth_stencil = { 1.0f, 0 };

    auto& cmd_buf = command_buffer;

    if (0)
    {
        // generate envmap cubemap from panorama hdr image.
        cmd_buf->addImageBarrier(
            rt_envmap_tex_.image,
            er::Helper::getImageAsSource(),
            er::Helper::getImageAsColorAttachment(),
            0, 1, 0, 6);

        cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS, envmap_pipeline_);

        std::vector<er::ClearValue> envmap_clear_values(6, clear_values[0]);
        cmd_buf->beginRenderPass(cubemap_render_pass_, rt_envmap_tex_.framebuffers[0], glm::uvec2(kCubemapSize, kCubemapSize), envmap_clear_values);

        glsl::IblParams ibl_params = {};
        cmd_buf->pushConstants(SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT), ibl_pipeline_layout_, &ibl_params, sizeof(ibl_params));

        cmd_buf->bindDescriptorSets(er::PipelineBindPoint::GRAPHICS, ibl_pipeline_layout_, { envmap_tex_desc_set_ });

        cmd_buf->draw(3);

        cmd_buf->endRenderPass();

        uint32_t num_mips = static_cast<uint32_t>(std::log2(kCubemapSize) + 1);

        er::Helper::generateMipmapLevels(
            cmd_buf,
            rt_envmap_tex_.image,
            num_mips,
            kCubemapSize,
            kCubemapSize,
            er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }
    else {
        // generate envmap from skybox.
        cmd_buf->addImageBarrier(
            rt_envmap_tex_.image,
            er::Helper::getImageAsSource(),
            er::Helper::getImageAsColorAttachment(),
            0, 1, 0, 6);

        cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS, cube_skybox_pipeline_);

        std::vector<er::ClearValue> envmap_clear_values(6, clear_values[0]);
        cmd_buf->beginRenderPass(cubemap_render_pass_, rt_envmap_tex_.framebuffers[0], glm::uvec2(kCubemapSize, kCubemapSize), envmap_clear_values);

        glsl::SunSkyParams sun_sky_params = {};
        sun_sky_params.sun_pos = glm::vec3(cos(s_sun_angle), sin(s_sun_angle), -0.3f);

        cmd_buf->pushConstants(SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT), cube_skybox_pipeline_layout_, &sun_sky_params, sizeof(sun_sky_params));

        cmd_buf->bindDescriptorSets(er::PipelineBindPoint::GRAPHICS, cube_skybox_pipeline_layout_, { envmap_tex_desc_set_ });

        cmd_buf->draw(3);

        cmd_buf->endRenderPass();

        uint32_t num_mips = static_cast<uint32_t>(std::log2(kCubemapSize) + 1);

        er::Helper::generateMipmapLevels(
            cmd_buf,
            rt_envmap_tex_.image,
            num_mips,
            kCubemapSize,
            kCubemapSize,
            er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }

    // generate ibl diffuse texture.
    {
        cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS, lambertian_pipeline_);

        cmd_buf->addImageBarrier(
            rt_ibl_diffuse_tex_.image,
            er::Helper::getImageAsSource(),
            er::Helper::getImageAsColorAttachment(),
            0, 1, 0, 6);

        std::vector<er::ClearValue> envmap_clear_values(6, clear_values[0]);
        cmd_buf->beginRenderPass(cubemap_render_pass_, rt_ibl_diffuse_tex_.framebuffers[0], glm::uvec2(kCubemapSize, kCubemapSize), envmap_clear_values);

        glsl::IblParams ibl_params = {};
        ibl_params.roughness = 1.0f;
        ibl_params.currentMipLevel = 0;
        ibl_params.width = kCubemapSize;
        ibl_params.lodBias = 0;
        cmd_buf->pushConstants(SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT), ibl_pipeline_layout_, &ibl_params, sizeof(ibl_params));

        cmd_buf->bindDescriptorSets(er::PipelineBindPoint::GRAPHICS, ibl_pipeline_layout_, { ibl_tex_desc_set_ });

        cmd_buf->draw(3);

        cmd_buf->endRenderPass();

        cmd_buf->addImageBarrier(
            rt_ibl_diffuse_tex_.image,
            er::Helper::getImageAsColorAttachment(),
            er::Helper::getImageAsShaderSampler(),
            0, 1, 0, 6);
    }

    // generate ibl specular texture.
    {
        uint32_t num_mips = static_cast<uint32_t>(std::log2(kCubemapSize) + 1);
        cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS, ggx_pipeline_);

        for (int i_mip = num_mips - 1; i_mip >= 0; i_mip--) {
            cmd_buf->addImageBarrier(
                rt_ibl_specular_tex_.image,
                er::Helper::getImageAsSource(),
                er::Helper::getImageAsColorAttachment(),
                i_mip, 1, 0, 6);

            uint32_t width = std::max(static_cast<uint32_t>(kCubemapSize) >> i_mip, 1u);
            uint32_t height = std::max(static_cast<uint32_t>(kCubemapSize) >> i_mip, 1u);

            std::vector<er::ClearValue> envmap_clear_values(6, clear_values[0]);
            cmd_buf->beginRenderPass(cubemap_render_pass_, rt_ibl_specular_tex_.framebuffers[i_mip], glm::uvec2(width, height), envmap_clear_values);

            glsl::IblParams ibl_params = {};
            ibl_params.roughness = static_cast<float>(i_mip) / static_cast<float>(num_mips - 1);
            ibl_params.currentMipLevel = i_mip;
            ibl_params.width = kCubemapSize;
            ibl_params.lodBias = 0;
            cmd_buf->pushConstants(SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT), ibl_pipeline_layout_, &ibl_params, sizeof(ibl_params));

            cmd_buf->bindDescriptorSets(er::PipelineBindPoint::GRAPHICS, ibl_pipeline_layout_, { ibl_tex_desc_set_ });

            cmd_buf->draw(3);

            cmd_buf->endRenderPass();
        }

        cmd_buf->addImageBarrier(
            rt_ibl_specular_tex_.image,
            er::Helper::getImageAsColorAttachment(),
            er::Helper::getImageAsShaderSampler(),
            0, num_mips, 0, 6);
    }

    // generate ibl sheen texture.
    {
        uint32_t num_mips = static_cast<uint32_t>(std::log2(kCubemapSize) + 1);
        cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS, charlie_pipeline_);

        for (int i_mip = num_mips - 1; i_mip >= 0; i_mip--) {
            uint32_t width = std::max(static_cast<uint32_t>(kCubemapSize) >> i_mip, 1u);
            uint32_t height = std::max(static_cast<uint32_t>(kCubemapSize) >> i_mip, 1u);

            cmd_buf->addImageBarrier(
                rt_ibl_sheen_tex_.image,
                er::Helper::getImageAsSource(),
                er::Helper::getImageAsColorAttachment(),
                i_mip, 1, 0, 6);

            std::vector<er::ClearValue> envmap_clear_values(6, clear_values[0]);
            cmd_buf->beginRenderPass(cubemap_render_pass_, rt_ibl_sheen_tex_.framebuffers[i_mip], glm::uvec2(width, height), envmap_clear_values);

            glsl::IblParams ibl_params = {};
            ibl_params.roughness = static_cast<float>(i_mip) / static_cast<float>(num_mips - 1);
            ibl_params.currentMipLevel = i_mip;
            ibl_params.width = kCubemapSize;
            ibl_params.lodBias = 0;
            cmd_buf->pushConstants(SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT), ibl_pipeline_layout_, &ibl_params, sizeof(ibl_params));

            cmd_buf->bindDescriptorSets(er::PipelineBindPoint::GRAPHICS, ibl_pipeline_layout_, { ibl_tex_desc_set_ });

            cmd_buf->draw(3);

            cmd_buf->endRenderPass();
        }

        cmd_buf->addImageBarrier(
            rt_ibl_sheen_tex_.image,
            er::Helper::getImageAsColorAttachment(),
            er::Helper::getImageAsShaderSampler(),
            0, num_mips, 0, 6);
    }

    {
        if (0)
        {
            cmd_buf->addImageBarrier(
                rt_ibl_diffuse_tex_.image,
                er::Helper::getImageAsSource(),
                er::Helper::getImageAsStore(),
                0, 1, 0, 6);

            cmd_buf->bindPipeline(er::PipelineBindPoint::COMPUTE, blur_comp_pipeline_);
            glsl::IblComputeParams ibl_comp_params = {};
            ibl_comp_params.size = glm::ivec4(kCubemapSize, kCubemapSize, 0, 0);
            cmd_buf->pushConstants(SET_FLAG_BIT(ShaderStage, COMPUTE_BIT), ibl_comp_pipeline_layout_, &ibl_comp_params, sizeof(ibl_comp_params));

            cmd_buf->bindDescriptorSets(er::PipelineBindPoint::COMPUTE, ibl_comp_pipeline_layout_, { ibl_diffuse_tex_desc_set_ });

            cmd_buf->dispatch((kCubemapSize + 7) / 8, (kCubemapSize + 7) / 8, 6);

            uint32_t num_mips = static_cast<uint32_t>(std::log2(kCubemapSize) + 1);
            cmd_buf->addImageBarrier(
                rt_ibl_diffuse_tex_.image,
                er::Helper::getImageAsStore(),
                er::Helper::getImageAsShaderSampler(),
                0, 1, 0, 6);
            cmd_buf->addImageBarrier(
                rt_ibl_specular_tex_.image,
                er::Helper::getImageAsSource(),
                er::Helper::getImageAsShaderSampler(),
                0, num_mips, 0, 6);
            cmd_buf->addImageBarrier(
                rt_ibl_sheen_tex_.image,
                er::Helper::getImageAsSource(),
                er::Helper::getImageAsShaderSampler(),
                0, num_mips, 0, 6);
        }
    }

    {
        for (auto& tile_obj : tile_objects_) {
            tile_obj->generateTileBuffers(cmd_buf);
        }
    }

    {
        ego::GltfObject::updateGameObjectsBuffer(cmd_buf, s_update_frame_count);
        if (s_update_frame_count >= 0) {
            s_update_frame_count++;
        }
    }

    {
        cmd_buf->beginRenderPass(
            hdr_render_pass_,
            hdr_frame_buffer_,
            screen_size,
            clear_values);

        er::DescriptorSetList desc_sets{ global_tex_desc_set_, frame_desc_set };
        // render gltf.
        {
            for (auto& gltf_obj : gltf_objects_) {
                gltf_obj->draw(cmd_buf, desc_sets);
            }
        }

        // render terrain opaque pass.
        {
            for (auto& tile_obj : tile_objects_) {
                tile_obj->draw(cmd_buf, desc_sets, screen_size, true);
            }
        }

        // render skybox.
        {
            cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS, skybox_pipeline_);
            std::vector<std::shared_ptr<er::Buffer>> buffers(1);
            std::vector<uint64_t> offsets(1);
            buffers[0] = vertex_buffer_.buffer;
            offsets[0] = 0;

            cmd_buf->bindVertexBuffers(0, buffers, offsets);
            cmd_buf->bindIndexBuffer(index_buffer_.buffer, 0, er::IndexType::UINT16);

            glsl::SunSkyParams sun_sky_params = {};
            sun_sky_params.sun_pos = glm::vec3(cos(s_sun_angle), sin(s_sun_angle), -0.3f);
            s_sun_angle += 0.0001f;
            cmd_buf->pushConstants(SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT), skybox_pipeline_layout_, &sun_sky_params, sizeof(sun_sky_params));

            er::DescriptorSetList desc_sets{ skybox_tex_desc_set_, frame_desc_set };
            cmd_buf->bindDescriptorSets(er::PipelineBindPoint::GRAPHICS, skybox_pipeline_layout_, desc_sets);

            cmd_buf->drawIndexed(36);
        }

        cmd_buf->endRenderPass();

        er::ImageResourceInfo src_info = {
            er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT) };

        er::ImageResourceInfo dst_info = {
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };

        er::Helper::blitImage(
            cmd_buf,
            hdr_color_buffer_.image,
            hdr_color_buffer_copy_.image,
            src_info,
            src_info,
            dst_info,
            dst_info,
            SET_FLAG_BIT(ImageAspect, COLOR_BIT),
            SET_FLAG_BIT(ImageAspect, COLOR_BIT),
            glm::ivec3(screen_size.x, screen_size.y, 1));

        src_info = {
            er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            SET_FLAG_BIT(Access, DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, EARLY_FRAGMENT_TESTS_BIT) };

        dst_info = {
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };

        er::Helper::blitImage(
            cmd_buf,
            depth_buffer_.image,
            depth_buffer_copy_.image,
            src_info,
            src_info,
            dst_info,
            dst_info,
            SET_FLAG_BIT(ImageAspect, DEPTH_BIT),
            SET_FLAG_BIT(ImageAspect, DEPTH_BIT),
            glm::ivec3(screen_size.x, screen_size.y, 1));

        cmd_buf->beginRenderPass(
            hdr_water_render_pass_,
            hdr_water_frame_buffer_,
            screen_size,
            clear_values);

        // render terrain water pass.
        {
            for (auto& tile_obj : tile_objects_) {
                tile_obj->draw(cmd_buf, desc_sets, screen_size, false);
            }
        }

        cmd_buf->endRenderPass();
    }

    er::ImageResourceInfo src_info = {
        er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT)};

    er::ImageResourceInfo dst_info = {
        er::ImageLayout::PRESENT_SRC_KHR,
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT) };

    er::Helper::blitImage(
        cmd_buf,
        hdr_color_buffer_.image,
        swap_chain_info.images[image_index],
        src_info,
        src_info,
        dst_info,
        dst_info,
        SET_FLAG_BIT(ImageAspect, COLOR_BIT),
        SET_FLAG_BIT(ImageAspect, COLOR_BIT),
        glm::ivec3(screen_size.x, screen_size.y, 1));
}

void RealWorldApplication::drawMenu(
    std::shared_ptr<er::CommandBuffer> command_buffer) {
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    static bool s_select_load_gltf = false;
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("Game Objects"))
        {
            if (ImGui::MenuItem("Load gltf", NULL)) {
                s_select_load_gltf = true;
            }

            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit"))
        {
            if (ImGui::MenuItem("Undo", "CTRL+Z")) {}
            if (ImGui::MenuItem("Redo", "CTRL+Y", false, false)) {}  // Disabled item
            ImGui::Separator();
            if (ImGui::MenuItem("Cut", "CTRL+X")) {}
            if (ImGui::MenuItem("Copy", "CTRL+C")) {}
            if (ImGui::MenuItem("Paste", "CTRL+V")) {}
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    s_camera_paused =
        ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow) ||
        ImGui::IsWindowFocused(ImGuiHoveredFlags_AnyWindow);

    if (s_select_load_gltf) {
        ImGui::OpenPopup("select gltf object");
        if (ImGui::BeginPopup("select gltf object"))
        {
            std::vector<const char*> listbox_items;
            for (const auto& name : gltf_file_names_) {
                listbox_items.push_back(name.c_str());
            }
            static int s_listbox_item_current = -1;
            ImGui::ListBox("",
                &s_listbox_item_current, listbox_items.data(),
                static_cast<int>(listbox_items.size()),
                static_cast<int>(listbox_items.size()));

            if (s_listbox_item_current >= 0) {
                auto file_name = "assets/" + gltf_file_names_[s_listbox_item_current];

                auto gltf_obj = std::make_shared<ego::GltfObject>(
                    device_info_,
                    descriptor_pool_,
                    hdr_render_pass_,
                    graphic_pipeline_info_,
                    texture_sampler_,
                    thin_film_lut_tex_,
                    file_name,
                    swap_chain_info_.extent,
                    glm::inverse(view_params_.view));

                gltf_objects_.push_back(gltf_obj);
                s_listbox_item_current = -1;
                s_select_load_gltf = false;
                if (s_update_frame_count < 0) {
                    s_update_frame_count = 0;
                }
            }

            ImGui::EndPopup();
        }

        if (!s_select_load_gltf) {
            ImGui::CloseCurrentPopup();
        }
    }

    er::Helper::addImGuiToCommandBuffer(command_buffer);
}

void RealWorldApplication::drawFrame() {
    std::vector<std::shared_ptr<er::Fence>> in_flight_fences(1);
    in_flight_fences[0] = in_flight_fences_[current_frame_];
    device_->waitForFences(in_flight_fences);

    uint32_t image_index = 0;
    bool need_recreate_swap_chain = er::Helper::acquireNextImage(
        device_,
        swap_chain_info_.swap_chain,
        image_available_semaphores_[current_frame_],
        image_index);

    if (need_recreate_swap_chain) {
        recreateSwapChain();
        return;
    }

    if (images_in_flight_[image_index] != VK_NULL_HANDLE) {
        std::vector<std::shared_ptr<er::Fence>> images_in_flight(1);
        images_in_flight[0] = images_in_flight_[image_index];
        device_->waitForFences(images_in_flight);
    }
    // Mark the image as now being in use by this frame
    images_in_flight_[image_index] = in_flight_fences_[current_frame_];

    updateViewConstBuffer(image_index);

    device_->resetFences(in_flight_fences);

    auto command_buffer = command_buffers_[image_index];
    std::vector<std::shared_ptr<er::CommandBuffer>>command_buffers(1, command_buffer);

    static auto start_time = std::chrono::high_resolution_clock::now();
    auto current_time = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(current_time - start_time).count();

    command_buffer->reset(0);
    command_buffer->beginCommandBuffer(SET_FLAG_BIT(CommandBufferUsage, ONE_TIME_SUBMIT_BIT));

    drawScene(command_buffer,
        swap_chain_info_,
        desc_sets_,
        swap_chain_info_.extent,
        image_index);

    //drawMenu(command_buffer);

    command_buffer->endCommandBuffer();

    er::Helper::submitQueue(
        graphics_queue_,
        in_flight_fences_[current_frame_],
        { image_available_semaphores_[current_frame_] },
        { command_buffer },
        { render_finished_semaphores_[current_frame_] });

    need_recreate_swap_chain = er::Helper::presentQueue(
        present_queue_,
        { swap_chain_info_.swap_chain },
        { render_finished_semaphores_[current_frame_] },
        image_index,
        framebuffer_resized_);

    if (need_recreate_swap_chain) {
        recreateSwapChain();
    }

    current_frame_ = (current_frame_ + 1) % kMaxFramesInFlight;
}

void RealWorldApplication::cleanupSwapChain() {
    assert(device_);
    depth_buffer_.destroy(device_);
    hdr_color_buffer_.destroy(device_);
    hdr_color_buffer_copy_.destroy(device_);
    device_->destroyFramebuffer(hdr_frame_buffer_);
    device_->destroyFramebuffer(hdr_water_frame_buffer_);

    for (auto framebuffer : swap_chain_info_.framebuffers) {
        device_->destroyFramebuffer(framebuffer);
    }

    device_->freeCommandBuffers(command_pool_, command_buffers_);
    device_->destroyPipeline(skybox_pipeline_);
    device_->destroyPipeline(blur_comp_pipeline_);
    device_->destroyPipelineLayout(skybox_pipeline_layout_);
    device_->destroyPipelineLayout(ibl_comp_pipeline_layout_);
    device_->destroyRenderPass(render_pass_);
    device_->destroyRenderPass(hdr_render_pass_);
    device_->destroyRenderPass(hdr_water_render_pass_);

    for (auto image_view : swap_chain_info_.image_views) {
        device_->destroyImageView(image_view);
    }

    device_->destroySwapchain(swap_chain_info_.swap_chain);

    for (auto& buffer : view_const_buffers_) {
        buffer.destroy(device_);
    }

    device_->destroyDescriptorPool(descriptor_pool_);
}

void RealWorldApplication::cleanup() {
    cleanupSwapChain();

    device_->destroyPipeline(envmap_pipeline_);
    device_->destroyPipeline(cube_skybox_pipeline_);
    device_->destroyPipeline(lambertian_pipeline_);
    device_->destroyPipeline(ggx_pipeline_);
    device_->destroyPipeline(charlie_pipeline_);
    device_->destroyPipelineLayout(ibl_pipeline_layout_);
    device_->destroyPipelineLayout(cube_skybox_pipeline_layout_);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    device_->destroyRenderPass(cubemap_render_pass_);

    assert(device_);
    device_->destroySampler(texture_sampler_);
    sample_tex_.destroy(device_);
    er::Helper::destroy(device_);
    ggx_lut_tex_.destroy(device_);
    brdf_lut_tex_.destroy(device_);
    charlie_lut_tex_.destroy(device_);
    thin_film_lut_tex_.destroy(device_);
    panorama_tex_.destroy(device_);
    ibl_diffuse_tex_.destroy(device_);
    ibl_specular_tex_.destroy(device_);
    ibl_sheen_tex_.destroy(device_);
    rt_envmap_tex_.destroy(device_);
    tmp_ibl_diffuse_tex_.destroy(device_);
    tmp_ibl_specular_tex_.destroy(device_);
    tmp_ibl_sheen_tex_.destroy(device_);
    rt_ibl_diffuse_tex_.destroy(device_);
    rt_ibl_specular_tex_.destroy(device_);
    rt_ibl_sheen_tex_.destroy(device_);
    device_->destroyDescriptorSetLayout(view_desc_set_layout_);
    device_->destroyDescriptorSetLayout(global_tex_desc_set_layout_);
    device_->destroyDescriptorSetLayout(skybox_desc_set_layout_);
    device_->destroyDescriptorSetLayout(ibl_desc_set_layout_);
    device_->destroyDescriptorSetLayout(ibl_comp_desc_set_layout_);
    
    tile_objects_.clear();
    ego::TileObject::destoryStaticMembers(device_);
    gltf_objects_.clear();
    ego::GltfObject::destoryStaticMembers(device_);

    vertex_buffer_.destroy(device_);
    index_buffer_.destroy(device_);

    for (uint64_t i = 0; i < kMaxFramesInFlight; i++) {
        device_->destroySemaphore(render_finished_semaphores_[i]);
        device_->destroySemaphore(image_available_semaphores_[i]);
        device_->destroyFence(in_flight_fences_[i]);
    }

    device_->destroyCommandPool(command_pool_);
    device_->destroy();

    instance_->destroySurface(surface_);
    instance_->destroy();

    glfwDestroyWindow(window_);
    glfwTerminate();
}

}//namespace app
}//namespace work
