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

namespace er = engine::renderer;
namespace ego = engine::game_object;

namespace {
constexpr int kWindowSizeX = 1920;
constexpr int kWindowSizeY = 1080;
static int s_update_frame_count = -1;

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
    bool clear = false,
    er::SampleCountFlagBits sample_count = er::SampleCountFlagBits::SC_1_BIT,
    er::ImageLayout color_image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) {
    auto color_attachment = FillAttachmentDescription(
        format,
        sample_count,
        clear ? er::ImageLayout::UNDEFINED : color_image_layout,
        color_image_layout,
        clear ? er::AttachmentLoadOp::CLEAR : er::AttachmentLoadOp::LOAD);

    er::AttachmentReference color_attachment_ref(0, er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    auto depth_attachment = FillAttachmentDescription(
        depth_format,
        sample_count,
        clear ? er::ImageLayout::UNDEFINED : er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        clear ? er::AttachmentLoadOp::CLEAR : er::AttachmentLoadOp::LOAD);

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
const float s_camera_speed = 1.0f;
static glm::vec3 s_camera_pos = glm::vec3(0, 500.0f, 0);
static glm::vec3 s_camera_dir = glm::normalize(glm::vec3(1.0f, 0.0f, 0.0f));
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
    final_render_pass_ = createRenderPass(
        device_,
        swap_chain_info_.format,
        depth_format_,
        false,
        er::SampleCountFlagBits::SC_1_BIT,
        er::ImageLayout::PRESENT_SRC_KHR);
    hdr_render_pass_ = createRenderPass(device_, hdr_format_, depth_format_, true);
    hdr_water_render_pass_ = createRenderPass(device_, hdr_format_, depth_format_, false);
}

void RealWorldApplication::initVulkan() {
    static auto color_blend_attachment = er::helper::fillPipelineColorBlendAttachmentState();
    static std::vector<er::PipelineColorBlendAttachmentState> color_blend_attachments(1, color_blend_attachment);
    static std::vector<er::PipelineColorBlendAttachmentState> cube_color_blend_attachments(6, color_blend_attachment);

    auto single_blend_state_info =
        std::make_shared<er::PipelineColorBlendStateCreateInfo>(
            er::helper::fillPipelineColorBlendStateCreateInfo(color_blend_attachments));

    auto cube_blend_state_info =
        std::make_shared<er::PipelineColorBlendStateCreateInfo>(
            er::helper::fillPipelineColorBlendStateCreateInfo(cube_color_blend_attachments));

    auto cull_rasterization_info =
        std::make_shared<er::PipelineRasterizationStateCreateInfo>(
            er::helper::fillPipelineRasterizationStateCreateInfo());

    auto no_cull_rasterization_info =
        std::make_shared<er::PipelineRasterizationStateCreateInfo>(
            er::helper::fillPipelineRasterizationStateCreateInfo(
                false,
                false,
                er::PolygonMode::FILL,
                SET_FLAG_BIT(CullMode, NONE)));

    auto ms_info = std::make_shared<er::PipelineMultisampleStateCreateInfo>(
        er::helper::fillPipelineMultisampleStateCreateInfo());

    auto depth_stencil_info =
        std::make_shared<er::PipelineDepthStencilStateCreateInfo>(
            er::helper::fillPipelineDepthStencilStateCreateInfo());

    auto fs_depth_stencil_info =
        std::make_shared<er::PipelineDepthStencilStateCreateInfo>(
            er::helper::fillPipelineDepthStencilStateCreateInfo(
                false,
                false,
                er::CompareOp::ALWAYS));

    graphic_pipeline_info_.blend_state_info = single_blend_state_info;
    graphic_pipeline_info_.rasterization_info = cull_rasterization_info;
    graphic_pipeline_info_.ms_info = ms_info;
    graphic_pipeline_info_.depth_stencil_info = depth_stencil_info;
        
    graphic_fs_pipeline_info_.blend_state_info = single_blend_state_info;
    graphic_fs_pipeline_info_.rasterization_info = no_cull_rasterization_info;
    graphic_fs_pipeline_info_.ms_info = ms_info;
    graphic_fs_pipeline_info_.depth_stencil_info = fs_depth_stencil_info;
        
    graphic_cubemap_pipeline_info_.blend_state_info = cube_blend_state_info;
    graphic_cubemap_pipeline_info_.rasterization_info = no_cull_rasterization_info;
    graphic_cubemap_pipeline_info_.ms_info = ms_info;
    graphic_cubemap_pipeline_info_.depth_stencil_info = fs_depth_stencil_info;

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
    createDescriptorSetLayout();
    createCommandPool();
    assert(command_pool_);
    device_info_.cmd_pool = command_pool_;
    er::Helper::init(device_info_);

    engine::helper::loadMtx2Texture(
        device_info_,
        cubemap_render_pass_,
        "assets/environments/doge2/lambertian/diffuse.ktx2",
        ibl_diffuse_tex_);
    engine::helper::loadMtx2Texture(
        device_info_,
        cubemap_render_pass_,
        "assets/environments/doge2/ggx/specular.ktx2",
        ibl_specular_tex_);
    engine::helper::loadMtx2Texture(
        device_info_,
        cubemap_render_pass_,
        "assets/environments/doge2/charlie/sheen.ktx2",
        ibl_sheen_tex_);
    recreateRenderBuffer(swap_chain_info_.extent);
    auto format = er::Format::R8G8B8A8_UNORM;
    engine::helper::createTextureImage(device_info_, "assets/statue.jpg", format, sample_tex_);
    engine::helper::createTextureImage(device_info_, "assets/brdfLUT.png", format, brdf_lut_tex_);
    engine::helper::createTextureImage(device_info_, "assets/lut_ggx.png", format, ggx_lut_tex_);
    engine::helper::createTextureImage(device_info_, "assets/lut_charlie.png", format, charlie_lut_tex_);
    engine::helper::createTextureImage(device_info_, "assets/lut_thin_film.png", format, thin_film_lut_tex_);
    createTextureSampler();
    createUniformBuffers();
    descriptor_pool_ = device_->createDescriptorPool();
    createCommandBuffers();
    createSyncObjects();

    clear_values_.resize(2);
    clear_values_[0].color = { 50.0f / 255.0f, 50.0f / 255.0f, 50.0f / 255.0f, 1.0f };
    clear_values_[1].depth_stencil = { 1.0f, 0 };

    ibl_creator_ = std::make_shared<es::IblCreator>(
        device_info_,
        descriptor_pool_,
        cubemap_render_pass_,
        graphic_cubemap_pipeline_info_,
        texture_sampler_,
        kCubemapSize);

    skydome_ = std::make_shared<es::Skydome>(
        device_info_,
        descriptor_pool_,
        hdr_render_pass_,
        cubemap_render_pass_,
        view_desc_set_layout_,
        ibl_creator_->getIblDescSetLayout(),
        graphic_pipeline_info_,
        graphic_cubemap_pipeline_info_,
        ibl_creator_->getEnvmapTexture(),
        texture_sampler_,
        swap_chain_info_.extent,
        kCubemapSize);

    createDescriptorSets();

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
        device_info_,
        hdr_render_pass_,
        hdr_water_render_pass_,
        graphic_pipeline_info_,
        desc_set_layouts,
        swap_chain_info_.extent);

    std::vector<std::shared_ptr<er::ImageView>> soil_water_texes(2);
    soil_water_texes[0] = ego::TileObject::getSoilWaterLayer(0).view;
    soil_water_texes[1] = ego::TileObject::getSoilWaterLayer(1).view;
    weather_system_ = std::make_shared<es::WeatherSystem>(
        device_info_,
        descriptor_pool_,
        texture_sampler_,
        ego::TileObject::getRockLayer().view,
        soil_water_texes);

    ego::GltfObject::initStaticMembers(
        device_,
        descriptor_pool_,
        desc_set_layouts,
        texture_sampler_,
        ego::TileObject::getRockLayer(),
        ego::TileObject::getSoilWaterLayer(0),
        ego::TileObject::getSoilWaterLayer(1),
        ego::TileObject::getWaterFlow(),
        weather_system_->getAirflowTex());

    ego::DebugDrawObject::initStaticMembers(
        device_info_,
        hdr_render_pass_,
        graphic_pipeline_info_,
        desc_set_layouts,
        swap_chain_info_.extent);

    ego::TileObject::updateStaticDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        hdr_color_buffer_copy_.view,
        depth_buffer_copy_.view,
        weather_system_->getTempMoistureTexes());

    volume_cloud_ = std::make_shared<es::VolumeCloud>(
        device_info_,
        descriptor_pool_,
        hdr_water_render_pass_,
        view_desc_set_layout_,
        ibl_creator_->getIblDescSetLayout(),
        graphic_fs_pipeline_info_,
        texture_sampler_,
        hdr_color_buffer_copy_.view,
        depth_buffer_copy_.view,
        weather_system_->getTempMoistureTexes(),
        swap_chain_info_.extent);

    ego::DebugDrawObject::updateStaticDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        weather_system_->getTempMoistureTex(0),
        weather_system_->getAirflowTex());

    menu_ = std::make_shared<es::Menu>(
        device_info_,
        instance_,
        window_, 
        queue_indices_,
        swap_chain_info_,
        graphics_queue_,
        descriptor_pool_,
        final_render_pass_,
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

    menu_->destroy();

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
    ego::DebugDrawObject::recreateStaticMembers(
        device_,
        hdr_render_pass_,
        graphic_pipeline_info_,
        desc_set_layouts,
        swap_chain_info_.extent);
    recreateRenderBuffer(swap_chain_info_.extent);
    createUniformBuffers();
    descriptor_pool_ = device_->createDescriptorPool();
    createCommandBuffers();

    skydome_->recreate(
        device_,
        descriptor_pool_,
        hdr_render_pass_,
        view_desc_set_layout_,
        graphic_pipeline_info_,
        ibl_creator_->getEnvmapTexture(),
        texture_sampler_,
        swap_chain_info_.extent);

    ibl_creator_->recreate(
        device_,
        descriptor_pool_,
        texture_sampler_);

    weather_system_->recreate(
        device_,
        descriptor_pool_,
        texture_sampler_,
        ego::TileObject::getRockLayer().view,
        { ego::TileObject::getSoilWaterLayer(0).view,
          ego::TileObject::getSoilWaterLayer(1).view });

    createDescriptorSets();

    for (auto& image : swap_chain_info_.images) {
        er::Helper::transitionImageLayout(
            device_info_,
            image,
            swap_chain_info_.format,
            er::ImageLayout::UNDEFINED,
            er::ImageLayout::PRESENT_SRC_KHR);
    }

    ego::TileObject::generateAllDescriptorSets(
        device_,
        descriptor_pool_,
        texture_sampler_);

    ego::DebugDrawObject::generateAllDescriptorSets(
        device_,
        descriptor_pool_,
        texture_sampler_,
        weather_system_->getTempMoistureTex(0),
        weather_system_->getAirflowTex());

    ego::GltfObject::generateDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        thin_film_lut_tex_,
        ego::TileObject::getRockLayer(),
        ego::TileObject::getSoilWaterLayer(0),
        ego::TileObject::getSoilWaterLayer(1),
        ego::TileObject::getWaterFlow(),
        weather_system_->getAirflowTex());

    ego::TileObject::updateStaticDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        hdr_color_buffer_copy_.view,
        depth_buffer_copy_.view,
        weather_system_->getTempMoistureTexes());

    ego::DebugDrawObject::updateStaticDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        weather_system_->getTempMoistureTex(0),
        weather_system_->getAirflowTex());

    volume_cloud_->recreate(
        device_info_.device,
        descriptor_pool_,
        hdr_water_render_pass_,
        view_desc_set_layout_,
        graphic_fs_pipeline_info_,
        texture_sampler_,
        hdr_color_buffer_copy_.view,
        depth_buffer_copy_.view,
        weather_system_->getTempMoistureTexes(),
        swap_chain_info_.extent);

    menu_->init(
        device_info_,
        instance_,
        window_,
        queue_indices_,
        swap_chain_info_,
        graphics_queue_,
        descriptor_pool_,
        final_render_pass_,
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
    assert(final_render_pass_);
    for (uint64_t i = 0; i < swap_chain_info_.image_views.size(); i++) {
        assert(swap_chain_info_.image_views[i]);
        std::vector<std::shared_ptr<er::ImageView>> attachments(2);
        attachments[0] = swap_chain_info_.image_views[i];
        attachments[1] = depth_buffer_.view;

        swap_chain_info_.framebuffers[i] =
            device_->createFrameBuffer(final_render_pass_, attachments, display_size);
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
    auto fov = glm::radians(45.0f);

    view_params_.camera_pos = glm::vec4(s_camera_pos, 0);
    view_params_.view = glm::lookAt(s_camera_pos, s_camera_pos + s_camera_dir, s_camera_up);
    view_params_.proj = glm::perspective(fov, aspect, near_z, 10000.0f);
    view_params_.proj[1][1] *= -1;
    view_params_.view_proj = view_params_.proj * view_params_.view;
    view_params_.inv_view_proj = glm::inverse(view_params_.view_proj);
    auto view_relative = glm::lookAt(vec3(0), s_camera_dir, s_camera_up);
    view_params_.inv_view_proj_relative = glm::inverse(view_params_.proj * view_relative);
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
    er::Helper::addOneTexture(
        descriptor_writes,
        GGX_LUT_INDEX,
        texture_sampler_,
        ggx_lut_tex_.view,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(
        descriptor_writes,
        CHARLIE_LUT_INDEX,
        texture_sampler_,
        charlie_lut_tex_.view,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    ibl_creator_->addToGlobalTextures(
        descriptor_writes,
        description_set,
        texture_sampler_);

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
}

void RealWorldApplication::createTextureSampler() {
    texture_sampler_ = device_->createSampler(
        er::Filter::LINEAR,
        er::SamplerAddressMode::CLAMP_TO_EDGE,
        er::SamplerMipmapMode::LINEAR, 16.0f);
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
    uint32_t image_index,
    float delta_t,
    float current_time) {

    auto frame_buffer = swap_chain_info.framebuffers[image_index];
    auto src_color = swap_chain_info.image_views[image_index];
    auto frame_desc_set = frame_desc_sets[image_index];

    auto& cmd_buf = command_buffer;

    static int s_dbuf_idx = 0;

    if (0)
    {
        ibl_creator_->drawEnvmapFromPanoramaImage(
            cmd_buf,
            cubemap_render_pass_,
            clear_values_,
            kCubemapSize);
    }
    else {
        skydome_->drawCubeSkyBox(
            cmd_buf,
            cubemap_render_pass_,
            ibl_creator_->getEnvmapTexDescSet(),
            ibl_creator_->getEnvmapTexture(),
            clear_values_,
            kCubemapSize);
    }

    ibl_creator_->createIblDiffuseMap(
        cmd_buf,
        cubemap_render_pass_,
        clear_values_,
        kCubemapSize);

    ibl_creator_->createIblSpecularMap(
        cmd_buf,
        cubemap_render_pass_,
        clear_values_,
        kCubemapSize);

    ibl_creator_->createIblSheenMap(
        cmd_buf,
        cubemap_render_pass_,
        clear_values_,
        kCubemapSize);
 
    {
        // only init one time.
        static bool s_tile_buffer_inited = false;
        if (!s_tile_buffer_inited) {
            ego::TileObject::generateTileBuffers(cmd_buf);
            weather_system_->initTemperatureBuffer(cmd_buf);
            s_tile_buffer_inited = true;
        }
        else {
            ego::TileObject::updateTileFlowBuffers(cmd_buf, current_time_, s_dbuf_idx);
            ego::TileObject::updateTileBuffers(cmd_buf, current_time_, s_dbuf_idx);
            weather_system_->updateAirflowBuffer(cmd_buf, menu_->getWeatherControls(), s_dbuf_idx, current_time);
        }
    }

    // this has to be happened after tile update, or you wont get the right height info.
    {
        ego::GltfObject::updateGameObjectsBuffer(
            cmd_buf,
            ego::TileObject::getWorldMin(),
            ego::TileObject::getWorldRange(),
            s_camera_pos,
            menu_->getAirFlowStrength(),
            menu_->getWaterFlowStrength(),
            s_update_frame_count,
            s_dbuf_idx,
            menu_->isAirfowOn());

        for (auto& gltf_obj : gltf_objects_) {
            gltf_obj->updateBuffers(cmd_buf);
        }

        if (s_update_frame_count >= 0) {
            s_update_frame_count++;
        }
    }

    {
        cmd_buf->beginRenderPass(
            hdr_render_pass_,
            hdr_frame_buffer_,
            screen_size,
            clear_values_);

        er::DescriptorSetList desc_sets{ global_tex_desc_set_, frame_desc_set };
        // render gltf.
        {
            for (auto& gltf_obj : gltf_objects_) {
                gltf_obj->draw(cmd_buf, desc_sets);
            }
        }

        // render terrain opaque pass.
        {
            ego::TileObject::drawAllVisibleTiles(
                cmd_buf,
                desc_sets,
                screen_size,
                s_dbuf_idx,
                delta_t,
                current_time,
                true);
        }

        if (menu_->getDebugDrawType() != NO_DEBUG_DRAW) {
            ego::DebugDrawObject::draw(
                cmd_buf,
                desc_sets,
                s_camera_pos,
                menu_->getDebugDrawType());
        }

        // render skybox.
        {
            skydome_->draw(cmd_buf, frame_desc_set);
        }

        cmd_buf->endRenderPass();

        er::ImageResourceInfo color_src_info = {
            er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT) };

        er::ImageResourceInfo color_dst_info = {
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };

        er::ImageResourceInfo depth_src_info = {
            er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            SET_FLAG_BIT(Access, DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, EARLY_FRAGMENT_TESTS_BIT) };

        er::ImageResourceInfo depth_dst_info = {
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };

        if (!menu_->isWaterPassTurnOff()) {
            er::Helper::blitImage(
                cmd_buf,
                hdr_color_buffer_.image,
                hdr_color_buffer_copy_.image,
                color_src_info,
                color_src_info,
                color_dst_info,
                color_dst_info,
                SET_FLAG_BIT(ImageAspect, COLOR_BIT),
                SET_FLAG_BIT(ImageAspect, COLOR_BIT),
                glm::ivec3(screen_size.x, screen_size.y, 1));

            er::Helper::blitImage(
                cmd_buf,
                depth_buffer_.image,
                depth_buffer_copy_.image,
                depth_src_info,
                depth_src_info,
                depth_dst_info,
                depth_dst_info,
                SET_FLAG_BIT(ImageAspect, DEPTH_BIT),
                SET_FLAG_BIT(ImageAspect, DEPTH_BIT),
                glm::ivec3(screen_size.x, screen_size.y, 1));
        }

        {
            // render terrain water pass.
            if (!menu_->isWaterPassTurnOff())
            {
                cmd_buf->beginRenderPass(
                    hdr_water_render_pass_,
                    hdr_water_frame_buffer_,
                    screen_size,
                    clear_values_);

                ego::TileObject::drawAllVisibleTiles(
                    cmd_buf,
                    desc_sets,
                    screen_size,
                    s_dbuf_idx,
                    delta_t,
                    current_time,
                    false);
                cmd_buf->endRenderPass();
            }

            er::Helper::blitImage(
                cmd_buf,
                hdr_color_buffer_.image,
                hdr_color_buffer_copy_.image,
                color_src_info,
                color_src_info,
                color_dst_info,
                color_dst_info,
                SET_FLAG_BIT(ImageAspect, COLOR_BIT),
                SET_FLAG_BIT(ImageAspect, COLOR_BIT),
                glm::ivec3(screen_size.x, screen_size.y, 1));

            er::Helper::blitImage(
                cmd_buf,
                depth_buffer_.image,
                depth_buffer_copy_.image,
                depth_src_info,
                depth_src_info,
                depth_dst_info,
                depth_dst_info,
                SET_FLAG_BIT(ImageAspect, DEPTH_BIT),
                SET_FLAG_BIT(ImageAspect, DEPTH_BIT),
                glm::ivec3(screen_size.x, screen_size.y, 1));

            if (!menu_->isVolumeMoistTurnOff()) {
                cmd_buf->beginRenderPass(
                    hdr_water_render_pass_,
                    hdr_water_frame_buffer_,
                    screen_size,
                    clear_values_);

                volume_cloud_->drawVolumeMoisture(
                    cmd_buf,
                    frame_desc_set,
                    s_dbuf_idx,
                    skydome_->getSunDir());

                cmd_buf->endRenderPass();
            }

            
        }
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

    s_dbuf_idx = 1 - s_dbuf_idx;
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

    time_t now = time(0);
    tm* localtm = gmtime(&now);

    float latitude = 37.4419f;
    float longtitude = -122.1430f; // west.

    skydome_->update(latitude, longtitude, localtm->tm_yday, 22/*localtm->tm_hour*/, localtm->tm_min, localtm->tm_sec);

    device_->resetFences(in_flight_fences);

    ego::TileObject::updateAllTiles(
        device_info_,
        descriptor_pool_,
        1024,
        glm::vec2(s_camera_pos.x, s_camera_pos.z));

    auto command_buffer = command_buffers_[image_index];
    std::vector<std::shared_ptr<er::CommandBuffer>>command_buffers(1, command_buffer);

    if (current_time_ == 0) {
        last_frame_time_point_ = std::chrono::high_resolution_clock::now();
    }

    auto current_time_point = std::chrono::high_resolution_clock::now();
    delta_t_ = std::chrono::duration<float, std::chrono::seconds::period>(
                    current_time_point - last_frame_time_point_).count();

    current_time_ += delta_t_;
    last_frame_time_point_ = current_time_point;

    auto to_load_gltf_names = menu_->getToLoadGltfNamesAndClear();
    for (auto& gltf_name : to_load_gltf_names) {
        auto gltf_obj = std::make_shared<ego::GltfObject>(
            device_info_,
            descriptor_pool_,
            hdr_render_pass_,
            graphic_pipeline_info_,
            texture_sampler_,
            thin_film_lut_tex_,
            gltf_name,
            swap_chain_info_.extent,
            glm::inverse(view_params_.view));

        s_update_frame_count = -1;
        gltf_objects_.push_back(gltf_obj);
    }

    command_buffer->reset(0);
    command_buffer->beginCommandBuffer(SET_FLAG_BIT(CommandBufferUsage, ONE_TIME_SUBMIT_BIT));

    drawScene(command_buffer,
        swap_chain_info_,
        desc_sets_,
        swap_chain_info_.extent,
        image_index,
        delta_t_,
        current_time_);

    s_camera_paused = menu_->draw(
        command_buffer,
        final_render_pass_,
        swap_chain_info_,
        swap_chain_info_.extent,
        image_index);

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
    if (s_update_frame_count < 0) {
        s_update_frame_count = 0;
    }
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
    device_->destroyRenderPass(final_render_pass_);
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

    skydome_->destroy(device_);
    menu_->destroy();
    device_->destroyRenderPass(cubemap_render_pass_);

    assert(device_);
    device_->destroySampler(texture_sampler_);
    sample_tex_.destroy(device_);
    er::Helper::destroy(device_);
    ggx_lut_tex_.destroy(device_);
    brdf_lut_tex_.destroy(device_);
    charlie_lut_tex_.destroy(device_);
    thin_film_lut_tex_.destroy(device_);
    ibl_diffuse_tex_.destroy(device_);
    ibl_specular_tex_.destroy(device_);
    ibl_sheen_tex_.destroy(device_);
    device_->destroyDescriptorSetLayout(view_desc_set_layout_);
    device_->destroyDescriptorSetLayout(global_tex_desc_set_layout_);
    
    ego::TileObject::destoryAllTiles();
    ego::TileObject::destoryStaticMembers(device_);
    gltf_objects_.clear();
    ego::GltfObject::destoryStaticMembers(device_);

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
