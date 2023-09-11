#include <iostream>
#include <vector>
#include <map>
#include <limits>
#include <chrono>
#include <string>
#include <filesystem>
#include "Windows.h"

#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "ray_tracing/raytracing_callable.h"
#include "ray_tracing/raytracing_shadow.h"
#include "engine_helper.h"
#include "application.h"

namespace er = engine::renderer;
namespace ego = engine::game_object;

namespace {
constexpr int kWindowSizeX = 2560;
constexpr int kWindowSizeY = 1440;
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
    auto error_strings =
        eh::initCompileGlobalShaders(
            "src\\sim_engine\\shaders",
            "lib\\shaders",
            "src\\sim_engine\\third_parties\\vulkan_lib");
    if (error_strings.length() > 0) {
        MessageBoxA(NULL, error_strings.c_str(), "Shader Error!", MB_OK);
    }
    initWindow();
    initVulkan();
    initDrawFrame();
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
static int s_key = 0;
static float s_mouse_wheel_offset = 0.0f;
const float s_camera_speed = 10.0f;

static void keyInputCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    s_key = 0;
    if (action != GLFW_RELEASE && !s_camera_paused) {
        s_key = key;
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

    s_last_mouse_pos = cur_mouse_pos;
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

void mouseWheelCallback(GLFWwindow* window, double xoffset, double yoffset) {
    s_mouse_wheel_offset = static_cast<float>(yoffset);
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
    glfwSetScrollCallback(window_, mouseWheelCallback);
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
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT) |
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
    static auto color_no_blend_attachment = er::helper::fillPipelineColorBlendAttachmentState();
    static auto color_blend_attachment =
        er::helper::fillPipelineColorBlendAttachmentState(
            SET_FLAG_BIT(ColorComponent, ALL_BITS),
            true,
            er::BlendFactor::ONE,
            er::BlendFactor::SRC_ALPHA,
            er::BlendOp::ADD,
            er::BlendFactor::ONE,
            er::BlendFactor::ZERO,
            er::BlendOp::ADD);
    static std::vector<er::PipelineColorBlendAttachmentState> color_no_blend_attachments(1, color_no_blend_attachment);
    static std::vector<er::PipelineColorBlendAttachmentState> color_blend_attachments(1, color_blend_attachment);
    static std::vector<er::PipelineColorBlendAttachmentState> cube_color_no_blend_attachments(6, color_no_blend_attachment);

    auto single_no_blend_state_info =
        std::make_shared<er::PipelineColorBlendStateCreateInfo>(
            er::helper::fillPipelineColorBlendStateCreateInfo(color_no_blend_attachments));

    auto single_blend_state_info =
        std::make_shared<er::PipelineColorBlendStateCreateInfo>(
            er::helper::fillPipelineColorBlendStateCreateInfo(color_blend_attachments));

    auto cube_no_blend_state_info =
        std::make_shared<er::PipelineColorBlendStateCreateInfo>(
            er::helper::fillPipelineColorBlendStateCreateInfo(cube_color_no_blend_attachments));

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

    auto depth_no_write_stencil_info =
        std::make_shared<er::PipelineDepthStencilStateCreateInfo>(
            er::helper::fillPipelineDepthStencilStateCreateInfo(
                true, false));

    auto fs_depth_stencil_info =
        std::make_shared<er::PipelineDepthStencilStateCreateInfo>(
            er::helper::fillPipelineDepthStencilStateCreateInfo(
                false,
                false,
                er::CompareOp::ALWAYS));

    graphic_pipeline_info_.blend_state_info = single_no_blend_state_info;
    graphic_pipeline_info_.rasterization_info = cull_rasterization_info;
    graphic_pipeline_info_.ms_info = ms_info;
    graphic_pipeline_info_.depth_stencil_info = depth_stencil_info;

    graphic_double_face_pipeline_info_.blend_state_info = single_no_blend_state_info;
    graphic_double_face_pipeline_info_.rasterization_info = no_cull_rasterization_info;
    graphic_double_face_pipeline_info_.ms_info = ms_info;
    graphic_double_face_pipeline_info_.depth_stencil_info = depth_stencil_info;

    graphic_no_depth_write_pipeline_info_.blend_state_info = single_no_blend_state_info;
    graphic_no_depth_write_pipeline_info_.rasterization_info = cull_rasterization_info;
    graphic_no_depth_write_pipeline_info_.ms_info = ms_info;
    graphic_no_depth_write_pipeline_info_.depth_stencil_info = depth_no_write_stencil_info;
        
    graphic_fs_pipeline_info_.blend_state_info = single_no_blend_state_info;
    graphic_fs_pipeline_info_.rasterization_info = no_cull_rasterization_info;
    graphic_fs_pipeline_info_.ms_info = ms_info;
    graphic_fs_pipeline_info_.depth_stencil_info = fs_depth_stencil_info;

    graphic_fs_blend_pipeline_info_.blend_state_info = single_blend_state_info;
    graphic_fs_blend_pipeline_info_.rasterization_info = no_cull_rasterization_info;
    graphic_fs_blend_pipeline_info_.ms_info = ms_info;
    graphic_fs_blend_pipeline_info_.depth_stencil_info = fs_depth_stencil_info;

    graphic_cubemap_pipeline_info_.blend_state_info = cube_no_blend_state_info;
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
    er::Helper::initRayTracingProperties(physical_device_, device_, rt_pipeline_properties_, as_features_);
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

    eh::loadMtx2Texture(
        device_info_,
        cubemap_render_pass_,
        "assets/environments/doge2/lambertian/diffuse.ktx2",
        ibl_diffuse_tex_);
    eh::loadMtx2Texture(
        device_info_,
        cubemap_render_pass_,
        "assets/environments/doge2/ggx/specular.ktx2",
        ibl_specular_tex_);
    eh::loadMtx2Texture(
        device_info_,
        cubemap_render_pass_,
        "assets/environments/doge2/charlie/sheen.ktx2",
        ibl_sheen_tex_);
    recreateRenderBuffer(swap_chain_info_.extent);
    auto format = er::Format::R8G8B8A8_UNORM;
    eh::createTextureImage(device_info_, "assets/statue.jpg", format, sample_tex_);
    eh::createTextureImage(device_info_, "assets/brdfLUT.png", format, brdf_lut_tex_);
    eh::createTextureImage(device_info_, "assets/lut_ggx.png", format, ggx_lut_tex_);
    eh::createTextureImage(device_info_, "assets/lut_charlie.png", format, charlie_lut_tex_);
    eh::createTextureImage(device_info_, "assets/lut_thin_film.png", format, thin_film_lut_tex_);
    eh::createTextureImage(device_info_, "assets/map_mask.png", format, map_mask_tex_);
    eh::createTextureImage(device_info_, "assets/map.png", er::Format::R16_UNORM, heightmap_tex_);
//    eh::createTextureImage(device_info_, "assets/tile1.jpg", format, prt_base_tex_);
//    eh::createTextureImage(device_info_, "assets/tile1.tga", format, prt_bump_tex_);
//    eh::createTextureImage(device_info_, "assets/T_Mat4Mural_C.PNG", format, prt_base_tex_);
//    eh::createTextureImage(device_info_, "assets/T_Mat4Mural_H.PNG", format, prt_bump_tex_);
    eh::createTextureImage(device_info_, "assets/T_Mat2Mountains_C.jpg", format, prt_base_tex_);
    eh::createTextureImage(device_info_, "assets/T_Mat2Mountains_ORH.jpg", format, prt_bump_tex_);
    createTextureSampler();
    descriptor_pool_ = device_->createDescriptorPool();
    createCommandBuffers();
    createSyncObjects();

    auto desc_set_layouts = { global_tex_desc_set_layout_, view_desc_set_layout_ };

    prt_gen_ =
        std::make_shared<es::Prt>(
            device_info_,
            descriptor_pool_,
            texture_sampler_);

    cone_map_obj_ =
        std::make_shared<ego::ConeMapObj>(
            device_info_,
            descriptor_pool_,
            texture_sampler_,
            prt_bump_tex_,
            prt_gen_,
            2,
            0.05f,
            0.15f);

    unit_plane_ = std::make_shared<ego::Plane>(device_info_);
    prt_test_ = std::make_shared<ego::PrtTest>(
        device_info_,
        descriptor_pool_,
        hdr_render_pass_,
        graphic_pipeline_info_,
        desc_set_layouts,
        texture_sampler_,
        prt_base_tex_,
        prt_bump_tex_,
        cone_map_obj_,
        swap_chain_info_.extent,
        unit_plane_);

    clear_values_.resize(2);
    clear_values_[0].color = { 50.0f / 255.0f, 50.0f / 255.0f, 50.0f / 255.0f, 1.0f };
    clear_values_[1].depth_stencil = { 1.0f, 0 };

    cone_map_gen_ =
        std::make_shared<es::ConeMap>(
            device_info_,
            descriptor_pool_,
            texture_sampler_,
            prt_bump_tex_,
            *cone_map_obj_->getConemapTexture());

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
        graphic_no_depth_write_pipeline_info_,
        graphic_cubemap_pipeline_info_,
        ibl_creator_->getEnvmapTexture(),
        texture_sampler_,
        swap_chain_info_.extent,
        kCubemapSize);

    for (auto& image : swap_chain_info_.images) {
        er::Helper::transitionImageLayout(
            device_info_,
            image,
            swap_chain_info_.format,
            er::ImageLayout::UNDEFINED,
            er::ImageLayout::PRESENT_SRC_KHR);
    }

    ego::TileObject::initStaticMembers(
        device_info_,
        hdr_render_pass_,
        hdr_water_render_pass_,
        graphic_pipeline_info_,
        graphic_double_face_pipeline_info_,
        desc_set_layouts,
        swap_chain_info_.extent);

    std::vector<std::shared_ptr<er::ImageView>> soil_water_texes(2);
    soil_water_texes[0] = ego::TileObject::getSoilWaterLayer(0).view;
    soil_water_texes[1] = ego::TileObject::getSoilWaterLayer(1).view;
    weather_system_ = std::make_shared<es::WeatherSystem>(
        device_info_,
        descriptor_pool_,
        desc_set_layouts,
        mirror_repeat_sampler_,
        ego::TileObject::getRockLayer().view,
        soil_water_texes);

    ego::GltfObject::initGameObjectBuffer(device_);
    ego::GameCamera::initGameCameraBuffer(device_);
    assert(ego::GltfObject::getGameObjectsBuffer());
    assert(ego::GameCamera::getGameCameraBuffer());

    ego::GltfObject::initStaticMembers(
        device_,
        descriptor_pool_,
        desc_set_layouts,
        texture_sampler_,
        ego::GameCamera::getGameCameraBuffer(),
        ego::TileObject::getRockLayer(),
        ego::TileObject::getSoilWaterLayer(0),
        ego::TileObject::getSoilWaterLayer(1),
        ego::TileObject::getWaterFlow(),
        weather_system_->getAirflowTex());

    ego::GameCamera::initStaticMembers(
        device_,
        descriptor_pool_,
        desc_set_layouts,
        texture_sampler_,
        ego::TileObject::getRockLayer(),
        ego::TileObject::getSoilWaterLayer(0),
        ego::TileObject::getSoilWaterLayer(1),
        *ego::GltfObject::getGameObjectsBuffer());

    ego::DebugDrawObject::initStaticMembers(
        device_info_,
        hdr_render_pass_,
        graphic_pipeline_info_,
        desc_set_layouts,
        swap_chain_info_.extent);

    createDescriptorSets();

    volume_noise_ = std::make_shared<es::VolumeNoise>(
        device_info_,
        descriptor_pool_,
        hdr_render_pass_,
        view_desc_set_layout_,
        global_tex_desc_set_layout_,
        graphic_pipeline_info_,
        texture_sampler_,
        texture_point_sampler_,
        swap_chain_info_.extent);

    ego::TileObject::updateStaticDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        repeat_texture_sampler_,
        hdr_color_buffer_copy_.view,
        depth_buffer_copy_.view,
        weather_system_->getTempTexes(),
        heightmap_tex_.view,
        map_mask_tex_.view,
        volume_noise_->getDetailNoiseTexture().view,
        volume_noise_->getRoughNoiseTexture().view);

    volume_cloud_ = std::make_shared<es::VolumeCloud>(
        device_info_,
        descriptor_pool_,
        view_desc_set_layout_,
        mirror_repeat_sampler_,
        texture_point_sampler_,
        depth_buffer_copy_.view,
        hdr_color_buffer_.view,
        weather_system_->getMoistureTexes(),
        weather_system_->getTempTexes(),
        weather_system_->getCloudLightingTex(),
        skydome_->getScatteringLutTex(),
        volume_noise_->getDetailNoiseTexture().view,
        volume_noise_->getRoughNoiseTexture().view,
        swap_chain_info_.extent);

    ego::DebugDrawObject::updateStaticDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        weather_system_->getTempTex(0),
        weather_system_->getMoistureTex(0),
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
        graphic_double_face_pipeline_info_,
        desc_set_layouts,
        swap_chain_info_.extent);
    ego::GltfObject::recreateStaticMembers(
        device_,
        hdr_render_pass_,
        graphic_pipeline_info_,
        desc_set_layouts,
        swap_chain_info_.extent);
    ego::GameCamera::recreateStaticMembers(
        device_);
    ego::DebugDrawObject::recreateStaticMembers(
        device_,
        hdr_render_pass_,
        graphic_pipeline_info_,
        desc_set_layouts,
        swap_chain_info_.extent);
    recreateRenderBuffer(swap_chain_info_.extent);
    descriptor_pool_ = device_->createDescriptorPool();
    createCommandBuffers();

    skydome_->recreate(
        device_,
        descriptor_pool_,
        hdr_render_pass_,
        view_desc_set_layout_,
        graphic_no_depth_write_pipeline_info_,
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
        desc_set_layouts,
        mirror_repeat_sampler_,
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
        texture_sampler_,
        heightmap_tex_.view);

    ego::DebugDrawObject::generateAllDescriptorSets(
        device_,
        descriptor_pool_,
        texture_sampler_,
        weather_system_->getTempTex(0),
        weather_system_->getMoistureTex(0),
        weather_system_->getAirflowTex());

    ego::GltfObject::generateDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        ego::GameCamera::getGameCameraBuffer(),
        thin_film_lut_tex_,
        ego::TileObject::getRockLayer(),
        ego::TileObject::getSoilWaterLayer(0),
        ego::TileObject::getSoilWaterLayer(1),
        ego::TileObject::getWaterFlow(),
        weather_system_->getAirflowTex());

    assert(ego::GltfObject::getGameObjectsBuffer());
    ego::GameCamera::generateDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        ego::TileObject::getRockLayer(),
        ego::TileObject::getSoilWaterLayer(0),
        ego::TileObject::getSoilWaterLayer(1),
        *ego::GltfObject::getGameObjectsBuffer());

    ego::TileObject::updateStaticDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        repeat_texture_sampler_,
        hdr_color_buffer_copy_.view,
        depth_buffer_copy_.view,
        weather_system_->getTempTexes(),
        heightmap_tex_.view,
        map_mask_tex_.view,
        volume_noise_->getDetailNoiseTexture().view,
        volume_noise_->getRoughNoiseTexture().view);

    ego::DebugDrawObject::updateStaticDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        weather_system_->getTempTex(0),
        weather_system_->getMoistureTex(0),
        weather_system_->getAirflowTex());

    volume_noise_->recreate(
        device_,
        descriptor_pool_,
        hdr_render_pass_,
        view_desc_set_layout_,
        graphic_pipeline_info_,
        texture_sampler_,
        texture_point_sampler_,
        swap_chain_info_.extent);

    volume_cloud_->recreate(
        device_info_,
        descriptor_pool_,
        view_desc_set_layout_,
        mirror_repeat_sampler_,
        texture_point_sampler_,
        depth_buffer_copy_.view,
        hdr_color_buffer_.view,
        weather_system_->getMoistureTexes(),
        weather_system_->getTempTexes(),
        weather_system_->getCloudLightingTex(),
        skydome_->getScatteringLutTex(),
        volume_noise_->getDetailNoiseTexture().view,
        volume_noise_->getRoughNoiseTexture().view,
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
    init_command_buffers_ =
        device_->allocateCommandBuffers(
            command_pool_,
            1);

    command_buffers_ = 
        device_->allocateCommandBuffers(
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
    init_fence_ = device_->createFence();
    init_semaphore_ = device_->createSemaphore();

    assert(device_);
    for (uint64_t i = 0; i < kMaxFramesInFlight; i++) {
        image_available_semaphores_[i] = device_->createSemaphore();
        render_finished_semaphores_[i] = device_->createSemaphore();
        in_flight_fences_[i] = device_->createFence(true);
    }
}

void RealWorldApplication::createDescriptorSetLayout() {
    // global texture descriptor set layout.
    {
        std::vector<er::DescriptorSetLayoutBinding> bindings;
        bindings.reserve(5);

        bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            GGX_LUT_INDEX,
            SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT)));
        bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            CHARLIE_LUT_INDEX,
            SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT)));
        bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            LAMBERTIAN_ENV_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT)));
        bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            GGX_ENV_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT)));
        bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            CHARLIE_ENV_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT)));

        global_tex_desc_set_layout_ = device_->createDescriptorSetLayout(bindings);
    }

    {
        std::vector<er::DescriptorSetLayoutBinding> bindings(1);
        bindings[0].binding = VIEW_CAMERA_BUFFER_INDEX;
        bindings[0].descriptor_count = 1;
        bindings[0].descriptor_type = er::DescriptorType::STORAGE_BUFFER;
        bindings[0].stage_flags =
            SET_FLAG_BIT(ShaderStage, VERTEX_BIT) |
            SET_FLAG_BIT(ShaderStage, MESH_BIT_NV) |
            SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) |
            SET_FLAG_BIT(ShaderStage, GEOMETRY_BIT) |
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
        bindings[0].immutable_samplers = nullptr; // Optional

        view_desc_set_layout_ = device_->createDescriptorSetLayout(bindings);
    }
}

er::WriteDescriptorList RealWorldApplication::addGlobalTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set)
{
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(5);
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        GGX_LUT_INDEX,
        texture_sampler_,
        ggx_lut_tex_.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        CHARLIE_LUT_INDEX,
        texture_sampler_,
        charlie_lut_tex_.view,
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
        device_->updateDescriptorSets(global_texture_descs);
    }

    {
        view_desc_set_ = device_->createDescriptorSets(descriptor_pool_, view_desc_set_layout_, 1)[0];
        er::WriteDescriptorList buffer_descs;
        buffer_descs.reserve(1);
        er::Helper::addOneBuffer(
            buffer_descs,
            view_desc_set_,
            er::DescriptorType::STORAGE_BUFFER,
            VIEW_CAMERA_BUFFER_INDEX,
            ego::GameCamera::getGameCameraBuffer()->buffer,
            sizeof(glsl::GameCameraInfo));
        device_->updateDescriptorSets(buffer_descs);
    }
}

void RealWorldApplication::createTextureSampler() {
    texture_sampler_ = device_->createSampler(
        er::Filter::LINEAR,
        er::SamplerAddressMode::CLAMP_TO_EDGE,
        er::SamplerMipmapMode::LINEAR, 16.0f);

    texture_point_sampler_ = device_->createSampler(
        er::Filter::NEAREST,
        er::SamplerAddressMode::REPEAT,
        er::SamplerMipmapMode::LINEAR, 16.0f);

    repeat_texture_sampler_ = device_->createSampler(
        er::Filter::LINEAR,
        er::SamplerAddressMode::REPEAT,
        er::SamplerMipmapMode::LINEAR, 16.0f);

    mirror_repeat_sampler_ = device_->createSampler(
        er::Filter::LINEAR,
        er::SamplerAddressMode::MIRRORED_REPEAT,
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
    const std::shared_ptr<er::DescriptorSet>& view_desc_set,
    const glm::uvec2& screen_size,
    uint32_t image_index,
    float delta_t,
    float current_time) {

    auto frame_buffer = swap_chain_info.framebuffers[image_index];
    auto src_color = swap_chain_info.image_views[image_index];

    auto& cmd_buf = command_buffer;

    static int s_dbuf_idx = 0;
    bool render_prt_test = true;

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
 
    er::DescriptorSetList desc_sets{ global_tex_desc_set_, view_desc_set };

    {
        // only init one time.
        static bool s_tile_buffer_inited = false;
        if (!s_tile_buffer_inited) {
            ego::TileObject::generateTileBuffers(cmd_buf);
            weather_system_->initTemperatureBuffer(cmd_buf);
            volume_noise_->initNoiseTexture(cmd_buf);
            s_tile_buffer_inited = true;
        }
        else {
            ego::TileObject::updateTileFlowBuffers(cmd_buf, current_time_, s_dbuf_idx);
            ego::TileObject::updateTileBuffers(cmd_buf, current_time_, s_dbuf_idx);
            weather_system_->updateAirflowBuffer(
                cmd_buf,
                menu_->getWeatherControls(),
                menu_->getGloalFlowDir(),
                menu_->getGlobalFlowSpeed(),
                menu_->getCloudMoistToPressureRatio(),
                s_dbuf_idx,
                current_time);
        }

        skydome_->updateSkyScatteringLut(cmd_buf);

        weather_system_->updateCloudShadow(
            cmd_buf,
            skydome_->getSunDir(),
            menu_->getLightExtFactor(),
            s_dbuf_idx,
            current_time);
    }

    // this has to be happened after tile update, or you wont get the right height info.
    {
        static std::chrono::time_point s_last_time = std::chrono::steady_clock::now();
        auto cur_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = cur_time - s_last_time;
        auto delta_t = static_cast<float>(elapsed_seconds.count());
        s_last_time = cur_time;

        ego::GltfObject::updateGameObjectsBuffer(
            cmd_buf,
            ego::TileObject::getWorldMin(),
            ego::TileObject::getWorldRange(),
            gpu_game_camera_info_.position,
            menu_->getAirFlowStrength(),
            menu_->getWaterFlowStrength(),
            s_update_frame_count,
            s_dbuf_idx,
            delta_t,
            menu_->isAirfowOn());

        glsl::GameCameraParams game_camera_params;
        game_camera_params.world_min = ego::TileObject::getWorldMin();
        game_camera_params.inv_world_range = 1.0f / ego::TileObject::getWorldRange();
        if (render_prt_test) {
            game_camera_params.init_camera_pos = glm::vec3(0, 5.0f, 0);
            game_camera_params.init_camera_dir = glm::vec3(0.0f, -1.0f, 0.0f);
            game_camera_params.init_camera_up = glm::vec3(1, 0, 0);
            game_camera_params.camera_speed = 0.1f;
        }
        else {
            game_camera_params.init_camera_pos = glm::vec3(0, 500.0f, 0);
            game_camera_params.init_camera_dir = glm::vec3(1.0f, 0.0f, 0.0f);
            game_camera_params.init_camera_up = glm::vec3(0, 1, 0);
            game_camera_params.camera_speed = s_camera_speed;
        }
        game_camera_params.z_near = 0.1f;
        game_camera_params.z_far = 40000.0f;
        game_camera_params.yaw = 0.0f;
        game_camera_params.pitch = 0.0f;
        game_camera_params.camera_follow_dist = 5.0f;
        game_camera_params.key = s_key;
        game_camera_params.frame_count = s_update_frame_count;
        game_camera_params.delta_t = delta_t;
        game_camera_params.mouse_pos = s_last_mouse_pos;
        game_camera_params.fov = glm::radians(45.0f);
        game_camera_params.aspect = swap_chain_info_.extent.x / (float)swap_chain_info_.extent.y;
        game_camera_params.sensitivity = 0.2f;
        game_camera_params.num_game_objs = static_cast<int32_t>(gltf_objects_.size());
        game_camera_params.game_obj_idx = 0;
        game_camera_params.camera_rot_update = (!s_camera_paused && s_mouse_right_button_pressed) ? 1 : 0;
        game_camera_params.mouse_wheel_offset = s_mouse_wheel_offset;
        s_mouse_wheel_offset = 0;

        ego::GameCamera::updateGameCameraBuffer(
            cmd_buf,
            game_camera_params,
            s_dbuf_idx);

        if (player_object_) {
            player_object_->updateBuffers(cmd_buf);
        }

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

        if (render_prt_test) {
            prt_test_->draw(
                cmd_buf,
                desc_sets,
                unit_plane_,
                cone_map_obj_);
            
        }
        else {
            // render gltf.
            {
                for (auto& gltf_obj : gltf_objects_) {
                    gltf_obj->draw(cmd_buf, desc_sets);
                }
            }

            if (player_object_) {
                player_object_->draw(cmd_buf, desc_sets);
            }

            // render terrain opaque pass.
            {
                ego::TileObject::drawAllVisibleTiles(
                    cmd_buf,
                    desc_sets,
                    glm::vec2(gpu_game_camera_info_.position.x, gpu_game_camera_info_.position.z),
                    screen_size,
                    s_dbuf_idx,
                    delta_t,
                    current_time,
                    true,
                    !menu_->isGrassPassTurnOff());
            }

            if (menu_->getDebugDrawType() != NO_DEBUG_DRAW) {
                ego::DebugDrawObject::draw(
                    cmd_buf,
                    desc_sets,
                    gpu_game_camera_info_.position,
                    menu_->getDebugDrawType());
            }

            // render skybox.
            {
                skydome_->draw(cmd_buf, view_desc_set);
            }
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

        if (!render_prt_test) {
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
                    glm::vec2(gpu_game_camera_info_.position.x, gpu_game_camera_info_.position.z),
                    screen_size,
                    s_dbuf_idx,
                    delta_t,
                    current_time,
                    false);

                cmd_buf->endRenderPass();
            }

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
                volume_cloud_->renderVolumeCloud(
                    cmd_buf,
                    view_desc_set,
                    hdr_color_buffer_.image,
                    skydome_,
                    menu_->getViewExtFactor(),
                    menu_->getViewExtExponent(),
                    menu_->getCloudAmbientIntensity(),
                    menu_->getCloudPhaseIntensity(),
                    menu_->getCloudMoistToPressureRatio(),
                    menu_->getCloudNoiseWeight(0),
                    menu_->getCloudNoiseWeight(1),
                    menu_->getCloudNoiseThresold(),
                    menu_->getCloudNoiseScrollingSpeed(),
                    menu_->getCloudNoiseScale(),
                    screen_size,
                    s_dbuf_idx,
                    current_time);
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

void RealWorldApplication::initDrawFrame() {
    auto& command_buffer = init_command_buffers_[0];

    command_buffer->reset(0);
    command_buffer->beginCommandBuffer(
        SET_FLAG_BIT(CommandBufferUsage, ONE_TIME_SUBMIT_BIT));

    cone_map_gen_->update(
        command_buffer,
        cone_map_obj_);

    prt_gen_->update(
        command_buffer,
        cone_map_obj_);

    command_buffer->endCommandBuffer();

    er::Helper::submitQueue(
        graphics_queue_,
        init_fence_,
        { },
        { command_buffer },
        { },
        { });

    device_->waitForFences({ init_fence_ });
    device_->resetFences({ init_fence_ });

    prt_gen_->destroy(device_);
}

void RealWorldApplication::drawFrame() {
    device_->waitForFences({ in_flight_fences_[current_frame_] });
    device_->resetFences({ in_flight_fences_[current_frame_] });

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
        device_->waitForFences({ images_in_flight_[image_index] });
    }
    // Mark the image as now being in use by this frame
    images_in_flight_[image_index] = in_flight_fences_[current_frame_];

    time_t now = time(0);
    tm localtm;
    gmtime_s(&localtm, &now);

    float latitude = 37.4419f;
    float longtitude = -122.1430f; // west.

    skydome_->update(latitude, longtitude, localtm.tm_yday, 22/*localtm->tm_hour*/, localtm.tm_min, localtm.tm_sec);

    gpu_game_camera_info_ = ego::GameCamera::readCameraInfo(
        device_info_.device,
        0);

    ego::TileObject::updateAllTiles(
        device_info_,
        descriptor_pool_,
        128,
        glm::vec2(gpu_game_camera_info_.position.x, gpu_game_camera_info_.position.z));

    if (dump_volume_noise_) {
        const auto noise_texture_size = kDetailNoiseTextureSize;
#if 1
        uint32_t pixel_count = noise_texture_size * noise_texture_size * noise_texture_size;
        std::vector<uint32_t> temp_buffer;
        temp_buffer.resize(pixel_count);
        er::Helper::dumpTextureImage(
            device_info_,
            volume_noise_->getDetailNoiseTexture().image,
            er::Format::R8G8B8A8_UNORM,
            glm::uvec3(noise_texture_size, noise_texture_size, noise_texture_size),
            4,
            temp_buffer.data());
        
        eh::saveDdsTexture(
            glm::uvec3(noise_texture_size, noise_texture_size, noise_texture_size),
            temp_buffer.data(),
            "volume_noise.dds");
#else
        uint32_t pixel_count = noise_texture_size * noise_texture_size * noise_texture_size;
        std::vector<uint32_t> temp_buffer;
        temp_buffer.resize(pixel_count);
        er::Helper::dumpTextureImage(
            device_info_,
            volume_noise_->getDetailNoiseTexture().image,
            er::Format::R8G8B8A8_UNORM,
            glm::uvec3(noise_texture_size, noise_texture_size, noise_texture_size),
            4,
            temp_buffer.data());

        // flatten 3d texture to 2d texture.
        uint32_t h = 1 << (31 - eh::clz(static_cast<uint32_t>(std::sqrt(noise_texture_size))));
        uint32_t w = noise_texture_size / h;
        std::vector<uint32_t> flattened_buffer;
        flattened_buffer.resize(pixel_count);
        for (uint32_t y = 0; y < h * noise_texture_size; y++) {
            for (uint32_t x = 0; x < w * noise_texture_size; x++) {
                uint32_t tile_y = y / noise_texture_size;
                uint32_t tile_x = x / noise_texture_size;
                uint32_t depth = tile_y * w + tile_x;
                uint32_t dst_idx = y * (w * noise_texture_size) + x;
                uint32_t src_idx = depth * noise_texture_size * noise_texture_size +
                    (y % noise_texture_size) * noise_texture_size + (x % noise_texture_size);
                flattened_buffer[dst_idx] = temp_buffer[src_idx];
            }
        }
        eh::saveDdsTexture(
            glm::uvec3(w * noise_texture_size, h * noise_texture_size, 1),
            flattened_buffer.data(),
            "volume_noise.dds");
#endif
        dump_volume_noise_ = false;
    }

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

    auto to_load_player_name = menu_->getToLoadPlayerNameAndClear();
    if (to_load_player_name != "") {
        if (player_object_) {
            player_object_->destroy();
            player_object_ = nullptr;
        }
        player_object_ = std::make_shared<ego::GltfObject>(
            device_info_,
            descriptor_pool_,
            hdr_render_pass_,
            graphic_pipeline_info_,
            texture_sampler_,
            thin_film_lut_tex_,
            to_load_player_name,
            swap_chain_info_.extent,
            glm::inverse(view_params_.view));
    }

    if (player_object_) {
        player_object_->update(device_info_, current_time_);
    }

    for (auto& gltf_obj : gltf_objects_) {
        gltf_obj->update(device_info_, current_time_);
    }

    command_buffer->reset(0);
    command_buffer->beginCommandBuffer(SET_FLAG_BIT(CommandBufferUsage, ONE_TIME_SUBMIT_BIT));

    drawScene(command_buffer,
        swap_chain_info_,
        view_desc_set_,
        swap_chain_info_.extent,
        image_index,
        delta_t_,
        current_time_);

    if (!menu_->isRayTracingTurnOff()) {
        // ray tracing test.
        if (ray_tracing_test_ == nullptr) {
            ray_tracing_test_ = std::make_shared<engine::ray_tracing::RayTracingShadowTest>();
            ray_tracing_test_->init(
                device_info_,
                descriptor_pool_,
                rt_pipeline_properties_,
                as_features_,
                glm::uvec2(1920, 1080));
        }

        auto result_image =
            ray_tracing_test_->draw(
                device_info_,
                command_buffer,
                view_params_);

        er::ImageResourceInfo src_info = {
            er::ImageLayout::GENERAL,
            SET_FLAG_BIT(Access, SHADER_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, RAY_TRACING_SHADER_BIT_KHR) };

        er::ImageResourceInfo dst_info = {
            er::ImageLayout::PRESENT_SRC_KHR,
            SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT) };

        er::Helper::blitImage(
            command_buffer,
            result_image.image,
            swap_chain_info_.images[image_index],
            src_info,
            src_info,
            dst_info,
            dst_info,
            SET_FLAG_BIT(ImageAspect, COLOR_BIT),
            SET_FLAG_BIT(ImageAspect, COLOR_BIT),
            glm::ivec3(1920, 1080, 1));
    }

    s_camera_paused = menu_->draw(
        command_buffer,
        final_render_pass_,
        swap_chain_info_,
        swap_chain_info_.extent,
        skydome_,
        image_index,
        dump_volume_noise_,
        delta_t_);

    command_buffer->endCommandBuffer();

    er::Helper::submitQueue(
        graphics_queue_,
        in_flight_fences_[current_frame_],
        { image_available_semaphores_[current_frame_] },
        { command_buffer },
        { render_finished_semaphores_[current_frame_] },
        { });

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
    depth_buffer_copy_.destroy(device_);
    hdr_color_buffer_.destroy(device_);
    hdr_color_buffer_copy_.destroy(device_);
    device_->destroyFramebuffer(hdr_frame_buffer_);
    device_->destroyFramebuffer(hdr_water_frame_buffer_);

    for (auto framebuffer : swap_chain_info_.framebuffers) {
        device_->destroyFramebuffer(framebuffer);
    }

    device_->freeCommandBuffers(command_pool_, command_buffers_);
    device_->freeCommandBuffers(command_pool_, init_command_buffers_);
    device_->destroyRenderPass(final_render_pass_);
    device_->destroyRenderPass(hdr_render_pass_);
    device_->destroyRenderPass(hdr_water_render_pass_);

    for (auto image_view : swap_chain_info_.image_views) {
        device_->destroyImageView(image_view);
    }

    device_->destroySwapchain(swap_chain_info_.swap_chain);

    device_->destroyDescriptorPool(descriptor_pool_);
}

void RealWorldApplication::cleanup() {
    cleanupSwapChain();

    skydome_->destroy(device_);
    menu_->destroy();
    device_->destroyRenderPass(cubemap_render_pass_);

    assert(device_);
    device_->destroySampler(texture_sampler_);
    device_->destroySampler(mirror_repeat_sampler_);
    sample_tex_.destroy(device_);
    er::Helper::destroy(device_);
    ggx_lut_tex_.destroy(device_);
    brdf_lut_tex_.destroy(device_);
    charlie_lut_tex_.destroy(device_);
    thin_film_lut_tex_.destroy(device_);
    map_mask_tex_.destroy(device_);
    heightmap_tex_.destroy(device_);
    prt_base_tex_.destroy(device_);
    prt_bump_tex_.destroy(device_);
    ibl_diffuse_tex_.destroy(device_);
    ibl_specular_tex_.destroy(device_);
    ibl_sheen_tex_.destroy(device_);
    device_->destroySampler(texture_sampler_);
    device_->destroySampler(texture_point_sampler_);
    device_->destroySampler(repeat_texture_sampler_);
    device_->destroySampler(mirror_repeat_sampler_);
    device_->destroyDescriptorSetLayout(view_desc_set_layout_);
    device_->destroyDescriptorSetLayout(global_tex_desc_set_layout_);
    
    ego::TileObject::destoryAllTiles();
    ego::TileObject::destoryStaticMembers(device_);
    if (player_object_) {
        player_object_->destroy();
        player_object_ = nullptr;
    }
    for (auto& obj : gltf_objects_) {
        obj->destroy();
    }
    gltf_objects_.clear();
    if (ray_tracing_test_) {
        ray_tracing_test_->destroy(device_);
    }
    ego::GltfObject::destoryStaticMembers(device_);
    ego::GameCamera::destoryStaticMembers(device_);
    ego::DebugDrawObject::destoryStaticMembers(device_);
    ibl_creator_->destroy(device_);
    weather_system_->destroy(device_);
    volume_noise_->destroy(device_);
    volume_cloud_->destroy(device_);
    unit_plane_->destroy(device_);
    cone_map_obj_->destroy(device_);
    cone_map_gen_->destroy(device_);
    prt_test_->destroy(device_);

    er::helper::clearCachedShaderModules(device_);

    device_->destroyFence(init_fence_);
    device_->destroySemaphore(init_semaphore_);

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
