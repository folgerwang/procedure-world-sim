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
static bool s_render_prt_test = false;
static bool s_render_hair_test = false;
static bool s_render_lbm_test = false;
static bool s_bistro_scene_test = true;

// global pbr texture descriptor set layout.
std::shared_ptr<er::DescriptorSetLayout> createPbrLightingDescriptorSetLayout(
    const std::shared_ptr<er::Device>& device) {
    std::vector<er::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(5);

    bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        GGX_LUT_INDEX,
        SET_2_FLAG_BITS(ShaderStage, FRAGMENT_BIT, COMPUTE_BIT)));
    bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        CHARLIE_LUT_INDEX,
        SET_2_FLAG_BITS(ShaderStage, FRAGMENT_BIT, COMPUTE_BIT)));
    bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        LAMBERTIAN_ENV_TEX_INDEX,
        SET_2_FLAG_BITS(ShaderStage, FRAGMENT_BIT, COMPUTE_BIT)));
    bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        GGX_ENV_TEX_INDEX,
        SET_2_FLAG_BITS(ShaderStage, FRAGMENT_BIT, COMPUTE_BIT)));
    bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        CHARLIE_ENV_TEX_INDEX,
        SET_2_FLAG_BITS(ShaderStage, FRAGMENT_BIT, COMPUTE_BIT)));

    return device->createDescriptorSetLayout(bindings);
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
static float s_mouse_wheel_offset = 0.0f;
static int s_key = 0;

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
    auto depth_format =
        er::Helper::findDepthFormat(device_);

    er::Helper::createDepthResources(
        device_,
        depth_format,
        display_size,
        depth_buffer_,
        std::source_location::current());

    er::Helper::create2DTextureImage(
        device_,
        er::Format::D32_SFLOAT,
        display_size,
        depth_buffer_copy_,
        SET_3_FLAG_BITS(ImageUsage, SAMPLED_BIT, DEPTH_STENCIL_ATTACHMENT_BIT, TRANSFER_DST_BIT),
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());
}

void RealWorldApplication::createHdrColorBuffer(const glm::uvec2& display_size) {
    er::Helper::create2DTextureImage(
        device_,
        hdr_format_,
        display_size,
        hdr_color_buffer_,
        SET_4_FLAG_BITS(ImageUsage, SAMPLED_BIT, STORAGE_BIT, COLOR_ATTACHMENT_BIT, TRANSFER_SRC_BIT),
        er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        std::source_location::current());
}

void RealWorldApplication::createColorBufferCopy(const glm::uvec2& display_size) {
    er::Helper::create2DTextureImage(
        device_,
        hdr_format_,
        display_size,
        hdr_color_buffer_copy_,
        SET_2_FLAG_BITS(ImageUsage, SAMPLED_BIT, TRANSFER_DST_BIT),
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());
}

void RealWorldApplication::recreateRenderBuffer(const glm::uvec2& display_size) {
    createDepthResources(display_size);
    createHdrColorBuffer(display_size);
    createColorBufferCopy(display_size);
    createFramebuffers(display_size);
}

void RealWorldApplication::createRenderPasses() {
    assert(device_);
    final_render_pass_ = er::helper::createRenderPass(
        device_,
        swap_chain_info_.format,
        depth_format_,
        std::source_location::current(),
        false,
        er::SampleCountFlagBits::SC_1_BIT,
        er::ImageLayout::PRESENT_SRC_KHR);
    hdr_render_pass_ =
        er::helper::createRenderPass(
            device_,
            hdr_format_,
            depth_format_,
            std::source_location::current(),
            true);
    hdr_water_render_pass_ =
        er::helper::createRenderPass(
            device_,
            hdr_format_,
            depth_format_,
            std::source_location::current(),
            false);
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
    queue_list_ = er::Helper::findQueueFamilies(physical_device_, surface_);
    device_ = er::Helper::createLogicalDevice(physical_device_, surface_, queue_list_);
    er::Helper::initRayTracingProperties(
        physical_device_,
        device_,
        rt_pipeline_properties_,
        as_features_);
    auto queue_list =
        queue_list_.getGraphicAndPresentFamilyIndex();
    assert(device_);
    depth_format_ = er::Helper::findDepthFormat(device_);
    graphics_queue_ = device_->getDeviceQueue(queue_list[0]);
    assert(graphics_queue_);
    present_queue_ = device_->getDeviceQueue(queue_list.back());
    er::Helper::createSwapChain(
        window_,
        device_,
        surface_,
        queue_list_,
        swap_chain_info_,
        SET_2_FLAG_BITS(ImageUsage, COLOR_ATTACHMENT_BIT, TRANSFER_DST_BIT));
    createRenderPasses();
    createImageViews();
    cubemap_render_pass_ =
        er::helper::createCubemapRenderPass(
            device_,
            std::source_location::current());

    pbr_lighting_desc_set_layout_ =
        createPbrLightingDescriptorSetLayout(device_);

    createCommandPool();
    assert(command_pool_);
    er::Helper::init(device_);

    eh::loadMtx2Texture(
        device_,
        cubemap_render_pass_,
        "assets/environments/doge2/lambertian/diffuse.ktx2",
        ibl_diffuse_tex_,
        std::source_location::current());
    eh::loadMtx2Texture(
        device_,
        cubemap_render_pass_,
        "assets/environments/doge2/ggx/specular.ktx2",
        ibl_specular_tex_,
        std::source_location::current());
    eh::loadMtx2Texture(
        device_,
        cubemap_render_pass_,
        "assets/environments/doge2/charlie/sheen.ktx2",
        ibl_sheen_tex_,
        std::source_location::current());
    recreateRenderBuffer(swap_chain_info_.extent);
    auto default_color_format = er::Format::R8G8B8A8_UNORM;
    auto height_map_format = er::Format::R16_UNORM;
    eh::createTextureImage(device_, "assets/statue.jpg", default_color_format, sample_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/brdfLUT.png", default_color_format, brdf_lut_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/lut_ggx.png", default_color_format, ggx_lut_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/lut_charlie.png", default_color_format, charlie_lut_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/lut_thin_film.png", default_color_format, thin_film_lut_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/map_mask.png", default_color_format, map_mask_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/map.png", height_map_format, heightmap_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/tile1.jpg", default_color_format, prt_base_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/tile1.tga", default_color_format, prt_bump_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/T_Mat4Mural_C.PNG", default_color_format, prt_base_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/T_Mat4Mural_H.PNG", default_color_format, prt_height_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/T_Mat4Mural_N.PNG", default_color_format, prt_normal_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/T_Mat4Mural_TRA.PNG", default_color_format, prt_orh_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/T_Mat1Ground_C.jpg", default_color_format, prt_base_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/T_Mat1Ground_ORH.jpg", default_color_format, prt_bump_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/T_Mat2Mountains_C.jpg", default_color_format, prt_base_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/T_Mat2Mountains_N.jpg", default_color_format, prt_normal_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/T_Mat2Mountains_ORH.jpg", default_color_format, prt_orh_tex_, std::source_location::current());
    createTextureSampler();
    descriptor_pool_ = device_->createDescriptorPool();
    createCommandBuffers();
    createSyncObjects();

    ego::ViewObject::createViewCameraDescriptorSetLayout(device_);

    prt_shadow_gen_ =
        std::make_shared<es::PrtShadow>(
            device_,
            descriptor_pool_,
            texture_sampler_);

    conemap_obj_ =
        std::make_shared<ego::ConemapObj>(
            device_,
            descriptor_pool_,
            texture_sampler_,
            prt_orh_tex_,//prt_height_tex_,
            prt_shadow_gen_,
            2,//0,
            true,
            0.05f,
            0.1f,
            8.0f / 256.0f);

    unit_plane_ =
        std::make_shared<ego::Plane>(
            device_,
            nullptr,
            10,
            10,
            std::source_location::current());

    auto v = std::make_shared<std::array<glm::vec3, 8>>();
    (*v)[0] = glm::vec3(-1.0f, -5.0f, -1.0f);
    (*v)[1] = glm::vec3(1.0f, -5.0f, -1.0f);
    (*v)[2] = glm::vec3(1.0f, -5.0f, 1.0f);
    (*v)[3] = glm::vec3(-1.0f, -5.0f, 1.0f);
    (*v)[4] = glm::vec3(-1.0f, 0.0f, -1.0f);
    (*v)[5] = glm::vec3(1.0f, 0.0f, -1.0f);
    (*v)[6] = glm::vec3(1.0f, 0.0f, 1.0f);
    (*v)[7] = glm::vec3(-1.0f, 0.0f, 1.0f);

    unit_box_ =
        std::make_shared<ego::Box>(
            device_,
            v,
            10,
            30,
            10,
            std::source_location::current());

    auto desc_set_layouts = {
        pbr_lighting_desc_set_layout_,
        ego::ViewObject::getViewCameraDescriptorSetLayout() };

    conemap_test_ =
        std::make_shared<ego::ConemapTest>(
            device_,
            descriptor_pool_,
            hdr_render_pass_,
            graphic_pipeline_info_,
            desc_set_layouts,
            texture_sampler_,
            prt_base_tex_,
            prt_normal_tex_,
            prt_orh_tex_,
            conemap_obj_,
            swap_chain_info_.extent,
            unit_plane_);

    hair_patch_ =
        std::make_shared<ego::HairPatch>(
            device_,
            descriptor_pool_,
            desc_set_layouts,
            texture_sampler_,
            glm::uvec2(1024, 1024));

    hair_test_ =
        std::make_shared<ego::HairTest>(
            device_,
            descriptor_pool_,
            hdr_render_pass_,
            graphic_pipeline_info_,
            desc_set_layouts,
            texture_sampler_,
            hair_patch_->getHairPatchColorTexture(),
            hair_patch_->getHairPatchWeightTexture(),
            swap_chain_info_.extent,
            unit_plane_);

    lbm_patch_ =
        std::make_shared<ego::LbmPatch>(
            device_,
            descriptor_pool_,
            desc_set_layouts,
            texture_sampler_,
            glm::uvec2(1024, 1024));

    lbm_test_ =
        std::make_shared<ego::LbmTest>(
            device_,
            descriptor_pool_,
            hdr_render_pass_,
            graphic_pipeline_info_,
            desc_set_layouts,
            texture_sampler_,
            lbm_patch_->getLbmPatchTexture(),
            swap_chain_info_.extent,
            unit_plane_);

    clear_values_.resize(2);
    clear_values_[0].color = { 50.0f / 255.0f, 50.0f / 255.0f, 50.0f / 255.0f, 1.0f };
    clear_values_[1].depth_stencil = { 1.0f, 0 };

    conemap_gen_ =
        std::make_shared<es::Conemap>(
            device_,
            descriptor_pool_,
            texture_sampler_,
            prt_orh_tex_,//prt_height_tex_,
            conemap_obj_);

    ibl_creator_ = std::make_shared<es::IblCreator>(
        device_,
        descriptor_pool_,
        cubemap_render_pass_,
        graphic_cubemap_pipeline_info_,
        texture_sampler_,
        kCubemapSize);

    skydome_ = std::make_shared<es::Skydome>(
        device_,
        descriptor_pool_,
        hdr_render_pass_,
        cubemap_render_pass_,
        ego::ViewObject::getViewCameraDescriptorSetLayout(),
        graphic_no_depth_write_pipeline_info_,
        graphic_cubemap_pipeline_info_,
        ibl_creator_->getEnvmapTexture(),
        texture_sampler_,
        swap_chain_info_.extent,
        kCubemapSize);

    for (auto& image : swap_chain_info_.images) {
        er::Helper::transitionImageLayout(
            device_,
            image,
            swap_chain_info_.format,
            er::ImageLayout::UNDEFINED,
            er::ImageLayout::PRESENT_SRC_KHR);
    }

    ego::TileObject::initStaticMembers(device_);

    std::vector<std::shared_ptr<er::ImageView>> soil_water_texes(2);
    soil_water_texes[0] = ego::TileObject::getSoilWaterLayer(0).view;
    soil_water_texes[1] = ego::TileObject::getSoilWaterLayer(1).view;
    weather_system_ = std::make_shared<es::WeatherSystem>(
        device_,
        descriptor_pool_,
        desc_set_layouts,
        mirror_repeat_sampler_,
        ego::TileObject::getRockLayer().view,
        soil_water_texes);

    ego::DrawableObject::initGameObjectBuffer(device_);
    assert(ego::DrawableObject::getGameObjectsBuffer());

    ego::DrawableObject::initStaticMembers(
        device_,
        descriptor_pool_,
        desc_set_layouts,
        texture_sampler_,
//        ego::ViewCamera::getViewCameraBuffer(),
        ego::TileObject::getRockLayer(),
        ego::TileObject::getSoilWaterLayer(0),
        ego::TileObject::getSoilWaterLayer(1),
        ego::TileObject::getWaterFlow(),
        weather_system_->getAirflowTex());

    ego::ViewCamera::initStaticMembers(
        device_,
        descriptor_pool_,
        desc_set_layouts);

    terrain_scene_view_ =
        std::make_shared<es::TerrainSceneView>(
            device_,
            descriptor_pool_,
            desc_set_layouts,
            nullptr,
            nullptr);

    terrain_scene_view_->createCameraDescSetWithTerrain(
        texture_sampler_,
        ego::TileObject::getRockLayer(),
        ego::TileObject::getSoilWaterLayer(0),
        ego::TileObject::getSoilWaterLayer(1),
        *ego::DrawableObject::getGameObjectsBuffer());

    ego::DebugDrawObject::initStaticMembers(
        device_,
        hdr_render_pass_,
        graphic_pipeline_info_,
        desc_set_layouts,
        swap_chain_info_.extent);

    createDescriptorSets();

    volume_noise_ = std::make_shared<es::VolumeNoise>(
        device_,
        descriptor_pool_,
        hdr_render_pass_,
        ego::ViewObject::getViewCameraDescriptorSetLayout(),
        pbr_lighting_desc_set_layout_,
        graphic_pipeline_info_,
        texture_sampler_,
        texture_point_sampler_,
        swap_chain_info_.extent);

    assert(volume_noise_);
    terrain_scene_view_->updateTileResDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        repeat_texture_sampler_,
        weather_system_->getTempTexes(),
        map_mask_tex_.view,
        volume_noise_->getDetailNoiseTexture().view,
        volume_noise_->getRoughNoiseTexture().view);

    ego::TileObject::updateStaticDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        heightmap_tex_.view);

    volume_cloud_ = std::make_shared<es::VolumeCloud>(
        device_,
        descriptor_pool_,
        ego::ViewObject::getViewCameraDescriptorSetLayout(),
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

    // ray tracing test.
    ray_tracing_test_ = std::make_shared<engine::ray_tracing::RayTracingShadowTest>();
    ray_tracing_test_->init(
        device_,
        descriptor_pool_,
        terrain_scene_view_->getViewCameraBuffer(),
        rt_pipeline_properties_,
        as_features_,
        glm::uvec2(1024, 768));

    bistro_exterior_scene_ =
        std::make_shared<ego::DrawableObject>(
            device_,
            descriptor_pool_,
            hdr_render_pass_,
            graphic_pipeline_info_,
            texture_sampler_,
            thin_film_lut_tex_,
            "assets/Bistro_v5_2/BistroExterior.fbx",
            swap_chain_info_.extent,
            glm::inverse(view_params_.view));

    menu_ = std::make_shared<es::Menu>(
        window_,
        device_,
        instance_,
        queue_list_,
        swap_chain_info_,
        graphics_queue_,
        descriptor_pool_,
        final_render_pass_,
        texture_sampler_,
        ray_tracing_test_->getFinalImage().view,
        nullptr);
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
        queue_list_,
        swap_chain_info_,
        SET_2_FLAG_BITS(ImageUsage, COLOR_ATTACHMENT_BIT, TRANSFER_DST_BIT));

    createRenderPasses();
    createImageViews();

    auto desc_set_layouts = {
        pbr_lighting_desc_set_layout_,
        ego::ViewObject::getViewCameraDescriptorSetLayout() };

    ego::TileObject::recreateStaticMembers(device_);
    ego::DrawableObject::recreateStaticMembers(
        device_,
        hdr_render_pass_,
        graphic_pipeline_info_,
        desc_set_layouts,
        swap_chain_info_.extent);
    ego::ViewCamera::recreateStaticMembers(
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
        ego::ViewObject::getViewCameraDescriptorSetLayout(),
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
            device_,
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

    ego::DrawableObject::generateDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
//        ego::ViewCamera::getViewCameraBuffer(),
        thin_film_lut_tex_,
        ego::TileObject::getRockLayer(),
        ego::TileObject::getSoilWaterLayer(0),
        ego::TileObject::getSoilWaterLayer(1),
        ego::TileObject::getWaterFlow(),
        weather_system_->getAirflowTex());

/*    assert(ego::DrawableObject::getGameObjectsBuffer());
    ego::ViewCamera::generateDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        ego::TileObject::getRockLayer(),
        ego::TileObject::getSoilWaterLayer(0),
        ego::TileObject::getSoilWaterLayer(1),
        *ego::DrawableObject::getGameObjectsBuffer());*/

    ego::TileObject::updateStaticDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        heightmap_tex_.view);

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
        ego::ViewObject::getViewCameraDescriptorSetLayout(),
        graphic_pipeline_info_,
        texture_sampler_,
        texture_point_sampler_,
        swap_chain_info_.extent);

    volume_cloud_->recreate(
        device_,
        descriptor_pool_,
        ego::ViewObject::getViewCameraDescriptorSetLayout(),
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
        window_,
        device_,
        instance_,
        queue_list_,
        swap_chain_info_,
        graphics_queue_,
        descriptor_pool_,
        final_render_pass_);
}

void RealWorldApplication::createImageViews() {
    swap_chain_info_.image_views.resize(swap_chain_info_.images.size());
    for (uint64_t i_img = 0; i_img < swap_chain_info_.images.size(); i_img++) {
        swap_chain_info_.image_views[i_img] = device_->createImageView(
            swap_chain_info_.images[i_img],
            er::ImageViewType::VIEW_2D,
            swap_chain_info_.format,
            SET_FLAG_BIT(ImageAspect, COLOR_BIT),
            std::source_location::current());
    }
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
            device_->createFrameBuffer(
                final_render_pass_,
                attachments,
                display_size,
                std::source_location::current());
    }

    assert(hdr_render_pass_);
    assert(hdr_color_buffer_.view);
    std::vector<std::shared_ptr<er::ImageView>> attachments(2);
    attachments[0] = hdr_color_buffer_.view;
    attachments[1] = depth_buffer_.view;

    hdr_frame_buffer_ =
        device_->createFrameBuffer(
            hdr_render_pass_,
            attachments,
            display_size,
            std::source_location::current());

    assert(hdr_water_render_pass_);
    hdr_water_frame_buffer_ =
        device_->createFrameBuffer(
            hdr_water_render_pass_,
            attachments,
            display_size,
            std::source_location::current());
}

void RealWorldApplication::createCommandPool() {
    command_pool_ = device_->createCommandPool(
        queue_list_.getGraphicAndPresentFamilyIndex()[0],
        SET_FLAG_BIT(CommandPoolCreate, RESET_COMMAND_BUFFER_BIT));
}

void RealWorldApplication::createCommandBuffers() {
    command_buffers_ = 
        device_->allocateCommandBuffers(
            command_pool_,
            static_cast<uint32_t>(swap_chain_info_.framebuffers.size()));
}

void RealWorldApplication::createSyncObjects() {
    image_available_semaphores_.resize(kMaxFramesInFlight);
    render_finished_semaphores_.resize(kMaxFramesInFlight);
    in_flight_fences_.resize(kMaxFramesInFlight);
    images_in_flight_.resize(swap_chain_info_.images.size(), VK_NULL_HANDLE);
    init_semaphore_ =
        device_->createSemaphore(
            std::source_location::current());

    assert(device_);
    for (uint64_t i = 0; i < kMaxFramesInFlight; i++) {
        image_available_semaphores_[i] =
            device_->createSemaphore(
                std::source_location::current());
        render_finished_semaphores_[i] =
            device_->createSemaphore(
                std::source_location::current());
        in_flight_fences_[i] =
            device_->createFence(
                std::source_location::current(),
                true);
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
        pbr_lighting_desc_set_ =
            device_->createDescriptorSets(
                descriptor_pool_,
                pbr_lighting_desc_set_layout_,
                1)[0];

        // create a global ibl texture descriptor set.
        auto pbr_lighting_descs = addGlobalTextures(pbr_lighting_desc_set_);
        device_->updateDescriptorSets(pbr_lighting_descs);
    }
}

void RealWorldApplication::createTextureSampler() {
    texture_sampler_ = device_->createSampler(
        er::Filter::LINEAR,
        er::SamplerAddressMode::CLAMP_TO_EDGE,
        er::SamplerMipmapMode::LINEAR, 16.0f,
        std::source_location::current());

    texture_point_sampler_ = device_->createSampler(
        er::Filter::NEAREST,
        er::SamplerAddressMode::REPEAT,
        er::SamplerMipmapMode::LINEAR, 16.0f,
        std::source_location::current());

    repeat_texture_sampler_ = device_->createSampler(
        er::Filter::LINEAR,
        er::SamplerAddressMode::REPEAT,
        er::SamplerMipmapMode::LINEAR, 16.0f,
        std::source_location::current());

    mirror_repeat_sampler_ = device_->createSampler(
        er::Filter::LINEAR,
        er::SamplerAddressMode::MIRRORED_REPEAT,
        er::SamplerMipmapMode::LINEAR, 16.0f,
        std::source_location::current());
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
 
    er::DescriptorSetList desc_sets{ pbr_lighting_desc_set_, view_desc_set };

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

        ego::DrawableObject::updateGameObjectsBuffer(
            cmd_buf,
            ego::TileObject::getWorldMin(),
            ego::TileObject::getWorldRange(),
            terrain_scene_view_->getCameraPosition(),
            menu_->getAirFlowStrength(),
            menu_->getWaterFlowStrength(),
            s_update_frame_count,
            s_dbuf_idx,
            delta_t,
            menu_->isAirfowOn());

        terrain_scene_view_->updateCamera(
            cmd_buf,
            s_dbuf_idx,
            s_key,
            s_update_frame_count,
            delta_t,
            s_last_mouse_pos,
            s_mouse_wheel_offset,
            !s_camera_paused && s_mouse_right_button_pressed);

        s_mouse_wheel_offset = 0;

        if (player_object_) {
            player_object_->updateBuffers(cmd_buf);
        }

        for (auto& drawable_obj : drawable_objects_) {
            drawable_obj->updateBuffers(cmd_buf);
        }

        if (s_update_frame_count >= 0) {
            s_update_frame_count++;
        }
    }

    {
        if (s_render_hair_test) {
            hair_patch_->update(
                device_,
                cmd_buf,
                desc_sets
            );
        }
        else if (s_render_lbm_test) {
            lbm_patch_->update(
                device_,
                cmd_buf,
                desc_sets
            );
        }

        terrain_scene_view_->draw(
            cmd_buf,
            desc_sets,
            s_dbuf_idx,
            delta_t,
            current_time);

#if 0
        cmd_buf->beginRenderPass(
            hdr_render_pass_,
            hdr_frame_buffer_,
            screen_size,
            clear_values_);

        if (s_render_prt_test) {
            conemap_test_->draw(
                device_,
                cmd_buf,
                desc_sets,
                unit_plane_,
                conemap_obj_);
        }
        else if (s_render_hair_test) {
            hair_test_->draw(
                device_,
                cmd_buf,
                desc_sets,
                unit_plane_,
                unit_box_);
        }
        else if (s_render_lbm_test) {
            lbm_test_->draw(
                device_,
                cmd_buf,
                desc_sets,
                unit_plane_,
                unit_box_);
        }
        else if (s_bistro_scene_test) {
            if (bistro_exterior_scene_) {
                bistro_exterior_scene_->draw(cmd_buf, desc_sets);
            }
        }
        else {
            // render drawable.
            {
                for (auto& drawable_obj : drawable_objects_) {
                    drawable_obj->draw(cmd_buf, desc_sets);
                }
            }

            if (player_object_) {
                player_object_->draw(cmd_buf, desc_sets);
            }

            // render debug draw.
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
#endif
        er::ImageResourceInfo depth_src_info = {
            er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            SET_FLAG_BIT(Access, DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, EARLY_FRAGMENT_TESTS_BIT) };

        er::ImageResourceInfo depth_dst_info = {
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };

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
            depth_buffer_.size,
            depth_buffer_copy_.size);

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
        terrain_scene_view_->getColorBuffer()->image,
        swap_chain_info.images[image_index],
        src_info,
        src_info,
        dst_info,
        dst_info,
        SET_FLAG_BIT(ImageAspect, COLOR_BIT),
        SET_FLAG_BIT(ImageAspect, COLOR_BIT),
        terrain_scene_view_->getColorBuffer()->size,
        glm::ivec3(screen_size.x, screen_size.y, 1));

    s_dbuf_idx = 1 - s_dbuf_idx;
}

void RealWorldApplication::initDrawFrame() {
    if (s_render_prt_test) {
        const auto full_buffer_size =
            conemap_obj_->getConemapTexture()->size;

        // generate minmax depth buffer.
        {
            const auto& cmd_buf =
                device_->setupTransientCommandBuffer();
            conemap_obj_->update(
                cmd_buf,
                conemap_obj_->getConemapTexture()->size);
            device_->submitAndWaitTransientCommandBuffer();
        }

        // conemap generation.
        {
            auto conemap_start_point_ =
                std::chrono::high_resolution_clock::now();

            auto dispatch_block_size =
                glm::uvec2(kConemapGenBlockSizeX, kConemapGenBlockSizeY);

            auto dispatch_block_count =
                (glm::uvec2(full_buffer_size) + dispatch_block_size - glm::uvec2(1)) / dispatch_block_size;

            auto num_passes =
                dispatch_block_count.x * dispatch_block_count.y;

            const uint32_t pass_step = 8;
            for (uint32_t i_pass = 0; i_pass < num_passes; i_pass+= pass_step) {
                auto pass_end = std::min(i_pass + pass_step, num_passes);
/*                std::cout <<
                    "conemap generation pass: " <<
                    i_pass <<
                    ", " <<
                    pass_end <<
                    std::endl;*/
                const auto& cmd_buf =
                    device_->setupTransientCommandBuffer();
                conemap_gen_->update(
                    cmd_buf,
                    conemap_obj_,
                    i_pass,
                    pass_end);
                device_->submitAndWaitTransientCommandBuffer();
            }

            auto conemap_end_point_ =
                std::chrono::high_resolution_clock::now();
            float delta_t_ =
                std::chrono::duration<float, std::chrono::seconds::period>(
                    conemap_end_point_ - conemap_start_point_).count();
            std::cout << "conemap generation time: " << delta_t_ << "s" << std::endl;
        }

        // prt shadow generation.
        {
            auto prt_start_point_ =
                std::chrono::high_resolution_clock::now();
            const auto& prt_gen_cmd_buf =
                device_->setupTransientCommandBuffer();
            prt_shadow_gen_->update(
                prt_gen_cmd_buf,
                conemap_obj_);
            device_->submitAndWaitTransientCommandBuffer();
            auto prt_end_point_ =
                std::chrono::high_resolution_clock::now();
            delta_t_ =
                std::chrono::duration<float, std::chrono::seconds::period>(
                    prt_end_point_ - prt_start_point_).count();
            std::cout << "prt generation time: " << delta_t_ << "s" << std::endl;
        }
    }

    prt_shadow_gen_->destroy(device_);
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

    skydome_->update(
        latitude,
        longtitude,
        localtm.tm_yday,
        22/*localtm->tm_hour*/,
        localtm.tm_min,
        localtm.tm_sec);

    terrain_scene_view_->readCameraInfo();

    auto visible_tiles =
        ego::TileObject::updateAllTiles(
            device_,
            descriptor_pool_,
            128,
            glm::vec2(terrain_scene_view_->getCameraPosition()));

    if (dump_volume_noise_) {
        const auto noise_texture_size = kDetailNoiseTextureSize;
#if 1
        uint32_t pixel_count = noise_texture_size * noise_texture_size * noise_texture_size;
        std::vector<uint32_t> temp_buffer;
        temp_buffer.resize(pixel_count);
        er::Helper::dumpTextureImage(
            device_,
            volume_noise_->getDetailNoiseTexture().image,
            er::Format::R8G8B8A8_UNORM,
            glm::uvec3(noise_texture_size, noise_texture_size, noise_texture_size),
            4,
            temp_buffer.data(),
            std::source_location::current());
        
        eh::saveDdsTexture(
            glm::uvec3(noise_texture_size, noise_texture_size, noise_texture_size),
            temp_buffer.data(),
            "volume_noise.dds");
#else
        uint32_t pixel_count = noise_texture_size * noise_texture_size * noise_texture_size;
        std::vector<uint32_t> temp_buffer;
        temp_buffer.resize(pixel_count);
        er::Helper::dumpTextureImage(
            device_,
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

    auto to_load_drawable_names = menu_->getToLoadGltfNamesAndClear();
    for (auto& drawable_name : to_load_drawable_names) {
        auto drawable_obj = std::make_shared<ego::DrawableObject>(
            device_,
            descriptor_pool_,
            hdr_render_pass_,
            graphic_pipeline_info_,
            texture_sampler_,
            thin_film_lut_tex_,
            drawable_name,
            swap_chain_info_.extent,
            glm::inverse(view_params_.view));

        s_update_frame_count = -1;
        drawable_objects_.push_back(drawable_obj);
    }

    auto to_load_player_name = menu_->getToLoadPlayerNameAndClear();
    if (to_load_player_name != "") {
        if (player_object_) {
            player_object_->destroy(device_);
            player_object_ = nullptr;
        }
        player_object_ = std::make_shared<ego::DrawableObject>(
            device_,
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
        player_object_->update(device_, current_time_);
    }

    for (auto& drawable_obj : drawable_objects_) {
        drawable_obj->update(device_, current_time_);
    }

    command_buffer->reset(0);
    command_buffer->beginCommandBuffer(SET_FLAG_BIT(CommandBufferUsage, ONE_TIME_SUBMIT_BIT));

    terrain_scene_view_->setVisibleTiles(visible_tiles);
    drawScene(command_buffer,
        swap_chain_info_,
        terrain_scene_view_->getViewCameraDescriptorSet(),
        swap_chain_info_.extent,
        image_index,
        delta_t_,
        current_time_);

    if (!menu_->isRayTracingTurnOff()) {
        auto result_image =
            ray_tracing_test_->getFinalImage();
#define BLIT_IMAGE_TO_DISPLAY 0

        er::BarrierList raytracing_barrier_list_start;
        raytracing_barrier_list_start.image_barriers.reserve(1);

        er::helper::addTexturesToBarrierList(
            raytracing_barrier_list_start,
            { result_image.image },
            er::ImageLayout::GENERAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_2_FLAG_BITS(Access, SHADER_READ_BIT, SHADER_WRITE_BIT));

        command_buffer->addBarriers(
            raytracing_barrier_list_start,
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT),
            SET_FLAG_BIT(PipelineStage, RAY_TRACING_SHADER_BIT_KHR));

        ray_tracing_test_->draw(
            device_,
            command_buffer,
            view_params_);

#if BLIT_IMAGE_TO_DISPLAY
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
            result_image.size);
#else
        er::BarrierList raytracing_barrier_list_end;
        raytracing_barrier_list_end.image_barriers.reserve(1);

        er::helper::addTexturesToBarrierList(
            raytracing_barrier_list_end,
            { result_image.image },
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            SET_2_FLAG_BITS(Access, SHADER_READ_BIT, SHADER_WRITE_BIT),
            SET_FLAG_BIT(Access, SHADER_READ_BIT));

        command_buffer->addBarriers(
            raytracing_barrier_list_end,
            SET_FLAG_BIT(PipelineStage, RAY_TRACING_SHADER_BIT_KHR),
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT));
#endif
    }

    //
    s_camera_paused = menu_->draw(
        command_buffer,
        final_render_pass_,
        swap_chain_info_.framebuffers[image_index],
        swap_chain_info_.extent,
        skydome_,
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
    menu_->destroy();
    cleanupSwapChain();

    skydome_->destroy(device_);
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
    prt_normal_tex_.destroy(device_);
    prt_height_tex_.destroy(device_);
    prt_orh_tex_.destroy(device_);
    ibl_diffuse_tex_.destroy(device_);
    ibl_specular_tex_.destroy(device_);
    ibl_sheen_tex_.destroy(device_);
    device_->destroySampler(texture_sampler_);
    device_->destroySampler(texture_point_sampler_);
    device_->destroySampler(repeat_texture_sampler_);
    device_->destroySampler(mirror_repeat_sampler_);
    device_->destroyDescriptorSetLayout(pbr_lighting_desc_set_layout_);
    
    ego::TileObject::destroyAllTiles(device_);
    ego::TileObject::destroyStaticMembers(device_);
    if (player_object_) {
        player_object_->destroy(device_);
        player_object_ = nullptr;
    }

    if (bistro_exterior_scene_) {
        bistro_exterior_scene_->destroy(device_);
        bistro_exterior_scene_ = nullptr;
    }

    for (auto& obj : drawable_objects_) {
        obj->destroy(device_);
    }
    drawable_objects_.clear();
    if (ray_tracing_test_) {
        ray_tracing_test_->destroy(device_);
    }
    ego::DrawableObject::destroyStaticMembers(device_);
    ego::ViewCamera::destroyStaticMembers(device_);
    ego::DebugDrawObject::destroyStaticMembers(device_);
    ibl_creator_->destroy(device_);
    weather_system_->destroy(device_);
    volume_noise_->destroy(device_);
    volume_cloud_->destroy(device_);
    unit_plane_->destroy(device_);
    unit_box_->destroy(device_);
    conemap_obj_->destroy(device_); 
    conemap_gen_->destroy(device_);
    conemap_test_->destroy(device_);
    hair_patch_->destroy(device_);
    hair_test_->destroy(device_);
    lbm_patch_->destroy(device_);
    lbm_test_->destroy(device_);

    er::helper::clearCachedShaderModules(device_);

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
