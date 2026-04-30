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
#include "helper/engine_helper.h"
#include "helper/cluster_mesh.h"
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
    bindings.reserve(6);

    bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        GGX_LUT_INDEX,
        SET_2_FLAG_BITS(ShaderStage, FRAGMENT_BIT, COMPUTE_BIT)));
    bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        CHARLIE_LUT_INDEX,
        SET_2_FLAG_BITS(ShaderStage, FRAGMENT_BIT, COMPUTE_BIT)));
    bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        DIRECT_SHADOW_INDEX,
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

// global pbr texture descriptor set layout.
std::shared_ptr<er::DescriptorSetLayout> createRuntimeLightsDescriptorSetLayout(
    const std::shared_ptr<er::Device>& device) {
    std::vector<er::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(1);

    bindings.push_back(er::helper::getBufferDescriptionSetLayoutBinding(
        RUNTIME_LIGHTS_CONSTANT_INDEX,
        SET_3_FLAG_BITS(ShaderStage, FRAGMENT_BIT, COMPUTE_BIT, GEOMETRY_BIT),
        er::DescriptorType::UNIFORM_BUFFER));

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
static bool s_toggle_profiler_pause = false;

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
        // Space toggles the GPU profiler flame-chart pause so you can
        // inspect any frame without it scrolling away.
        s_toggle_profiler_pause = true;
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
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
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
        1,
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
        1,
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
        1,
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
    renderbuffer_formats_[int(er::RenderPasses::kDepthOnly)].depth_format =
        renderbuffer_formats_[int(er::RenderPasses::kForward)].depth_format =
        er::Format::D24_UNORM_S8_UINT;
    renderbuffer_formats_[int(er::RenderPasses::kForward)].color_formats.
        push_back(er::Format::B10G11R11_UFLOAT_PACK32);
    renderbuffer_formats_[int(er::RenderPasses::kShadow)].depth_format =
        er::Format::D16_UNORM;

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

    runtime_lights_desc_set_layout_ =
        createRuntimeLightsDescriptorSetLayout(device_);

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
    eh::createTextureImage(device_, "assets/statue.jpg", default_color_format, false, sample_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/brdfLUT.png", default_color_format, false, brdf_lut_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/lut_ggx.png", default_color_format, false, ggx_lut_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/lut_charlie.png", default_color_format, false, charlie_lut_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/lut_thin_film.png", default_color_format, false, thin_film_lut_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/map_mask.png", default_color_format, false, map_mask_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/map.png", height_map_format, false, heightmap_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/tile1.jpg", default_color_format, false, prt_base_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/tile1.tga", default_color_format, false, prt_bump_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/T_Mat4Mural_C.PNG", default_color_format, false, prt_base_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/T_Mat4Mural_H.PNG", default_color_format, false, prt_height_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/T_Mat4Mural_N.PNG", default_color_format, false, prt_normal_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/T_Mat4Mural_TRA.PNG", default_color_format, false, prt_orh_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/T_Mat1Ground_C.jpg", default_color_format, false, prt_base_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/T_Mat1Ground_ORH.jpg", default_color_format, false, prt_bump_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/T_Mat2Mountains_C.jpg", default_color_format, false, prt_base_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/T_Mat2Mountains_N.jpg", default_color_format, false, prt_normal_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/T_Mat2Mountains_ORH.jpg", default_color_format, false, prt_orh_tex_, std::source_location::current());
    createTextureSampler();
    descriptor_pool_ = device_->createDescriptorPool();
    createCommandBuffers();
    createSyncObjects();

    ego::CameraObject::createViewCameraDescriptorSetLayout(device_);

    auto desc_set_layouts = {
        pbr_lighting_desc_set_layout_,
        ego::CameraObject::getViewCameraDescriptorSetLayout() };

    ego::ShapeBase::initStaticMembers(
        device_,
        desc_set_layouts,
        graphic_pipeline_info_,
        renderbuffer_formats_[int(er::RenderPasses::kForward)]);

    // "Nanite-lite" cluster-debug pipeline. Cheap to create unconditionally
    // -- the per-mesh GPU buffers are only allocated when the
    // --cluster-debug flag is set. Shares the same descriptor-set layout
    // list as ShapeBase since the vertex shader only needs the camera SSBO.
    ego::ClusterDebugDraw::initStaticMembers(
        device_,
        desc_set_layouts,
        graphic_pipeline_info_,
        renderbuffer_formats_[int(er::RenderPasses::kForward)]);

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

    unit_sphere_ =
        std::make_shared<ego::Sphere>(
            device_,
            1.0f,
            2,
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
        ego::CameraObject::getViewCameraDescriptorSetLayout(),
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

    er::DescriptorSetLayoutList object_desc_set_layouts(MAX_NUM_PARAMS_SETS);
    object_desc_set_layouts[PBR_GLOBAL_PARAMS_SET] = pbr_lighting_desc_set_layout_;
    object_desc_set_layouts[VIEW_PARAMS_SET] = ego::CameraObject::getViewCameraDescriptorSetLayout();
    object_desc_set_layouts[RUNTIME_LIGHTS_PARAMS_SET] = runtime_lights_desc_set_layout_;

    ego::DrawableObject::initStaticMembers(
        device_,
        descriptor_pool_,
        object_desc_set_layouts,
        texture_sampler_,
        ego::TileObject::getRockLayer(),
        ego::TileObject::getSoilWaterLayer(0),
        ego::TileObject::getSoilWaterLayer(1),
        ego::TileObject::getWaterFlow(),
        weather_system_->getAirflowTex());

    ego::ViewCamera::initStaticMembers(
        device_,
        descriptor_pool_,
        desc_set_layouts);

    main_camera_object_ =
        std::make_shared<ego::ObjectViewCameraObject>(
            device_,
            descriptor_pool_,
            45.0f,
            float(kWindowSizeX)/float(kWindowSizeY));

    shadow_camera_object_ =
        std::make_shared<ego::ShadowViewCameraObject>(
            device_,
            descriptor_pool_,
            glm::vec3(0.3f, -0.8f, 0.0f));

    // Allocate 4 independent per-cascade storage buffers + descriptor sets.
    // Without these each cascade draw would overwrite the same host-coherent
    // buffer, and by the time the GPU runs all 4 draws every cascade would
    // use the last-written VP matrix.
    shadow_camera_object_->initCascadeDescriptorSets(device_, descriptor_pool_);

    terrain_scene_view_ =
        std::make_shared<es::TerrainSceneView>(
            device_,
            descriptor_pool_,
            renderbuffer_formats_,
            main_camera_object_,
            desc_set_layouts,
            nullptr,
            nullptr,
            glm::uvec2(kWindowSizeX, kWindowSizeY));

    object_scene_view_ = 
        std::make_shared<es::ObjectSceneView>(
            device_,
            descriptor_pool_,
            renderbuffer_formats_[int(er::RenderPasses::kForward)],
            main_camera_object_,
            nullptr,
            nullptr,
            glm::uvec2(kWindowSizeX, kWindowSizeY));

    // ── Create CSM shadow depth array (2048×2048×CSM_CASCADE_COUNT) ──────────
    {
        const er::Format csm_fmt =
            renderbuffer_formats_[int(er::RenderPasses::kShadow)].depth_format;
        constexpr uint32_t kCsmSize = 2048u;

        csm_shadow_tex_ = std::make_shared<er::TextureInfo>();
        csm_shadow_tex_->image = device_->createImage(
            er::ImageType::TYPE_2D,
            glm::uvec3(kCsmSize, kCsmSize, 1),
            csm_fmt,
            SET_2_FLAG_BITS(ImageUsage,
                            DEPTH_STENCIL_ATTACHMENT_BIT,
                            SAMPLED_BIT),
            er::ImageTiling::OPTIMAL,
            er::ImageLayout::UNDEFINED,
            std::source_location::current(),
            0, false, 1, 1,
            CSM_CASCADE_COUNT);

        auto mem_req = device_->getImageMemoryRequirements(csm_shadow_tex_->image);
        csm_shadow_tex_->memory = device_->allocateMemory(
            mem_req.size,
            mem_req.memory_type_bits,
            SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
            0);
        device_->bindImageMemory(csm_shadow_tex_->image, csm_shadow_tex_->memory);
        csm_shadow_tex_->size = glm::uvec3(kCsmSize, kCsmSize, 1);

        // Full-array view — used as the sampler2DArray in the fragment shader.
        csm_shadow_tex_->view = device_->createImageView(
            csm_shadow_tex_->image,
            er::ImageViewType::VIEW_2D_ARRAY,
            csm_fmt,
            SET_FLAG_BIT(ImageAspect, DEPTH_BIT),
            std::source_location::current(),
            0, 1, 0, CSM_CASCADE_COUNT);

        // Per-layer views — used as render-target attachments.
        for (int k = 0; k < CSM_CASCADE_COUNT; ++k) {
            csm_layer_views_[k] = device_->createImageView(
                csm_shadow_tex_->image,
                er::ImageViewType::VIEW_2D,
                csm_fmt,
                SET_FLAG_BIT(ImageAspect, DEPTH_BIT),
                std::source_location::current(),
                0, 1, k, 1);
        }

        // Prime all layers to DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
        er::Helper::transitionImageLayout(
            device_,
            csm_shadow_tex_->image,
            csm_fmt,
            er::ImageLayout::UNDEFINED,
            er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            0, 1, 0, CSM_CASCADE_COUNT);
    }

    shadow_object_scene_view_ =
        std::make_shared<es::ObjectSceneView>(
            device_,
            descriptor_pool_,
            renderbuffer_formats_[int(er::RenderPasses::kShadow)],
            shadow_camera_object_,
            nullptr,
            csm_shadow_tex_,   // provide our array texture as the depth buffer
            glm::uvec2(2048),
            true);

    main_camera_object_->createCameraDescSetWithTerrain(
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
    setupCsmDebugDraw();

    volume_noise_ = std::make_shared<es::VolumeNoise>(
        device_,
        descriptor_pool_,
        hdr_render_pass_,
        ego::CameraObject::getViewCameraDescriptorSetLayout(),
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
        ego::CameraObject::getViewCameraDescriptorSetLayout(),
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

    ssao_ = std::make_shared<es::SSAO>(
        device_,
        descriptor_pool_,
        ego::CameraObject::getViewCameraDescriptorSetLayout(),
        texture_sampler_,
        depth_buffer_copy_.view,
        hdr_color_buffer_.view,
        swap_chain_info_.extent);

    cluster_renderer_ = std::make_shared<es::ClusterRenderer>(
        device_,
        descriptor_pool_);

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
        main_camera_object_->getViewCameraBuffer(),
        rt_pipeline_properties_,
        as_features_,
        glm::uvec2(1024, 768));

    // Spin up the async mesh loader before kicking off any load.
    // Its ctor probes device_->hasLoaderQueue(): on hardware with the
    // dedicated loader queue the worker thread starts here and the
    // startup assets below stream in asynchronously; on single-queue
    // hardware the manager transparently falls back to inline loads
    // so this block behaves exactly like the pre-async code path.
    mesh_load_task_manager_ =
        std::make_unique<ego::MeshLoadTaskManager>(device_);

    // Async startup loads. createAsync returns a shell DrawableObject
    // immediately; the render loop's isReady() check skips it until
    // the worker thread finishes parsing + GPU upload and the main
    // thread finalizes descriptors/pipelines on the next poll(). The
    // HUD menu spinner (see Menu::drawMeshLoadProgress) reads
    // MeshLoadTaskManager::inFlightFilenames() to tell the user
    // what's still loading.
    bistro_exterior_scene_ =
        ego::DrawableObject::createAsync(
            *mesh_load_task_manager_,
            device_,
            descriptor_pool_,
            renderbuffer_formats_,
            graphic_pipeline_info_,
            repeat_texture_sampler_,
            thin_film_lut_tex_,
            "assets/Bistro_v5_2/BistroExterior.fbx",
            glm::inverse(view_params_.view));

    bistro_interior_scene_ =
        ego::DrawableObject::createAsync(
            *mesh_load_task_manager_,
            device_,
            descriptor_pool_,
            renderbuffer_formats_,
            graphic_pipeline_info_,
            repeat_texture_sampler_,
            thin_film_lut_tex_,
            "assets/Bistro_v5_2/BistroInterior.fbx",
            glm::inverse(view_params_.view));

    player_object_ =
        ego::DrawableObject::createAsync(
            *mesh_load_task_manager_,
            device_,
            descriptor_pool_,
            renderbuffer_formats_,
            graphic_pipeline_info_,
            repeat_texture_sampler_,
            thin_film_lut_tex_,
            "assets/Characters/scifi_girl_v.01.glb",
            glm::inverse(view_params_.view));

    // Bistro scenes contain a sky dome mesh that draws a purple atmospheric
    // background.  Disabled so only game objects and the player render.
    // object_scene_view_->addDrawableObject(bistro_exterior_scene_);
    // object_scene_view_->addDrawableObject(bistro_interior_scene_);
    // shadow_object_scene_view_->addDrawableObject(bistro_exterior_scene_);

    object_scene_view_->addDrawableObject(
        player_object_);

    shadow_object_scene_view_->addDrawableObject(
        player_object_);

    menu_ = std::make_shared<engine::ui::Menu>(
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

    // Register CSM debug colour targets as ImGui textures (ImGui must be
    // initialised by the Menu constructor before this can be called).
    registerCsmDebugImTextureIds();

    // Wire the GPU profiler into the menu so it can display itself.
    menu_->setGpuProfiler(&gpu_profiler_);

    // Wire the async mesh loader so the menu can draw a HUD spinner
    // for in-flight loads (see Menu::draw). Passing a non-owning
    // pointer — the manager's lifetime is tied to the Application.
    if (mesh_load_task_manager_) {
        menu_->setMeshLoadTaskManager(mesh_load_task_manager_.get());
    }

    // ---- Plugin system --------------------------------------------------------
    plugin_manager_.registerPlugin(
        std::make_unique<plugins::auto_rig::AutoRigPlugin>());
    plugin_manager_.initAll(device_);
    menu_->setPluginManager(&plugin_manager_);

    if (ssao_) {
        menu_->setSSAO(ssao_.get());
    }
    if (cluster_renderer_) {
        menu_->setClusterRenderer(cluster_renderer_.get());
    }
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
        ego::CameraObject::getViewCameraDescriptorSetLayout() };

    ego::TileObject::recreateStaticMembers(device_);
    ego::DrawableObject::recreateStaticMembers(
        device_,
        &renderbuffer_formats_[0],
        graphic_pipeline_info_,
        desc_set_layouts);
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
        ego::CameraObject::getViewCameraDescriptorSetLayout(),
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

    // ---- Scene views: re-allocate descriptor sets + resize buffers ----
    {
        const auto& fwd_fmt =
            renderbuffer_formats_[int(er::RenderPasses::kForward)];
        object_scene_view_->recreate(fwd_fmt, swap_chain_info_.extent);
        terrain_scene_view_->recreate(fwd_fmt, swap_chain_info_.extent);
        // Shadow view keeps its fixed 2048×2048 size but needs new desc sets.
        const auto& shd_fmt =
            renderbuffer_formats_[int(er::RenderPasses::kShadow)];
        shadow_object_scene_view_->recreate(shd_fmt, glm::uvec2(2048));
    }

    // ---- Camera descriptor sets ----
    main_camera_object_->recreateDescriptorSet();
    shadow_camera_object_->recreateDescriptorSet();
    shadow_camera_object_->recreateCascadeDescriptorSets(device_, descriptor_pool_);

    // re-create the terrain camera descriptor (binds terrain textures).
    main_camera_object_->createCameraDescSetWithTerrain(
        texture_sampler_,
        ego::TileObject::getRockLayer(),
        ego::TileObject::getSoilWaterLayer(0),
        ego::TileObject::getSoilWaterLayer(1),
        *ego::DrawableObject::getGameObjectsBuffer());

    // ---- Ray tracing ----
    ray_tracing_test_->recreate(
        device_,
        descriptor_pool_,
        main_camera_object_->getViewCameraBuffer(),
        rt_pipeline_properties_,
        as_features_,
        glm::uvec2(1024, 768));

    // ---- LBM test ----
    lbm_test_->recreate(
        device_,
        descriptor_pool_,
        texture_sampler_,
        lbm_patch_->getLbmPatchTexture());

    // ---- CSM debug descriptor set (re-alloc from new pool) ----
    if (csm_debug_desc_set_layout_) {
        csm_debug_desc_set_ = device_->createDescriptorSets(
            descriptor_pool_, csm_debug_desc_set_layout_, 1)[0];
        er::WriteDescriptorList writes;
        er::Helper::addOneTexture(
            writes,
            csm_debug_desc_set_,
            er::DescriptorType::COMBINED_IMAGE_SAMPLER,
            0,
            texture_sampler_,
            csm_shadow_tex_->view,
            er::ImageLayout::DEPTH_READ_ONLY_OPTIMAL);
        device_->updateDescriptorSets(writes);
    }

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
        ego::CameraObject::getViewCameraDescriptorSetLayout(),
        graphic_pipeline_info_,
        texture_sampler_,
        texture_point_sampler_,
        swap_chain_info_.extent);

    // Terrain tile descriptors need the noise textures just recreated above.
    terrain_scene_view_->updateTileResDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        repeat_texture_sampler_,
        weather_system_->getTempTexes(),
        map_mask_tex_.view,
        volume_noise_->getDetailNoiseTexture().view,
        volume_noise_->getRoughNoiseTexture().view);

    volume_cloud_->recreate(
        device_,
        descriptor_pool_,
        ego::CameraObject::getViewCameraDescriptorSetLayout(),
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

    if (ssao_) {
        ssao_->recreate(
            device_,
            descriptor_pool_,
            ego::CameraObject::getViewCameraDescriptorSetLayout(),
            texture_sampler_,
            depth_buffer_copy_.view,
            hdr_color_buffer_.view,
            swap_chain_info_.extent);
    }

    if (cluster_renderer_) {
        cluster_renderer_->recreate(descriptor_pool_);
    }

    menu_->init(
        window_,
        device_,
        instance_,
        queue_list_,
        swap_chain_info_,
        graphics_queue_,
        descriptor_pool_,
        final_render_pass_);

    // Re-register CSM debug textures — ImGui was re-initialized above.
    registerCsmDebugImTextureIds();

    // Re-wire profiler after menu re-init.
    menu_->setGpuProfiler(&gpu_profiler_);
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
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::TextureInfo>& direct_shadow_tex )
{
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(6);
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
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        DIRECT_SHADOW_INDEX,
        texture_sampler_,
        direct_shadow_tex->view,
        er::ImageLayout::DEPTH_READ_ONLY_OPTIMAL);

    ibl_creator_->addToGlobalTextures(
        descriptor_writes,
        description_set,
        texture_sampler_);

    return descriptor_writes;
}

void RealWorldApplication::createDescriptorSets() {
    auto buffer_count = swap_chain_info_.images.size();

    pbr_lighting_desc_set_ =
        device_->createDescriptorSets(
            descriptor_pool_,
            pbr_lighting_desc_set_layout_,
            1)[0];

    // create a global ibl texture descriptor set.
    auto pbr_lighting_descs = 
        addGlobalTextures(
            pbr_lighting_desc_set_,
            shadow_object_scene_view_->getDepthBuffer());
    device_->updateDescriptorSets(pbr_lighting_descs);

    runtime_lights_desc_set_ =
        device_->createDescriptorSets(
        descriptor_pool_,
        runtime_lights_desc_set_layout_,
        1)[0];

    if (!runtime_lights_buffer_) {
        runtime_lights_buffer_ = std::make_shared<er::BufferInfo>();
        device_->createBuffer(
            sizeof(glsl::RuntimeLightsParams),
            SET_FLAG_BIT(BufferUsage, UNIFORM_BUFFER_BIT),
            SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
            0,
            runtime_lights_buffer_->buffer,
            runtime_lights_buffer_->memory,
            std::source_location::current());
    }

    // create a global ibl texture descriptor set.
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(1);

    er::Helper::addOneBuffer(
        descriptor_writes,
        runtime_lights_desc_set_,
        engine::renderer::DescriptorType::UNIFORM_BUFFER,
        RUNTIME_LIGHTS_CONSTANT_INDEX,
        runtime_lights_buffer_->buffer,
        runtime_lights_buffer_->buffer->getSize());

    device_->updateDescriptorSets(descriptor_writes);
}

void RealWorldApplication::setupCsmDebugDraw() {
    // Idempotent: skip if already set up (e.g. called twice during init).
    if (csm_debug_pipeline_) return;

    // ---- 4 small R8G8B8A8_UNORM colour render targets (one per cascade) ----
    for (int k = 0; k < CSM_CASCADE_COUNT; ++k) {
        er::Helper::create2DTextureImage(
            device_,
            er::Format::R8G8B8A8_UNORM,
            glm::uvec2(kCsmDebugSize),
            1,
            csm_debug_color_[k],
            SET_3_FLAG_BITS(ImageUsage,
                COLOR_ATTACHMENT_BIT,
                SAMPLED_BIT,
                TRANSFER_SRC_BIT),
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            std::source_location::current());
    }

    // ---- Descriptor set layout: binding 0 = sampler2DArray (frag only) ----
    {
        std::vector<er::DescriptorSetLayoutBinding> bindings(1);
        bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            0,
            SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT));
        csm_debug_desc_set_layout_ = device_->createDescriptorSetLayout(bindings);
    }

    // ---- Descriptor set: bind the CSM depth array ----
    csm_debug_desc_set_ = device_->createDescriptorSets(
        descriptor_pool_, csm_debug_desc_set_layout_, 1)[0];
    {
        er::WriteDescriptorList writes;
        er::Helper::addOneTexture(
            writes,
            csm_debug_desc_set_,
            er::DescriptorType::COMBINED_IMAGE_SAMPLER,
            0,
            texture_sampler_,
            csm_shadow_tex_->view,
            er::ImageLayout::DEPTH_READ_ONLY_OPTIMAL);
        device_->updateDescriptorSets(writes);
    }

    // ---- Pipeline layout: our desc set + push constant (cascade index) ----
    {
        er::PushConstantRange push_const{};
        push_const.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
        push_const.offset = 0;
        push_const.size = sizeof(int32_t);
        csm_debug_pipeline_layout_ = device_->createPipelineLayout(
            { csm_debug_desc_set_layout_ },
            { push_const },
            std::source_location::current());
    }

    // ---- Pipeline: full_screen.vert + csm_debug.frag, no depth buffer ----
    {
        er::ShaderModuleList shaders(2);
        shaders[0] = er::helper::loadShaderModule(
            device_,
            "full_screen_vert.spv",
            er::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());
        shaders[1] = er::helper::loadShaderModule(
            device_,
            "csm_debug_frag.spv",
            er::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());

        er::PipelineRenderbufferFormats fmt;
        fmt.color_formats = { er::Format::R8G8B8A8_UNORM };
        fmt.depth_format   = er::Format::UNDEFINED;   // no depth attachment

        er::PipelineInputAssemblyStateCreateInfo ia;
        ia.topology = er::PrimitiveTopology::TRIANGLE_LIST;
        ia.restart_enable = false;

        // graphic_fs_pipeline_info_ = no cull, no depth test/write, single colour.
        csm_debug_pipeline_ = device_->createPipeline(
            csm_debug_pipeline_layout_,
            {},   // no vertex bindings
            {},   // no vertex attributes
            ia,
            graphic_fs_pipeline_info_,
            shaders,
            fmt,
            {},   // default RasterizationStateOverride
            std::source_location::current());
    }
}

void RealWorldApplication::registerCsmDebugImTextureIds() {
    if (!csm_debug_pipeline_) return;  // not set up yet
    std::array<ImTextureID, CSM_CASCADE_COUNT> ids;
    for (int k = 0; k < CSM_CASCADE_COUNT; ++k) {
        ids[k] = er::Helper::addImTextureID(
            texture_sampler_,
            csm_debug_color_[k].view);
    }
    menu_->setCsmDebugTextureIds(ids);
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

    // ----- GPU Profiler: begin frame -----------------------------------------
    if (!gpu_profiler_initialized_) {
        gpu_profiler_.init(device_, kMaxFramesInFlight);
        gpu_profiler_initialized_ = true;
    }
    // Consume Space-key pause toggle.
    if (s_toggle_profiler_pause) {
        gpu_profiler_.togglePause();
        s_toggle_profiler_pause = false;
    }
    gpu_profiler_.beginFrame(cmd_buf, static_cast<uint32_t>(current_frame_ % kMaxFramesInFlight));

    {
        auto _scope = gpu_profiler_.beginScope(cmd_buf, "IBL / Skydome");
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
        gpu_profiler_.endScope(cmd_buf, _scope);
    }
 
    er::DescriptorSetList desc_sets{
        pbr_lighting_desc_set_,
        view_desc_set };

    {
        auto _scope_terrain = gpu_profiler_.beginScope(cmd_buf, "Terrain / Weather Update");
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
        gpu_profiler_.endScope(cmd_buf, _scope_terrain);
    }

    // Declared here so both the update block and the render block can access them.
    const glm::vec4 csm_cascade_splits(5.0f, 20.0f, 80.0f, 300.0f);
    std::array<glm::mat4, CSM_CASCADE_COUNT> csm_cascade_vps;

    // this has to be happened after tile update, or you wont get the right height info.
    {
        auto _scope_go = gpu_profiler_.beginScope(cmd_buf, "Game Object Updates");
        static std::chrono::time_point s_last_time = std::chrono::steady_clock::now();
        auto cur_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = cur_time - s_last_time;
        auto delta_t = static_cast<float>(elapsed_seconds.count());
        s_last_time = cur_time;

        ego::DrawableObject::updateGameObjectsBuffer(
            cmd_buf,
            ego::TileObject::getWorldMin(),
            ego::TileObject::getWorldRange(),
            main_camera_object_->getCameraPosition(),
            menu_->getAirFlowStrength(),
            menu_->getWaterFlowStrength(),
            0,//s_update_frame_count,
            s_dbuf_idx,
            delta_t,
            menu_->isAirfowOn());

        main_camera_object_->updateCamera(
            cmd_buf,
            s_dbuf_idx,
            s_key,
            s_update_frame_count,
            delta_t,
            s_last_mouse_pos,
            s_mouse_wheel_offset,
            !s_camera_paused && s_mouse_right_button_pressed);

        // Push per-frame feature flags into the camera buffer so shaders can
        // conditionally skip expensive passes (e.g. shadow sampling).
        {
            uint32_t input_flags = 0u;
            if (menu_->isShadowPassTurnOff())
                input_flags |= FEATURE_INPUT_SHADOW_DISABLED;
            main_camera_object_->setInputFeatureFlags(input_flags);
        }

        // Tell the forward pass whether to skip clustered meshes.
        // When the cluster indirect draw is active the cluster renderer owns
        // those meshes; drawing them in the forward pass as well causes
        // double-rendering with z-fighting.
        engine::helper::clusterIndirectActive() =
            cluster_renderer_ &&
            cluster_renderer_->isEnabled() &&
            !engine::helper::clusterRenderingEnabled();  // not debug-draw mode

        // Update the base shadow camera position (facing dir, up vector).
        shadow_camera_object_->updateCamera(
            cmd_buf,
            main_camera_object_->getCameraPos());

        s_mouse_wheel_offset = 0;

        // ── Compute CSM cascade matrices ──────────────────────────────────────
        main_camera_object_->readGpuCameraInfo();
        const auto& main_cam = main_camera_object_->getCameraViewInfo();

        // View-space far depths for each cascade (tune to scene scale).
        shadow_camera_object_->computeCascadeMatrices(
            main_cam.view,
            main_cam.proj,
            csm_cascade_splits,
            0.1f,  // main camera z_near
            csm_cascade_vps);

        glsl::RuntimeLightsParams runtime_lights_params = {};
        {
            for (int k = 0; k < CSM_CASCADE_COUNT; ++k) {
                runtime_lights_params.light_view_proj[k] = csm_cascade_vps[k];
            }
            runtime_lights_params.cascade_splits = csm_cascade_splits;
            for (int l = 0; l < LIGHT_COUNT; l++) {
                runtime_lights_params.lights[l].type = glsl::LightType_Directional;
                runtime_lights_params.lights[l].color = glm::vec3(255.0f, 250.0f, 240.0f) / 255.0f;
                runtime_lights_params.lights[l].direction = shadow_camera_object_->getLightDir();
                runtime_lights_params.lights[l].intensity = 20.0f;
                runtime_lights_params.lights[l].position = glm::vec3(0, 0, 0);
            }
        }

        device_->updateBufferMemory(
            runtime_lights_buffer_->memory,
            sizeof(runtime_lights_params),
            &runtime_lights_params);

        if (player_object_) {
            player_object_->updateBuffers(cmd_buf);
        }

        // Bistro scene buffer updates — enabled once game has started.
        if (bistro_exterior_scene_ && bistro_exterior_scene_->isReady()) {
            bistro_exterior_scene_->updateBuffers(cmd_buf);
        }
        if (bistro_interior_scene_ && bistro_interior_scene_->isReady()) {
            bistro_interior_scene_->updateBuffers(cmd_buf);
        }

        for (auto& drawable_obj : drawable_objects_) {
            drawable_obj->updateBuffers(cmd_buf);
        }

        if (s_update_frame_count >= 0) {
            s_update_frame_count++;
        }
        gpu_profiler_.endScope(cmd_buf, _scope_go);
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

        er::DescriptorSetList desc_sets(MAX_NUM_PARAMS_SETS);
        desc_sets[PBR_GLOBAL_PARAMS_SET] = pbr_lighting_desc_set_;
        desc_sets[RUNTIME_LIGHTS_PARAMS_SET] = runtime_lights_desc_set_;

        // ── Single-pass CSM shadow render ─────────────────────────────────────
        // The geometry shader (base_depthonly_csm.geom) reads all 4 VP matrices
        // from RuntimeLightsParams and broadcasts each triangle to all 4 depth
        // array layers in one draw.  Vertex transform runs once per vertex
        // instead of 4×, giving roughly 4× the throughput of separate passes.
        // The full 4-layer array view is used so the GS can write to all layers.
        if (!menu_->isShadowPassTurnOff()) {
            auto _scope_shadow = gpu_profiler_.beginScope(cmd_buf, "CSM Shadow");
            shadow_object_scene_view_->draw(
                cmd_buf,
                desc_sets,
                nullptr,
                s_dbuf_idx,
                delta_t,
                current_time,
                true,
                csm_shadow_tex_->view,    // full CSM_CASCADE_COUNT-layer array view
                CSM_CASCADE_COUNT);       // layer_count triggers GS layered path
            gpu_profiler_.endScope(cmd_buf, _scope_shadow);
        }

        // Custom resource infos matching the shadow pass attachment layout.
        // The shadow depth buffer uses DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        // (not DEPTH_ATTACHMENT_OPTIMAL) because object_scene_view uses
        // a combined depth/stencil format.
        er::ImageResourceInfo csm_as_depth_attachment = {
            er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            SET_2_FLAG_BITS(Access, DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, DEPTH_STENCIL_ATTACHMENT_READ_BIT),
            SET_2_FLAG_BITS(PipelineStage, EARLY_FRAGMENT_TESTS_BIT, LATE_FRAGMENT_TESTS_BIT) };
        er::ImageResourceInfo csm_as_shader_sampler = {
            er::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_2_FLAG_BITS(PipelineStage, FRAGMENT_SHADER_BIT, COMPUTE_SHADER_BIT) };

        // Transition ALL cascade layers from depth-attachment to shader-read.
        cmd_buf->addImageBarrier(
            shadow_object_scene_view_->getDepthBuffer()->image,
            csm_as_depth_attachment,
            csm_as_shader_sampler,
            0, 1, 0, CSM_CASCADE_COUNT);

        // ---- CSM debug visualisation ----------------------------------------
        // While the depth array is in DEPTH_READ_ONLY_OPTIMAL, blit each
        // cascade to its small R8G8B8A8 colour target so the ImGui window can
        // display them.
        if (menu_->showCsmDebug() && csm_debug_pipeline_) {
            er::ImageResourceInfo dbg_color_attach = {
                er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT),
                SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT) };
            er::ImageResourceInfo dbg_shader_read = {
                er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                SET_FLAG_BIT(Access, SHADER_READ_BIT),
                SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };

            for (int k = 0; k < CSM_CASCADE_COUNT; ++k) {
                // SHADER_READ_ONLY → COLOR_ATTACHMENT
                cmd_buf->addImageBarrier(
                    csm_debug_color_[k].image,
                    dbg_shader_read, dbg_color_attach,
                    0, 1, 0, 1);

                // Render cascade k depth → colour target.
                er::RenderingAttachmentInfo color_attach;
                color_attach.image_view   = csm_debug_color_[k].view;
                color_attach.image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
                color_attach.load_op      = er::AttachmentLoadOp::CLEAR;
                color_attach.store_op     = er::AttachmentStoreOp::STORE;
                color_attach.clear_value.color = { {0.0f, 0.0f, 0.0f, 1.0f} };

                er::RenderingInfo ri = {};
                ri.render_area_offset = { 0, 0 };
                ri.render_area_extent = { kCsmDebugSize, kCsmDebugSize };
                ri.layer_count        = 1;
                ri.view_mask          = 0;
                ri.color_attachments  = { color_attach };
                ri.depth_attachments  = {};
                ri.stencil_attachments = {};
                cmd_buf->beginDynamicRendering(ri);

                er::Viewport vp;
                vp.x = 0; vp.y = 0;
                vp.width  = float(kCsmDebugSize);
                vp.height = float(kCsmDebugSize);
                vp.min_depth = 0.0f;
                vp.max_depth = 1.0f;
                er::Scissor sc;
                sc.offset = glm::ivec2(0);
                sc.extent = glm::uvec2(kCsmDebugSize);
                cmd_buf->setViewports({ vp });
                cmd_buf->setScissors({ sc });

                cmd_buf->bindPipeline(
                    er::PipelineBindPoint::GRAPHICS, csm_debug_pipeline_);

                int32_t cascade_idx = k;
                cmd_buf->pushConstants(
                    SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
                    csm_debug_pipeline_layout_,
                    &cascade_idx,
                    sizeof(int32_t));
                cmd_buf->bindDescriptorSets(
                    er::PipelineBindPoint::GRAPHICS,
                    csm_debug_pipeline_layout_,
                    { csm_debug_desc_set_ });

                cmd_buf->draw(3); // fullscreen triangle (no vertex buffer needed)

                cmd_buf->endDynamicRendering();

                // COLOR_ATTACHMENT → SHADER_READ_ONLY (for ImGui)
                cmd_buf->addImageBarrier(
                    csm_debug_color_[k].image,
                    dbg_color_attach, dbg_shader_read,
                    0, 1, 0, 1);
            }
        }
        // ---- End CSM debug --------------------------------------------------

        // ── GPU cluster culling (Nanite-lite) ──
        if (cluster_renderer_ && cluster_renderer_->isEnabled()) {
            auto _scope_cull = gpu_profiler_.beginScope(cmd_buf, "Cluster Cull");
            cluster_renderer_->cull(
                cmd_buf,
                main_camera_object_->getViewProjMatrix(),
                main_camera_object_->getCameraPosition());
            gpu_profiler_.endScope(cmd_buf, _scope_cull);
        }

        // ── Per-mesh frustum culling ──────────────────────────────────
        // When cluster rendering is enabled, extract world-space frustum
        // planes and pass them to DrawableObject. drawMesh() will transform
        // each mesh's local-space bounding sphere by its model matrix and
        // cull against these planes — correct regardless of node transform.
        if (cluster_renderer_ && cluster_renderer_->isEnabled()) {
            const auto& vp = main_camera_object_->getViewProjMatrix();
            glm::vec4 fplanes[6];
            fplanes[0] = glm::vec4(vp[0][3]+vp[0][0], vp[1][3]+vp[1][0],
                                    vp[2][3]+vp[2][0], vp[3][3]+vp[3][0]);
            fplanes[1] = glm::vec4(vp[0][3]-vp[0][0], vp[1][3]-vp[1][0],
                                    vp[2][3]-vp[2][0], vp[3][3]-vp[3][0]);
            fplanes[2] = glm::vec4(vp[0][3]+vp[0][1], vp[1][3]+vp[1][1],
                                    vp[2][3]+vp[2][1], vp[3][3]+vp[3][1]);
            fplanes[3] = glm::vec4(vp[0][3]-vp[0][1], vp[1][3]-vp[1][1],
                                    vp[2][3]-vp[2][1], vp[3][3]-vp[3][1]);
            fplanes[4] = glm::vec4(vp[0][3]+vp[0][2], vp[1][3]+vp[1][2],
                                    vp[2][3]+vp[2][2], vp[3][3]+vp[3][2]);
            fplanes[5] = glm::vec4(vp[0][3]-vp[0][2], vp[1][3]-vp[1][2],
                                    vp[2][3]-vp[2][2], vp[3][3]-vp[3][2]);
            for (int i = 0; i < 6; ++i) {
                float len = glm::length(glm::vec3(fplanes[i]));
                if (len > 0.0001f) fplanes[i] /= len;
            }
            ego::DrawableObject::setFrustumCullPlanes(fplanes);
        }

        {
            auto _scope_forward = gpu_profiler_.beginScope(cmd_buf, "Forward Pass");

            object_scene_view_->draw(
                cmd_buf,
                desc_sets,
                nullptr,
                s_dbuf_idx,
                delta_t,
                current_time);

            gpu_profiler_.endScope(cmd_buf, _scope_forward);
        }

        // Clear frustum cull state so depth-only / shadow passes are not culled.
        ego::DrawableObject::clearFrustumCull();

        // ── Bindless cluster draw (single drawIndexedIndirectCount) ──
        // Rendered in its own dynamic rendering pass, on top of the
        // forward pass output (LOAD, not CLEAR).
        // Skipped when cluster debug draw is active — the forward pass already
        // drew the cluster visualisation and the indirect draw would overwrite it.
        if (cluster_renderer_ && cluster_renderer_->isEnabled() &&
            !engine::helper::clusterRenderingEnabled()) {
            auto _scope_cluster = gpu_profiler_.beginScope(
                cmd_buf, "Cluster Bindless Draw");

            auto color_buf = object_scene_view_->getColorBuffer();
            auto depth_buf = object_scene_view_->getDepthBuffer();

            er::RenderingAttachmentInfo color_attach;
            color_attach.image_view   = color_buf->view;
            color_attach.image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
            color_attach.load_op      = er::AttachmentLoadOp::LOAD;
            color_attach.store_op     = er::AttachmentStoreOp::STORE;

            er::RenderingAttachmentInfo depth_attach;
            depth_attach.image_view   = depth_buf->view;
            depth_attach.image_layout = er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depth_attach.load_op      = er::AttachmentLoadOp::LOAD;
            depth_attach.store_op     = er::AttachmentStoreOp::STORE;

            er::RenderingInfo ri = {};
            ri.render_area_offset  = { 0, 0 };
            ri.render_area_extent  = screen_size;
            ri.layer_count         = 1;
            ri.view_mask           = 0;
            ri.color_attachments   = { color_attach };
            ri.depth_attachments   = { depth_attach };
            ri.stencil_attachments = {};

            cmd_buf->beginDynamicRendering(ri);

            // Build descriptor set list matching the bindless pipeline layout:
            //   set 0 = PBR global (IBL + shadow sampler)
            //   set 1 = view camera (VP matrices)
            //   set 2 = cluster bindless (injected by draw())
            //   set 3 = nullptr (SKIN — unused by cluster pass)
            //   set 4 = runtime lights (sun direction/color + CSM data)
            er::DescriptorSetList cluster_desc_sets(
                RUNTIME_LIGHTS_PARAMS_SET + 1, nullptr);
            cluster_desc_sets[PBR_GLOBAL_PARAMS_SET] = pbr_lighting_desc_set_;
            cluster_desc_sets[VIEW_PARAMS_SET] =
                main_camera_object_->getViewCameraDescriptorSet();
            cluster_desc_sets[RUNTIME_LIGHTS_PARAMS_SET] =
                runtime_lights_desc_set_;

            // Viewport and scissor for the full screen.
            er::Viewport vp;
            vp.x = 0; vp.y = 0;
            vp.width  = static_cast<float>(screen_size.x);
            vp.height = static_cast<float>(screen_size.y);
            vp.min_depth = 0.0f;
            vp.max_depth = 1.0f;
            er::Scissor sc;
            sc.offset = glm::ivec2(0);
            sc.extent = screen_size;

            cluster_renderer_->draw(
                cmd_buf,
                cluster_desc_sets,
                { vp },
                { sc });

            cmd_buf->endDynamicRendering();

            gpu_profiler_.endScope(cmd_buf, _scope_cluster);
        }

        // Transition ALL cascade layers back to depth-attachment for next frame.
        cmd_buf->addImageBarrier(
            shadow_object_scene_view_->getDepthBuffer()->image,
            csm_as_shader_sampler,
            csm_as_depth_attachment,
            0, 1, 0, CSM_CASCADE_COUNT);

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
                    focus_scene_view->getCameraPosition(),
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

        // ── SSAO: generate, blur, apply to HDR color ──
        if (ssao_ && ssao_->enabled) {
            auto _scope_ssao = gpu_profiler_.beginScope(cmd_buf, "SSAO");
            ssao_->render(cmd_buf, view_desc_set,
                          hdr_color_buffer_.image, screen_size);
            gpu_profiler_.endScope(cmd_buf, _scope_ssao);
        }

        if (!menu_->isVolumeMoistTurnOff()) {
            auto _scope_cloud = gpu_profiler_.beginScope(cmd_buf, "Volume Cloud");
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
            gpu_profiler_.endScope(cmd_buf, _scope_cloud);
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

    {
        auto _scope_blit = gpu_profiler_.beginScope(cmd_buf, "Final Blit");
        er::Helper::blitImage(
            cmd_buf,
            object_scene_view_->getColorBuffer()->image,
            swap_chain_info.images[image_index],
            src_info,
            src_info,
            dst_info,
            dst_info,
            SET_FLAG_BIT(ImageAspect, COLOR_BIT),
            SET_FLAG_BIT(ImageAspect, COLOR_BIT),
            object_scene_view_->getColorBuffer()->size,
            glm::ivec3(screen_size.x, screen_size.y, 1));
        gpu_profiler_.endScope(cmd_buf, _scope_blit);
    }

    // ----- GPU Profiler: end frame -------------------------------------------
    gpu_profiler_.endFrame(cmd_buf, static_cast<uint32_t>(current_frame_ % kMaxFramesInFlight));

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

    main_camera_object_->readGpuCameraInfo();

    auto visible_tiles =
        ego::TileObject::updateAllTiles(
            device_,
            descriptor_pool_,
            128,
            glm::vec2(main_camera_object_->getCameraPosition()));

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

    // Drain any in-flight async mesh loads whose GPU fence has signaled:
    // this runs phase 3 (descriptor-set + pipeline creation) on the main
    // thread. Cheap when nothing is ready. Must run before we touch
    // drawable_objects_ for draw so that objects finalized this frame
    // pop in immediately instead of waiting one extra frame.
    if (mesh_load_task_manager_) {
        mesh_load_task_manager_->poll();
    }

    // ---- "New Game" handler: add the bistro scenes (already created at
    // startup and streaming in the background) into the scene views so
    // they start rendering.  The XML mesh list is available for any
    // additional meshes that weren't pre-loaded.
    if (menu_->consumeNewGameRequest()) {
        // Restore the bistro exterior/interior into the scene views.
        // They were created at startup but NOT added to the views so the
        // title screen stays clean.
        if (bistro_exterior_scene_) {
            object_scene_view_->addDrawableObject(bistro_exterior_scene_);
            shadow_object_scene_view_->addDrawableObject(bistro_exterior_scene_);
        }
        if (bistro_interior_scene_) {
            object_scene_view_->addDrawableObject(bistro_interior_scene_);
        }

        s_update_frame_count = -1;
    }

    // Transition Loading → InGame once bistro scenes have finished streaming.
    if (menu_->getGameState() == engine::ui::GameState::Loading) {
        bool all_ready = true;
        if (bistro_exterior_scene_ && !bistro_exterior_scene_->isReady())
            all_ready = false;
        if (bistro_interior_scene_ && !bistro_interior_scene_->isReady())
            all_ready = false;
        if (all_ready) {
            menu_->setGameState(engine::ui::GameState::InGame);
            menu_->setBackgroundEnabled(false);

            // Upload cluster data to the GPU cluster renderer now that
            // meshes are fully loaded. Iterates every mesh in every
            // drawable and uploads its ClusterMesh sidecar.
            if (cluster_renderer_) {
                // Cluster bounds from buildClusterMesh() are already in
                // the mesh's local space, which for FBX scenes IS world
                // space (the FBX loader bakes node transforms into
                // vertex positions). Using identity here avoids the
                auto uploadClusters = [&](
                    const std::shared_ptr<ego::DrawableObject>& obj) {
                    if (!obj || !obj->isReady()) return;
                    auto& meshes  = obj->getMutableMeshes();
                    const auto& data = obj->getDrawableData();

                    // Build mesh_idx → first-node world transform table.
                    // node.cached_matrix_ is set by DrawableData::update()
                    // and already includes the full parent hierarchy.
                    // Defaults to identity for meshes with no owning node.
                    std::vector<glm::mat4> mesh_transforms(
                        meshes.size(), glm::mat4(1.0f));
                    for (const auto& node : data.nodes_) {
                        if (node.mesh_idx_ >= 0 &&
                            static_cast<uint32_t>(node.mesh_idx_) < meshes.size()) {
                            // First node wins — handles non-instanced meshes.
                            if (mesh_transforms[node.mesh_idx_] == glm::mat4(1.0f))
                                mesh_transforms[node.mesh_idx_] = node.cached_matrix_;
                        }
                    }

                    for (uint32_t mi = 0; mi < meshes.size(); ++mi) {
                        auto& cm = meshes[mi].cluster_mesh_;
                        if (cm.empty()) continue;
                        // Record global mesh index BEFORE upload (it
                        // increments uploaded_mesh_count_ internally).
                        meshes[mi].cluster_global_mesh_idx_ =
                            static_cast<int32_t>(cluster_renderer_->getMeshCount());
                        cluster_renderer_->uploadMeshClusters(
                            cm, obj->getDrawableData(), mi, 0,
                            mesh_transforms[mi]);
                    }
                };
                uploadClusters(bistro_exterior_scene_);
                uploadClusters(bistro_interior_scene_);
                for (auto& d : drawable_objects_) {
                    uploadClusters(d);
                }

                // Merge all staged cluster data into single flat GPU SSBOs.
                cluster_renderer_->finalizeUploads();

                // Initialize the bindless graphics pipeline for cluster
                // rendering. Needs the global descriptor set layouts
                // (PBR_GLOBAL at set 0, VIEW_PARAMS at set 1) so the
                // pipeline layout matches the forward pass.
                {
                    // Provide all sets 0..RUNTIME_LIGHTS so the cluster
                    // pipeline can read the same sun light + shadow data
                    // as the standard forward pass.
                    er::DescriptorSetLayoutList global_layouts(
                        RUNTIME_LIGHTS_PARAMS_SET + 1, nullptr);
                    global_layouts[PBR_GLOBAL_PARAMS_SET] =
                        pbr_lighting_desc_set_layout_;
                    global_layouts[VIEW_PARAMS_SET] =
                        ego::CameraObject::getViewCameraDescriptorSetLayout();
                    // Set 2 (PBR_MATERIAL_PARAMS_SET) is replaced by the
                    // cluster bindless set inside initBindlessPipeline.
                    // Set 3 (SKIN_PARAMS_SET) stays nullptr.
                    global_layouts[RUNTIME_LIGHTS_PARAMS_SET] =
                        runtime_lights_desc_set_layout_;
                    cluster_renderer_->initBindlessPipeline(
                        global_layouts,
                        graphic_pipeline_info_,
                        renderbuffer_formats_[
                            int(er::RenderPasses::kForward)]);
                }
            }
        }
    }

    // Legacy background-disable check (for non-title-screen mesh loads).
    if (menu_->isBackgroundEnabled() &&
        menu_->getGameState() == engine::ui::GameState::InGame) {
        menu_->setBackgroundEnabled(false);
    }

    auto to_load_drawable_names = menu_->getToLoadGltfNamesAndClear();
    for (auto& drawable_name : to_load_drawable_names) {
        auto drawable_obj = ego::DrawableObject::createAsync(
            *mesh_load_task_manager_,
            device_,
            descriptor_pool_,
            renderbuffer_formats_,
            graphic_pipeline_info_,
            texture_sampler_,
            thin_film_lut_tex_,
            drawable_name,
            glm::inverse(view_params_.view));

        s_update_frame_count = -1;
        drawable_objects_.push_back(drawable_obj);
        object_scene_view_->addDrawableObject(drawable_obj);
        shadow_object_scene_view_->addDrawableObject(drawable_obj);
    }

    auto to_load_player_name = menu_->getToLoadPlayerNameAndClear();
    if (to_load_player_name != "") {
        if (player_object_) {
            // Destroy has to wait until the previous async load (if any)
            // finished — tearing down an object whose GPU upload is
            // still in flight would use-after-free the staging buffers.
            // In practice this is rare (the user can't spam the menu
            // faster than a load completes) but being defensive here
            // costs nothing.
            if (!player_object_->isReady()) {
                mesh_load_task_manager_->waitAll();
            }
            object_scene_view_->removeDrawableObject(player_object_);
            shadow_object_scene_view_->removeDrawableObject(player_object_);
            player_object_->destroy(device_);
            player_object_ = nullptr;
        }
        player_object_ = ego::DrawableObject::createAsync(
            *mesh_load_task_manager_,
            device_,
            descriptor_pool_,
            renderbuffer_formats_,
            graphic_pipeline_info_,
            texture_sampler_,
            thin_film_lut_tex_,
            to_load_player_name,
            glm::inverse(view_params_.view));
        object_scene_view_->addDrawableObject(player_object_);
        shadow_object_scene_view_->addDrawableObject(player_object_);
    }

    if (player_object_) {
        player_object_->update(device_, current_time_);
    }

    for (auto& drawable_obj : drawable_objects_) {
        drawable_obj->update(device_, current_time_);
    }

    command_buffer->reset(0);
    command_buffer->beginCommandBuffer(SET_FLAG_BIT(CommandBufferUsage, ONE_TIME_SUBMIT_BIT));

    std::shared_ptr<ego::ViewObject> display_scene_view = nullptr;

    display_scene_view = object_scene_view_;
    //display_scene_view = terrain_scene_view_;


    terrain_scene_view_->setVisibleTiles(visible_tiles);

    // Skip the full 3D scene (forward pass, shadows, terrain, volume
    // cloud, etc.) while on the title screen — only the ImGui title UI
    // and background image are shown. This avoids partially-loaded
    // meshes drawing garbage geometry through the background.
    if (menu_->getGameState() != engine::ui::GameState::TitleScreen) {
        drawScene(command_buffer,
            swap_chain_info_,
            main_camera_object_->getViewCameraDescriptorSet(),
            swap_chain_info_.extent,
            image_index,
            delta_t_,
            current_time_);
    }

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

    // Collect GPU timestamp results from the *previous* frame to avoid stalls.
    // current_frame_ still holds the frame just submitted; the "previous" slot
    // (one kMaxFramesInFlight period ago) should already be done.
    if (gpu_profiler_initialized_) {
        uint32_t collect_frame = static_cast<uint32_t>(
            (current_frame_ + kMaxFramesInFlight - 1) % kMaxFramesInFlight);
        gpu_profiler_.collectResults(device_, collect_frame);
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
    // Tear down the async mesh loader first. Its dtor drains the
    // pending queue, joins the worker thread, and waitAll()'s any
    // in-flight GPU fences. Doing this before waitIdle/destroyXxx
    // means no phase 2/phase 3 work can race with the destroy
    // sequence below (which would otherwise wipe buffers/images
    // the worker still holds shared_ptrs to).
    if (mesh_load_task_manager_) {
        mesh_load_task_manager_.reset();
    }

    // Wait for the GPU to finish before tearing anything down, otherwise
    // destroying resources still referenced by in-flight command buffers
    // (query pools, images, buffers, etc.) will crash.
    if (device_) {
        device_->waitIdle();
    }

    plugin_manager_.shutdownAll();

    if (gpu_profiler_initialized_) {
        gpu_profiler_.destroy(device_);
        gpu_profiler_initialized_ = false;
    }
    menu_->destroyResources();
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
    runtime_lights_buffer_->destroy(device_);
    device_->destroySampler(texture_sampler_);
    device_->destroySampler(texture_point_sampler_);
    device_->destroySampler(repeat_texture_sampler_);
    device_->destroySampler(mirror_repeat_sampler_);
    device_->destroyDescriptorSetLayout(pbr_lighting_desc_set_layout_);
    device_->destroyDescriptorSetLayout(runtime_lights_desc_set_layout_);

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

    if (bistro_interior_scene_) {
        bistro_interior_scene_->destroy(device_);
        bistro_interior_scene_ = nullptr;
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
    ego::ShapeBase::destroyStaticMembers(device_);
    ibl_creator_->destroy(device_);
    weather_system_->destroy(device_);
    volume_noise_->destroy(device_);
    volume_cloud_->destroy(device_);
    if (ssao_) ssao_->destroy(device_);
    if (cluster_renderer_) cluster_renderer_->destroy();
    unit_plane_->destroy(device_);
    unit_box_->destroy(device_);
    unit_sphere_->destroy(device_);
    conemap_obj_->destroy(device_); 
    conemap_gen_->destroy(device_);
    conemap_test_->destroy(device_);
    hair_patch_->destroy(device_);
    hair_test_->destroy(device_);
    lbm_patch_->destroy(device_);
    lbm_test_->destroy(device_);

    main_camera_object_->destroy(device_);
    shadow_camera_object_->destroy(device_);
    // Destroy the shared static descriptor set layout exactly once (after
    // both camera instances have released their descriptor sets).
    ego::CameraObject::destroyStaticMembers(device_);
    terrain_scene_view_->destroy(device_);
    object_scene_view_->destroy(device_);

    // Destroy CSM debug colour targets and pipeline.
    for (int k = 0; k < CSM_CASCADE_COUNT; ++k) {
        if (csm_debug_color_[k].image) {
            csm_debug_color_[k].destroy(device_);
        }
    }
    if (csm_debug_pipeline_) {
        device_->destroyPipeline(csm_debug_pipeline_);
        csm_debug_pipeline_ = nullptr;
    }
    if (csm_debug_pipeline_layout_) {
        device_->destroyPipelineLayout(csm_debug_pipeline_layout_);
        csm_debug_pipeline_layout_ = nullptr;
    }
    if (csm_debug_desc_set_layout_) {
        device_->destroyDescriptorSetLayout(csm_debug_desc_set_layout_);
        csm_debug_desc_set_layout_ = nullptr;
    }

    // Destroy CSM per-layer image views BEFORE destroying the scene view that
    // owns the underlying array image — Vulkan requires views are destroyed
    // before the image they reference.
    for (int k = 0; k < CSM_CASCADE_COUNT; ++k) {
        if (csm_layer_views_[k]) {
            device_->destroyImageView(csm_layer_views_[k]);
            csm_layer_views_[k] = nullptr;
        }
    }

    // Destroys the array texture (csm_shadow_tex_) via m_depth_buffer_.
    shadow_object_scene_view_->destroy(device_);

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
