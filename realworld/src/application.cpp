#include <iostream>
#include <vector>
#include <map>
#include <limits>
#include <fstream>
#include <chrono>

#include "renderer/renderer.h"
#include "application.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"
#include "tiny_mtx2.h"

namespace {
constexpr int kWindowSizeX = 1920;
constexpr int kWindowSizeY = 1080;
static float s_sun_angle = 0.0f;

struct SkyBoxVertex {
    glm::vec3 pos;

    static std::vector<work::renderer::VertexInputBindingDescription> getBindingDescription() {
        std::vector<work::renderer::VertexInputBindingDescription> binding_description(1);
        binding_description[0].binding = 0;
        binding_description[0].stride = sizeof(SkyBoxVertex);
        binding_description[0].input_rate = work::renderer::VertexInputRate::VERTEX;
        return binding_description;
    }

    static std::vector<work::renderer::VertexInputAttributeDescription> getAttributeDescriptions() {
        std::vector<work::renderer::VertexInputAttributeDescription> attribute_descriptions(1);
        attribute_descriptions[0].binding = 0;
        attribute_descriptions[0].location = 0;
        attribute_descriptions[0].format = work::renderer::Format::R32G32B32_SFLOAT;
        attribute_descriptions[0].offset = offsetof(SkyBoxVertex, pos);
        return attribute_descriptions;
    }
};

struct TileVertex {
    glm::vec2 height;

    static std::vector<work::renderer::VertexInputBindingDescription> getBindingDescription() {
        std::vector<work::renderer::VertexInputBindingDescription> binding_description(1);
        binding_description[0].binding = 0;
        binding_description[0].stride = sizeof(TileVertex);
        binding_description[0].input_rate = work::renderer::VertexInputRate::VERTEX;
        return binding_description;
    }

    static std::vector<work::renderer::VertexInputAttributeDescription> getAttributeDescriptions() {
        std::vector<work::renderer::VertexInputAttributeDescription> attribute_descriptions(1);
        attribute_descriptions[0].binding = 0;
        attribute_descriptions[0].location = 0;
        attribute_descriptions[0].format = work::renderer::Format::R32G32_SFLOAT;
        attribute_descriptions[0].offset = offsetof(TileVertex, height);
        return attribute_descriptions;
    }
};

std::vector<uint64_t> readFile(const std::string& file_name, uint64_t& file_size) {
    std::ifstream file(file_name, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        std::string error_message = std::string("failed to open file! :") + file_name;
        throw std::runtime_error(error_message);
    }

    file_size = (uint64_t)file.tellg();
    std::vector<uint64_t> buffer((file_size + sizeof(uint64_t) - 1) / sizeof(uint64_t));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);

    file.close();
    return buffer;
}

static void check_vk_result(VkResult err)
{
    if (err == 0)
        return;
    fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
    if (err < 0)
        abort();
}

}

namespace work {
namespace renderer {

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

void ObjectData::destroy(const std::shared_ptr<Device>& device) {
    for (auto& texture : textures_) {
        texture.destroy(device);
    }

    for (auto& material : materials_) {
        material.uniform_buffer_.destroy(device);
    }

    for (auto& buffer : buffers_) {
        buffer.destroy(device);
    }
}

}

namespace app {

void RealWorldApplication::run() {
    initWindow();
    initVulkan();
    initImgui();
    mainLoop();
    cleanup();
}

static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<work::app::RealWorldApplication*>(glfwGetWindowUserPointer(window));
    app->setFrameBufferResized(true);
}

static bool s_exit_game = false;
static bool s_mouse_init = false;
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
    if (action != GLFW_RELEASE) {
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

    float sensitivity = 0.02f;
    mouse_offset *= sensitivity;

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

void RealWorldApplication::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(kWindowSizeX, kWindowSizeY, "Real World", nullptr, nullptr);
    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
    glfwSetKeyCallback(window_, keyInputCallback);
    glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetCursorPosCallback(window_, mouseInputCallback);
}

void RealWorldApplication::createDepthResources()
{
    auto depth_format = renderer::Helper::findDepthFormat(device_info_.device);
    renderer::Helper::createDepthResources(
        device_info_,
        depth_format,
        swap_chain_info_.extent,
        depth_buffer_);
}

void RealWorldApplication::initVulkan() {
    // the initialization order has to be strict.
    instance_ = renderer::Helper::createInstance();
    physical_devices_ = renderer::Helper::collectPhysicalDevices(instance_);
    surface_ = renderer::Helper::createSurface(instance_, window_);
    physical_device_ = renderer::Helper::pickPhysicalDevice(physical_devices_, surface_);
    queue_indices_ = renderer::Helper::findQueueFamilies(physical_device_, surface_);
    device_ = renderer::Helper::createLogicalDevice(physical_device_, surface_, queue_indices_);
    assert(device_);
    device_info_.device = device_;
    graphics_queue_ = device_->getDeviceQueue(queue_indices_.graphics_family_.value());
    assert(graphics_queue_);
    device_info_.cmd_queue = graphics_queue_;
    present_queue_ = device_->getDeviceQueue(queue_indices_.present_family_.value());
    renderer::Helper::createSwapChain(window_, device_, surface_, queue_indices_, swap_chain_info_);
    createRenderPass();
    createImageViews();
    createCubemapRenderPass();
    createCubemapFramebuffers();
    createDescriptorSetLayout();
    createCommandPool();
    assert(command_pool_);
    device_info_.cmd_pool = command_pool_;
    renderer::Helper::init(device_info_);

//    loadGltfModel(device_info_, "assets/Avocado.glb");
//    loadGltfModel(device_info_, "assets/BoomBox.glb");
    gltf_object_ = renderer::loadGltfModel(device_info_, "assets/DamagedHelmet.glb");
//    loadGltfModel(device_info_, "assets/Duck.glb");
//    loadGltfModel(device_info_, "assets/MetalRoughSpheres.glb");
//    loadGltfModel(device_info_, "assets/BarramundiFish.glb");
//    loadGltfModel(device_info_, "assets/Lantern.glb");
//    *loadGltfModel(device_info_, "assets/MetalRoughSpheresNoTextures.glb");
//    loadGltfModel(device_info_, "assets/BrainStem.glb"); 
//    *loadGltfModel(device_info_, "assets/AnimatedTriangle.gltf");
    loadMtx2Texture("assets/environments/doge2/lambertian/diffuse.ktx2", ibl_diffuse_tex_);
    loadMtx2Texture("assets/environments/doge2/ggx/specular.ktx2", ibl_specular_tex_);
    loadMtx2Texture("assets/environments/doge2/charlie/sheen.ktx2", ibl_sheen_tex_);
    createGltfPipelineLayout();
    createTileMeshPipelineLayout();
    createSkyboxPipelineLayout();
    createCubemapPipelineLayout();
    createCubeSkyboxPipelineLayout();
    createCubemapComputePipelineLayout();
    createGraphicsPipeline();
    createComputePipeline();
    createDepthResources();
    createFramebuffers();
    auto format = work::renderer::Format::R8G8B8A8_UNORM;
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

    tile_mesh_ = std::make_shared<renderer::TileMesh> (device_info_, glm::uvec2(256, 256), glm::vec2(-100.0f, -100.0f), glm::vec2(100.0f, 100.0f));
}

void RealWorldApplication::initImgui() {
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
    auto logic_device = RENDER_TYPE_CAST(Device, device_);
    ImGui_ImplGlfw_InitForVulkan(window_, true);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = RENDER_TYPE_CAST(Instance, instance_)->get();
    init_info.PhysicalDevice = RENDER_TYPE_CAST(PhysicalDevice, logic_device->getPhysicalDevice())->get();
    init_info.Device = logic_device->get();
    init_info.QueueFamily = queue_indices_.graphics_family_.value();
    init_info.Queue = RENDER_TYPE_CAST(Queue, graphics_queue_)->get();
    init_info.PipelineCache = nullptr;// g_PipelineCache;
    init_info.DescriptorPool = RENDER_TYPE_CAST(DescriptorPool, descriptor_pool_)->get();
    init_info.Allocator = nullptr; // g_Allocator;
    init_info.MinImageCount = swap_chain_info_.framebuffers.size();
    init_info.ImageCount = swap_chain_info_.framebuffers.size();
    init_info.CheckVkResultFn = check_vk_result;
    ImGui_ImplVulkan_Init(&init_info, RENDER_TYPE_CAST(RenderPass, render_pass_)->get());

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
        auto current_cmd_buf = command_buffers_[0];
        VkCommandBuffer command_buffer = RENDER_TYPE_CAST(CommandBuffer, current_cmd_buf)->get();

        //auto err = vkResetCommandPool(init_info.Device, command_pool, 0);
        //check_vk_result(err);
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        auto err = vkBeginCommandBuffer(command_buffer, &begin_info);
        check_vk_result(err);

        ImGui_ImplVulkan_CreateFontsTexture(command_buffer);

        VkSubmitInfo end_info = {};
        end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        end_info.commandBufferCount = 1;
        end_info.pCommandBuffers = &command_buffer;
        err = vkEndCommandBuffer(command_buffer);
        check_vk_result(err);
        err = vkQueueSubmit(init_info.Queue, 1, &end_info, VK_NULL_HANDLE);
        check_vk_result(err);

        err = vkDeviceWaitIdle(init_info.Device);
        check_vk_result(err);
        ImGui_ImplVulkan_DestroyFontUploadObjects();
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

    cleanupSwapChain();

    renderer::Helper::createSwapChain(window_, device_, surface_, queue_indices_, swap_chain_info_);
    createRenderPass();
    createImageViews();
    createGltfPipelineLayout();
    createSkyboxPipelineLayout();
    createCubemapPipelineLayout();
    createCubemapComputePipelineLayout();
    createGraphicsPipeline();
    createComputePipeline();
    createDepthResources();
    createFramebuffers();
    createUniformBuffers();
    descriptor_pool_ = device_->createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
}

void RealWorldApplication::createImageViews() {
    swap_chain_info_.image_views.resize(swap_chain_info_.images.size());
    for (uint64_t i_img = 0; i_img < swap_chain_info_.images.size(); i_img++) {
        swap_chain_info_.image_views[i_img] = device_->createImageView(
            swap_chain_info_.images[i_img],
            renderer::ImageViewType::VIEW_2D,
            swap_chain_info_.format,
            SET_FLAG_BIT(ImageAspect, COLOR_BIT));
    }
}

void RealWorldApplication::createCubemapFramebuffers() {
    uint32_t num_mips = static_cast<uint32_t>(std::log2(kCubemapSize) + 1);
    std::vector<work::renderer::BufferImageCopyInfo> dump_copies;

    renderer::Helper::createCubemapTexture(
        device_info_,
        cubemap_render_pass_,
        kCubemapSize,
        kCubemapSize,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_envmap_tex_);

    renderer::Helper::createCubemapTexture(
        device_info_,
        cubemap_render_pass_,
        kCubemapSize,
        kCubemapSize,
        1,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        tmp_ibl_diffuse_tex_);

    renderer::Helper::createCubemapTexture(
        device_info_,
        cubemap_render_pass_,
        kCubemapSize,
        kCubemapSize,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        tmp_ibl_specular_tex_);

    renderer::Helper::createCubemapTexture(
        device_info_,
        cubemap_render_pass_,
        kCubemapSize,
        kCubemapSize,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        tmp_ibl_sheen_tex_);

    renderer::Helper::createCubemapTexture(
        device_info_,
        cubemap_render_pass_,
        kCubemapSize,
        kCubemapSize,
        1,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_ibl_diffuse_tex_);

    renderer::Helper::createCubemapTexture(
        device_info_,
        cubemap_render_pass_,
        kCubemapSize,
        kCubemapSize,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_ibl_specular_tex_);

    renderer::Helper::createCubemapTexture(
        device_info_,
        cubemap_render_pass_,
        kCubemapSize,
        kCubemapSize,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_ibl_sheen_tex_);
}

renderer::DescriptorSetLayoutBinding getTextureSamplerDescriptionSetLayoutBinding(
    uint32_t binding, 
    renderer::ShaderStageFlags stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
    renderer::DescriptorType descript_type = renderer::DescriptorType::COMBINED_IMAGE_SAMPLER) {
    renderer::DescriptorSetLayoutBinding texture_binding{};
    texture_binding.binding = binding;
    texture_binding.descriptor_count = 1;
    texture_binding.descriptor_type = descript_type;
    texture_binding.immutable_samplers = nullptr;
    texture_binding.stage_flags = stage_flags;

    return texture_binding;
}

renderer::ShaderModuleList getGltfShaderModules(
    std::shared_ptr<renderer::Device> device, 
    bool has_normals, 
    bool has_tangent, 
    bool has_texcoord_0,
    bool has_skin_set_0)
{
    renderer::ShaderModuleList shader_modules(2);
    std::string feature_str = std::string(has_texcoord_0 ? "_TEX" : "") + 
        (has_tangent ? "_TN" : (has_normals ? "_N" : "")) +
        (has_skin_set_0 ? "_SKIN" : "");
    uint64_t vert_code_size, frag_code_size;
    auto vert_shader_code = readFile("lib/shaders/base_vert" + feature_str + ".spv", vert_code_size);
    auto frag_shader_code = readFile("lib/shaders/base_frag" + feature_str + ".spv", frag_code_size);

    shader_modules[0] = device->createShaderModule(vert_code_size, vert_shader_code.data());
    shader_modules[1] = device->createShaderModule(frag_code_size, frag_shader_code.data());

    return shader_modules;
}

std::vector<std::shared_ptr<renderer::ShaderModule>> getSkyboxShaderModules(
    std::shared_ptr<renderer::Device> device)
{
    uint64_t vert_code_size, frag_code_size;
    std::vector<std::shared_ptr<renderer::ShaderModule>> shader_modules(2);
    auto vert_shader_code = readFile("lib/shaders/skybox_vert.spv", vert_code_size);
    auto frag_shader_code = readFile("lib/shaders/skybox_frag.spv", frag_code_size);

    shader_modules[0] = device->createShaderModule(vert_code_size, vert_shader_code.data());
    shader_modules[1] = device->createShaderModule(frag_code_size, frag_shader_code.data());

    return shader_modules;
}

std::vector<std::shared_ptr<renderer::ShaderModule>> getTileShaderModules(
    std::shared_ptr<renderer::Device> device)
{
    uint64_t vert_code_size, frag_code_size;
    std::vector<std::shared_ptr<renderer::ShaderModule>> shader_modules(2);
    auto vert_shader_code = readFile("lib/shaders/tile_vert.spv", vert_code_size);
    auto frag_shader_code = readFile("lib/shaders/tile_frag.spv", frag_code_size);

    shader_modules[0] = device->createShaderModule(vert_code_size, vert_shader_code.data());
    shader_modules[1] = device->createShaderModule(frag_code_size, frag_shader_code.data());

    return shader_modules;
}

std::vector<std::shared_ptr<renderer::ShaderModule>> getIblShaderModules(
    std::shared_ptr<renderer::Device> device)
{
    uint64_t vert_code_size, frag_code_size;
    std::vector<std::shared_ptr<renderer::ShaderModule>> shader_modules;
    shader_modules.reserve(7);
    auto vert_shader_code = readFile("lib/shaders/ibl_vert.spv", vert_code_size);
    shader_modules.push_back(device->createShaderModule(vert_code_size, vert_shader_code.data()));
    auto frag_shader_code = readFile("lib/shaders/panorama_to_cubemap_frag.spv", frag_code_size);
    shader_modules.push_back(device->createShaderModule(frag_code_size, frag_shader_code.data()));
    auto labertian_frag_shader_code = readFile("lib/shaders/ibl_labertian_frag.spv", frag_code_size);
    shader_modules.push_back(device->createShaderModule(frag_code_size, labertian_frag_shader_code.data()));
    auto ggx_frag_shader_code = readFile("lib/shaders/ibl_ggx_frag.spv", frag_code_size);
    shader_modules.push_back(device->createShaderModule(frag_code_size, ggx_frag_shader_code.data()));
    auto charlie_frag_shader_code = readFile("lib/shaders/ibl_charlie_frag.spv", frag_code_size);
    shader_modules.push_back(device->createShaderModule(frag_code_size, charlie_frag_shader_code.data()));
    auto cube_skybox_shader_code = readFile("lib/shaders/cube_skybox.spv", frag_code_size);
    shader_modules.push_back(device->createShaderModule(frag_code_size, cube_skybox_shader_code.data()));

    return shader_modules;
}

std::vector<std::shared_ptr<renderer::ShaderModule>> getIblComputeShaderModules(
    std::shared_ptr<renderer::Device> device)
{
    uint64_t compute_code_size;
    std::vector<std::shared_ptr<renderer::ShaderModule>> shader_modules;
    shader_modules.reserve(1);
    auto compute_shader_code = readFile("lib/shaders/ibl_smooth_comp.spv", compute_code_size);
    shader_modules.push_back(device->createShaderModule(compute_code_size, compute_shader_code.data()));

    return shader_modules;
}

void RealWorldApplication::createGltfPipelineLayout()
{
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, VERTEX_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(ModelParams);

    renderer::DescriptorSetLayoutList desc_set_layouts;
    desc_set_layouts.reserve(3);
    desc_set_layouts.push_back(global_tex_desc_set_layout_);
    desc_set_layouts.push_back(desc_set_layout_);
    if (gltf_object_->materials_.size() > 0) {
        desc_set_layouts.push_back(material_tex_desc_set_layout_);
    }

    gltf_pipeline_layout_ = device_->createPipelineLayout(desc_set_layouts, { push_const_range });
}

void RealWorldApplication::createTileMeshPipelineLayout()
{
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, VERTEX_BIT) |
                                   SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(TileParams);

    renderer::DescriptorSetLayoutList desc_set_layouts;
    desc_set_layouts.reserve(3);
    desc_set_layouts.push_back(global_tex_desc_set_layout_);
    desc_set_layouts.push_back(desc_set_layout_);
    desc_set_layouts.push_back(skybox_desc_set_layout_);

    tile_pipeline_layout_ = device_->createPipelineLayout(desc_set_layouts, { push_const_range });
}

void RealWorldApplication::createSkyboxPipelineLayout()
{
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(SunSkyParams);

    renderer::DescriptorSetLayoutList desc_set_layouts;
    desc_set_layouts.reserve(2);
    desc_set_layouts.push_back(skybox_desc_set_layout_);
    desc_set_layouts.push_back(desc_set_layout_);

    skybox_pipeline_layout_ = device_->createPipelineLayout(desc_set_layouts, { push_const_range });
}

void RealWorldApplication::createCubemapPipelineLayout()
{
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(IblParams);

    renderer::DescriptorSetLayoutList desc_set_layouts(1);
    desc_set_layouts[0] = ibl_desc_set_layout_;

    ibl_pipeline_layout_ = device_->createPipelineLayout(desc_set_layouts, { push_const_range });
}

void RealWorldApplication::createCubeSkyboxPipelineLayout()
{
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(SunSkyParams);

    renderer::DescriptorSetLayoutList desc_set_layouts(1);
    desc_set_layouts[0] = ibl_desc_set_layout_;

    cube_skybox_pipeline_layout_ = device_->createPipelineLayout(desc_set_layouts, { push_const_range });
}

void RealWorldApplication::createCubemapComputePipelineLayout()
{
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(IblComputeParams);

    renderer::DescriptorSetLayoutList desc_set_layouts(1);
    desc_set_layouts[0] = ibl_comp_desc_set_layout_;

    ibl_comp_pipeline_layout_ = device_->createPipelineLayout(desc_set_layouts, { push_const_range });
}

renderer::PipelineColorBlendAttachmentState fillPipelineColorBlendAttachmentState(
    renderer::ColorComponentFlags color_write_mask = SET_FLAG_BIT(ColorComponent, ALL_BITS),
    bool blend_enable = false,
    renderer::BlendFactor src_color_blend_factor = renderer::BlendFactor::ONE,
    renderer::BlendFactor dst_color_blend_factor = renderer::BlendFactor::ZERO,
    renderer::BlendOp color_blend_op = renderer::BlendOp::ADD,
    renderer::BlendFactor src_alpha_blend_factor = renderer::BlendFactor::ONE,
    renderer::BlendFactor dst_alpha_blend_factor = renderer::BlendFactor::ZERO,
    renderer::BlendOp alpha_blend_op = renderer::BlendOp::ADD) {
    renderer::PipelineColorBlendAttachmentState color_blend_attachment{};
    color_blend_attachment.color_write_mask = color_write_mask;
    color_blend_attachment.blend_enable = blend_enable;
    color_blend_attachment.src_color_blend_factor = src_color_blend_factor; // Optional
    color_blend_attachment.dst_color_blend_factor = dst_color_blend_factor; // Optional
    color_blend_attachment.color_blend_op = color_blend_op; // Optional
    color_blend_attachment.src_alpha_blend_factor = src_alpha_blend_factor; // Optional
    color_blend_attachment.dst_alpha_blend_factor = dst_alpha_blend_factor; // Optional
    color_blend_attachment.alpha_blend_op = alpha_blend_op; // Optional

    return color_blend_attachment;
}

renderer::PipelineColorBlendStateCreateInfo fillPipelineColorBlendStateCreateInfo(
    const std::vector<renderer::PipelineColorBlendAttachmentState>& color_blend_attachments,
    bool logic_op_enable = false,
    renderer::LogicOp logic_op = renderer::LogicOp::NO_OP,
    glm::vec4 blend_constants = glm::vec4(0.0f)) {
    renderer::PipelineColorBlendStateCreateInfo color_blending{};
    color_blending.logic_op_enable = logic_op_enable;
    color_blending.logic_op = logic_op; // Optional
    color_blending.attachment_count = static_cast<uint32_t>(color_blend_attachments.size());
    color_blending.attachments = color_blend_attachments.data();
    color_blending.blend_constants = blend_constants; // Optional

    return color_blending;
}

renderer::PipelineRasterizationStateCreateInfo fillPipelineRasterizationStateCreateInfo(
    bool depth_clamp_enable = false,
    bool rasterizer_discard_enable = false,
    renderer::PolygonMode polygon_mode = renderer::PolygonMode::FILL,
    renderer::CullModeFlags cull_mode = SET_FLAG_BIT(CullMode, BACK_BIT),
    renderer::FrontFace front_face = renderer::FrontFace::COUNTER_CLOCKWISE,
    bool  depth_bias_enable = false,
    float depth_bias_constant_factor = 0.0f,
    float depth_bias_clamp = 0.0f,
    float depth_bias_slope_factor = 0.0f,
    float line_width = 1.0f) {
    renderer::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.depth_clamp_enable = depth_clamp_enable;
    rasterizer.rasterizer_discard_enable = rasterizer_discard_enable;
    rasterizer.polygon_mode = polygon_mode;
    rasterizer.line_width = line_width;
    rasterizer.cull_mode = cull_mode;
    rasterizer.front_face = front_face;
    rasterizer.depth_bias_enable = depth_bias_enable;
    rasterizer.depth_bias_constant_factor = depth_bias_constant_factor; // Optional
    rasterizer.depth_bias_clamp = depth_bias_clamp; // Optional
    rasterizer.depth_bias_slope_factor = depth_bias_slope_factor; // Optional

    return rasterizer;
}

renderer::PipelineMultisampleStateCreateInfo fillPipelineMultisampleStateCreateInfo(
    renderer::SampleCountFlagBits rasterization_samples = renderer::SampleCountFlagBits::SC_1_BIT,
    bool sample_shading_enable = false,
    float min_sample_shading = 1.0f,
    const renderer::SampleMask* sample_mask = nullptr,
    bool alpha_to_coverage_enable = false,
    bool alpha_to_one_enable = false) {
    renderer::PipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sample_shading_enable = sample_shading_enable;
    multisampling.rasterization_samples = rasterization_samples;
    multisampling.min_sample_shading = min_sample_shading; // Optional
    multisampling.sample_mask = sample_mask; // Optional
    multisampling.alpha_to_coverage_enable = alpha_to_coverage_enable; // Optional
    multisampling.alpha_to_one_enable = alpha_to_one_enable; // Optional

    return multisampling;
}

renderer::StencilOpState fillDepthStencilInfo(
    renderer::StencilOp fail_op = renderer::StencilOp::KEEP,
    renderer::StencilOp pass_op = renderer::StencilOp::KEEP,
    renderer::StencilOp depth_fail_op = renderer::StencilOp::KEEP,
    renderer::CompareOp compare_op = renderer::CompareOp::NEVER,
    uint32_t        compare_mask = 0xff,
    uint32_t        write_mask = 0xff,
    uint32_t        reference = 0x00) {
    renderer::StencilOpState stencil_op_state;
    stencil_op_state.fail_op = fail_op;
    stencil_op_state.pass_op = pass_op;
    stencil_op_state.depth_fail_op = depth_fail_op;
    stencil_op_state.compare_op = compare_op;
    stencil_op_state.compare_mask = compare_mask;
    stencil_op_state.write_mask = write_mask;
    stencil_op_state.reference = reference;

    return stencil_op_state;
}

renderer::PipelineDepthStencilStateCreateInfo fillPipelineDepthStencilStateCreateInfo(
    bool depth_test_enable = true,
    bool depth_write_enable = true,
    renderer::CompareOp depth_compare_op = renderer::CompareOp::LESS,
    bool depth_bounds_test_enable = false,
    float min_depth_bounds = 0.0f,
    float max_depth_bounds = 1.0f,
    bool stencil_test_enable = false,
    renderer::StencilOpState front = {},
    renderer::StencilOpState back = {})
{
    renderer::PipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.depth_test_enable = depth_test_enable;
    depth_stencil.depth_write_enable = depth_write_enable;
    depth_stencil.depth_compare_op = depth_compare_op;
    depth_stencil.depth_bounds_test_enable = depth_bounds_test_enable;
    depth_stencil.min_depth_bounds = min_depth_bounds; // Optional
    depth_stencil.max_depth_bounds = max_depth_bounds; // Optional
    depth_stencil.stencil_test_enable = stencil_test_enable;
    depth_stencil.front = front; // Optional
    depth_stencil.back = back; // Optional

    return depth_stencil;
}

void RealWorldApplication::createGraphicsPipeline() {
    auto color_blend_attachment = fillPipelineColorBlendAttachmentState();
    std::vector<renderer::PipelineColorBlendAttachmentState> color_blend_attachments(1, color_blend_attachment);
    std::vector<renderer::PipelineColorBlendAttachmentState> cube_color_blend_attachments(6, color_blend_attachment);
    auto color_blending = fillPipelineColorBlendStateCreateInfo(color_blend_attachments);
    auto cubemap_color_blending = fillPipelineColorBlendStateCreateInfo(cube_color_blend_attachments);
    auto rasterizer = fillPipelineRasterizationStateCreateInfo();
    auto ibl_rasterizer = fillPipelineRasterizationStateCreateInfo(
        false, false, renderer::PolygonMode::FILL,
        SET_FLAG_BIT(CullMode, NONE));
    auto multisampling = fillPipelineMultisampleStateCreateInfo();
    auto depth_stencil_info = fillDepthStencilInfo();
    auto depth_stencil = fillPipelineDepthStencilStateCreateInfo();
    auto cubemap_depth_stencil = fillPipelineDepthStencilStateCreateInfo(
        false, false, renderer::CompareOp::ALWAYS, false);
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
    {
        auto shader_modules = getGltfShaderModules(
            device_,
            gltf_object_->meshes_[0].primitives_[0].has_normal_,
            gltf_object_->meshes_[0].primitives_[0].has_tangent_,
            gltf_object_->meshes_[0].primitives_[0].has_texcoord_0_,
            gltf_object_->meshes_[0].primitives_[0].has_skin_set_0_);

        const auto& primitives = gltf_object_->meshes_[0].primitives_[0];
        gltf_pipeline_ = device_->createPipeline(
            render_pass_,
            gltf_pipeline_layout_,
            primitives.binding_descs_,
            primitives.attribute_descs_,
            primitives.topology_info_,
            color_blending,
            rasterizer,
            multisampling,
            depth_stencil,
            shader_modules,
            swap_chain_info_.extent);

        for (auto& shader_module : shader_modules) {
            device_->destroyShaderModule(shader_module);
        }
    }

    renderer::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = renderer::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;
    {
        auto shader_modules = getTileShaderModules(device_);
        tile_pipeline_ = device_->createPipeline(
            render_pass_,
            tile_pipeline_layout_,
            TileVertex::getBindingDescription(),
            TileVertex::getAttributeDescriptions(),
            input_assembly,
            color_blending,
            rasterizer,
            multisampling,
            depth_stencil,
            shader_modules,
            swap_chain_info_.extent);

        for (auto& shader_module : shader_modules) {
            device_->destroyShaderModule(shader_module);
        }
    }

    {
        auto shader_modules = getSkyboxShaderModules(device_);
        skybox_pipeline_ = device_->createPipeline(
            render_pass_,
            skybox_pipeline_layout_,
            SkyBoxVertex::getBindingDescription(),
            SkyBoxVertex::getAttributeDescriptions(),
            input_assembly,
            color_blending,
            rasterizer,
            multisampling,
            depth_stencil,
            shader_modules,
            swap_chain_info_.extent);

        for (auto& shader_module : shader_modules) {
            device_->destroyShaderModule(shader_module);
        }
    }

    {
        auto ibl_shader_modules = getIblShaderModules(device_);
        cube_skybox_pipeline_ = device_->createPipeline(
            cubemap_render_pass_,
            cube_skybox_pipeline_layout_,
            {}, {},
            input_assembly,
            cubemap_color_blending,
            ibl_rasterizer,
            multisampling,
            cubemap_depth_stencil,
            { ibl_shader_modules[0], ibl_shader_modules[5] },
            glm::uvec2(kCubemapSize, kCubemapSize));

        envmap_pipeline_ = device_->createPipeline(
            cubemap_render_pass_,
            ibl_pipeline_layout_,
            {}, {},
            input_assembly,
            cubemap_color_blending,
            ibl_rasterizer,
            multisampling,
            cubemap_depth_stencil,
            { ibl_shader_modules[0], ibl_shader_modules[1] },
            glm::uvec2(kCubemapSize, kCubemapSize));

        lambertian_pipeline_ = device_->createPipeline(
            cubemap_render_pass_,
            ibl_pipeline_layout_,
            {}, {},
            input_assembly,
            cubemap_color_blending,
            ibl_rasterizer,
            multisampling,
            cubemap_depth_stencil,
            { ibl_shader_modules[0], ibl_shader_modules[2] },
            glm::uvec2(kCubemapSize, kCubemapSize));

        ggx_pipeline_ = device_->createPipeline(
            cubemap_render_pass_,
            ibl_pipeline_layout_,
            {}, {},
            input_assembly,
            cubemap_color_blending,
            ibl_rasterizer,
            multisampling,
            cubemap_depth_stencil,
            { ibl_shader_modules[0], ibl_shader_modules[3] },
            glm::uvec2(kCubemapSize, kCubemapSize));

        charlie_pipeline_ = device_->createPipeline(
            cubemap_render_pass_,
            ibl_pipeline_layout_,
            {}, {},
            input_assembly,
            cubemap_color_blending,
            ibl_rasterizer,
            multisampling,
            cubemap_depth_stencil,
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

renderer::AttachmentDescription FillAttachmentDescription(
    renderer::Format format,
    renderer::SampleCountFlagBits samples = renderer::SampleCountFlagBits::SC_1_BIT,
    renderer::ImageLayout initial_layout = renderer::ImageLayout::UNDEFINED,
    renderer::ImageLayout final_layout = renderer::ImageLayout::PRESENT_SRC_KHR,
    renderer::AttachmentLoadOp load_op = renderer::AttachmentLoadOp::CLEAR,
    renderer::AttachmentStoreOp store_op = renderer::AttachmentStoreOp::STORE,
    renderer::AttachmentLoadOp stencil_load_op = renderer::AttachmentLoadOp::DONT_CARE,
    renderer::AttachmentStoreOp stencil_store_op = renderer::AttachmentStoreOp::DONT_CARE) {

    renderer::AttachmentDescription attachment{};
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

renderer::SubpassDescription FillSubpassDescription(
    renderer::PipelineBindPoint pipeline_bind_point,
    const std::vector<renderer::AttachmentReference>& color_attachments,
    const renderer::AttachmentReference* depth_stencil_attachment,
    renderer::SubpassDescriptionFlags flags = static_cast<renderer::SubpassDescriptionFlags>(0),
    const std::vector<renderer::AttachmentReference>& input_attachments = {},
    const std::vector<renderer::AttachmentReference>& resolve_attachments = {}) {
    renderer::SubpassDescription desc{};
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

renderer::SubpassDependency FillSubpassDependency(
    uint32_t src_subpass,
    uint32_t dst_subpass,
    renderer::PipelineStageFlags src_stage_mask,
    renderer::PipelineStageFlags dst_stage_mask,
    renderer::AccessFlags src_access_mask,
    renderer::AccessFlags dst_access_mask,
    renderer::DependencyFlags dependency_flags = 0){
    renderer::SubpassDependency dependency{};
    dependency.src_subpass = src_subpass;
    dependency.dst_subpass = dst_subpass;
    dependency.src_stage_mask = src_stage_mask;
    dependency.dst_stage_mask = dst_stage_mask;
    dependency.src_access_mask = src_access_mask;
    dependency.dst_access_mask = dst_access_mask;
    dependency.dependency_flags = dependency_flags;
    return dependency;
}

void RealWorldApplication::createRenderPass() {
    renderer::AttachmentDescription color_attachment = FillAttachmentDescription(
        swap_chain_info_.format);

    renderer::AttachmentReference color_attachment_ref(0, renderer::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    auto depth_attachment = FillAttachmentDescription(
        renderer::Helper::findDepthFormat(device_),
        renderer::SampleCountFlagBits::SC_1_BIT,
        renderer::ImageLayout::UNDEFINED,
        renderer::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    renderer::AttachmentReference depth_attachment_ref(1, renderer::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    auto subpass = FillSubpassDescription(
        renderer::PipelineBindPoint::GRAPHICS,
        { color_attachment_ref },
        &depth_attachment_ref);

    auto depency = FillSubpassDependency(~0U, 0,
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT),
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT),
        0,
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT) |
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_READ_BIT));

    std::vector<renderer::AttachmentDescription> attachments(2);
    attachments[0] = color_attachment;
    attachments[1] = depth_attachment;

    render_pass_ = device_->createRenderPass(attachments, { subpass }, { depency });
}

void RealWorldApplication::createCubemapRenderPass() {
    auto color_attachment = FillAttachmentDescription(
        renderer::Format::R16G16B16A16_SFLOAT,
        renderer::SampleCountFlagBits::SC_1_BIT,
        renderer::ImageLayout::UNDEFINED,
        renderer::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    std::vector<renderer::AttachmentReference> color_attachment_refs(6);
    for (uint32_t i = 0; i < 6; i++) {
        color_attachment_refs[i].attachment_ = i;
        color_attachment_refs[i].layout_ = renderer::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    }

    auto subpass = FillSubpassDescription(
        renderer::PipelineBindPoint::GRAPHICS,
        color_attachment_refs,
        nullptr);

    auto depency = FillSubpassDependency(~0U, 0,
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT),
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT),
        0,
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT));

    std::vector<renderer::AttachmentDescription> attachments = {6, color_attachment};

    cubemap_render_pass_ = device_->createRenderPass(attachments, { subpass }, { depency });
}

void RealWorldApplication::createFramebuffers() {
    swap_chain_info_.framebuffers.resize(swap_chain_info_.image_views.size());
    for (uint64_t i = 0; i < swap_chain_info_.image_views.size(); i++) {
        assert(swap_chain_info_.image_views[i]);
        assert(depth_buffer_.view);
        assert(render_pass_);
        std::vector<std::shared_ptr<renderer::ImageView>> attachments(2);
        attachments[0] = swap_chain_info_.image_views[i];
        attachments[1] = depth_buffer_.view;

        swap_chain_info_.framebuffers[i] =
            device_->createFrameBuffer(render_pass_, attachments, swap_chain_info_.extent);
    }
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

void RealWorldApplication::mainLoop() {
    while (!glfwWindowShouldClose(window_) && !s_exit_game) {
        glfwPollEvents();
        drawFrame();
    }

    device_->waitIdle();
}

void RealWorldApplication::drawFrame() {
    std::vector<std::shared_ptr<renderer::Fence>> in_flight_fences(1);
    in_flight_fences[0] = in_flight_fences_[current_frame_];
    device_->waitForFences(in_flight_fences);

    auto vk_device = RENDER_TYPE_CAST(Device, device_);
    auto vk_swap_chain = RENDER_TYPE_CAST(Swapchain, swap_chain_info_.swap_chain);
    auto vk_img_available_semphores = RENDER_TYPE_CAST(Semaphore, image_available_semaphores_[current_frame_]);
    assert(vk_device);

    uint32_t image_index;
    auto result = vkAcquireNextImageKHR(vk_device->get(), vk_swap_chain->get(), UINT64_MAX, vk_img_available_semphores->get(), VK_NULL_HANDLE, &image_index);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain();
        return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    if (images_in_flight_[image_index] != VK_NULL_HANDLE) {
        std::vector<std::shared_ptr<renderer::Fence>> images_in_flight(1);
        images_in_flight[0] = images_in_flight_[image_index];
        device_->waitForFences(images_in_flight);
    }
    // Mark the image as now being in use by this frame
    images_in_flight_[image_index] = in_flight_fences_[current_frame_];

    int32_t root_node = gltf_object_->default_scene_ >= 0 ? gltf_object_->default_scene_ : 0;
    auto min_t = gltf_object_->scenes_[root_node].bbox_min_;
    auto max_t = gltf_object_->scenes_[root_node].bbox_max_;

    auto center = (min_t + max_t) * 0.5f;
    auto extent = (max_t - min_t) * 0.5f;
    float radius = max(max(extent.x, extent.y), extent.z);

    updateViewConstBuffer(image_index, center, radius);

    device_->resetFences(in_flight_fences);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    // todo.
    auto command_buffer = command_buffers_[image_index];
    auto vk_command_buffer = RENDER_TYPE_CAST(CommandBuffer, command_buffer);
    std::vector<std::shared_ptr<renderer::CommandBuffer>>command_buffers(1, command_buffer);

    std::vector<renderer::ClearValue> clear_values(2);
    clear_values[0].color = { 50.0f / 255.0f, 50.0f / 255.0f, 50.0f / 255.0f, 1.0f };
    clear_values[1].depth_stencil = { 1.0f, 0 };

    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    auto& cmd_buf = command_buffer;

    cmd_buf->reset(0);
    cmd_buf->beginCommandBuffer(SET_FLAG_BIT(CommandBufferUsage, ONE_TIME_SUBMIT_BIT));

    if (0)
    {
        // generate envmap cubemap from panorama hdr image.
        cmd_buf->addImageBarrier(
            rt_envmap_tex_.image,
            renderer::Helper::getImageAsSource(),
            renderer::Helper::getImageAsColorAttachment(),
            0, 1, 0, 6);

        cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, envmap_pipeline_);

        std::vector<renderer::ClearValue> envmap_clear_values(6, clear_values[0]);
        cmd_buf->beginRenderPass(cubemap_render_pass_, rt_envmap_tex_.framebuffers[0], glm::uvec2(kCubemapSize, kCubemapSize), envmap_clear_values);

        IblParams ibl_params = {};
        cmd_buf->pushConstants(SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT), ibl_pipeline_layout_, &ibl_params, sizeof(ibl_params));

        cmd_buf->bindDescriptorSets(renderer::PipelineBindPoint::GRAPHICS, ibl_pipeline_layout_, { envmap_tex_desc_set_ });

        cmd_buf->draw(3);

        cmd_buf->endRenderPass();

        uint32_t num_mips = static_cast<uint32_t>(std::log2(kCubemapSize) + 1);

        renderer::Helper::generateMipmapLevels(
            cmd_buf,
            rt_envmap_tex_.image,
            num_mips,
            kCubemapSize,
            kCubemapSize,
            renderer::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }
    else {
        // generate envmap from skybox.
        cmd_buf->addImageBarrier(
            rt_envmap_tex_.image,
            renderer::Helper::getImageAsSource(),
            renderer::Helper::getImageAsColorAttachment(),
            0, 1, 0, 6);

        cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, cube_skybox_pipeline_);

        std::vector<renderer::ClearValue> envmap_clear_values(6, clear_values[0]);
        cmd_buf->beginRenderPass(cubemap_render_pass_, rt_envmap_tex_.framebuffers[0], glm::uvec2(kCubemapSize, kCubemapSize), envmap_clear_values);

        SunSkyParams sun_sky_params = {};
        sun_sky_params.sun_pos = glm::vec3(cos(s_sun_angle), sin(s_sun_angle), -0.3f);

        cmd_buf->pushConstants(SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT), cube_skybox_pipeline_layout_, &sun_sky_params, sizeof(sun_sky_params));

        cmd_buf->bindDescriptorSets(renderer::PipelineBindPoint::GRAPHICS, cube_skybox_pipeline_layout_, { envmap_tex_desc_set_ });

        cmd_buf->draw(3);

        cmd_buf->endRenderPass();

        uint32_t num_mips = static_cast<uint32_t>(std::log2(kCubemapSize) + 1);

        renderer::Helper::generateMipmapLevels(
            cmd_buf,
            rt_envmap_tex_.image,
            num_mips,
            kCubemapSize,
            kCubemapSize,
            renderer::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }

    // generate ibl diffuse texture.
    {
        cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, lambertian_pipeline_);

        cmd_buf->addImageBarrier(
            rt_ibl_diffuse_tex_.image,
            renderer::Helper::getImageAsSource(),
            renderer::Helper::getImageAsColorAttachment(),
            0, 1, 0, 6);

        std::vector<renderer::ClearValue> envmap_clear_values(6, clear_values[0]);
        cmd_buf->beginRenderPass(cubemap_render_pass_, rt_ibl_diffuse_tex_.framebuffers[0], glm::uvec2(kCubemapSize, kCubemapSize), envmap_clear_values);

        IblParams ibl_params = {};
        ibl_params.roughness = 1.0f;
        ibl_params.currentMipLevel = 0;
        ibl_params.width = kCubemapSize;
        ibl_params.lodBias = 0;
        cmd_buf->pushConstants(SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT), ibl_pipeline_layout_, &ibl_params, sizeof(ibl_params));

        cmd_buf->bindDescriptorSets(renderer::PipelineBindPoint::GRAPHICS, ibl_pipeline_layout_, { ibl_tex_desc_set_ });

        cmd_buf->draw(3);

        cmd_buf->endRenderPass();

        cmd_buf->addImageBarrier(
            rt_ibl_diffuse_tex_.image,
            renderer::Helper::getImageAsColorAttachment(),
            renderer::Helper::getImageAsShaderSampler(),
            0, 1, 0, 6);
    }

    // generate ibl specular texture.
    {
        uint32_t num_mips = static_cast<uint32_t>(std::log2(kCubemapSize) + 1);
        cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, ggx_pipeline_);

        for (int i_mip = num_mips-1; i_mip >= 0; i_mip--) {
            cmd_buf->addImageBarrier(
                rt_ibl_specular_tex_.image,
                renderer::Helper::getImageAsSource(),
                renderer::Helper::getImageAsColorAttachment(),
                i_mip, 1, 0, 6);

            uint32_t width = std::max(static_cast<uint32_t>(kCubemapSize) >> i_mip, 1u);
            uint32_t height = std::max(static_cast<uint32_t>(kCubemapSize) >> i_mip, 1u);

            std::vector<renderer::ClearValue> envmap_clear_values(6, clear_values[0]);
            cmd_buf->beginRenderPass(cubemap_render_pass_, rt_ibl_specular_tex_.framebuffers[i_mip], glm::uvec2(width, height), envmap_clear_values);

            IblParams ibl_params = {};
            ibl_params.roughness = static_cast<float>(i_mip) / static_cast<float>(num_mips - 1);
            ibl_params.currentMipLevel = i_mip;
            ibl_params.width = kCubemapSize;
            ibl_params.lodBias = 0;
            cmd_buf->pushConstants(SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT), ibl_pipeline_layout_, &ibl_params, sizeof(ibl_params));

            cmd_buf->bindDescriptorSets(renderer::PipelineBindPoint::GRAPHICS, ibl_pipeline_layout_, { ibl_tex_desc_set_ });

            cmd_buf->draw(3);

            cmd_buf->endRenderPass();
        }

        cmd_buf->addImageBarrier(
            rt_ibl_specular_tex_.image,
            renderer::Helper::getImageAsColorAttachment(),
            renderer::Helper::getImageAsShaderSampler(),
            0, num_mips, 0, 6);
    }

    // generate ibl sheen texture.
    {
        uint32_t num_mips = static_cast<uint32_t>(std::log2(kCubemapSize) + 1);
        cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, charlie_pipeline_);

        for (int i_mip = num_mips - 1; i_mip >= 0; i_mip--) {
            uint32_t width = std::max(static_cast<uint32_t>(kCubemapSize) >> i_mip, 1u);
            uint32_t height = std::max(static_cast<uint32_t>(kCubemapSize) >> i_mip, 1u);

            cmd_buf->addImageBarrier(
                rt_ibl_sheen_tex_.image,
                renderer::Helper::getImageAsSource(),
                renderer::Helper::getImageAsColorAttachment(),
                i_mip, 1, 0, 6);

            std::vector<renderer::ClearValue> envmap_clear_values(6, clear_values[0]);
            cmd_buf->beginRenderPass(cubemap_render_pass_, rt_ibl_sheen_tex_.framebuffers[i_mip], glm::uvec2(width, height), envmap_clear_values);

            IblParams ibl_params = {};
            ibl_params.roughness = static_cast<float>(i_mip) / static_cast<float>(num_mips - 1);
            ibl_params.currentMipLevel = i_mip;
            ibl_params.width = kCubemapSize;
            ibl_params.lodBias = 0;
            cmd_buf->pushConstants(SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT), ibl_pipeline_layout_, &ibl_params, sizeof(ibl_params));

            cmd_buf->bindDescriptorSets(renderer::PipelineBindPoint::GRAPHICS, ibl_pipeline_layout_, { ibl_tex_desc_set_ });

            cmd_buf->draw(3);

            cmd_buf->endRenderPass();
        }

        cmd_buf->addImageBarrier(
            rt_ibl_sheen_tex_.image,
            renderer::Helper::getImageAsColorAttachment(),
            renderer::Helper::getImageAsShaderSampler(),
            0, num_mips, 0, 6);
    }

    {
        if (0)
        {
            cmd_buf->addImageBarrier(
                rt_ibl_diffuse_tex_.image,
                renderer::Helper::getImageAsSource(),
                renderer::Helper::getImageAsStore(),
                0, 1, 0, 6);

            cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, blur_comp_pipeline_);
            IblComputeParams ibl_comp_params = {};
            ibl_comp_params.size = glm::ivec4(kCubemapSize, kCubemapSize, 0, 0);
            cmd_buf->pushConstants(SET_FLAG_BIT(ShaderStage, COMPUTE_BIT), ibl_comp_pipeline_layout_, &ibl_comp_params, sizeof(ibl_comp_params));

            cmd_buf->bindDescriptorSets(renderer::PipelineBindPoint::COMPUTE, ibl_comp_pipeline_layout_, { ibl_diffuse_tex_desc_set_ });

            cmd_buf->dispatch((kCubemapSize + 7) / 8, (kCubemapSize + 7) / 8, 6);

            uint32_t num_mips = static_cast<uint32_t>(std::log2(kCubemapSize) + 1);
            cmd_buf->addImageBarrier(
                rt_ibl_diffuse_tex_.image,
                renderer::Helper::getImageAsStore(),
                renderer::Helper::getImageAsShaderSampler(),
                0, 1, 0, 6);
            cmd_buf->addImageBarrier(
                rt_ibl_specular_tex_.image,
                renderer::Helper::getImageAsSource(),
                renderer::Helper::getImageAsShaderSampler(),
                0, num_mips, 0, 6);
            cmd_buf->addImageBarrier(
                rt_ibl_sheen_tex_.image,
                renderer::Helper::getImageAsSource(),
                renderer::Helper::getImageAsShaderSampler(),
                0, num_mips, 0, 6);
        }
    }

    {
        cmd_buf->beginRenderPass(
            render_pass_,
            swap_chain_info_.framebuffers[image_index],
            swap_chain_info_.extent, clear_values);

        // render gltf meshes.
        cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, gltf_pipeline_);

        auto model_mat = glm::translate(glm::mat4(1.0f), s_camera_pos + s_camera_dir * 5.0f);
        for (auto node_idx : gltf_object_->scenes_[root_node].nodes_) {
            renderer::drawNodes(cmd_buf,
                gltf_object_,
                gltf_pipeline_layout_,
                global_tex_desc_set_,
                desc_sets_,
                node_idx,
                image_index,
                model_mat);
        }

        // render terrain.
        {
            cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, tile_pipeline_);
            renderer::DescriptorSetList desc_sets{ global_tex_desc_set_, desc_sets_[image_index], skybox_tex_desc_set_ };
            tile_mesh_->draw(cmd_buf, tile_pipeline_layout_, desc_sets);
        }

        // render skybox.
        {
            cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, skybox_pipeline_);
            std::vector<std::shared_ptr<renderer::Buffer>> buffers(1);
            std::vector<uint64_t> offsets(1);
            buffers[0] = vertex_buffer_.buffer;
            offsets[0] = 0;

            cmd_buf->bindVertexBuffers(0, buffers, offsets);
            cmd_buf->bindIndexBuffer(index_buffer_.buffer, 0, renderer::IndexType::UINT16);

            SunSkyParams sun_sky_params = {};
            sun_sky_params.sun_pos = glm::vec3(cos(s_sun_angle), sin(s_sun_angle), -0.3f);
            s_sun_angle += 0.001f;
            cmd_buf->pushConstants(SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT), skybox_pipeline_layout_, &sun_sky_params, sizeof(sun_sky_params));

            renderer::DescriptorSetList desc_sets{ skybox_tex_desc_set_, desc_sets_[image_index] };
            cmd_buf->bindDescriptorSets(renderer::PipelineBindPoint::GRAPHICS, skybox_pipeline_layout_, desc_sets);

            cmd_buf->drawIndexed(36);
        }

        cmd_buf->endRenderPass();
    }

    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            //ShowExampleMenuFile();
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

    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();

    ImGui_ImplVulkan_RenderDrawData(draw_data, RENDER_TYPE_CAST(CommandBuffer, cmd_buf)->get());

    cmd_buf->endCommandBuffer();

    auto vk_render_finished_semphores = RENDER_TYPE_CAST(Semaphore, render_finished_semaphores_[current_frame_]);
    auto vk_in_flight_fence = RENDER_TYPE_CAST(Fence, in_flight_fences_[current_frame_]);
    auto vk_graphic_queue = RENDER_TYPE_CAST(Queue, graphics_queue_);
    auto vk_present_queue = RENDER_TYPE_CAST(Queue, present_queue_);

    std::vector<VkCommandBuffer> cmd_bufs(1, vk_command_buffer->get());
    VkSemaphore wait_semaphores[] = { vk_img_available_semphores->get() };
    VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = wait_semaphores;
    submit_info.pWaitDstStageMask = wait_stages;
    submit_info.commandBufferCount = static_cast<uint32_t>(cmd_bufs.size());
    submit_info.pCommandBuffers = cmd_bufs.data();
    VkSemaphore signal_semaphores[] = { vk_render_finished_semphores->get() };
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = signal_semaphores;

    if (vkQueueSubmit(vk_graphic_queue->get(), 1, &submit_info, vk_in_flight_fence->get()) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = signal_semaphores;
    VkSwapchainKHR swapChains[] = { vk_swap_chain->get() };
    present_info.swapchainCount = 1;
    present_info.pSwapchains = swapChains;
    present_info.pImageIndices = &image_index;
    present_info.pResults = nullptr; // Optional

    result = vkQueuePresentKHR(vk_present_queue->get(), &present_info);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebuffer_resized_) {
        framebuffer_resized_ = false;
        recreateSwapChain();
    }
    else if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image!");
    }

    current_frame_ = (current_frame_ + 1) % kMaxFramesInFlight;
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

    renderer::Helper::createBufferWithSrcData(
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
        0, 2, 4, 4, 2, 6};

    uint64_t buffer_size =
        sizeof(indices[0]) * indices.size();

    renderer::Helper::createBufferWithSrcData(
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
        std::vector<renderer::DescriptorSetLayoutBinding> bindings;
        bindings.reserve(5);

        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(GGX_LUT_INDEX));
        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(CHARLIE_LUT_INDEX));
        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(LAMBERTIAN_ENV_TEX_INDEX));
        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(GGX_ENV_TEX_INDEX));
        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(CHARLIE_ENV_TEX_INDEX));

        global_tex_desc_set_layout_ = device_->createDescriptorSetLayout(bindings);
    }

    // material texture descriptor set layout.
    {
        std::vector<renderer::DescriptorSetLayoutBinding> bindings;
        bindings.reserve(7);

        renderer::DescriptorSetLayoutBinding ubo_pbr_layout_binding{};
        ubo_pbr_layout_binding.binding = PBR_CONSTANT_INDEX;
        ubo_pbr_layout_binding.descriptor_count = 1;
        ubo_pbr_layout_binding.descriptor_type = renderer::DescriptorType::UNIFORM_BUFFER;
        ubo_pbr_layout_binding.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
        ubo_pbr_layout_binding.immutable_samplers = nullptr; // Optional
        bindings.push_back(ubo_pbr_layout_binding);

        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(BASE_COLOR_TEX_INDEX));
        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(NORMAL_TEX_INDEX));
        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(METAL_ROUGHNESS_TEX_INDEX));
        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(EMISSIVE_TEX_INDEX));
        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(OCCLUSION_TEX_INDEX));
        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(THIN_FILM_LUT_INDEX));

        material_tex_desc_set_layout_ = device_->createDescriptorSetLayout(bindings);
    }

    {
        std::vector<renderer::DescriptorSetLayoutBinding> bindings(1);

        renderer::DescriptorSetLayoutBinding ubo_layout_binding{};
        ubo_layout_binding.binding = VIEW_CONSTANT_INDEX;
        ubo_layout_binding.descriptor_count = 1;
        ubo_layout_binding.descriptor_type = renderer::DescriptorType::UNIFORM_BUFFER;
        ubo_layout_binding.stage_flags = SET_FLAG_BIT(ShaderStage, VERTEX_BIT) | 
                                         SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
        ubo_layout_binding.immutable_samplers = nullptr; // Optional
        bindings[0] = ubo_layout_binding;

        desc_set_layout_ = device_->createDescriptorSetLayout(bindings);
    }

    {
        std::vector<renderer::DescriptorSetLayoutBinding> bindings(1);

        bindings[0] = getTextureSamplerDescriptionSetLayoutBinding(BASE_COLOR_TEX_INDEX);

        skybox_desc_set_layout_ = device_->createDescriptorSetLayout(bindings);
    }

    // ibl texture descriptor set layout.
    {
        std::vector<renderer::DescriptorSetLayoutBinding> bindings(1);
        bindings[0] = getTextureSamplerDescriptionSetLayoutBinding(PANORAMA_TEX_INDEX);
        //bindings[1] = getTextureSamplerDescriptionSetLayoutBinding(ENVMAP_TEX_INDEX);

        ibl_desc_set_layout_ = device_->createDescriptorSetLayout(bindings);
    }

    // ibl compute texture descriptor set layout.
    {
        std::vector<renderer::DescriptorSetLayoutBinding> bindings(2);
        bindings[0] = getTextureSamplerDescriptionSetLayoutBinding(SRC_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            renderer::DescriptorType::STORAGE_IMAGE);
        bindings[1] = getTextureSamplerDescriptionSetLayoutBinding(DST_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            renderer::DescriptorType::STORAGE_IMAGE);

        ibl_comp_desc_set_layout_ = device_->createDescriptorSetLayout(bindings);
    }
}

void RealWorldApplication::createUniformBuffers() {
    uint64_t buffer_size = sizeof(ViewParams);
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

void RealWorldApplication::updateViewConstBuffer(uint32_t current_image, const glm::vec3& center, float radius) {
    auto aspect = swap_chain_info_.extent.x / (float)swap_chain_info_.extent.y;

    ViewParams view_params{};
    view_params.camera_pos = glm::vec4(s_camera_pos, 0);
    view_params.view = glm::lookAt(s_camera_pos, s_camera_pos + s_camera_dir, s_camera_up);
    view_params.proj = glm::perspective(glm::radians(45.0f), aspect, 1.0f * radius, 10000.0f);
    view_params.proj[1][1] *= -1;
    view_params.input_features = glm::vec4(gltf_object_->meshes_[0].primitives_[0].has_tangent_ ? FEATURE_INPUT_HAS_TANGENT : 0, 0, 0, 0);

    device_->updateBufferMemory(view_const_buffers_[current_image].memory, sizeof(view_params), &view_params);
}

std::vector<renderer::TextureDescriptor> RealWorldApplication::addGlobalTextures(
    const std::shared_ptr<renderer::DescriptorSet>& description_set)
{
    std::vector<renderer::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(5);
    renderer::Helper::addOneTexture(descriptor_writes, GGX_LUT_INDEX, texture_sampler_, ggx_lut_tex_.view, description_set);
    renderer::Helper::addOneTexture(descriptor_writes, CHARLIE_LUT_INDEX, texture_sampler_, charlie_lut_tex_.view, description_set);
    renderer::Helper::addOneTexture(descriptor_writes, LAMBERTIAN_ENV_TEX_INDEX, texture_sampler_, rt_ibl_diffuse_tex_.view, description_set);
    renderer::Helper::addOneTexture(descriptor_writes, GGX_ENV_TEX_INDEX, texture_sampler_, rt_ibl_specular_tex_.view, description_set);
    renderer::Helper::addOneTexture(descriptor_writes, CHARLIE_ENV_TEX_INDEX, texture_sampler_, rt_ibl_sheen_tex_.view, description_set);

    return descriptor_writes;
}

std::vector<renderer::TextureDescriptor> RealWorldApplication::addSkyboxTextures(
    const std::shared_ptr<renderer::DescriptorSet>& description_set)
{
    std::vector<renderer::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(1);

    // envmap texture.
    renderer::Helper::addOneTexture(descriptor_writes, BASE_COLOR_TEX_INDEX, texture_sampler_, rt_envmap_tex_.view, description_set);

    return descriptor_writes;
}

std::vector<renderer::TextureDescriptor> RealWorldApplication::addPanoramaTextures(
    const std::shared_ptr<renderer::DescriptorSet>& description_set)
{
    std::vector<renderer::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(1);

    // envmap texture.
    renderer::Helper::addOneTexture(descriptor_writes, PANORAMA_TEX_INDEX, texture_sampler_, panorama_tex_.view, description_set);

    return descriptor_writes;
}

std::vector<renderer::TextureDescriptor> RealWorldApplication::addIblTextures(
    const std::shared_ptr<renderer::DescriptorSet>& description_set)
{
    std::vector<renderer::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(1);

    renderer::Helper::addOneTexture(descriptor_writes, ENVMAP_TEX_INDEX, texture_sampler_, rt_envmap_tex_.view, description_set);

    return descriptor_writes;
}

std::vector<renderer::TextureDescriptor> RealWorldApplication::addIblComputeTextures(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const renderer::TextureInfo& src_tex,
    const renderer::TextureInfo& dst_tex)
{
    std::vector<renderer::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(2);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        SRC_TEX_INDEX,
        texture_sampler_,
        src_tex.view,
        description_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        renderer::ImageLayout::GENERAL);
    renderer::Helper::addOneTexture(
        descriptor_writes,
        DST_TEX_INDEX,
        texture_sampler_,
        dst_tex.view,
        description_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        renderer::ImageLayout::GENERAL);

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
        for (uint32_t i_mat = 0; i_mat < gltf_object_->materials_.size(); i_mat++) {
            auto& material = gltf_object_->materials_[i_mat];
            material.desc_set_ = device_->createDescriptorSets(descriptor_pool_, material_tex_desc_set_layout_, 1)[0];

            std::vector<renderer::BufferDescriptor> material_buffer_descs;
            renderer::Helper::addOneBuffer(
                material_buffer_descs,
                PBR_CONSTANT_INDEX,
                material.uniform_buffer_.buffer,
                material.desc_set_,
                renderer::DescriptorType::UNIFORM_BUFFER,
                sizeof(PbrMaterialParams));

            // create a global ibl texture descriptor set.
            auto material_tex_descs = renderer::addGltfTextures(gltf_object_, material, texture_sampler_, thin_film_lut_tex_);

            device_->updateDescriptorSets(material_tex_descs, material_buffer_descs);
        }
    }

    {
        desc_sets_ = device_->createDescriptorSets(descriptor_pool_, desc_set_layout_, buffer_count);
        for (uint64_t i = 0; i < buffer_count; i++) {
            std::vector<renderer::BufferDescriptor> buffer_descs;
            renderer::Helper::addOneBuffer(
                buffer_descs,
                VIEW_CONSTANT_INDEX,
                view_const_buffers_[i].buffer,
                desc_sets_[i],
                renderer::DescriptorType::UNIFORM_BUFFER,
                sizeof(ViewParams));

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

void RealWorldApplication::createTextureImage(const std::string& file_name, renderer::Format format, renderer::TextureInfo& texture) {
    int tex_width, tex_height, tex_channels;
    stbi_uc* pixels = stbi_load(file_name.c_str(), &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);

    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }
    renderer::Helper::create2DTextureImage(device_info_, format, tex_width, tex_height, tex_channels, pixels, texture.image, texture.memory);

    stbi_image_free(pixels);

    texture.view = device_->createImageView(
        texture.image,
        renderer::ImageViewType::VIEW_2D,
        format,
        SET_FLAG_BIT(ImageAspect, COLOR_BIT));
}

void RealWorldApplication::createTextureSampler() {
    texture_sampler_ = device_->createSampler(renderer::Filter::LINEAR, renderer::SamplerAddressMode::REPEAT, renderer::SamplerMipmapMode::LINEAR, 16.0f);
}

void RealWorldApplication::cleanupSwapChain() {
    assert(device_);
    depth_buffer_.destroy(device_);

    for (auto framebuffer : swap_chain_info_.framebuffers) {
        device_->destroyFramebuffer(framebuffer);
    }

    device_->freeCommandBuffers(command_pool_, command_buffers_);
    device_->destroyPipeline(gltf_pipeline_);
    device_->destroyPipeline(tile_pipeline_);
    device_->destroyPipeline(skybox_pipeline_);
    device_->destroyPipeline(envmap_pipeline_);
    device_->destroyPipeline(cube_skybox_pipeline_);
    device_->destroyPipeline(lambertian_pipeline_);
    device_->destroyPipeline(ggx_pipeline_);
    device_->destroyPipeline(charlie_pipeline_);
    device_->destroyPipeline(blur_comp_pipeline_);
    device_->destroyPipelineLayout(gltf_pipeline_layout_);
    device_->destroyPipelineLayout(tile_pipeline_layout_);
    device_->destroyPipelineLayout(skybox_pipeline_layout_);
    device_->destroyPipelineLayout(ibl_pipeline_layout_);
    device_->destroyPipelineLayout(cube_skybox_pipeline_layout_);
    device_->destroyPipelineLayout(ibl_comp_pipeline_layout_);
    device_->destroyRenderPass(render_pass_);

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

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    auto vk_instance = RENDER_TYPE_CAST(Instance, instance_);
    auto vk_surface = RENDER_TYPE_CAST(Surface, surface_);

    device_->destroyRenderPass(cubemap_render_pass_);

    gltf_object_->destroy(device_);

    assert(device_);
    device_->destroySampler(texture_sampler_);
    sample_tex_.destroy(device_);
    renderer::Helper::destroy(device_);
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
    device_->destroyDescriptorSetLayout(desc_set_layout_);
    device_->destroyDescriptorSetLayout(global_tex_desc_set_layout_);
    device_->destroyDescriptorSetLayout(material_tex_desc_set_layout_);
    device_->destroyDescriptorSetLayout(skybox_desc_set_layout_);
    device_->destroyDescriptorSetLayout(ibl_desc_set_layout_);
    device_->destroyDescriptorSetLayout(ibl_comp_desc_set_layout_);

    vertex_buffer_.destroy(device_);
    index_buffer_.destroy(device_);

    tile_mesh_->destory();

    for (uint64_t i = 0; i < kMaxFramesInFlight; i++) {
        device_->destroySemaphore(render_finished_semaphores_[i]);
        device_->destroySemaphore(image_available_semaphores_[i]);
        device_->destroyFence(in_flight_fences_[i]);
    }

    device_->destroyCommandPool(command_pool_);

    // todo.
    auto vk_device = RENDER_TYPE_CAST(Device, device_);
    vkDestroyDevice(vk_device->get(), nullptr);

    // todo.
    vkDestroySurfaceKHR(vk_instance->get(), vk_surface->get(), nullptr);

    vk_instance->destroy();
    glfwDestroyWindow(window_);
    glfwTerminate();
}

void RealWorldApplication::loadMtx2Texture(const std::string& input_filename, renderer::TextureInfo& texture) {
    uint64_t buffer_size;
    auto mtx2_data = readFile(input_filename, buffer_size);
    auto src_data = (char*)mtx2_data.data();

    // header block
    Mtx2HeaderBlock* header_block = reinterpret_cast<Mtx2HeaderBlock*>(src_data);
    src_data += sizeof(Mtx2HeaderBlock);

    assert(header_block->format == renderer::Format::R16G16B16A16_SFLOAT);

    // index block
    Mtx2IndexBlock* index_block = reinterpret_cast<Mtx2IndexBlock*>(src_data);
    src_data += sizeof(Mtx2IndexBlock);

    uint32_t width = header_block->pixel_width;
    uint32_t height = header_block->pixel_height;
    // level index block.
    uint32_t num_level_blocks = std::max(1u, header_block->level_count);
    std::vector<work::renderer::BufferImageCopyInfo> copy_regions(num_level_blocks);
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

    renderer::Helper::createCubemapTexture(
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

}//namespace app
}//namespace work
