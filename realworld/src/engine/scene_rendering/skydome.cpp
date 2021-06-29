#include <iostream>
#include <vector>
#include <chrono>

#include "engine/renderer/renderer_helper.h"
#include "engine/engine_helper.h"
#include "shaders/global_definition.glsl.h"
#include "skydome.h"

namespace {
namespace er = engine::renderer;
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

er::ShaderModuleList getCubeSkyboxShaderModules(
    std::shared_ptr<er::Device> device)
{
    uint64_t vert_code_size, frag_code_size;
    er::ShaderModuleList shader_modules(2);
    auto vert_shader_code = engine::helper::readFile("lib/shaders/ibl_vert.spv", vert_code_size);
    auto cube_skybox_shader_code = engine::helper::readFile("lib/shaders/cube_skybox.spv", frag_code_size);

    shader_modules[0] = device->createShaderModule(vert_code_size, vert_shader_code.data());
    shader_modules[1] = device->createShaderModule(frag_code_size, cube_skybox_shader_code.data());

    return shader_modules;
}

er::BufferInfo createVertexBuffer(
    const er::DeviceInfo& device_info) {
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

    er::BufferInfo buffer;
    er::Helper::createBufferWithSrcData(
        device_info,
        SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
        buffer_size,
        vertices.data(),
        buffer.buffer,
        buffer.memory);

    return buffer;
}

er::BufferInfo createIndexBuffer(
    const er::DeviceInfo& device_info) {
    const std::vector<uint16_t> indices = {
        0, 1, 2, 2, 1, 3,
        4, 6, 5, 5, 6, 7,
        0, 4, 1, 1, 4, 5,
        2, 3, 6, 6, 3, 7,
        1, 5, 3, 3, 5, 7,
        0, 2, 4, 4, 2, 6 };

    uint64_t buffer_size =
        sizeof(indices[0]) * indices.size();

    er::BufferInfo buffer;
    er::Helper::createBufferWithSrcData(
        device_info,
        SET_FLAG_BIT(BufferUsage, INDEX_BUFFER_BIT),
        buffer_size,
        indices.data(),
        buffer.buffer,
        buffer.memory);

    return buffer;
}

std::vector<er::TextureDescriptor> addSkyboxTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const er::TextureInfo& envmap_tex) {
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(1);

    // envmap texture.
    er::Helper::addOneTexture(
        descriptor_writes,
        BASE_COLOR_TEX_INDEX,
        texture_sampler,
        envmap_tex.view,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

std::shared_ptr<er::PipelineLayout>
    createSkydomePipelineLayout(
        const std::shared_ptr<er::Device>& device,
        const std::shared_ptr<er::DescriptorSetLayout>& desc_set_layout,
        const std::shared_ptr<er::DescriptorSetLayout>& view_desc_set_layout) {
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::SunSkyParams);

    return device->createPipelineLayout(
        {desc_set_layout , view_desc_set_layout},
        { push_const_range });
}

std::shared_ptr<er::PipelineLayout>
    createCubeSkyboxPipelineLayout(
        const std::shared_ptr<er::Device>& device,
        const std::shared_ptr<er::DescriptorSetLayout> ibl_desc_set_layout)
{
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::SunSkyParams);

    er::DescriptorSetLayoutList desc_set_layouts(1);
    desc_set_layouts[0] = ibl_desc_set_layout;

    return device->createPipelineLayout(desc_set_layouts, { push_const_range });
}

std::shared_ptr<er::Pipeline> createGraphicsPipeline(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::RenderPass>& render_pass,
    const std::shared_ptr<er::PipelineLayout>& pipeline_layout,
    const er::GraphicPipelineInfo& graphic_pipeline_info,
    const glm::uvec2& display_size) {
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

    auto shader_modules = getSkyboxShaderModules(device);
    std::shared_ptr<er::Pipeline> skybox_pipeline =
        device->createPipeline(
        render_pass,
        pipeline_layout,
        SkyBoxVertex::getBindingDescription(),
        SkyBoxVertex::getAttributeDescriptions(),
        input_assembly,
        graphic_pipeline_info,
        shader_modules,
        display_size);

    for (auto& shader_module : shader_modules) {
        device->destroyShaderModule(shader_module);
    }

    return skybox_pipeline;
}

std::shared_ptr<er::Pipeline> createCubeGraphicsPipeline(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::RenderPass>& render_pass,
    const std::shared_ptr<er::PipelineLayout>& pipeline_layout,
    const er::GraphicPipelineInfo& graphic_pipeline_info,
    const uint32_t& cube_size) {
    er::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = er::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;
    auto cube_shader_modules = getCubeSkyboxShaderModules(device);
    return device->createPipeline(
        render_pass,
        pipeline_layout,
        {}, {},
        input_assembly,
        graphic_pipeline_info,
        cube_shader_modules,
        glm::uvec2(cube_size, cube_size));
}

} // namespace

namespace engine {
namespace scene_rendering {

Skydome::Skydome(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
    const std::shared_ptr<renderer::DescriptorSetLayout>& ibl_desc_set_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::GraphicPipelineInfo& cube_graphic_pipeline_info,
    const renderer::TextureInfo& rt_envmap_tex,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const glm::uvec2& display_size,
    const uint32_t& cube_size) {

    const auto& device = device_info.device;

    vertex_buffer_ = createVertexBuffer(device_info);
    index_buffer_ = createIndexBuffer(device_info);

    std::vector<renderer::DescriptorSetLayoutBinding> bindings(1);
    bindings[0] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(BASE_COLOR_TEX_INDEX);

    skybox_desc_set_layout_ =
        device->createDescriptorSetLayout(bindings);

    recreate(
        device,
        descriptor_pool,
        render_pass,
        view_desc_set_layout,
        graphic_pipeline_info,
        rt_envmap_tex,
        texture_sampler,
        display_size);

    cube_skybox_pipeline_layout_ =
        createCubeSkyboxPipelineLayout(
            device,
            ibl_desc_set_layout);

    cube_skybox_pipeline_ = createCubeGraphicsPipeline(
        device,
        cube_render_pass,
        cube_skybox_pipeline_layout_,
        cube_graphic_pipeline_info,
        cube_size);
}

void Skydome::recreate(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::TextureInfo& rt_envmap_tex,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const glm::uvec2& display_size) {

    if (skybox_pipeline_layout_ != nullptr) {
        device->destroyPipelineLayout(skybox_pipeline_layout_);
        skybox_pipeline_layout_ = nullptr;
    }
    
    if (skybox_pipeline_ != nullptr) {
        device->destroyPipeline(skybox_pipeline_);
        skybox_pipeline_ = nullptr;
    }

    skybox_tex_desc_set_ = nullptr;

    // skybox
    skybox_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            skybox_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto skybox_texture_descs = addSkyboxTextures(
        skybox_tex_desc_set_,
        texture_sampler,
        rt_envmap_tex);
    device->updateDescriptorSets(skybox_texture_descs, {});

    assert(view_desc_set_layout);
    skybox_pipeline_layout_ =
        createSkydomePipelineLayout(
            device,
            skybox_desc_set_layout_,
            view_desc_set_layout);

    skybox_pipeline_ = createGraphicsPipeline(
        device,
        render_pass,
        skybox_pipeline_layout_,
        graphic_pipeline_info,
        display_size);
}

// render skybox.
void Skydome::draw(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::DescriptorSet>& frame_desc_set) {
    cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, skybox_pipeline_);
    std::vector<std::shared_ptr<renderer::Buffer>> buffers(1);
    std::vector<uint64_t> offsets(1);
    buffers[0] = vertex_buffer_.buffer;
    offsets[0] = 0;

    cmd_buf->bindVertexBuffers(0, buffers, offsets);
    cmd_buf->bindIndexBuffer(index_buffer_.buffer, 0, renderer::IndexType::UINT16);

    glsl::SunSkyParams sun_sky_params = {};
    sun_sky_params.sun_pos = sun_dir_;

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        skybox_pipeline_layout_,
        &sun_sky_params,
        sizeof(sun_sky_params));

    renderer::DescriptorSetList desc_sets{
        skybox_tex_desc_set_,
        frame_desc_set };
    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        skybox_pipeline_layout_,
        desc_sets);

    cmd_buf->drawIndexed(36);
}

void Skydome::drawCubeSkyBox(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::DescriptorSet>& envmap_tex_desc_set,
    const renderer::TextureInfo& rt_envmap_tex,
    const std::vector<er::ClearValue>& clear_values,
    const uint32_t& cube_size)
{
    // generate envmap from skybox.
    cmd_buf->addImageBarrier(
        rt_envmap_tex.image,
        renderer::Helper::getImageAsSource(),
        renderer::Helper::getImageAsColorAttachment(),
        0, 1, 0, 6);

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::GRAPHICS,
        cube_skybox_pipeline_);

    std::vector<renderer::ClearValue> envmap_clear_values(6, clear_values[0]);
    cmd_buf->beginRenderPass(
        render_pass,
        rt_envmap_tex.framebuffers[0],
        glm::uvec2(cube_size),
        envmap_clear_values);

    glsl::SunSkyParams sun_sky_params = {};
    sun_sky_params.sun_pos = sun_dir_;

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        cube_skybox_pipeline_layout_,
        &sun_sky_params,
        sizeof(sun_sky_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        cube_skybox_pipeline_layout_,
        { envmap_tex_desc_set });

    cmd_buf->draw(3);

    cmd_buf->endRenderPass();

    uint32_t num_mips = static_cast<uint32_t>(std::log2(cube_size) + 1);

    renderer::Helper::generateMipmapLevels(
        cmd_buf,
        rt_envmap_tex.image,
        num_mips,
        cube_size,
        cube_size,
        renderer::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
}

void Skydome::update(float latitude, float longtitude, int d, int th, int tm, int ts) {
    float latitude_r = glm::radians(latitude);
    float decline_angle = glm::radians(
        -23.44f * cos(2.0f * PI / 365.0f * (d + 10.0f)));
    float t = th + tm / 60.0f + ts / 3600.0f;
    float h = -15.0f * (t - 12.0f);
    float delta_h = glm::radians(h - longtitude);

    sun_dir_.x = cos(latitude_r) * sin(decline_angle) - sin(latitude_r) * cos(decline_angle) * cos(delta_h);
    sun_dir_.z = cos(decline_angle) * sin(delta_h);
    sun_dir_.y = sin(latitude_r) * sin(decline_angle) + cos(latitude_r) * cos(decline_angle) * cos(delta_h);
}

void Skydome::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    vertex_buffer_.destroy(device);
    index_buffer_.destroy(device);
    device->destroyDescriptorSetLayout(skybox_desc_set_layout_);
    device->destroyPipelineLayout(skybox_pipeline_layout_);
    device->destroyPipeline(skybox_pipeline_);
    device->destroyPipelineLayout(cube_skybox_pipeline_layout_);
    device->destroyPipeline(cube_skybox_pipeline_);
}

}//namespace scene_rendering
}//namespace engine
