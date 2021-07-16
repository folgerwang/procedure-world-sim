#include <vector>

#include "engine/renderer/renderer_helper.h"
#include "engine/engine_helper.h"
#include "engine/game_object/terrain.h"
#include "shaders/global_definition.glsl.h"
#include "volume_cloud.h"

namespace {
namespace er = engine::renderer;
struct CloudVertex {
    glm::vec3 pos;

    static std::vector<er::VertexInputBindingDescription> getBindingDescription() {
        std::vector<er::VertexInputBindingDescription> binding_description(1);
        binding_description[0].binding = 0;
        binding_description[0].stride = sizeof(CloudVertex);
        binding_description[0].input_rate = er::VertexInputRate::VERTEX;
        return binding_description;
    }

    static std::vector<er::VertexInputAttributeDescription> getAttributeDescriptions() {
        std::vector<er::VertexInputAttributeDescription> attribute_descriptions(1);
        attribute_descriptions[0].binding = 0;
        attribute_descriptions[0].location = 0;
        attribute_descriptions[0].format = er::Format::R32G32B32_SFLOAT;
        attribute_descriptions[0].offset = offsetof(CloudVertex, pos);
        return attribute_descriptions;
    }
};

er::ShaderModuleList getCloudShaderModules(
    std::shared_ptr<er::Device> device)
{
    uint64_t vert_code_size, frag_code_size;
    er::ShaderModuleList shader_modules(2);
    auto vert_shader_code = engine::helper::readFile("lib/shaders/cloud_vert.spv", vert_code_size);
    auto frag_shader_code = engine::helper::readFile("lib/shaders/cloud_frag.spv", frag_code_size);

    shader_modules[0] = device->createShaderModule(vert_code_size, vert_shader_code.data());
    shader_modules[1] = device->createShaderModule(frag_code_size, frag_shader_code.data());

    return shader_modules;
}

er::ShaderModuleList getDrawVolumeMoistureShaderModules(
    std::shared_ptr<er::Device> device)
{
    uint64_t vert_code_size, frag_code_size;
    er::ShaderModuleList shader_modules(2);
    auto vert_shader_code = engine::helper::readFile("lib/shaders/full_screen_vert.spv", vert_code_size);
    auto frag_shader_code = engine::helper::readFile("lib/shaders/draw_volume_moist_frag.spv", frag_code_size);

    shader_modules[0] = device->createShaderModule(vert_code_size, vert_shader_code.data());
    shader_modules[1] = device->createShaderModule(frag_code_size, frag_shader_code.data());

    return shader_modules;
}

er::BufferInfo createVertexBuffer(
    const er::DeviceInfo& device_info) {
    const std::vector<CloudVertex> vertices = {
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

std::vector<er::TextureDescriptor> addCloudTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::shared_ptr<er::ImageView>& volume_moist_tex) {
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(1);

    // envmap texture.
    er::Helper::addOneTexture(
        descriptor_writes,
        BASE_COLOR_TEX_INDEX,
        texture_sampler,
        volume_moist_tex,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

std::vector<er::TextureDescriptor> addVolumeMoistureTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::shared_ptr<er::ImageView>& src_depth,
    const std::shared_ptr<er::ImageView>& volume_moist_tex,
    const std::shared_ptr<er::ImageView>& cloud_lighting_tex) {
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(3);

    // envmap texture.
    er::Helper::addOneTexture(
        descriptor_writes,
        SRC_TEMP_MOISTURE_INDEX,
        texture_sampler,
        volume_moist_tex,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        SRC_DEPTH_TEX_INDEX,
        texture_sampler,
        src_depth,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        SRC_CLOUD_LIGHTING_TEX_INDEX,
        texture_sampler,
        cloud_lighting_tex,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

std::shared_ptr<er::PipelineLayout>
    createCloudPipelineLayout(
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

    auto shader_modules = getCloudShaderModules(device);
    std::shared_ptr<er::Pipeline> cloud_pipeline =
        device->createPipeline(
        render_pass,
        pipeline_layout,
        CloudVertex::getBindingDescription(),
        CloudVertex::getAttributeDescriptions(),
        input_assembly,
        graphic_pipeline_info,
        shader_modules,
        display_size);

    for (auto& shader_module : shader_modules) {
        device->destroyShaderModule(shader_module);
    }

    return cloud_pipeline;
}

std::shared_ptr<er::PipelineLayout> createDrawVolumeMoisturePipelineLayout(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorSetLayout>& desc_set_layout,
    const std::shared_ptr<er::DescriptorSetLayout>& view_desc_set_layout)
{
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::VolumeMoistrueParams);

    return device->createPipelineLayout(
        { desc_set_layout , view_desc_set_layout },
        { push_const_range });
}

std::shared_ptr<er::Pipeline> createDrawVolumeMoisturePipeline(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::RenderPass>& render_pass,
    const std::shared_ptr<er::PipelineLayout>& pipeline_layout,
    const er::GraphicPipelineInfo& graphic_pipeline_info,
    const glm::uvec2& display_size) {
    er::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = er::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;
    auto draw_volume_moist_shader_modules =
        getDrawVolumeMoistureShaderModules(device);
    return device->createPipeline(
        render_pass,
        pipeline_layout,
        {}, {},
        input_assembly,
        graphic_pipeline_info,
        draw_volume_moist_shader_modules,
        display_size);
}

} // namespace

namespace engine {
namespace scene_rendering {

VolumeCloud::VolumeCloud(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
    const std::shared_ptr<renderer::DescriptorSetLayout>& ibl_desc_set_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& src_depth,
    const std::vector<std::shared_ptr<renderer::ImageView>>& temp_moisture_texes,
    const std::shared_ptr<renderer::ImageView>& cloud_lighting_tex,
    const glm::uvec2& display_size) {

    const auto& device = device_info.device;

    vertex_buffer_ = createVertexBuffer(device_info);
    index_buffer_ = createIndexBuffer(device_info);

    cloud_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(BASE_COLOR_TEX_INDEX) });

    draw_volume_moist_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(SRC_TEMP_MOISTURE_INDEX),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(SRC_DEPTH_TEX_INDEX),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(SRC_CLOUD_LIGHTING_TEX_INDEX)});

    recreate(
        device,
        descriptor_pool,
        render_pass,
        view_desc_set_layout,
        graphic_pipeline_info,
        texture_sampler,
        src_depth,
        temp_moisture_texes,
        cloud_lighting_tex,
        display_size);
}

void VolumeCloud::recreate(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& src_depth,
    const std::vector<std::shared_ptr<renderer::ImageView>>& temp_moisture_texes,
    const std::shared_ptr<renderer::ImageView>& cloud_lighting_tex,
    const glm::uvec2& display_size) {

    if (cloud_pipeline_layout_ != nullptr) {
        device->destroyPipelineLayout(cloud_pipeline_layout_);
        cloud_pipeline_layout_ = nullptr;
    }
    
    if (cloud_pipeline_ != nullptr) {
        device->destroyPipeline(cloud_pipeline_);
        cloud_pipeline_ = nullptr;
    }

    if (draw_volume_moist_pipeline_layout_ != nullptr) {
        device->destroyPipelineLayout(draw_volume_moist_pipeline_layout_);
        draw_volume_moist_pipeline_layout_ = nullptr;
    }

    if (draw_volume_moist_pipeline_ != nullptr) {
        device->destroyPipeline(draw_volume_moist_pipeline_);
        draw_volume_moist_pipeline_ = nullptr;
    }
    
#if 0
    cloud_tex_desc_set_ = nullptr;

    // skybox
    cloud_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            cloud_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto cloud_texture_descs = addCloudTextures(
        cloud_tex_desc_set_,
        texture_sampler,
        temp_moisture_texes[0]);
    device->updateDescriptorSets(cloud_texture_descs, {});

    assert(view_desc_set_layout);
    cloud_pipeline_layout_ =
        createCloudPipelineLayout(
            device,
            cloud_desc_set_layout_,
            view_desc_set_layout);

    cloud_pipeline_ = createGraphicsPipeline(
        device,
        render_pass,
        cloud_pipeline_layout_,
        graphic_pipeline_info,
        display_size);
#endif

    for (auto dbuf_idx = 0; dbuf_idx < 2; dbuf_idx++) {
        draw_volume_moist_desc_set_[dbuf_idx] = nullptr;

        // skybox
        draw_volume_moist_desc_set_[dbuf_idx] =
            device->createDescriptorSets(
                descriptor_pool,
                draw_volume_moist_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto draw_volume_moist_texture_descs = addVolumeMoistureTextures(
            draw_volume_moist_desc_set_[dbuf_idx],
            texture_sampler,
            src_depth,
            temp_moisture_texes[dbuf_idx],
            cloud_lighting_tex);
        device->updateDescriptorSets(draw_volume_moist_texture_descs, {});
    }

    draw_volume_moist_pipeline_layout_ =
        createDrawVolumeMoisturePipelineLayout(
            device,
            draw_volume_moist_desc_set_layout_,
            view_desc_set_layout);

    draw_volume_moist_pipeline_ =
        createDrawVolumeMoisturePipeline(
            device,
            render_pass,
            draw_volume_moist_pipeline_layout_,
            graphic_pipeline_info,
            display_size);
}

// render skybox.
void VolumeCloud::draw(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::DescriptorSet>& frame_desc_set) {
    cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, cloud_pipeline_);
    std::vector<std::shared_ptr<renderer::Buffer>> buffers(1);
    std::vector<uint64_t> offsets(1);
    buffers[0] = vertex_buffer_.buffer;
    offsets[0] = 0;

    cmd_buf->bindVertexBuffers(0, buffers, offsets);
    cmd_buf->bindIndexBuffer(index_buffer_.buffer, 0, renderer::IndexType::UINT16);

    glsl::CloudParams cloud_params = {};

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        cloud_pipeline_layout_,
        &cloud_params,
        sizeof(cloud_params));

    renderer::DescriptorSetList desc_sets{
        cloud_tex_desc_set_,
        frame_desc_set };
    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        cloud_pipeline_layout_,
        desc_sets);

    cmd_buf->drawIndexed(36);
}

void VolumeCloud::drawVolumeMoisture(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::DescriptorSet>& frame_desc_set,
    const glm::uvec2& display_size,
    int dbuf_idx,
    float current_time) {
    // render moisture volume.

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::GRAPHICS,
        draw_volume_moist_pipeline_);

    glsl::VolumeMoistrueParams params = {};
    params.world_min = glm::vec2(-kWorldMapSize / 2.0f);
    params.inv_world_range = 1.0f / (glm::vec2(kWorldMapSize / 2.0f) - params.world_min);
    params.inv_screen_size = 1.0f / glm::vec2(display_size);
    params.time = current_time;

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        draw_volume_moist_pipeline_layout_,
        &params,
        sizeof(params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        draw_volume_moist_pipeline_layout_,
        { draw_volume_moist_desc_set_[dbuf_idx], frame_desc_set });

    cmd_buf->draw(3);
}

void VolumeCloud::update() {
}

void VolumeCloud::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    vertex_buffer_.destroy(device);
    index_buffer_.destroy(device);
    device->destroyDescriptorSetLayout(cloud_desc_set_layout_);
    device->destroyPipelineLayout(cloud_pipeline_layout_);
    device->destroyPipeline(cloud_pipeline_);
    device->destroyDescriptorSetLayout(draw_volume_moist_desc_set_layout_);
    device->destroyPipelineLayout(draw_volume_moist_pipeline_layout_);
    device->destroyPipeline(draw_volume_moist_pipeline_);
}

}//namespace scene_rendering
}//namespace engine
