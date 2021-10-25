#include <vector>

#include "engine/renderer/renderer_helper.h"
#include "engine/engine_helper.h"
#include "shaders/global_definition.glsl.h"
#include "volume_noise.h"

namespace {
namespace er = engine::renderer;
namespace erh = er::helper;

#if 0
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
#endif

std::vector<er::TextureDescriptor> addPerlinNoiseInitTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const er::TextureInfo& perlin_noise_tex) {
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(1);

    er::Helper::addOneTexture(
        descriptor_writes,
        DST_PERLIN_NOISE_TEX_INDEX,
        nullptr,
        perlin_noise_tex.view,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

#if 0
std::vector<er::TextureDescriptor> addCloudTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler) {
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(1);

    // envmap texture.
    er::Helper::addOneTexture(
        descriptor_writes,
        BASE_COLOR_TEX_INDEX,
        texture_sampler,
        nullptr,
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
#endif

} // namespace

namespace engine {
namespace scene_rendering {

VolumeNoise::VolumeNoise(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
    const std::shared_ptr<renderer::DescriptorSetLayout>& ibl_desc_set_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const glm::uvec2& display_size) {

    const auto& device = device_info.device;
#if 0
    vertex_buffer_ = createVertexBuffer(device_info);
    index_buffer_ = createIndexBuffer(device_info);
#endif
    renderer::Helper::create3DTextureImage(
        device_info,
        renderer::Format::R8_UNORM,
        glm::uvec3(
            kNoiseTextureSize,
            kNoiseTextureSize,
            kNoiseTextureSize),
        perlin_noise_tex_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::create3DTextureImage(
        device_info,
        renderer::Format::R8_UNORM,
        glm::uvec3(
            kNoiseTextureSize,
            kNoiseTextureSize,
            kNoiseTextureSize),
        worley_noise_tex_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    perlin_noise_init_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_PERLIN_NOISE_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE) });

#if 0
    {
        std::vector<renderer::DescriptorSetLayoutBinding> bindings(1);
        bindings[0] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(BASE_COLOR_TEX_INDEX);

        cloud_desc_set_layout_ =
            device->createDescriptorSetLayout(bindings);
    }
#endif

    recreate(
        device,
        descriptor_pool,
        render_pass,
        view_desc_set_layout,
        graphic_pipeline_info,
        texture_sampler,
        display_size);
}

void VolumeNoise::recreate(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const glm::uvec2& display_size) {

    erh::releasePipelineLayout(device, perlin_noise_init_pipeline_layout_);
    erh::releasePipeline(device, perlin_noise_init_pipeline_);

    // perlin noise texture init.
    perlin_noise_init_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            perlin_noise_init_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto perlin_noise_init_texture_descs =
        addPerlinNoiseInitTextures(
            perlin_noise_init_tex_desc_set_,
            perlin_noise_tex_);
    device->updateDescriptorSets(perlin_noise_init_texture_descs, {});

    perlin_noise_init_pipeline_layout_ =
        erh::createComputePipelineLayout(
            device,
            { perlin_noise_init_desc_set_layout_ },
            sizeof(glsl::PerlinNoiseInitParams));

    perlin_noise_init_pipeline_ =
        erh::createComputePipeline(
            device,
            perlin_noise_init_pipeline_layout_,
            "perlin_noise_init");

#if 0
    // create a global ibl texture descriptor set.
    auto cloud_texture_descs = addCloudTextures(
        cloud_tex_desc_set_,
        texture_sampler);
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
}

void VolumeNoise::initPerlinNoiseTexture(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf) {
    erh::transitMapTextureToStoreImage(
        cmd_buf,
        { perlin_noise_tex_.image });

    auto w = static_cast<uint32_t>(kNoiseTextureSize);
    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, perlin_noise_init_pipeline_);
    glsl::PerlinNoiseInitParams perlin_noise_init_params = {};

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        perlin_noise_init_pipeline_layout_,
        &perlin_noise_init_params,
        sizeof(perlin_noise_init_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        perlin_noise_init_pipeline_layout_,
        { perlin_noise_init_tex_desc_set_ });

    cmd_buf->dispatch(
        (w + 3) / 4,
        (w + 3) / 4,
        (w + 3) / 4);

    erh::transitMapTextureFromStoreImage(
        cmd_buf,
        { perlin_noise_tex_.image });
}

// render skybox.
void VolumeNoise::draw(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::DescriptorSet>& frame_desc_set) {
#if 0
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
#endif
}

void VolumeNoise::update() {
}

void VolumeNoise::destroy(
    const std::shared_ptr<renderer::Device>& device) {
#if 0
    vertex_buffer_.destroy(device);
    index_buffer_.destroy(device);
    device->destroyDescriptorSetLayout(cloud_desc_set_layout_);
    device->destroyPipelineLayout(cloud_pipeline_layout_);
    device->destroyPipeline(cloud_pipeline_);
#endif
    device->destroyDescriptorSetLayout(perlin_noise_init_desc_set_layout_);
    device->destroyPipelineLayout(perlin_noise_init_pipeline_layout_);
    device->destroyPipeline(perlin_noise_init_pipeline_);
}

}//namespace scene_rendering
}//namespace engine
