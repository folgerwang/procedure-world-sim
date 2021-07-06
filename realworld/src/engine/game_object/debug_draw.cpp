// Mountains. By David Hoskins - 2013
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

// https://www.shadertoy.com/view/4slGD4
// A ray-marched version of my terrain renderer which uses
// streaming texture normals for speed:-
// http://www.youtube.com/watch?v=qzkBnCBpQAM

// It uses binary subdivision to accurately find the height map.
// Lots of thanks to Inigo and his noise functions!

// Video of my OpenGL version that 
// http://www.youtube.com/watch?v=qzkBnCBpQAM


#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "engine/renderer/renderer.h"
#include "engine/renderer/renderer_helper.h"
#include "engine/engine_helper.h"
#include "debug_draw.h"
#include "shaders/global_definition.glsl.h"

//#define STATIC_CAMERA
#define LOWQUALITY

namespace engine {
namespace game_object {
namespace {

std::vector<renderer::TextureDescriptor> addTileCreatorBuffers(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer,
    const renderer::TextureInfo& grass_snow_layer) {
    std::vector<renderer::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(3);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        ROCK_LAYER_BUFFER_INDEX,
        nullptr,
        rock_layer.view,
        description_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        renderer::ImageLayout::GENERAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        SOIL_WATER_LAYER_BUFFER_INDEX,
        nullptr,
        soil_water_layer.view,
        description_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        renderer::ImageLayout::GENERAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        ORTHER_INFO_LAYER_BUFFER_INDEX,
        nullptr,
        grass_snow_layer.view,
        description_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        renderer::ImageLayout::GENERAL);

    return descriptor_writes;
}

std::vector<renderer::TextureDescriptor> addTileUpdateBuffers(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer,
    const renderer::TextureInfo& dst_water_normal) {
    std::vector<renderer::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(3);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        ROCK_LAYER_BUFFER_INDEX,
        texture_sampler,
        rock_layer.view,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        SOIL_WATER_LAYER_BUFFER_INDEX,
        nullptr,
        soil_water_layer.view,
        description_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        renderer::ImageLayout::GENERAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        DST_WATER_NORMAL_BUFFER_INDEX,
        nullptr,
        dst_water_normal.view,
        description_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        renderer::ImageLayout::GENERAL);

    return descriptor_writes;
}

std::vector<renderer::TextureDescriptor> addTileFlowUpdateBuffers(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer,
    const renderer::TextureInfo& dst_soil_water_layer,
    const renderer::TextureInfo& dst_water_flow) {
    std::vector<renderer::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(4);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        ROCK_LAYER_BUFFER_INDEX,
        texture_sampler,
        rock_layer.view,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        SOIL_WATER_LAYER_BUFFER_INDEX,
        nullptr,
        soil_water_layer.view,
        description_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        renderer::ImageLayout::GENERAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        DST_SOIL_WATER_LAYER_BUFFER_INDEX,
        nullptr,
        dst_soil_water_layer.view,
        description_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        renderer::ImageLayout::GENERAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        DST_WATER_FLOW_BUFFER_INDEX,
        nullptr,
        dst_water_flow.view,
        description_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        renderer::ImageLayout::GENERAL);

    return descriptor_writes;
}

std::vector<renderer::TextureDescriptor> addTileResourceTextures(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& src_texture,
    const std::shared_ptr<renderer::ImageView>& src_depth,
    const std::shared_ptr<renderer::ImageView>& rock_layer,
    const std::shared_ptr<renderer::ImageView>& soil_water_layer,
    const std::shared_ptr<renderer::ImageView>& grass_snow_layer,
    const std::shared_ptr<renderer::ImageView>& water_normal,
    const std::shared_ptr<renderer::ImageView>& water_flow,
    const std::shared_ptr<renderer::ImageView>& airflow_tex) {
    std::vector<renderer::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(8);

    // src color.
    renderer::Helper::addOneTexture(
        descriptor_writes,
        SRC_COLOR_TEX_INDEX,
        texture_sampler,
        src_texture,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // src depth.
    renderer::Helper::addOneTexture(
        descriptor_writes,
        SRC_DEPTH_TEX_INDEX,
        texture_sampler,
        src_depth,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        ROCK_LAYER_BUFFER_INDEX,
        texture_sampler,
        rock_layer,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        SOIL_WATER_LAYER_BUFFER_INDEX,
        texture_sampler,
        soil_water_layer,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        ORTHER_INFO_LAYER_BUFFER_INDEX,
        texture_sampler,
        grass_snow_layer,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        WATER_NORMAL_BUFFER_INDEX,
        texture_sampler,
        water_normal,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        WATER_FLOW_BUFFER_INDEX,
        texture_sampler,
        water_flow,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        SRC_VOLUME_TEST_INDEX,
        texture_sampler,
        airflow_tex,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

static renderer::ShaderModuleList getTileCreatorCsModules(
    const std::shared_ptr<renderer::Device>& device) {
    uint64_t compute_code_size;
    renderer::ShaderModuleList shader_modules;
    shader_modules.reserve(1);
    auto compute_shader_code = engine::helper::readFile("lib/shaders/tile_creator_comp.spv", compute_code_size);
    shader_modules.push_back(device->createShaderModule(compute_code_size, compute_shader_code.data()));

    return shader_modules;
}

static renderer::ShaderModuleList getTileUpdateCsModules(
    const std::shared_ptr<renderer::Device>& device) {
    uint64_t compute_code_size;
    renderer::ShaderModuleList shader_modules;
    shader_modules.reserve(1);
    auto compute_shader_code = engine::helper::readFile("lib/shaders/tile_update_comp.spv", compute_code_size);
    shader_modules.push_back(device->createShaderModule(compute_code_size, compute_shader_code.data()));

    return shader_modules;
}

static renderer::ShaderModuleList getTileFlowUpdateCsModules(
    const std::shared_ptr<renderer::Device>& device) {
    uint64_t compute_code_size;
    renderer::ShaderModuleList shader_modules;
    shader_modules.reserve(1);
    auto compute_shader_code = engine::helper::readFile("lib/shaders/tile_flow_update_comp.spv", compute_code_size);
    shader_modules.push_back(device->createShaderModule(compute_code_size, compute_shader_code.data()));

    return shader_modules;
}

static renderer::ShaderModuleList getTileShaderModules(
    std::shared_ptr<renderer::Device> device) {
    uint64_t vert_code_size, frag_code_size;
    renderer::ShaderModuleList shader_modules(2);
    auto vert_shader_code = engine::helper::readFile("lib/shaders/tile_soil_vert.spv", vert_code_size);
    auto frag_shader_code = engine::helper::readFile("lib/shaders/tile_frag.spv", frag_code_size);

    shader_modules[0] = device->createShaderModule(vert_code_size, vert_shader_code.data());
    shader_modules[1] = device->createShaderModule(frag_code_size, frag_shader_code.data());

    return shader_modules;
}

static renderer::ShaderModuleList getTileWaterShaderModules(
    std::shared_ptr<renderer::Device> device) {
    uint64_t vert_code_size, frag_code_size;
    renderer::ShaderModuleList shader_modules(2);
    auto vert_shader_code = engine::helper::readFile("lib/shaders/tile_water_vert.spv", vert_code_size);
    auto frag_shader_code = engine::helper::readFile("lib/shaders/tile_water_frag.spv", frag_code_size);

    shader_modules[0] = device->createShaderModule(vert_code_size, vert_shader_code.data());
    shader_modules[1] = device->createShaderModule(frag_code_size, frag_shader_code.data());

    return shader_modules;
}

static std::shared_ptr<renderer::DescriptorSetLayout> createTileCreateDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings(3);
    bindings[0] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        ROCK_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_IMAGE);
    bindings[1] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        SOIL_WATER_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_IMAGE);
    bindings[2] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        ORTHER_INFO_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_IMAGE);

    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<renderer::DescriptorSetLayout> createTileUpdateDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings(3);
    bindings[0] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        ROCK_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER);
    bindings[1] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        SOIL_WATER_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_IMAGE);
    bindings[2] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        DST_WATER_NORMAL_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_IMAGE);
    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<renderer::DescriptorSetLayout> createTileFlowUpdateDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings(4);
    bindings[0] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        ROCK_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER);
    bindings[1] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        SOIL_WATER_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_IMAGE);
    bindings[2] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        DST_SOIL_WATER_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_IMAGE);
    bindings[3] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        DST_WATER_FLOW_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_IMAGE);  
    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<renderer::DescriptorSetLayout> CreateTileResourceDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings(8);
    bindings[0] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        SRC_COLOR_TEX_INDEX);
    bindings[1] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        SRC_DEPTH_TEX_INDEX);
    bindings[2] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        ROCK_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[3] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        SOIL_WATER_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[4] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        ORTHER_INFO_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[5] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        WATER_NORMAL_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[6] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        WATER_FLOW_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[7] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        SRC_VOLUME_TEST_INDEX,
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<renderer::PipelineLayout> createTileCreatorPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& desc_set_layouts) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::TileCreateParams);

    return device->createPipelineLayout(desc_set_layouts, { push_const_range });
}

static std::shared_ptr<renderer::PipelineLayout> createTileUpdatePipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& desc_set_layouts) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::TileUpdateParams);

    return device->createPipelineLayout(desc_set_layouts, { push_const_range });
}

static std::shared_ptr<renderer::PipelineLayout> createTileFlowUpdatePipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& desc_set_layouts) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::TileUpdateParams);

    return device->createPipelineLayout(desc_set_layouts, { push_const_range });
}

static std::shared_ptr<renderer::Pipeline> createTileCreatorPipeline(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout) {
    auto tile_creator_compute_shader_modules = getTileCreatorCsModules(device);
    assert(tile_creator_compute_shader_modules.size() == 1);

    auto pipeline = device->createPipeline(
        pipeline_layout,
        tile_creator_compute_shader_modules[0]);

    for (auto& shader_module : tile_creator_compute_shader_modules) {
        device->destroyShaderModule(shader_module);
    }

    return pipeline;
}

static std::shared_ptr<renderer::Pipeline> createTileUpdatePipeline(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout) {
    auto tile_update_compute_shader_modules = getTileUpdateCsModules(device);
    assert(tile_update_compute_shader_modules.size() == 1);

    auto pipeline = device->createPipeline(
        pipeline_layout,
        tile_update_compute_shader_modules[0]);

    for (auto& shader_module : tile_update_compute_shader_modules) {
        device->destroyShaderModule(shader_module);
    }

    return pipeline;
}

static std::shared_ptr<renderer::Pipeline> createTileFlowUpdatePipeline(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout) {
    auto tile_flow_update_compute_shader_modules = getTileFlowUpdateCsModules(device);
    assert(tile_flow_update_compute_shader_modules.size() == 1);

    auto pipeline = device->createPipeline(
        pipeline_layout,
        tile_flow_update_compute_shader_modules[0]);

    for (auto& shader_module : tile_flow_update_compute_shader_modules) {
        device->destroyShaderModule(shader_module);
    }

    return pipeline;
}

static std::shared_ptr<renderer::PipelineLayout> createTilePipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& desc_set_layouts) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags =
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) |
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::TileParams);

    return device->createPipelineLayout(desc_set_layouts, { push_const_range });
}

static std::shared_ptr<renderer::Pipeline> createTilePipeline(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const glm::uvec2& display_size) {
    renderer::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = renderer::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;

    auto shader_modules = getTileShaderModules(device);
    auto pipeline = device->createPipeline(
        render_pass,
        pipeline_layout,
        {},
        {},
        input_assembly,
        graphic_pipeline_info,
        shader_modules,
        display_size);

    for (auto& shader_module : shader_modules) {
        device->destroyShaderModule(shader_module);
    }

    return pipeline;
}

static std::shared_ptr<renderer::Pipeline> createTileWaterPipeline(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const glm::uvec2& display_size) {
    renderer::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = renderer::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;

    renderer::GraphicPipelineInfo new_graphic_pipeline_info = graphic_pipeline_info;

    auto shader_modules = getTileWaterShaderModules(device);
    auto pipeline = device->createPipeline(
        render_pass,
        pipeline_layout,
        {},
        {},
        input_assembly,
        new_graphic_pipeline_info,
        shader_modules,
        display_size);

    for (auto& shader_module : shader_modules) {
        device->destroyShaderModule(shader_module);
    }

    return pipeline;
}

size_t generateHash(
    const glm::vec2& min,
    const glm::vec2& max,
    const uint32_t& segment_count) {
    size_t hash;
    hash = std::hash<float>{}(min.x);
    hash_combine(hash, min.y);
    hash_combine(hash, max.x);
    hash_combine(hash, max.y);
    hash_combine(hash, segment_count);
    return hash;
}

} // namespace

// static member definition.
std::shared_ptr<renderer::DescriptorSet> DebugDrawObject::creator_buffer_desc_set_;
std::shared_ptr<renderer::DescriptorSet> DebugDrawObject::tile_update_buffer_desc_set_[2];
std::shared_ptr<renderer::DescriptorSet> DebugDrawObject::tile_flow_update_buffer_desc_set_[2];
std::shared_ptr<renderer::DescriptorSetLayout> DebugDrawObject::tile_creator_desc_set_layout_;
std::shared_ptr<renderer::PipelineLayout> DebugDrawObject::tile_creator_pipeline_layout_;
std::shared_ptr<renderer::Pipeline> DebugDrawObject::tile_creator_pipeline_;
std::shared_ptr<renderer::DescriptorSetLayout> DebugDrawObject::tile_update_desc_set_layout_;
std::shared_ptr<renderer::PipelineLayout> DebugDrawObject::tile_update_pipeline_layout_;
std::shared_ptr<renderer::Pipeline> DebugDrawObject::tile_update_pipeline_;
std::shared_ptr<renderer::DescriptorSetLayout> DebugDrawObject::tile_flow_update_desc_set_layout_;
std::shared_ptr<renderer::PipelineLayout> DebugDrawObject::tile_flow_update_pipeline_layout_;
std::shared_ptr<renderer::Pipeline> DebugDrawObject::tile_flow_update_pipeline_;
std::shared_ptr<renderer::PipelineLayout> DebugDrawObject::tile_pipeline_layout_;
std::shared_ptr<renderer::Pipeline> DebugDrawObject::tile_pipeline_;
std::shared_ptr<renderer::DescriptorSetLayout> DebugDrawObject::tile_res_desc_set_layout_;
std::shared_ptr<renderer::DescriptorSet> DebugDrawObject::tile_res_desc_set_[2];
std::shared_ptr<renderer::Pipeline> DebugDrawObject::tile_water_pipeline_;

DebugDrawObject::DebugDrawObject(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::DescriptorPool> descriptor_pool,
    const glm::vec2& min,
    const glm::vec2& max,
    const size_t& hash_value,
    const uint32_t& block_idx) :
    device_info_(device_info),
    min_(min),
    max_(max){
    createMeshBuffers();
    assert(tile_creator_desc_set_layout_);
    assert(tile_update_desc_set_layout_);
    assert(tile_flow_update_desc_set_layout_);
    assert(tile_res_desc_set_layout_);
}

void DebugDrawObject::destory() {
}

void DebugDrawObject::createStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::RenderPass>& water_render_pass,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const glm::uvec2& display_size) {
    if (tile_creator_pipeline_layout_ == nullptr) {
        assert(tile_creator_desc_set_layout_);
        tile_creator_pipeline_layout_ =
            createTileCreatorPipelineLayout(
                device,
                { tile_creator_desc_set_layout_ });
    }

    if (tile_creator_pipeline_ == nullptr) {
        assert(tile_creator_pipeline_layout_);
        tile_creator_pipeline_ =
            createTileCreatorPipeline(
                device,
                tile_creator_pipeline_layout_);
    }

    if (tile_update_pipeline_layout_ == nullptr) {
        assert(tile_update_desc_set_layout_);
        tile_update_pipeline_layout_ =
            createTileUpdatePipelineLayout(
                device,
                { tile_update_desc_set_layout_ });
    }

    if (tile_update_pipeline_ == nullptr) {
        assert(tile_update_pipeline_layout_);
        tile_update_pipeline_ =
            createTileUpdatePipeline(
                device,
                tile_update_pipeline_layout_);
    }

    if (tile_flow_update_pipeline_layout_ == nullptr) {
        assert(tile_flow_update_desc_set_layout_);
        tile_flow_update_pipeline_layout_ =
            createTileFlowUpdatePipelineLayout(
                device,
                { tile_flow_update_desc_set_layout_ });
    }

    if (tile_flow_update_pipeline_ == nullptr) {
        assert(tile_flow_update_pipeline_layout_);
        tile_flow_update_pipeline_ =
            createTileFlowUpdatePipeline(
                device,
                tile_flow_update_pipeline_layout_);
    }

    auto desc_set_layouts = global_desc_set_layouts;
    desc_set_layouts.push_back(tile_res_desc_set_layout_);

    if (tile_pipeline_layout_ == nullptr) {
        tile_pipeline_layout_ =
            createTilePipelineLayout(
                device,
                desc_set_layouts);
    }

    if (tile_pipeline_ == nullptr) {
        assert(tile_pipeline_layout_);
        tile_pipeline_ =
            createTilePipeline(
                device,
                render_pass,
                tile_pipeline_layout_,
                graphic_pipeline_info,
                display_size);
    }

    if (tile_water_pipeline_ == nullptr) {
        assert(tile_pipeline_layout_);
        tile_water_pipeline_ =
            createTileWaterPipeline(
                device,
                water_render_pass,
                tile_pipeline_layout_,
                graphic_pipeline_info,
                display_size);
    }
}

void DebugDrawObject::initStaticMembers(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::RenderPass>& water_render_pass,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const glm::uvec2& display_size) {
    auto& device = device_info.device;

     if (tile_creator_desc_set_layout_ == nullptr) {
        tile_creator_desc_set_layout_ =
            createTileCreateDescriptorSetLayout(device);
    }

    if (tile_update_desc_set_layout_ == nullptr) {
        tile_update_desc_set_layout_ =
            createTileUpdateDescriptorSetLayout(device);
    }

    if (tile_flow_update_desc_set_layout_ == nullptr) {
        tile_flow_update_desc_set_layout_ =
            createTileFlowUpdateDescriptorSetLayout(device);
    }

    if (tile_res_desc_set_layout_ == nullptr) {
        tile_res_desc_set_layout_ =
            CreateTileResourceDescriptorSetLayout(device);
    }

    createStaticMembers(
        device,
        render_pass,
        water_render_pass,
        graphic_pipeline_info,
        global_desc_set_layouts,
        display_size);
}

void DebugDrawObject::recreateStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::RenderPass>& wateer_render_pass,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const glm::uvec2& display_size) {
    if (tile_creator_pipeline_layout_) {
        device->destroyPipelineLayout(tile_creator_pipeline_layout_);
        tile_creator_pipeline_layout_ = nullptr;
    }

    if (tile_creator_pipeline_) {
        device->destroyPipeline(tile_creator_pipeline_);
        tile_creator_pipeline_ = nullptr;
    }

    if (tile_update_pipeline_layout_) {
        device->destroyPipelineLayout(tile_update_pipeline_layout_);
        tile_update_pipeline_layout_ = nullptr;
    }

    if (tile_update_pipeline_) {
        device->destroyPipeline(tile_update_pipeline_);
        tile_update_pipeline_ = nullptr;
    }

    if (tile_flow_update_pipeline_layout_) {
        device->destroyPipelineLayout(tile_flow_update_pipeline_layout_);
        tile_flow_update_pipeline_layout_ = nullptr;
    }

    if (tile_flow_update_pipeline_) {
        device->destroyPipeline(tile_flow_update_pipeline_);
        tile_flow_update_pipeline_ = nullptr;
    }

    if (tile_pipeline_layout_) {
        device->destroyPipelineLayout(tile_pipeline_layout_);
        tile_pipeline_layout_ = nullptr;
    }

    if (tile_pipeline_) {
        device->destroyPipeline(tile_pipeline_);
        tile_pipeline_ = nullptr;
    }

    if (tile_water_pipeline_) {
        device->destroyPipeline(tile_water_pipeline_);
        tile_water_pipeline_ = nullptr;
    }

    createStaticMembers(
        device,
        render_pass,
        wateer_render_pass,
        graphic_pipeline_info,
        global_desc_set_layouts,
        display_size);
}

void DebugDrawObject::destoryStaticMembers(
    const std::shared_ptr<renderer::Device>& device) {
    device->destroyDescriptorSetLayout(tile_creator_desc_set_layout_);
    device->destroyDescriptorSetLayout(tile_update_desc_set_layout_);
    device->destroyDescriptorSetLayout(tile_flow_update_desc_set_layout_);
    device->destroyDescriptorSetLayout(tile_res_desc_set_layout_);
    device->destroyPipelineLayout(tile_creator_pipeline_layout_);
    device->destroyPipelineLayout(tile_update_pipeline_layout_);
    device->destroyPipelineLayout(tile_flow_update_pipeline_layout_);
    device->destroyPipeline(tile_creator_pipeline_);
    device->destroyPipeline(tile_update_pipeline_);
    device->destroyPipeline(tile_flow_update_pipeline_);
    device->destroyPipelineLayout(tile_pipeline_layout_);
    device->destroyPipeline(tile_pipeline_);
    device->destroyPipeline(tile_water_pipeline_);
}

void DebugDrawObject::generateStaticDescriptorSet(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler) {
    // tile creator buffer set.
    creator_buffer_desc_set_ = device->createDescriptorSets(
        descriptor_pool, tile_creator_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    // all world map buffer only create once, so always pick the first one.
    auto texture_descs = addTileCreatorBuffers(
        creator_buffer_desc_set_,
        rock_layer_,
        soil_water_layer_[0], 
        grass_snow_layer_);
    device->updateDescriptorSets(texture_descs, {});

    // tile creator buffer set.
    for (int soil_water = 0; soil_water < 2; soil_water++) {
        tile_update_buffer_desc_set_[soil_water] = device->createDescriptorSets(
            descriptor_pool, tile_update_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        texture_descs = addTileUpdateBuffers(
            tile_update_buffer_desc_set_[soil_water],
            texture_sampler,
            rock_layer_,
            soil_water_layer_[soil_water],
            water_normal_);
        device->updateDescriptorSets(texture_descs, {});

        tile_flow_update_buffer_desc_set_[soil_water] = device->createDescriptorSets(
            descriptor_pool, tile_flow_update_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        texture_descs = addTileFlowUpdateBuffers(
            tile_flow_update_buffer_desc_set_[soil_water],
            texture_sampler,
            rock_layer_,
            soil_water_layer_[1 - soil_water],
            soil_water_layer_[soil_water],
            water_flow_);
        device->updateDescriptorSets(texture_descs, {});

        // tile params set.
        tile_res_desc_set_[soil_water] = device->createDescriptorSets(
            descriptor_pool, tile_res_desc_set_layout_, 1)[0];
    }
}

void DebugDrawObject::updateStaticDescriptorSet(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& src_texture,
    const std::shared_ptr<renderer::ImageView>& src_depth,
    const std::shared_ptr<renderer::ImageView>& airflow_tex) {

    if (tile_res_desc_set_[0] == nullptr) {
        generateStaticDescriptorSet(
            device,
            descriptor_pool,
            texture_sampler);
    }

    for (int soil_water = 0; soil_water < 2; soil_water++) {
        // create a global ibl texture descriptor set.
        auto tile_res_descs = addTileResourceTextures(
            tile_res_desc_set_[soil_water],
            texture_sampler,
            src_texture,
            src_depth,
            rock_layer_.view,
            soil_water_layer_[soil_water].view,
            grass_snow_layer_.view,
            water_normal_.view,
            water_flow_.view,
            airflow_tex);
        device->updateDescriptorSets(tile_res_descs, {});
    }
}

bool DebugDrawObject::validTileBySize(
    const glm::ivec2& min_tile_idx,
    const glm::ivec2& max_tile_idx,
    const float& tile_size) {

    glm::ivec2 tile_index =
        glm::ivec2((min_ + glm::vec2(tile_size / 2 + 1)) * glm::vec2(1.0f / tile_size));

    return
        (tile_index.x >= min_tile_idx.x && tile_index.x <= max_tile_idx.x) &&
        (tile_index.y >= min_tile_idx.y && tile_index.y <= max_tile_idx.y);
}

void DebugDrawObject::generateTileBuffers(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf) {

    renderer::helper::transitMapTextureToStoreImage(
        cmd_buf,
        {rock_layer_.image,
         soil_water_layer_[0].image,
         grass_snow_layer_.image});

    auto dispatch_count = static_cast<uint32_t>(TileConst::kWaterlayerSize);
    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, tile_creator_pipeline_);
    glsl::TileCreateParams tile_params = {};
    tile_params.world_min = glm::vec2(-kWorldMapSize / 2.0f);
    tile_params.world_range = glm::vec2(kWorldMapSize / 2.0f) - tile_params.world_min;
    tile_params.width_pixel_count = dispatch_count;
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        tile_creator_pipeline_layout_,
        &tile_params,
        sizeof(tile_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        tile_creator_pipeline_layout_,
        { creator_buffer_desc_set_ });

    cmd_buf->dispatch(
        (dispatch_count + 7) / 8,
        (dispatch_count + 7) / 8, 1);

    renderer::helper::transitMapTextureFromStoreImage(
        cmd_buf,
        { rock_layer_.image,
         soil_water_layer_[0].image,
         grass_snow_layer_.image });
}

void DebugDrawObject::updateTileBuffers(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    float current_time,
    int soil_water) {

    renderer::helper::transitMapTextureToStoreImage(
        cmd_buf,
        {soil_water_layer_[soil_water].image,
         water_normal_.image});

    auto dispatch_count = static_cast<uint32_t>(TileConst::kWaterlayerSize);

    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, tile_update_pipeline_);
    glsl::TileUpdateParams tile_params = {};
    tile_params.world_min = glm::vec2(-kWorldMapSize / 2.0f);
    tile_params.world_range = glm::vec2(kWorldMapSize / 2.0f) - tile_params.world_min;
    tile_params.width_pixel_count = dispatch_count;
    tile_params.inv_width_pixel_count = 1.0f / dispatch_count;
    tile_params.range_per_pixel = tile_params.world_range * tile_params.inv_width_pixel_count;
    tile_params.current_time = current_time;
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        tile_update_pipeline_layout_,
        &tile_params,
        sizeof(tile_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        tile_update_pipeline_layout_,
        { tile_update_buffer_desc_set_[soil_water] });

    cmd_buf->dispatch(
        (dispatch_count + 15) / 16,
        (dispatch_count + 15) / 16, 1);

    renderer::helper::transitMapTextureFromStoreImage(
        cmd_buf,
        {soil_water_layer_[soil_water].image,
         water_normal_.image});
}

void DebugDrawObject::updateTileFlowBuffers(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    float current_time,
    int soil_water) {

    renderer::helper::transitMapTextureToStoreImage(
        cmd_buf,
        {soil_water_layer_[soil_water].image,
         soil_water_layer_[1 - soil_water].image,
         water_flow_.image});

    auto dispatch_count = static_cast<uint32_t>(TileConst::kWaterlayerSize);

    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, tile_flow_update_pipeline_);
    glsl::TileUpdateParams tile_params = {};
    tile_params.world_min = glm::vec2(-kWorldMapSize / 2.0f);
    tile_params.world_range = glm::vec2(kWorldMapSize / 2.0f) - tile_params.world_min;
    tile_params.width_pixel_count = dispatch_count;
    tile_params.inv_width_pixel_count = 1.0f / dispatch_count;
    tile_params.range_per_pixel = tile_params.world_range * tile_params.inv_width_pixel_count;
    tile_params.flow_speed_factor = 1.0f / (1024.0f * tile_params.range_per_pixel);
    tile_params.current_time = current_time;
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        tile_flow_update_pipeline_layout_,
        &tile_params,
        sizeof(tile_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        tile_flow_update_pipeline_layout_,
        { tile_flow_update_buffer_desc_set_[soil_water] });

    cmd_buf->dispatch(
        (dispatch_count + 15) / 16,
        (dispatch_count + 15) / 16, 1);

    renderer::helper::transitMapTextureFromStoreImage(
        cmd_buf,
        {soil_water_layer_[soil_water].image,
         soil_water_layer_[1 - soil_water].image,
         water_flow_.image });
}

void DebugDrawObject::draw(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    const glm::uvec2 display_size,
    int soil_water,
    float delta_t,
    float cur_time,
    bool is_base_pass) {
    auto segment_count = static_cast<uint32_t>(TileConst::kSegmentCount);

    cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, is_base_pass ? tile_pipeline_ : tile_water_pipeline_);
    cmd_buf->bindIndexBuffer(index_buffer_.buffer, 0, renderer::IndexType::UINT32);

    glsl::TileParams tile_params = {};
    tile_params.world_min = glm::vec2(-kWorldMapSize / 2.0f);
    tile_params.inv_world_range = 1.0f / (glm::vec2(kWorldMapSize / 2.0f) - tile_params.world_min);
    tile_params.min = min_;
    tile_params.range = max_ - min_;
    tile_params.segment_count = segment_count;
    tile_params.offset = 0;
    tile_params.inv_screen_size = glm::vec2(1.0f / display_size.x, 1.0f / display_size.y);
    tile_params.delta_t = delta_t;
    tile_params.time = cur_time;
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) | 
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        tile_pipeline_layout_, 
        &tile_params, 
        sizeof(tile_params));

    auto new_desc_sets = desc_set_list;
    new_desc_sets.push_back(tile_res_desc_set_[soil_water]);

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        tile_pipeline_layout_, 
        new_desc_sets);

    cmd_buf->drawIndexed(segment_count * segment_count * 6);
}

void DebugDrawObject::generateAllDescriptorSets(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler) {
    generateStaticDescriptorSet(device, descriptor_pool, texture_sampler);
}

void DebugDrawObject::drawAllVisibleTiles(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    const glm::uvec2 display_size,
    int soil_water,
    float delta_t,
    float cur_time,
    bool is_base_pass) {

    for (auto& tile : visible_tiles_) {
        tile->draw(
            cmd_buf,
            desc_set_list,
            display_size,
            soil_water,
            delta_t,
            cur_time,
            is_base_pass);
    }
}

void DebugDrawObject::updateAllTiles(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::DescriptorPool> descriptor_pool,
    const float& tile_size,
    const glm::vec2& camera_pos) {

    uint32_t segment_count = static_cast<uint32_t>(TileConst::kSegmentCount);
    int32_t cache_tile_size = static_cast<int32_t>(TileConst::kCacheTileSize);
    int32_t visible_tile_size = static_cast<int32_t>(TileConst::kVisibleTileSize);
    int32_t num_cached_blocks = static_cast<int32_t>(TileConst::kNumCachedBlocks);

    glm::ivec2 center_tile_index = camera_pos * (1.0f / tile_size);
    glm::ivec2 min_cache_tile_idx = center_tile_index - glm::ivec2(cache_tile_size);
    glm::ivec2 max_cache_tile_idx = center_tile_index + glm::ivec2(cache_tile_size);
    glm::ivec2 min_visi_tile_idx = center_tile_index - glm::ivec2(visible_tile_size);
    glm::ivec2 max_visi_tile_idx = center_tile_index + glm::ivec2(visible_tile_size);

    visible_tiles_.clear();

    std::vector<size_t> to_delete_tiles;
    // remove all the tiles outside of cache zone.
    for (auto& tile : tile_meshes_) {
        bool inside = tile.second->validTileBySize(
            min_cache_tile_idx,
            max_cache_tile_idx,
            tile_size);

        if (!inside) {
            to_delete_tiles.push_back(tile.second->getHash());
        }
    }

    for (auto& hash_value : to_delete_tiles) {
        available_block_indexes_.push_back(tile_meshes_[hash_value]->block_idx_);
        tile_meshes_.erase(hash_value);
    }

    // add (kCacheTileSize * 2 + 1) x (kCacheTileSize * 2 + 1) tiles for caching.
    std::vector<std::shared_ptr<DebugDrawObject>> blocks(num_cached_blocks);
    int32_t i = 0;
    for (int y = min_cache_tile_idx.y; y <= max_cache_tile_idx.y; y++) {
        for (int x = min_cache_tile_idx.x; x <= max_cache_tile_idx.x; x++) {
            auto tile = addOneTile(
                device_info,
                descriptor_pool,
                glm::vec2(-tile_size / 2 + x * tile_size, -tile_size / 2 + y * tile_size),
                glm::vec2(tile_size / 2 + x * tile_size, tile_size / 2 + y * tile_size));
            blocks[i++] = tile;
        }
    }

    i = 0;
    auto row_size = cache_tile_size * 2 + 1;
    for (int y = min_cache_tile_idx.y; y <= max_cache_tile_idx.y; y++) {
        for (int x = min_cache_tile_idx.x; x <= max_cache_tile_idx.x; x++) {
            glm::ivec4 neighbors = glm::ivec4(
                x == min_cache_tile_idx.x ? -1 : blocks[i - 1]->block_idx_,
                x == max_cache_tile_idx.x ? -1 : blocks[i + 1]->block_idx_,
                y == min_cache_tile_idx.y ? -1 : blocks[i - row_size]->block_idx_,
                y == max_cache_tile_idx.y ? -1 : blocks[i + row_size]->block_idx_);
            blocks[i++]->setNeighbors(neighbors);
        }
    }

    for (auto& tile : tile_meshes_) {
        bool inside = tile.second->validTileBySize(
            min_visi_tile_idx,
            max_visi_tile_idx,
            tile_size);

        if (inside) {
            visible_tiles_.push_back(tile.second);
        }
    }
}

void DebugDrawObject::destoryAllTiles() {
}

} // namespace game_object
} // namespace engine