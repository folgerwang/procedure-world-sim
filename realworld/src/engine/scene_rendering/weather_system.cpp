#include <vector>

#include "engine/renderer/renderer_helper.h"
#include "engine/engine_helper.h"
#include "shaders/global_definition.glsl.h"
#include "engine/game_object/terrain.h"
#include "weather_system.h"

namespace {
namespace er = engine::renderer;

er::ShaderModuleList getAirflowUpdateShaderModules(
    std::shared_ptr<er::Device> device)
{
    uint64_t compute_code_size;
    er::ShaderModuleList shader_modules;
    shader_modules.reserve(1);
    auto compute_shader_code = engine::helper::readFile("lib/shaders/airflow_update_comp.spv", compute_code_size);
    shader_modules.push_back(device->createShaderModule(compute_code_size, compute_shader_code.data()));

    return shader_modules;
}

std::vector<er::TextureDescriptor> addAirflowTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const er::TextureInfo& airflow_tex,
    const std::shared_ptr<er::Sampler>& texture_sampler) {
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(1);

    er::Helper::addOneTexture(
        descriptor_writes,
        DST_AIRFLOW_TEX_INDEX,
        nullptr,
        airflow_tex.view,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

static std::shared_ptr<er::PipelineLayout> createAirflowUpdatePipelineLayout(
    const std::shared_ptr<er::Device>& device,
    const er::DescriptorSetLayoutList& desc_set_layouts) {
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::AirflowUpdateParams);

    return device->createPipelineLayout(desc_set_layouts, { push_const_range });
}

static std::shared_ptr<er::Pipeline> createAirflowUpdatePipeline(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::PipelineLayout>& pipeline_layout) {

    auto airflow_update_compute_shader_modules = getAirflowUpdateShaderModules(device);
    assert(airflow_update_compute_shader_modules.size() == 1);

    auto pipeline = device->createPipeline(
        pipeline_layout,
        airflow_update_compute_shader_modules[0]);

    for (auto& shader_module : airflow_update_compute_shader_modules) {
        device->destroyShaderModule(shader_module);
    }

    return pipeline;
}

} // namespace

namespace engine {
namespace scene_rendering {

WeatherSystem::WeatherSystem(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler) {

    const auto& device = device_info.device;

    renderer::Helper::create3DTextureImage(
        device_info,
        renderer::Format::R16G16_UNORM,
        glm::uvec3(
            WeatherSystemConst::kAirflowBufferWidth,
            WeatherSystemConst::kAirflowBufferWidth,
            WeatherSystemConst::kAirflowBufferHeight),
        airflow_volume_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    std::vector<renderer::DescriptorSetLayoutBinding> bindings(1);
    bindings[0] =
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            DST_AIRFLOW_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            renderer::DescriptorType::STORAGE_IMAGE);

    airflow_desc_set_layout_ =
        device->createDescriptorSetLayout(bindings);

    recreate(
        device,
        descriptor_pool,
        texture_sampler);
}

void WeatherSystem::recreate(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler) {

    if (airflow_pipeline_layout_ != nullptr) {
        device->destroyPipelineLayout(airflow_pipeline_layout_);
        airflow_pipeline_layout_ = nullptr;
    }
    
    if (airflow_pipeline_ != nullptr) {
        device->destroyPipeline(airflow_pipeline_);
        airflow_pipeline_ = nullptr;
    }

    airflow_tex_desc_set_ = nullptr;

    // skybox
    airflow_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            airflow_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto airflow_texture_descs = addAirflowTextures(
        airflow_tex_desc_set_,
        airflow_volume_,
        texture_sampler);
    device->updateDescriptorSets(airflow_texture_descs, {});

    airflow_pipeline_layout_ =
        createAirflowUpdatePipelineLayout(
            device,
            { airflow_desc_set_layout_ } );

    airflow_pipeline_ = createAirflowUpdatePipeline(
        device,
        airflow_pipeline_layout_);
}

// render skybox.
void WeatherSystem::updateAirflowBuffer(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf) {

    renderer::helper::transitMapTextureToStoreImage(
        cmd_buf,
        { airflow_volume_.image });

    auto w = static_cast<uint32_t>(WeatherSystemConst::kAirflowBufferWidth);
    auto h = static_cast<uint32_t>(WeatherSystemConst::kAirflowBufferHeight);
    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, airflow_pipeline_);
    glsl::AirflowUpdateParams airflow_params = {};
    airflow_params.world_min =
        glm::vec3(-kWorldMapSize / 2.0f, -kWorldMapSize / 2.0f, kAirflowLowHeight);
    airflow_params.world_range =
        glm::vec3(kWorldMapSize / 2.0f, kWorldMapSize / 2.0f, kAirflowMaxHeight) - airflow_params.world_min;
    airflow_params.inv_size = glm::vec3(1.0f / w, 1.0f / w, 1.0f / h);
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        airflow_pipeline_layout_,
        &airflow_params,
        sizeof(airflow_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        airflow_pipeline_layout_,
        { airflow_tex_desc_set_ });

    cmd_buf->dispatch(
        (w + 7) / 8,
        (w + 7) / 8,
        h);

    renderer::helper::transitMapTextureFromStoreImage(
        cmd_buf,
        { airflow_volume_.image });
}

void WeatherSystem::update() {
}

void WeatherSystem::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    airflow_volume_.destroy(device);
    device->destroyDescriptorSetLayout(airflow_desc_set_layout_);
    device->destroyPipelineLayout(airflow_pipeline_layout_);
    device->destroyPipeline(airflow_pipeline_);
}

}//namespace scene_rendering
}//namespace engine
