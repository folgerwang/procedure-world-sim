#include <vector>

#include "engine/renderer/renderer_helper.h"
#include "engine/engine_helper.h"
#include "shaders/global_definition.glsl.h"
#include "shaders/weather_common.glsl.h"
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

er::ShaderModuleList getTemperatureInitShaderModules(
    std::shared_ptr<er::Device> device)
{
    uint64_t compute_code_size;
    er::ShaderModuleList shader_modules;
    shader_modules.reserve(1);
    auto compute_shader_code = engine::helper::readFile("lib/shaders/temperature_init_comp.spv", compute_code_size);
    shader_modules.push_back(device->createShaderModule(compute_code_size, compute_shader_code.data()));

    return shader_modules;
}

std::vector<er::TextureDescriptor> addTemperatureInitTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const er::TextureInfo& temp_moisture_tex) {
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(1);

    er::Helper::addOneTexture(
        descriptor_writes,
        DST_TEMP_MOISTURE_TEX_INDEX,
        nullptr,
        temp_moisture_tex.view,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

std::vector<er::TextureDescriptor> addAirflowTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const er::TextureInfo& src_temp_moisture_tex,
    const er::TextureInfo& dst_temp_moisture_tex,
    const er::TextureInfo& dst_airflow_tex,
    const std::shared_ptr<er::ImageView>& rock_layer_tex,
    const std::shared_ptr<er::ImageView>& soil_water_layer_tex) {
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(5);

    er::Helper::addOneTexture(
        descriptor_writes,
        SRC_TEMP_MOISTURE_TEX_INDEX,
        nullptr,
        src_temp_moisture_tex.view,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        DST_TEMP_MOISTURE_TEX_INDEX,
        nullptr,
        dst_temp_moisture_tex.view,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        DST_AIRFLOW_TEX_INDEX,
        nullptr,
        dst_airflow_tex.view,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        ROCK_LAYER_BUFFER_INDEX,
        texture_sampler,
        rock_layer_tex,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        SOIL_WATER_LAYER_BUFFER_INDEX,
        texture_sampler,
        soil_water_layer_tex,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

static std::shared_ptr<er::DescriptorSetLayout> createTemperatureInitDescSetLayout(
    const std::shared_ptr<er::Device>& device) {
    std::vector<er::DescriptorSetLayoutBinding> bindings(1);
    bindings[0] =
        er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            DST_TEMP_MOISTURE_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);

    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<er::DescriptorSetLayout> createAirflowUpdateDescSetLayout(
    const std::shared_ptr<er::Device>& device) {
    std::vector<er::DescriptorSetLayoutBinding> bindings(5);
    bindings[0] =
        er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            DST_TEMP_MOISTURE_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);

    bindings[1] =
        er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            SRC_TEMP_MOISTURE_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);

    bindings[2] =
        er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            DST_AIRFLOW_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);

    bindings[3] =
        er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            ROCK_LAYER_BUFFER_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);

    bindings[4] =
        er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            SOIL_WATER_LAYER_BUFFER_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);

    return device->createDescriptorSetLayout(bindings);
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

static std::shared_ptr<er::PipelineLayout> createTemperatureInitPipelineLayout(
    const std::shared_ptr<er::Device>& device,
    const er::DescriptorSetLayoutList& desc_set_layouts) {
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::AirflowUpdateParams);

    return device->createPipelineLayout(desc_set_layouts, { push_const_range });
}

static std::shared_ptr<er::Pipeline> createTemperatureInitPipeline(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::PipelineLayout>& pipeline_layout) {

    auto temperature_init_compute_shader_modules = getTemperatureInitShaderModules(device);
    assert(temperature_init_compute_shader_modules.size() == 1);

    auto pipeline = device->createPipeline(
        pipeline_layout,
        temperature_init_compute_shader_modules[0]);

    for (auto& shader_module : temperature_init_compute_shader_modules) {
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
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& rock_layer_tex,
    const std::vector<std::shared_ptr<renderer::ImageView>>& soil_water_layer_tex) {

    const auto& device = device_info.device;

    for (int i = 0; i < 2; i++) {
        renderer::Helper::create3DTextureImage(
            device_info,
            renderer::Format::R16G16_UNORM,
            glm::uvec3(
                WeatherSystemConst::kAirflowBufferWidth,
                WeatherSystemConst::kAirflowBufferWidth,
                WeatherSystemConst::kAirflowBufferHeight),
            temp_moisture_volume_[i],
            SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
            SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }

    renderer::Helper::create3DTextureImage(
        device_info,
        renderer::Format::R8G8B8A8_UNORM,
        glm::uvec3(
            WeatherSystemConst::kAirflowBufferWidth,
            WeatherSystemConst::kAirflowBufferWidth,
            WeatherSystemConst::kAirflowBufferHeight),
        airflow_volume_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    temperature_init_desc_set_layout_ = createTemperatureInitDescSetLayout(device);
    airflow_desc_set_layout_ = createAirflowUpdateDescSetLayout(device);

    recreate(
        device,
        descriptor_pool,
        texture_sampler,
        rock_layer_tex,
        soil_water_layer_tex);
}

void WeatherSystem::recreate(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& rock_layer_tex,
    const std::vector<std::shared_ptr<renderer::ImageView>>& soil_water_layer_tex) {

    if (airflow_pipeline_layout_ != nullptr) {
        device->destroyPipelineLayout(airflow_pipeline_layout_);
        airflow_pipeline_layout_ = nullptr;
    }
    
    if (airflow_pipeline_ != nullptr) {
        device->destroyPipeline(airflow_pipeline_);
        airflow_pipeline_ = nullptr;
    }

    temperature_init_tex_desc_set_ = nullptr;
    temperature_init_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            temperature_init_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto temperature_init_texture_descs = addTemperatureInitTextures(
        temperature_init_tex_desc_set_,
        temp_moisture_volume_[0]);
    device->updateDescriptorSets(temperature_init_texture_descs, {});

    for (int dbuf_idx = 0; dbuf_idx < 2; dbuf_idx++) {
        airflow_tex_desc_set_[dbuf_idx] = nullptr;
        airflow_tex_desc_set_[dbuf_idx] =
            device->createDescriptorSets(
                descriptor_pool,
                airflow_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto airflow_texture_descs = addAirflowTextures(
            airflow_tex_desc_set_[dbuf_idx],
            texture_sampler,
            temp_moisture_volume_[1-dbuf_idx],
            temp_moisture_volume_[dbuf_idx],
            airflow_volume_,
            rock_layer_tex,
            soil_water_layer_tex[dbuf_idx]);
        device->updateDescriptorSets(airflow_texture_descs, {});
    }

    temperature_init_pipeline_layout_ =
        createTemperatureInitPipelineLayout(
            device,
            { temperature_init_desc_set_layout_ } );

    temperature_init_pipeline_ = createTemperatureInitPipeline(
        device,
        temperature_init_pipeline_layout_);

    airflow_pipeline_layout_ =
        createAirflowUpdatePipelineLayout(
            device,
            { airflow_desc_set_layout_ });

    airflow_pipeline_ = createAirflowUpdatePipeline(
        device,
        airflow_pipeline_layout_);
}

// update air flow buffer.
void WeatherSystem::initTemperatureBuffer(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf) {

    renderer::helper::transitMapTextureToStoreImage(
        cmd_buf,
        { temp_moisture_volume_[0].image });

    auto w = static_cast<uint32_t>(WeatherSystemConst::kAirflowBufferWidth);
    auto h = static_cast<uint32_t>(WeatherSystemConst::kAirflowBufferHeight);
    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, temperature_init_pipeline_);
    glsl::AirflowUpdateParams airflow_params = {};
    airflow_params.world_min =
        glm::vec3(-kWorldMapSize / 2.0f, -kWorldMapSize / 2.0f, kAirflowLowHeight);
    airflow_params.world_range =
        glm::vec3(kWorldMapSize / 2.0f, kWorldMapSize / 2.0f, kAirflowMaxHeight) - airflow_params.world_min;
    airflow_params.inv_size = glm::vec3(1.0f / w, 1.0f / w, 1.0f / h);
    airflow_params.size = glm::uvec3(w, w, h);
    airflow_params.sea_level_temperature = 30.0f;
    airflow_params.soil_temp_adj = 0;
    airflow_params.water_temp_adj = 0;
    airflow_params.air_temp_adj = 0;
    airflow_params.heat_transfer_ratio = 0;
    airflow_params.moist_transfer_ratio = 0;
    airflow_params.current_time = 0;
    airflow_params.height_params =
        glm::vec2(log2(1.0f + airflow_params.world_range.z),
            -1.0f + airflow_params.world_min.z);

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        temperature_init_pipeline_layout_,
        &airflow_params,
        sizeof(airflow_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        temperature_init_pipeline_layout_,
        { temperature_init_tex_desc_set_ });

    cmd_buf->dispatch(
        (w + 7) / 8,
        (w + 7) / 8,
        h);

    renderer::helper::transitMapTextureFromStoreImage(
        cmd_buf,
        { temp_moisture_volume_[0].image });
}

// update air flow buffer.
void WeatherSystem::updateAirflowBuffer(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    int dbuf_idx,
    float current_time) {

    renderer::helper::transitMapTextureToStoreImage(
        cmd_buf,
        { temp_moisture_volume_[dbuf_idx].image,
          temp_moisture_volume_[1-dbuf_idx].image,
          airflow_volume_.image});

    auto w = static_cast<uint32_t>(WeatherSystemConst::kAirflowBufferWidth);
    auto h = static_cast<uint32_t>(WeatherSystemConst::kAirflowBufferHeight);
    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, airflow_pipeline_);
    glsl::AirflowUpdateParams airflow_params = {};
    airflow_params.world_min =
        glm::vec3(-kWorldMapSize / 2.0f, -kWorldMapSize / 2.0f, kAirflowLowHeight);
    airflow_params.world_range =
        glm::vec3(kWorldMapSize / 2.0f, kWorldMapSize / 2.0f, kAirflowMaxHeight) - airflow_params.world_min;
    airflow_params.inv_size = glm::vec3(1.0f / w, 1.0f / w, 1.0f / h);
    airflow_params.size = glm::ivec3(w, w, h);
    airflow_params.sea_level_temperature = 30.0f;
    airflow_params.soil_temp_adj = 0.20f;
    airflow_params.water_temp_adj = 1.02f;
    airflow_params.air_temp_adj = 0.000f;
    airflow_params.water_moist_adj = 1.0f;
    airflow_params.soil_moist_adj = 0.2f;
    airflow_params.heat_transfer_ratio = 1.0f;
    airflow_params.moist_transfer_ratio = 0.8f;
    airflow_params.current_time = current_time;
    airflow_params.height_params =
        glm::vec2(log2(1.0f + airflow_params.world_range.z),
            -1.0f + airflow_params.world_min.z);

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        airflow_pipeline_layout_,
        &airflow_params,
        sizeof(airflow_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        airflow_pipeline_layout_,
        { airflow_tex_desc_set_[dbuf_idx] });

    cmd_buf->dispatch(
        (w + 7) / 8,
        (w + 7) / 8,
        (h + 15) / 16);

    renderer::helper::transitMapTextureFromStoreImage(
        cmd_buf,
        { temp_moisture_volume_[dbuf_idx].image,
          temp_moisture_volume_[1-dbuf_idx].image,
          airflow_volume_.image });
}

void WeatherSystem::update() {
}

void WeatherSystem::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    for (int i = 0; i < 2; i++) {
        temp_moisture_volume_[i].destroy(device);
    }
    airflow_volume_.destroy(device);
    device->destroyDescriptorSetLayout(airflow_desc_set_layout_);
    device->destroyPipelineLayout(airflow_pipeline_layout_);
    device->destroyPipeline(airflow_pipeline_);
    device->destroyDescriptorSetLayout(temperature_init_desc_set_layout_);
    device->destroyPipelineLayout(temperature_init_pipeline_layout_);
    device->destroyPipeline(temperature_init_pipeline_);
}

}//namespace scene_rendering
}//namespace engine
