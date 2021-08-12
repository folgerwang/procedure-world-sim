#include <vector>

#include "engine/renderer/renderer_helper.h"
#include "engine/engine_helper.h"
#include "engine/game_object/terrain.h"
#include "weather_system.h"
#include "shaders/weather_common.glsl.h"

namespace {
namespace er = engine::renderer;

er::ShaderModuleList getComputeShaderModules(
    const std::shared_ptr<er::Device>& device,
    const std::string& compute_shader_name) {
    uint64_t compute_code_size;
    er::ShaderModuleList shader_modules;
    shader_modules.reserve(1);
    std::string file_name = std::string("lib/shaders/") + compute_shader_name + "_comp.spv";
    auto compute_shader_code = engine::helper::readFile(file_name.c_str(), compute_code_size);
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

std::vector<er::TextureDescriptor> addCloudLightingTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const er::TextureInfo& src_temp_moisture_tex,
    const er::TextureInfo& dst_cloud_shadow_tex,
    const er::TextureInfo& dst_cloud_lighting_tex) {
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(3);

    er::Helper::addOneTexture(
        descriptor_writes,
        SRC_TEMP_MOISTURE_TEX_INDEX,
        texture_sampler,
        src_temp_moisture_tex.view,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        SRC_CLOUD_SHADOW_TEX_INDEX,
        texture_sampler,
        dst_cloud_shadow_tex.view,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        DST_CLOUD_LIGHTING_TEX_INDEX,
        nullptr,
        dst_cloud_lighting_tex.view,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

std::vector<er::TextureDescriptor> addCloudShadowTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const er::TextureInfo& src_temp_moisture_tex,
    const er::TextureInfo& dst_cloud_shadow_tex) {
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(2);

    er::Helper::addOneTexture(
        descriptor_writes,
        SRC_TEMP_MOISTURE_TEX_INDEX,
        texture_sampler,
        src_temp_moisture_tex.view,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        DST_CLOUD_SHADOW_TEX_INDEX,
        nullptr,
        dst_cloud_shadow_tex.view,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

std::vector<er::TextureDescriptor> addCloudShadowMergeTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const er::TextureInfo& dst_cloud_shadow_tex) {
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(1);

    er::Helper::addOneTexture(
        descriptor_writes,
        DST_CLOUD_SHADOW_TEX_INDEX,
        nullptr,
        dst_cloud_shadow_tex.view,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        er::ImageLayout::GENERAL);

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

static std::shared_ptr<er::DescriptorSetLayout> createCloudLightingUpdateDescSetLayout(
    const std::shared_ptr<er::Device>& device) {
    std::vector<er::DescriptorSetLayoutBinding> bindings(3);
    bindings[0] =
        er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            SRC_TEMP_MOISTURE_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);

    bindings[1] =
        er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            SRC_CLOUD_SHADOW_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);

    bindings[2] =
        er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            DST_CLOUD_LIGHTING_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);

    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<er::DescriptorSetLayout> createCloudShadowUpdateDescSetLayout(
    const std::shared_ptr<er::Device>& device) {
    std::vector<er::DescriptorSetLayoutBinding> bindings(2);
    bindings[0] =
        er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            SRC_TEMP_MOISTURE_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);

    bindings[1] =
        er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            DST_CLOUD_SHADOW_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);

    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<er::DescriptorSetLayout> createCloudShadowMergeDescSetLayout(
    const std::shared_ptr<er::Device>& device) {
    std::vector<er::DescriptorSetLayoutBinding> bindings(1);
    bindings[0] =
        er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            DST_CLOUD_SHADOW_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);

    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<er::PipelineLayout> createComputePipelineLayout(
    const std::shared_ptr<er::Device>& device,
    const er::DescriptorSetLayoutList& desc_set_layouts,
    const uint32_t& push_const_range_size) {
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = push_const_range_size;

    return device->createPipelineLayout(desc_set_layouts, { push_const_range });
}

static std::shared_ptr<er::Pipeline> createComputePipeline(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::PipelineLayout>& pipeline_layout,
    const std::string& compute_shader_name) {

    auto compute_shader_modules =
        getComputeShaderModules(device, compute_shader_name);
    assert(compute_shader_modules.size() == 1);

    auto pipeline = device->createPipeline(
        pipeline_layout,
        compute_shader_modules[0]);

    for (auto& shader_module : compute_shader_modules) {
        device->destroyShaderModule(shader_module);
    }

    return pipeline;
}

static void releasePipelineLayout(
    const std::shared_ptr<er::Device>& device,
    std::shared_ptr<er::PipelineLayout>& pipeline_layout) {
    if (pipeline_layout != nullptr) {
        device->destroyPipelineLayout(pipeline_layout);
        pipeline_layout = nullptr;
    }
}

static void releasePipeline(
    const std::shared_ptr<er::Device>& device,
    std::shared_ptr<er::Pipeline>& pipeline) {
    if (pipeline != nullptr) {
        device->destroyPipeline(pipeline);
        pipeline = nullptr;
    }
}

} // namespace

namespace engine {
namespace scene_rendering {

WeatherSystem::WeatherSystem(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& rock_layer_tex,
    const std::vector<std::shared_ptr<renderer::ImageView>>& soil_water_layer_tex) {

    const auto& device = device_info.device;

    for (int i = 0; i < 2; i++) {
        renderer::Helper::create3DTextureImage(
            device_info,
            renderer::Format::R16G16_UNORM,
            glm::uvec3(
                kAirflowBufferWidth,
                kAirflowBufferWidth,
                kAirflowBufferHeight),
            temp_moisture_volume_[i],
            SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
            SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }

    renderer::Helper::create3DTextureImage(
        device_info,
        renderer::Format::R8G8B8A8_UNORM,
        glm::uvec3(
            kAirflowBufferWidth,
            kAirflowBufferWidth,
            kAirflowBufferHeight),
        airflow_volume_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    uint32_t buffer_height = kAirflowBufferHeight;
    renderer::Helper::create3DTextureImage(
        device_info,
        renderer::Format::R16_SFLOAT,
        glm::uvec3(
            kAirflowBufferWidth,
            kAirflowBufferWidth,
            buffer_height),
        cloud_shadow_volume_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::create3DTextureImage(
        device_info,
        renderer::Format::B10G11R11_UFLOAT_PACK32,
        glm::uvec3(
            kAirflowBufferWidth,
            kAirflowBufferWidth,
            kAirflowBufferHeight),
        cloud_lighting_volume_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    temperature_init_desc_set_layout_ = createTemperatureInitDescSetLayout(device);
    airflow_desc_set_layout_ = createAirflowUpdateDescSetLayout(device);
    cloud_lighting_desc_set_layout_ = createCloudLightingUpdateDescSetLayout(device);
    cloud_shadow_desc_set_layout_ = createCloudShadowUpdateDescSetLayout(device);
    cloud_shadow_merge_desc_set_layout_ = createCloudShadowMergeDescSetLayout(device);

    recreate(
        device,
        descriptor_pool,
        global_desc_set_layouts,
        texture_sampler,
        rock_layer_tex,
        soil_water_layer_tex);
}

void WeatherSystem::recreate(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& rock_layer_tex,
    const std::vector<std::shared_ptr<renderer::ImageView>>& soil_water_layer_tex) {

    releasePipelineLayout(device, temperature_init_pipeline_layout_);
    releasePipeline(device, temperature_init_pipeline_);
    releasePipelineLayout(device, airflow_pipeline_layout_);
    releasePipeline(device, airflow_pipeline_);
    releasePipelineLayout(device, cloud_lighting_pipeline_layout_);
    releasePipeline(device, cloud_lighting_pipeline_);
    releasePipelineLayout(device, cloud_shadow_pipeline_layout_);
    releasePipeline(device, cloud_shadow_pipeline_);
    releasePipeline(device, cloud_shadow_init_pipeline_);
    releasePipelineLayout(device, cloud_shadow_merge_pipeline_layout_);
    releasePipeline(device, cloud_shadow_merge_pipeline_);

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

        cloud_lighting_tex_desc_set_[dbuf_idx] = nullptr;
        cloud_lighting_tex_desc_set_[dbuf_idx] =
            device->createDescriptorSets(
                descriptor_pool,
                cloud_lighting_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto cloud_lighting_texture_descs = addCloudLightingTextures(
            cloud_lighting_tex_desc_set_[dbuf_idx],
            texture_sampler,
            temp_moisture_volume_[dbuf_idx],
            cloud_shadow_volume_,
            cloud_lighting_volume_);
        device->updateDescriptorSets(cloud_lighting_texture_descs, {});

        cloud_shadow_tex_desc_set_[dbuf_idx] = nullptr;
        cloud_shadow_tex_desc_set_[dbuf_idx] =
            device->createDescriptorSets(
                descriptor_pool,
                cloud_shadow_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto cloud_shadow_texture_descs = addCloudShadowTextures(
            cloud_shadow_tex_desc_set_[dbuf_idx],
            texture_sampler,
            temp_moisture_volume_[dbuf_idx],
            cloud_shadow_volume_);
        device->updateDescriptorSets(cloud_shadow_texture_descs, {});
    }

    cloud_shadow_merge_tex_desc_set_ = nullptr;
    cloud_shadow_merge_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            cloud_shadow_merge_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto cloud_shadow_merge_tex_descs = addCloudShadowMergeTextures(
        cloud_shadow_merge_tex_desc_set_,
        texture_sampler,
        cloud_shadow_volume_);
    device->updateDescriptorSets(cloud_shadow_merge_tex_descs, {});

    temperature_init_pipeline_layout_ =
        createComputePipelineLayout(
            device,
            { temperature_init_desc_set_layout_ },
            sizeof(glsl::AirflowUpdateParams));

    temperature_init_pipeline_ = createComputePipeline(
        device,
        temperature_init_pipeline_layout_,
        "temperature_init");

    airflow_pipeline_layout_ =
        createComputePipelineLayout(
            device,
            { airflow_desc_set_layout_ },
            sizeof(glsl::AirflowUpdateParams));

    airflow_pipeline_ = createComputePipeline(
        device,
        airflow_pipeline_layout_,
        "airflow_update");

    auto desc_set_layouts = global_desc_set_layouts;
    desc_set_layouts.push_back(cloud_lighting_desc_set_layout_);
    cloud_lighting_pipeline_layout_ =
        createComputePipelineLayout(
            device,
            desc_set_layouts,
            sizeof(glsl::CloudLightingParams));

    cloud_lighting_pipeline_ = createComputePipeline(
        device,
        cloud_lighting_pipeline_layout_,
        "cloud_lighting");

    cloud_shadow_pipeline_layout_ =
        createComputePipelineLayout(
            device,
            { cloud_shadow_desc_set_layout_ },
            sizeof(glsl::CloudShadowParams));

    cloud_shadow_pipeline_ = createComputePipeline(
        device,
        cloud_shadow_pipeline_layout_,
        "cloud_shadow");

    cloud_shadow_init_pipeline_ = createComputePipeline(
        device,
        cloud_shadow_pipeline_layout_,
        "cloud_shadow_init");

    cloud_shadow_merge_pipeline_layout_ =
        createComputePipelineLayout(
            device,
            { cloud_shadow_merge_desc_set_layout_ },
            sizeof(glsl::CloudShadowParams));

    cloud_shadow_merge_pipeline_ = createComputePipeline(
        device,
        cloud_shadow_merge_pipeline_layout_,
        "cloud_shadow_merge");
}

// update air flow buffer.
void WeatherSystem::initTemperatureBuffer(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf) {

    renderer::helper::transitMapTextureToStoreImage(
        cmd_buf,
        { temp_moisture_volume_[0].image });

    auto w = static_cast<uint32_t>(kAirflowBufferWidth);
    auto h = static_cast<uint32_t>(kAirflowBufferHeight);
    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, temperature_init_pipeline_);
    glsl::AirflowUpdateParams airflow_params = {};
    airflow_params.world_min =
        glm::vec3(-kWorldMapSize / 2.0f, -kWorldMapSize / 2.0f, kAirflowLowHeight);
    airflow_params.world_range =
        glm::vec3(kWorldMapSize / 2.0f, kWorldMapSize / 2.0f, kAirflowMaxHeight) - airflow_params.world_min;
    airflow_params.inv_size = glm::vec3(1.0f / w, 1.0f / w, 1.0f / h);
    airflow_params.size = glm::uvec3(w, w, h);
    airflow_params.controls = {0};
    airflow_params.controls.sea_level_temperature = 30.0f;
    airflow_params.current_time = 0;

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
    const glsl::WeatherControl& weather_controls,
    int dbuf_idx,
    float current_time) {

    renderer::helper::transitMapTextureToStoreImage(
        cmd_buf,
        { temp_moisture_volume_[dbuf_idx].image,
          temp_moisture_volume_[1-dbuf_idx].image,
          airflow_volume_.image});

    auto w = static_cast<uint32_t>(kAirflowBufferWidth);
    auto h = static_cast<uint32_t>(kAirflowBufferHeight);
    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, airflow_pipeline_);
    glsl::AirflowUpdateParams airflow_params = {};
    airflow_params.world_min =
        glm::vec3(-kWorldMapSize / 2.0f, -kWorldMapSize / 2.0f, kAirflowLowHeight);
    airflow_params.world_range =
        glm::vec3(kWorldMapSize / 2.0f, kWorldMapSize / 2.0f, kAirflowMaxHeight) - airflow_params.world_min;
    airflow_params.inv_size = glm::vec3(1.0f / w, 1.0f / w, 1.0f / h);
    airflow_params.size = glm::ivec3(w, w, h);
    airflow_params.controls = weather_controls;
    airflow_params.current_time = current_time;

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

void WeatherSystem::updateCloudShadow(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const glm::vec3& sun_dir,
    const float& light_ext_factor,
    const int& dbuf_idx,
    const float& current_time) {

    auto w = static_cast<uint32_t>(kAirflowBufferWidth);
    auto h = static_cast<uint32_t>(kAirflowBufferHeight);
    // create light passing each layer.
    {
        renderer::helper::transitMapTextureToStoreImage(
            cmd_buf,
            { cloud_shadow_volume_.image });

        cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, cloud_shadow_init_pipeline_);
        glsl::CloudShadowParams params = {};
        params.world_min =
            glm::vec3(-kWorldMapSize / 2.0f, -kWorldMapSize / 2.0f, kAirflowLowHeight);
        params.world_range =
            glm::vec3(kWorldMapSize / 2.0f, kWorldMapSize / 2.0f, kAirflowMaxHeight) - params.world_min;
        params.inv_world_range = 1.0f / params.world_range;
        params.inv_size = glm::vec3(1.0f / w, 1.0f / w, 1.0f / h);
        params.size = glm::ivec3(w, w, h);
        params.current_time = current_time;
        params.sun_dir = sun_dir;
        params.light_ext_factor = light_ext_factor;

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            cloud_shadow_pipeline_layout_,
            { cloud_shadow_tex_desc_set_[dbuf_idx] });

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            cloud_shadow_pipeline_layout_,
            &params,
            sizeof(params));

        cmd_buf->dispatch(
            (w + 7) / 8,
            (w + 7) / 8,
            h);

        renderer::helper::transitMapTextureFromStoreImage(
            cmd_buf,
            { cloud_shadow_volume_.image });
    }

    // merge two layers to one combined layer.
    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, cloud_shadow_merge_pipeline_);
    for (uint32_t i_layer = 0; i_layer < kAirflowBufferCount-1; i_layer++) {
        renderer::helper::transitMapTextureToStoreImage(
            cmd_buf,
            { cloud_shadow_volume_.image });

        glsl::CloudShadowParams params = {};
        params.world_min =
            glm::vec3(-kWorldMapSize / 2.0f, -kWorldMapSize / 2.0f, kAirflowLowHeight);
        params.world_range =
            glm::vec3(kWorldMapSize / 2.0f, kWorldMapSize / 2.0f, kAirflowMaxHeight) - params.world_min;
        params.inv_world_range = 1.0f / params.world_range;
        params.inv_size = glm::vec3(1.0f / w, 1.0f / w, 1.0f / h);
        params.size = glm::ivec3(w, w, h);
        params.current_time = current_time;
        params.sun_dir = sun_dir;
        params.light_ext_factor = light_ext_factor;
        params.layer_idx = i_layer;

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            cloud_shadow_merge_pipeline_layout_,
            { cloud_shadow_merge_tex_desc_set_ });

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            cloud_shadow_merge_pipeline_layout_,
            &params,
            sizeof(params));

        cmd_buf->dispatch(
            (w + 7) / 8,
            (w + 7) / 8,
            h >> 1);

        renderer::helper::transitMapTextureFromStoreImage(
            cmd_buf,
            { cloud_shadow_volume_.image });
    }
}

void WeatherSystem::updateCloudLighting(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    const std::shared_ptr<scene_rendering::Skydome>& skydome,
    int dbuf_idx,
    float current_time) {
    renderer::helper::transitMapTextureToStoreImage(
        cmd_buf,
        { cloud_lighting_volume_.image });

    auto w = static_cast<uint32_t>(kAirflowBufferWidth);
    auto h = static_cast<uint32_t>(kAirflowBufferHeight);
    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, cloud_lighting_pipeline_);
    glsl::CloudLightingParams params = {};
    params.world_min =
        glm::vec3(-kWorldMapSize / 2.0f, -kWorldMapSize / 2.0f, kAirflowLowHeight);
    params.world_range =
        glm::vec3(kWorldMapSize / 2.0f, kWorldMapSize / 2.0f, kAirflowMaxHeight) - params.world_min;
    params.inv_world_range = 1.0f / params.world_range;
    params.inv_size = glm::vec3(1.0f / w, 1.0f / w, 1.0f / h);
    params.size = glm::ivec3(w, w, h);
    params.current_time = current_time;
    params.sun_dir = skydome->getSunDir();
    params.g = skydome->getG();

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        cloud_lighting_pipeline_layout_,
        &params,
        sizeof(params));

    auto new_desc_sets = desc_set_list;
    new_desc_sets.push_back(cloud_lighting_tex_desc_set_[dbuf_idx]);

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        cloud_lighting_pipeline_layout_,
        new_desc_sets);

    cmd_buf->dispatch(
        (w + 7) / 8,
        (w + 7) / 8,
        h);

    renderer::helper::transitMapTextureFromStoreImage(
        cmd_buf,
        { cloud_lighting_volume_.image });
}

void WeatherSystem::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    for (int i = 0; i < 2; i++) {
        temp_moisture_volume_[i].destroy(device);
    }
    airflow_volume_.destroy(device);
    cloud_lighting_volume_.destroy(device);
    cloud_shadow_volume_.destroy(device);
    device->destroyDescriptorSetLayout(airflow_desc_set_layout_);
    device->destroyPipelineLayout(airflow_pipeline_layout_);
    device->destroyPipeline(airflow_pipeline_);
    device->destroyDescriptorSetLayout(cloud_lighting_desc_set_layout_);
    device->destroyPipelineLayout(cloud_lighting_pipeline_layout_);
    device->destroyPipeline(cloud_lighting_pipeline_);
    device->destroyDescriptorSetLayout(temperature_init_desc_set_layout_);
    device->destroyPipelineLayout(temperature_init_pipeline_layout_);
    device->destroyPipeline(temperature_init_pipeline_);
    device->destroyDescriptorSetLayout(cloud_shadow_desc_set_layout_);
    device->destroyPipelineLayout(cloud_shadow_pipeline_layout_);
    device->destroyPipeline(cloud_shadow_pipeline_);
    device->destroyPipeline(cloud_shadow_init_pipeline_);
    device->destroyDescriptorSetLayout(cloud_shadow_merge_desc_set_layout_);
    device->destroyPipelineLayout(cloud_shadow_merge_pipeline_layout_);
    device->destroyPipeline(cloud_shadow_merge_pipeline_);
}

}//namespace scene_rendering
}//namespace engine
