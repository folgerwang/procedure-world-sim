#include <iostream>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <chrono>

#include "engine/engine_helper.h"
#include "engine/game_object/camera.h"
#include "engine/renderer/renderer_helper.h"

#include "tiny_gltf.h"
#include "engine/tiny_mtx2.h"

namespace ego = engine::game_object;
namespace engine {

namespace {
static renderer::ShaderModuleList getComputeShaderModules(
    const std::shared_ptr<renderer::Device>& device,
    const std::string& compute_shader_name) {
    uint64_t compute_code_size;
    renderer::ShaderModuleList shader_modules;
    shader_modules.reserve(1);
    auto file_path = "lib/shaders/" + compute_shader_name + "_comp.spv";
    auto compute_shader_code = engine::helper::readFile(file_path, compute_code_size);
    shader_modules.push_back(device->createShaderModule(compute_code_size, compute_shader_code.data()));

    return shader_modules;
}

std::vector<renderer::BufferDescriptor> addGameCameraInfoBuffer(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const renderer::BufferInfo& game_objects_buffer,
    const renderer::BufferInfo& game_camera_buffer) {
    std::vector<renderer::BufferDescriptor> descriptor_writes;
    descriptor_writes.reserve(2);

    renderer::Helper::addOneBuffer(
        descriptor_writes,
        GAME_OBJECTS_BUFFER_INDEX,
        game_objects_buffer.buffer,
        description_set,
        engine::renderer::DescriptorType::STORAGE_BUFFER,
        game_objects_buffer.buffer->getSize());

    renderer::Helper::addOneBuffer(
        descriptor_writes,
        CAMERA_OBJECT_BUFFER_INDEX,
        game_camera_buffer.buffer,
        description_set,
        engine::renderer::DescriptorType::STORAGE_BUFFER,
        game_camera_buffer.buffer->getSize());

    return descriptor_writes;
}

std::vector<renderer::TextureDescriptor> addTileHeightmapTexture(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer) {
    std::vector<renderer::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(2);

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
        texture_sampler,
        soil_water_layer.view,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

static std::shared_ptr<renderer::DescriptorSetLayout> createUpdateGameCameraDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings(4);
    bindings[0] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        GAME_OBJECTS_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_BUFFER);
    bindings[1] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        CAMERA_OBJECT_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_BUFFER);
    bindings[2] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        ROCK_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[3] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        SOIL_WATER_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));

    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<renderer::PipelineLayout> createGameCameraPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& desc_set_layouts) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::GameCameraParams);

    return device->createPipelineLayout(desc_set_layouts, { push_const_range });
}

static std::shared_ptr<renderer::Pipeline> createGameCameraPipeline(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout) {
    auto update_camera_cs_modules = getComputeShaderModules(device, "update_camera");
    assert(update_camera_cs_modules.size() == 1);

    auto pipeline = device->createPipeline(
        pipeline_layout,
        update_camera_cs_modules[0]);

    for (auto& shader_module : update_camera_cs_modules) {
        device->destroyShaderModule(shader_module);
    }

    return pipeline;
}

} // namespace

namespace game_object {

// static member definition.
std::shared_ptr<renderer::DescriptorSet> GameCamera::update_game_camera_desc_set_[2];
std::shared_ptr<renderer::DescriptorSetLayout> GameCamera::update_game_camera_desc_set_layout_;
std::shared_ptr<renderer::PipelineLayout> GameCamera::update_game_camera_pipeline_layout_;
std::shared_ptr<renderer::Pipeline> GameCamera::update_game_camera_pipeline_;
std::shared_ptr<renderer::BufferInfo> GameCamera::game_camera_buffer_;

GameCamera::GameCamera(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool)
    : device_info_(device_info) {
}

void GameCamera::createGameCameraUpdateDescSet(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer_0,
    const renderer::TextureInfo& soil_water_layer_1,
    const renderer::BufferInfo& game_objects_buffer) {
    // create a global ibl texture descriptor set.
    for (int soil_water = 0; soil_water < 2; soil_water++) {
        // game objects buffer update set.
        update_game_camera_desc_set_[soil_water] =
            device->createDescriptorSets(
                descriptor_pool, update_game_camera_desc_set_layout_, 1)[0];

        assert(game_camera_buffer_);
        auto buffer_descs = addGameCameraInfoBuffer(
            update_game_camera_desc_set_[soil_water],
            game_objects_buffer,
            *game_camera_buffer_);

        auto texture_descs = addTileHeightmapTexture(
            update_game_camera_desc_set_[soil_water],
            texture_sampler,
            rock_layer,
            soil_water == 0 ? soil_water_layer_0 : soil_water_layer_1);

        device->updateDescriptorSets(texture_descs, buffer_descs);
    }
}

void GameCamera::initStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer_0,
    const renderer::TextureInfo& soil_water_layer_1,
    const renderer::BufferInfo& game_objects_buffer) {
    if (!game_camera_buffer_) {
        game_camera_buffer_ = std::make_shared<renderer::BufferInfo>();
        device->createBuffer(
            sizeof(glsl::GameCameraInfo),
            SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
            SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
            game_camera_buffer_->buffer,
            game_camera_buffer_->memory);
    }

    if (update_game_camera_desc_set_layout_ == nullptr) {
        update_game_camera_desc_set_layout_ =
            createUpdateGameCameraDescriptorSetLayout(device);
    }

    createStaticMembers(device);

    createGameCameraUpdateDescSet(
        device,
        descriptor_pool,
        texture_sampler,
        rock_layer,
        soil_water_layer_0,
        soil_water_layer_1,
        game_objects_buffer);
}

void GameCamera::createStaticMembers(
    const std::shared_ptr<renderer::Device>& device) {

    {
        if (update_game_camera_pipeline_layout_) {
            device->destroyPipelineLayout(update_game_camera_pipeline_layout_);
            update_game_camera_pipeline_layout_ = nullptr;
        }

        if (update_game_camera_pipeline_layout_ == nullptr) {
            update_game_camera_pipeline_layout_ =
                createGameCameraPipelineLayout(
                    device,
                    { update_game_camera_desc_set_layout_ });
        }
    }

    {
        if (update_game_camera_pipeline_) {
            device->destroyPipeline(update_game_camera_pipeline_);
            update_game_camera_pipeline_ = nullptr;
        }

        if (update_game_camera_pipeline_ == nullptr) {
            assert(update_game_camera_pipeline_layout_);
            update_game_camera_pipeline_ =
                createGameCameraPipeline(
                    device,
                    update_game_camera_pipeline_layout_);
        }
    }
}

void GameCamera::recreateStaticMembers(
    const std::shared_ptr<renderer::Device>& device) {

    createStaticMembers(device);
}

void GameCamera::generateDescriptorSet(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer_0,
    const renderer::TextureInfo& soil_water_layer_1,
    const renderer::BufferInfo& game_objects_buffer) {

    createGameCameraUpdateDescSet(
        device,
        descriptor_pool,
        texture_sampler,
        rock_layer,
        soil_water_layer_0,
        soil_water_layer_1,
        game_objects_buffer);
}

void GameCamera::destoryStaticMembers(
    const std::shared_ptr<renderer::Device>& device) {
    device->destroyDescriptorSetLayout(update_game_camera_desc_set_layout_);
    device->destroyPipelineLayout(update_game_camera_pipeline_layout_);
    device->destroyPipeline(update_game_camera_pipeline_);
}

void GameCamera::updateGameCameraBuffer(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const glsl::GameCameraParams& game_camera_params,
    int soil_water) {

    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, update_game_camera_pipeline_);

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        update_game_camera_pipeline_layout_,
        &game_camera_params,
        sizeof(game_camera_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        update_game_camera_pipeline_layout_,
        { update_game_camera_desc_set_[soil_water] });

    cmd_buf->dispatch(1, 1);

    cmd_buf->addBufferBarrier(
        game_camera_buffer_->buffer,
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        game_camera_buffer_->buffer->getSize());
}

void GameCamera::update(
    const renderer::DeviceInfo& device_info,
    const float& time) {
}

std::shared_ptr<renderer::BufferInfo> GameCamera::getGameCameraBuffer() {
    return game_camera_buffer_;
}

} // game_object
} // engine