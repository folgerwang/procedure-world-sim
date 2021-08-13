#include <vector>

#include "engine/renderer/renderer_helper.h"
#include "engine/engine_helper.h"
#include "engine/game_object/terrain.h"
#include "shaders/global_definition.glsl.h"
#include "volume_cloud.h"

namespace {
namespace er = engine::renderer;

er::ShaderModuleList getBlurImageXShaderModules(
    std::shared_ptr<er::Device> device)
{
    uint64_t compute_code_size;
    er::ShaderModuleList shader_modules;
    shader_modules.reserve(1);
    auto compute_shader_code = engine::helper::readFile("lib/shaders/blur_image_x_comp.spv", compute_code_size);
    shader_modules.push_back(device->createShaderModule(compute_code_size, compute_shader_code.data()));

    return shader_modules;
}

er::ShaderModuleList getBlurImageYMergeShaderModules(
    std::shared_ptr<er::Device> device)
{
    uint64_t compute_code_size;
    er::ShaderModuleList shader_modules;
    shader_modules.reserve(1);
    auto compute_shader_code = engine::helper::readFile("lib/shaders/blur_image_y_merge_comp.spv", compute_code_size);
    shader_modules.push_back(device->createShaderModule(compute_code_size, compute_shader_code.data()));

    return shader_modules;
}

er::ShaderModuleList getRenderCloudFogShaderModules(
    std::shared_ptr<er::Device> device)
{
    uint64_t compute_code_size;
    er::ShaderModuleList shader_modules;
    shader_modules.reserve(1);
    auto compute_shader_code = engine::helper::readFile("lib/shaders/render_volume_cloud_comp.spv", compute_code_size);
    shader_modules.push_back(device->createShaderModule(compute_code_size, compute_shader_code.data()));

    return shader_modules;
}

std::vector<er::TextureDescriptor> addBlurImageTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::shared_ptr<er::ImageView>& src_image,
    const std::shared_ptr<er::ImageView>& dst_image,
    const std::shared_ptr<er::ImageView>& depth_image) {
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(2);

    // envmap texture.
    er::Helper::addOneTexture(
        descriptor_writes,
        SRC_TEX_INDEX,
        texture_sampler,
        src_image,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // envmap texture.
    er::Helper::addOneTexture(
        descriptor_writes,
        DST_TEX_INDEX,
        nullptr,
        dst_image,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        SRC_DEPTH_TEX_INDEX,
        texture_sampler,
        depth_image,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

std::vector<er::TextureDescriptor> addCloudFogTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::shared_ptr<er::ImageView>& src_depth,
    const std::shared_ptr<er::ImageView>& cloud_fog_tex,
    const std::shared_ptr<er::ImageView>& volume_moist_tex,
    const std::shared_ptr<er::ImageView>& cloud_lighting_tex,
    const std::shared_ptr<er::ImageView>& scattering_lut_tex) {
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(5);

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

    er::Helper::addOneTexture(
        descriptor_writes,
        SRC_SCATTERING_LUT_INDEX,
        texture_sampler,
        scattering_lut_tex,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        DST_FOG_CLOUD_INDEX,
        texture_sampler,
        cloud_fog_tex,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

std::shared_ptr<er::PipelineLayout>
    createBlurImagePipelineLayout(
        const std::shared_ptr<er::Device>& device,
        const std::shared_ptr<er::DescriptorSetLayout>& desc_set_layout,
        const std::shared_ptr<er::DescriptorSetLayout>& view_desc_set_layout) {
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::BlurImageParams);

    return device->createPipelineLayout(
        { desc_set_layout, view_desc_set_layout },
        { push_const_range });
}

std::shared_ptr<er::Pipeline> createBlurImageXPipeline(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::PipelineLayout>& pipeline_layout) {
    auto cs_modules = getBlurImageXShaderModules(device);
    assert(cs_modules.size() == 1);

    auto pipeline = device->createPipeline(
        pipeline_layout,
        cs_modules[0]);

    for (auto& shader_module : cs_modules) {
        device->destroyShaderModule(shader_module);
    }

    return pipeline;
}

std::shared_ptr<er::Pipeline> createBlurImageYMergePipeline(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::PipelineLayout>& pipeline_layout) {
    auto cs_modules = getBlurImageYMergeShaderModules(device);
    assert(cs_modules.size() == 1);

    auto pipeline = device->createPipeline(
        pipeline_layout,
        cs_modules[0]);

    for (auto& shader_module : cs_modules) {
        device->destroyShaderModule(shader_module);
    }

    return pipeline;
}

std::shared_ptr<er::PipelineLayout> createRenderCloudFogPipelineLayout(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorSetLayout>& desc_set_layout,
    const std::shared_ptr<er::DescriptorSetLayout>& view_desc_set_layout) {
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::VolumeMoistrueParams);

    return device->createPipelineLayout(
        { desc_set_layout , view_desc_set_layout },
        { push_const_range });
}

std::shared_ptr<er::Pipeline> createRenderCloudFogPipeline(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::PipelineLayout>& pipeline_layout) {
    auto cs_modules = getRenderCloudFogShaderModules(device);
    assert(cs_modules.size() == 1);

    auto pipeline = device->createPipeline(
        pipeline_layout,
        cs_modules[0]);

    for (auto& shader_module : cs_modules) {
        device->destroyShaderModule(shader_module);
    }

    return pipeline;
}

} // namespace

namespace engine {
namespace scene_rendering {

VolumeCloud::VolumeCloud(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& src_depth,
    const std::shared_ptr<renderer::ImageView>& hdr_color,
    const std::vector<std::shared_ptr<renderer::ImageView>>& temp_moisture_texes,
    const std::shared_ptr<renderer::ImageView>& cloud_lighting_tex,
    const std::shared_ptr<renderer::ImageView>& scattering_lut_tex,
    const glm::uvec2& display_size) {

    const auto& device = device_info.device;

    blur_image_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_DEPTH_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER) });

    render_cloud_fog_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_TEMP_MOISTURE_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_DEPTH_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_CLOUD_LIGHTING_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_FOG_CLOUD_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_SCATTERING_LUT_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER) });

    recreate(
        device_info,
        descriptor_pool,
        view_desc_set_layout,
        texture_sampler,
        src_depth,
        hdr_color,
        temp_moisture_texes,
        cloud_lighting_tex,
        scattering_lut_tex,
        display_size);
}

void VolumeCloud::recreate(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& src_depth,
    const std::shared_ptr<renderer::ImageView>& hdr_color,
    const std::vector<std::shared_ptr<renderer::ImageView>>& temp_moisture_texes,
    const std::shared_ptr<renderer::ImageView>& cloud_lighting_tex,
    const std::shared_ptr<renderer::ImageView>& scattering_lut_tex,
    const glm::uvec2& display_size) {

    auto& device = device_info.device;
    fog_cloud_tex_.destroy(device);
    blurred_fog_cloud_tex_.destroy(device);

    renderer::Helper::create2DTextureImage(
        device_info,
        renderer::Format::R16G16B16A16_SFLOAT,
        display_size,
        fog_cloud_tex_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::create2DTextureImage(
        device_info,
        renderer::Format::R16G16B16A16_SFLOAT,
        display_size,
        blurred_fog_cloud_tex_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    if (blur_image_pipeline_layout_ != nullptr) {
        device->destroyPipelineLayout(blur_image_pipeline_layout_);
        blur_image_pipeline_layout_ = nullptr;
    }
    
    if (blur_image_x_pipeline_ != nullptr) {
        device->destroyPipeline(blur_image_x_pipeline_);
        blur_image_x_pipeline_ = nullptr;
    }

    if (blur_image_y_merge_pipeline_ != nullptr) {
        device->destroyPipeline(blur_image_y_merge_pipeline_);
        blur_image_y_merge_pipeline_ = nullptr;
    }

    blur_image_x_tex_desc_set_ = nullptr;
    blur_image_y_merge_tex_desc_set_ = nullptr;

    // fog cloud
    blur_image_x_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            blur_image_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto cloud_texture_descs = addBlurImageTextures(
        blur_image_x_tex_desc_set_,
        texture_sampler,
        fog_cloud_tex_.view,
        blurred_fog_cloud_tex_.view,
        src_depth);
    device->updateDescriptorSets(cloud_texture_descs, {});

    blur_image_y_merge_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            blur_image_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    cloud_texture_descs = addBlurImageTextures(
        blur_image_y_merge_tex_desc_set_,
        texture_sampler,
        blurred_fog_cloud_tex_.view,
        hdr_color,
        src_depth);
    device->updateDescriptorSets(cloud_texture_descs, {});

    assert(view_desc_set_layout);
    blur_image_pipeline_layout_ =
        createBlurImagePipelineLayout(
            device,
            blur_image_desc_set_layout_,
            view_desc_set_layout);

    blur_image_x_pipeline_ = createBlurImageXPipeline(
        device,
        blur_image_pipeline_layout_);

    blur_image_y_merge_pipeline_ = createBlurImageYMergePipeline(
        device,
        blur_image_pipeline_layout_);

    for (auto dbuf_idx = 0; dbuf_idx < 2; dbuf_idx++) {
        render_cloud_fog_desc_set_[dbuf_idx] = nullptr;

        render_cloud_fog_desc_set_[dbuf_idx] =
            device->createDescriptorSets(
                descriptor_pool,
                render_cloud_fog_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto render_cloud_fog_texture_descs = addCloudFogTextures(
            render_cloud_fog_desc_set_[dbuf_idx],
            texture_sampler,
            src_depth,
            fog_cloud_tex_.view,
            temp_moisture_texes[dbuf_idx],
            cloud_lighting_tex,
            scattering_lut_tex);
        device->updateDescriptorSets(render_cloud_fog_texture_descs, {});
    }

    render_cloud_fog_pipeline_layout_ =
        createRenderCloudFogPipelineLayout(
            device,
            render_cloud_fog_desc_set_layout_,
            view_desc_set_layout);

    render_cloud_fog_pipeline_ =
        createRenderCloudFogPipeline(
            device,
            render_cloud_fog_pipeline_layout_);
}

void VolumeCloud::renderVolumeCloud(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::DescriptorSet>& frame_desc_set,
    const std::shared_ptr<renderer::Image>& hdr_color,
    const std::shared_ptr<scene_rendering::Skydome>& skydome,
    const float& view_ext_factor,
    const glm::uvec2& display_size,
    int dbuf_idx,
    float current_time) {
    // render moisture volume.
    {
        renderer::helper::transitMapTextureToStoreImage(
            cmd_buf,
            { fog_cloud_tex_.image });

        cmd_buf->bindPipeline(
            renderer::PipelineBindPoint::COMPUTE,
            render_cloud_fog_pipeline_);
        glsl::VolumeMoistrueParams params = {};
        params.world_min = glm::vec2(-kCloudMapSize / 2.0f);
        params.inv_world_range = 1.0f / (glm::vec2(kCloudMapSize / 2.0f) - params.world_min);
        params.size = display_size;
        params.inv_screen_size = 1.0f / glm::vec2(display_size);
        params.time = current_time;
        params.g = skydome->getG();
        params.inv_rayleigh_scale_height = 1.0f / skydome->getRayleighScaleHeight();
        params.inv_mie_scale_height = 1.0f / skydome->getMieScaleHeight();
        params.sun_pos = skydome->getSunDir();
        params.view_ext_factor = view_ext_factor;

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            render_cloud_fog_pipeline_layout_,
            &params,
            sizeof(params));

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            render_cloud_fog_pipeline_layout_,
            { render_cloud_fog_desc_set_[dbuf_idx],
              frame_desc_set });

        cmd_buf->dispatch(
            (display_size.x + 7) / 8,
            (display_size.y + 7) / 8,
            1);

        renderer::helper::transitMapTextureFromStoreImage(
            cmd_buf,
            { fog_cloud_tex_.image });
    }

    {
        renderer::helper::transitMapTextureToStoreImage(
            cmd_buf,
            { blurred_fog_cloud_tex_.image });

        cmd_buf->bindPipeline(
            renderer::PipelineBindPoint::COMPUTE,
            blur_image_x_pipeline_);
        glsl::BlurImageParams params = {};
        params.size = display_size;
        params.inv_size = 1.0f / glm::vec2(display_size);

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            blur_image_pipeline_layout_,
            &params,
            sizeof(params));

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            blur_image_pipeline_layout_,
            { blur_image_x_tex_desc_set_,
              frame_desc_set });

        cmd_buf->dispatch(
            (display_size.x + 63) / 64,
            display_size.y,
            1);

        renderer::helper::transitMapTextureFromStoreImage(
            cmd_buf,
            { blurred_fog_cloud_tex_.image });
    }

    {
        renderer::helper::transitMapTextureToStoreImage(
            cmd_buf,
            { hdr_color },
            renderer::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        cmd_buf->bindPipeline(
            renderer::PipelineBindPoint::COMPUTE,
            blur_image_y_merge_pipeline_);
        glsl::BlurImageParams params = {};
        params.size = display_size;
        params.inv_size = 1.0f / glm::vec2(display_size);

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            blur_image_pipeline_layout_,
            &params,
            sizeof(params));

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            blur_image_pipeline_layout_,
            { blur_image_y_merge_tex_desc_set_,
              frame_desc_set });

        cmd_buf->dispatch(
            display_size.x,
            (display_size.y + 63) / 64,
            1);

        renderer::helper::transitMapTextureFromStoreImage(
            cmd_buf,
            { hdr_color },
            renderer::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }
}

void VolumeCloud::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    fog_cloud_tex_.destroy(device);
    blurred_fog_cloud_tex_.destroy(device);
    device->destroyDescriptorSetLayout(blur_image_desc_set_layout_);
    device->destroyPipelineLayout(blur_image_pipeline_layout_);
    device->destroyPipeline(blur_image_x_pipeline_);
    device->destroyPipeline(blur_image_y_merge_pipeline_);
    device->destroyDescriptorSetLayout(render_cloud_fog_desc_set_layout_);
    device->destroyPipelineLayout(render_cloud_fog_pipeline_layout_);
    device->destroyPipeline(render_cloud_fog_pipeline_);
}

}//namespace scene_rendering
}//namespace engine
