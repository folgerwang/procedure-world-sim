#include <vector>

#include "engine/renderer/renderer.h"
#include "engine/renderer/renderer_helper.h"
#include "engine/engine_helper.h"
#include "shaders/global_definition.glsl.h"
#include "ibl_creator.h"

namespace {
namespace er = engine::renderer;
er::ShaderModuleList getIblShaderModules(
    const std::shared_ptr<er::Device>& device)
{
    uint64_t vert_code_size, frag_code_size;
    er::ShaderModuleList shader_modules;
    shader_modules.reserve(6);
    auto vert_shader_code = engine::helper::readFile("lib/shaders/ibl_vert.spv", vert_code_size);
    shader_modules.push_back(device->createShaderModule(vert_code_size, vert_shader_code.data()));
    auto frag_shader_code = engine::helper::readFile("lib/shaders/panorama_to_cubemap_frag.spv", frag_code_size);
    shader_modules.push_back(device->createShaderModule(frag_code_size, frag_shader_code.data()));
    auto labertian_frag_shader_code = engine::helper::readFile("lib/shaders/ibl_labertian_frag.spv", frag_code_size);
    shader_modules.push_back(device->createShaderModule(frag_code_size, labertian_frag_shader_code.data()));
    auto ggx_frag_shader_code = engine::helper::readFile("lib/shaders/ibl_ggx_frag.spv", frag_code_size);
    shader_modules.push_back(device->createShaderModule(frag_code_size, ggx_frag_shader_code.data()));
    auto charlie_frag_shader_code = engine::helper::readFile("lib/shaders/ibl_charlie_frag.spv", frag_code_size);
    shader_modules.push_back(device->createShaderModule(frag_code_size, charlie_frag_shader_code.data()));

    return shader_modules;
}

er::ShaderModuleList getIblComputeShaderModules(
    const std::shared_ptr<er::Device>& device)
{
    uint64_t compute_code_size;
    er::ShaderModuleList shader_modules;
    shader_modules.reserve(1);
    auto compute_shader_code = engine::helper::readFile("lib/shaders/ibl_smooth_comp.spv", compute_code_size);
    shader_modules.push_back(device->createShaderModule(compute_code_size, compute_shader_code.data()));

    return shader_modules;
}

std::shared_ptr<er::PipelineLayout> createCubemapPipelineLayout(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorSetLayout>& ibl_desc_set_layout) {
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::IblParams);

    return device->createPipelineLayout(
        { ibl_desc_set_layout },
        { push_const_range });
}

std::vector<er::TextureDescriptor> addPanoramaTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const er::TextureInfo& panorama_tex) {
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(1);

    // envmap texture.
    er::Helper::addOneTexture(
        descriptor_writes,
        PANORAMA_TEX_INDEX,
        texture_sampler,
        panorama_tex.view,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

std::vector<er::TextureDescriptor> addIblTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const er::TextureInfo& rt_envmap_tex)
{
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(1);

    er::Helper::addOneTexture(
        descriptor_writes,
        ENVMAP_TEX_INDEX,
        texture_sampler,
        rt_envmap_tex.view,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

std::vector<er::TextureDescriptor> addIblComputeTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const er::TextureInfo& src_tex,
    const er::TextureInfo& dst_tex)
{
    std::vector<er::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(2);

    er::Helper::addOneTexture(
        descriptor_writes,
        SRC_TEX_INDEX,
        texture_sampler,
        src_tex.view,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        DST_TEX_INDEX,
        texture_sampler,
        dst_tex.view,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

std::shared_ptr<er::PipelineLayout> createCubemapComputePipelineLayout(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorSetLayout>& ibl_comp_desc_set_layout)
{
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::IblComputeParams);

    er::DescriptorSetLayoutList desc_set_layouts(1);
    desc_set_layouts[0] = ibl_comp_desc_set_layout;

    return device->createPipelineLayout(
        desc_set_layouts,
        { push_const_range });
}

std::shared_ptr<er::Pipeline> createComputePipeline(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::PipelineLayout>& ibl_comp_pipeline_layout) {
    auto ibl_compute_shader_modules = getIblComputeShaderModules(device);
    assert(ibl_compute_shader_modules.size() == 1);

    auto blur_comp_pipeline = device->createPipeline(
        ibl_comp_pipeline_layout,
        ibl_compute_shader_modules[0]);

    for (auto& shader_module : ibl_compute_shader_modules) {
        device->destroyShaderModule(shader_module);
    }

    return blur_comp_pipeline;
}

}

namespace engine {
namespace scene_rendering {

IblCreator::IblCreator(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const renderer::GraphicPipelineInfo& cube_graphic_pipeline_info,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const uint32_t& cube_size) {

    createCubeTextures(
        device_info,
        cube_render_pass,
        cube_size);

    auto format = er::Format::R8G8B8A8_UNORM;
    helper::createTextureImage(
        device_info,
        "assets/environments/doge2.hdr",
        format,
        panorama_tex_);

    // ibl texture descriptor set layout.
    {
        std::vector<er::DescriptorSetLayoutBinding> bindings(1);
        bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(PANORAMA_TEX_INDEX);
        //bindings[1] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(ENVMAP_TEX_INDEX);

        ibl_desc_set_layout_ =
            device_info.device->createDescriptorSetLayout(bindings);
    }

    // ibl compute texture descriptor set layout.
    {
        std::vector<er::DescriptorSetLayoutBinding> bindings(2);
        bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(SRC_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);
        bindings[1] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(DST_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);

        ibl_comp_desc_set_layout_ =
            device_info.device->createDescriptorSetLayout(bindings);
    }

    createDescriptorSets(
        device_info.device,
        descriptor_pool,
        texture_sampler);

    ibl_comp_pipeline_layout_ = createCubemapComputePipelineLayout(
        device_info.device,
        ibl_comp_desc_set_layout_);

    blur_comp_pipeline_ = createComputePipeline(
        device_info.device,
        ibl_comp_pipeline_layout_);

    ibl_pipeline_layout_ = createCubemapPipelineLayout(
        device_info.device,
        ibl_desc_set_layout_);

    createIblGraphicsPipelines(
        device_info.device,
        cube_render_pass,
        cube_graphic_pipeline_info,
        cube_size);
}

void IblCreator::createCubeTextures(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const uint32_t& cube_size) {
    uint32_t num_mips = static_cast<uint32_t>(std::log2(cube_size) + 1);
    std::vector<renderer::BufferImageCopyInfo> dump_copies;

    renderer::Helper::createCubemapTexture(
        device_info,
        cube_render_pass,
        cube_size,
        cube_size,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_envmap_tex_);

    renderer::Helper::createCubemapTexture(
        device_info,
        cube_render_pass,
        cube_size,
        cube_size,
        1,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        tmp_ibl_diffuse_tex_);

    renderer::Helper::createCubemapTexture(
        device_info,
        cube_render_pass,
        cube_size,
        cube_size,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        tmp_ibl_specular_tex_);

    renderer::Helper::createCubemapTexture(
        device_info,
        cube_render_pass,
        cube_size,
        cube_size,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        tmp_ibl_sheen_tex_);

    renderer::Helper::createCubemapTexture(
        device_info,
        cube_render_pass,
        cube_size,
        cube_size,
        1,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_ibl_diffuse_tex_);

    renderer::Helper::createCubemapTexture(
        device_info,
        cube_render_pass,
        cube_size,
        cube_size,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_ibl_specular_tex_);

    renderer::Helper::createCubemapTexture(
        device_info,
        cube_render_pass,
        cube_size,
        cube_size,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_ibl_sheen_tex_);
}

void IblCreator::createIblGraphicsPipelines(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const renderer::GraphicPipelineInfo& cube_graphic_pipeline_info,
    const uint32_t& cube_size) {
    er::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = er::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;
    {
        auto ibl_shader_modules = getIblShaderModules(device);

        envmap_pipeline_ = device->createPipeline(
            cube_render_pass,
            ibl_pipeline_layout_,
            {}, {},
            input_assembly,
            cube_graphic_pipeline_info,
            { ibl_shader_modules[0], ibl_shader_modules[1] },
            glm::uvec2(cube_size, cube_size));

        lambertian_pipeline_ = device->createPipeline(
            cube_render_pass,
            ibl_pipeline_layout_,
            {}, {},
            input_assembly,
            cube_graphic_pipeline_info,
            { ibl_shader_modules[0], ibl_shader_modules[2] },
            glm::uvec2(cube_size, cube_size));

        ggx_pipeline_ = device->createPipeline(
            cube_render_pass,
            ibl_pipeline_layout_,
            {}, {},
            input_assembly,
            cube_graphic_pipeline_info,
            { ibl_shader_modules[0], ibl_shader_modules[3] },
            glm::uvec2(cube_size, cube_size));

        charlie_pipeline_ = device->createPipeline(
            cube_render_pass,
            ibl_pipeline_layout_,
            {}, {},
            input_assembly,
            cube_graphic_pipeline_info,
            { ibl_shader_modules[0], ibl_shader_modules[4] },
            glm::uvec2(cube_size, cube_size));

        for (auto& shader_module : ibl_shader_modules) {
            device->destroyShaderModule(shader_module);
        }
    }
}

void IblCreator::createDescriptorSets(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler) {
    // envmap
    {
        // only one descriptor layout.
        envmap_tex_desc_set_ =
            device->createDescriptorSets(
                descriptor_pool, ibl_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs = addPanoramaTextures(
            envmap_tex_desc_set_,
            texture_sampler,
            panorama_tex_);
        device->updateDescriptorSets(ibl_texture_descs, {});
    }

    // ibl
    {
        // only one descriptor layout.
        ibl_tex_desc_set_ = device->createDescriptorSets(
            descriptor_pool, ibl_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs = addIblTextures(
            ibl_tex_desc_set_,
            texture_sampler,
            rt_envmap_tex_);
        device->updateDescriptorSets(ibl_texture_descs, {});
    }

    // ibl diffuse compute
    {
        // only one descriptor layout.
        ibl_diffuse_tex_desc_set_ = device->createDescriptorSets(
            descriptor_pool, ibl_comp_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs = addIblComputeTextures(
            ibl_diffuse_tex_desc_set_,
            texture_sampler,
            tmp_ibl_diffuse_tex_,
            rt_ibl_diffuse_tex_);
        device->updateDescriptorSets(ibl_texture_descs, {});
    }

    // ibl specular compute
    {
        // only one descriptor layout.
        ibl_specular_tex_desc_set_ = device->createDescriptorSets(
            descriptor_pool, ibl_comp_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs = addIblComputeTextures(
            ibl_specular_tex_desc_set_,
            texture_sampler,
            tmp_ibl_specular_tex_,
            rt_ibl_specular_tex_);
        device->updateDescriptorSets(ibl_texture_descs, {});
    }

    // ibl sheen compute
    {
        // only one descriptor layout.
        ibl_sheen_tex_desc_set_ = device->createDescriptorSets(
            descriptor_pool, ibl_comp_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs = addIblComputeTextures(
            ibl_sheen_tex_desc_set_,
            texture_sampler,
            tmp_ibl_sheen_tex_,
            rt_ibl_sheen_tex_);
        device->updateDescriptorSets(ibl_texture_descs, {});
    }
}

void IblCreator::addToGlobalTextures(
    std::vector<renderer::TextureDescriptor>& descriptor_writes,
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const std::shared_ptr<renderer::Sampler>& texture_sampler) {
    er::Helper::addOneTexture(
        descriptor_writes,
        LAMBERTIAN_ENV_TEX_INDEX,
        texture_sampler,
        rt_ibl_diffuse_tex_.view,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(
        descriptor_writes,
        GGX_ENV_TEX_INDEX,
        texture_sampler,
        rt_ibl_specular_tex_.view,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(
        descriptor_writes,
        CHARLIE_ENV_TEX_INDEX,
        texture_sampler,
        rt_ibl_sheen_tex_.view,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
}

// generate envmap cubemap from panorama hdr image.
void IblCreator::drawEnvmapFromPanoramaImage(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const std::vector<er::ClearValue>& clear_values,
    const uint32_t& cube_size) {

    cmd_buf->addImageBarrier(
        rt_envmap_tex_.image,
        er::Helper::getImageAsSource(),
        er::Helper::getImageAsColorAttachment(),
        0, 1, 0, 6);

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::GRAPHICS,
        envmap_pipeline_);

    std::vector<er::ClearValue> envmap_clear_values(6, clear_values[0]);
    cmd_buf->beginRenderPass(
        cube_render_pass,
        rt_envmap_tex_.framebuffers[0],
        glm::uvec2(cube_size, cube_size),
        envmap_clear_values);

    glsl::IblParams ibl_params = {};
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        ibl_pipeline_layout_,
        &ibl_params,
        sizeof(ibl_params));

    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::GRAPHICS,
        ibl_pipeline_layout_,
        { envmap_tex_desc_set_ });

    cmd_buf->draw(3);

    cmd_buf->endRenderPass();

    uint32_t num_mips = static_cast<uint32_t>(std::log2(cube_size) + 1);

    er::Helper::generateMipmapLevels(
        cmd_buf,
        rt_envmap_tex_.image,
        num_mips,
        cube_size,
        cube_size,
        er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
}

// generate ibl diffuse texture.
void IblCreator::createIblDiffuseMap(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const std::vector<er::ClearValue>& clear_values,
    const uint32_t& cube_size)
{
    cmd_buf->bindPipeline(
        er::PipelineBindPoint::GRAPHICS,
        lambertian_pipeline_);

    cmd_buf->addImageBarrier(
        rt_ibl_diffuse_tex_.image,
        er::Helper::getImageAsSource(),
        er::Helper::getImageAsColorAttachment(),
        0, 1, 0, 6);

    std::vector<er::ClearValue> envmap_clear_values(6, clear_values[0]);
    cmd_buf->beginRenderPass(cube_render_pass,
        rt_ibl_diffuse_tex_.framebuffers[0],
        glm::uvec2(cube_size, cube_size),
        envmap_clear_values);

    glsl::IblParams ibl_params = {};
    ibl_params.roughness = 1.0f;
    ibl_params.currentMipLevel = 0;
    ibl_params.width = cube_size;
    ibl_params.lodBias = 0;
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        ibl_pipeline_layout_,
        &ibl_params,
        sizeof(ibl_params));

    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::GRAPHICS,
        ibl_pipeline_layout_,
        { ibl_tex_desc_set_ });

    cmd_buf->draw(3);

    cmd_buf->endRenderPass();

    cmd_buf->addImageBarrier(
        rt_ibl_diffuse_tex_.image,
        er::Helper::getImageAsColorAttachment(),
        er::Helper::getImageAsShaderSampler(),
        0, 1, 0, 6);
}

// generate ibl specular texture.
void IblCreator::createIblSpecularMap(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const std::vector<er::ClearValue>& clear_values,
    const uint32_t& cube_size)
{
    uint32_t num_mips = static_cast<uint32_t>(std::log2(cube_size) + 1);
    cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS, ggx_pipeline_);

    for (int i_mip = num_mips - 1; i_mip >= 0; i_mip--) {
        cmd_buf->addImageBarrier(
            rt_ibl_specular_tex_.image,
            er::Helper::getImageAsSource(),
            er::Helper::getImageAsColorAttachment(),
            i_mip, 1, 0, 6);

        uint32_t width = std::max(static_cast<uint32_t>(cube_size) >> i_mip, 1u);
        uint32_t height = std::max(static_cast<uint32_t>(cube_size) >> i_mip, 1u);

        std::vector<er::ClearValue> envmap_clear_values(6, clear_values[0]);
        cmd_buf->beginRenderPass(
            cube_render_pass,
            rt_ibl_specular_tex_.framebuffers[i_mip],
            glm::uvec2(width, height),
            envmap_clear_values);

        glsl::IblParams ibl_params = {};
        ibl_params.roughness = static_cast<float>(i_mip) / static_cast<float>(num_mips - 1);
        ibl_params.currentMipLevel = i_mip;
        ibl_params.width = cube_size;
        ibl_params.lodBias = 0;
        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
            ibl_pipeline_layout_,
            &ibl_params,
            sizeof(ibl_params));

        cmd_buf->bindDescriptorSets(
            er::PipelineBindPoint::GRAPHICS,
            ibl_pipeline_layout_,
            { ibl_tex_desc_set_ });

        cmd_buf->draw(3);

        cmd_buf->endRenderPass();
    }

    cmd_buf->addImageBarrier(
        rt_ibl_specular_tex_.image,
        er::Helper::getImageAsColorAttachment(),
        er::Helper::getImageAsShaderSampler(),
        0, num_mips, 0, 6);
}

// generate ibl sheen texture.
void IblCreator::createIblSheenMap(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const std::vector<er::ClearValue>& clear_values,
    const uint32_t& cube_size) {
    uint32_t num_mips = static_cast<uint32_t>(std::log2(cube_size) + 1);
    cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS, charlie_pipeline_);

    for (int i_mip = num_mips - 1; i_mip >= 0; i_mip--) {
        uint32_t width = std::max(static_cast<uint32_t>(cube_size) >> i_mip, 1u);
        uint32_t height = std::max(static_cast<uint32_t>(cube_size) >> i_mip, 1u);

        cmd_buf->addImageBarrier(
            rt_ibl_sheen_tex_.image,
            er::Helper::getImageAsSource(),
            er::Helper::getImageAsColorAttachment(),
            i_mip, 1, 0, 6);

        std::vector<er::ClearValue> envmap_clear_values(6, clear_values[0]);
        cmd_buf->beginRenderPass(
            cube_render_pass,
            rt_ibl_sheen_tex_.framebuffers[i_mip],
            glm::uvec2(width, height),
            envmap_clear_values);

        glsl::IblParams ibl_params = {};
        ibl_params.roughness = static_cast<float>(i_mip) / static_cast<float>(num_mips - 1);
        ibl_params.currentMipLevel = i_mip;
        ibl_params.width = cube_size;
        ibl_params.lodBias = 0;
        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
            ibl_pipeline_layout_,
            &ibl_params,
            sizeof(ibl_params));

        cmd_buf->bindDescriptorSets(
            er::PipelineBindPoint::GRAPHICS,
            ibl_pipeline_layout_,
            { ibl_tex_desc_set_ });

        cmd_buf->draw(3);

        cmd_buf->endRenderPass();
    }

    cmd_buf->addImageBarrier(
        rt_ibl_sheen_tex_.image,
        er::Helper::getImageAsColorAttachment(),
        er::Helper::getImageAsShaderSampler(),
        0, num_mips, 0, 6);
}

void IblCreator::blurIblMaps(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const uint32_t& cube_size) {

    cmd_buf->addImageBarrier(
        rt_ibl_diffuse_tex_.image,
        er::Helper::getImageAsSource(),
        er::Helper::getImageAsStore(),
        0, 1, 0, 6);

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::COMPUTE,
        blur_comp_pipeline_);

    glsl::IblComputeParams ibl_comp_params = {};
    ibl_comp_params.size = glm::ivec4(cube_size, cube_size, 0, 0);
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        ibl_comp_pipeline_layout_,
        &ibl_comp_params,
        sizeof(ibl_comp_params));

    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::COMPUTE,
        ibl_comp_pipeline_layout_,
        { ibl_diffuse_tex_desc_set_ });

    cmd_buf->dispatch((cube_size + 7) / 8, (cube_size + 7) / 8, 6);

    uint32_t num_mips = static_cast<uint32_t>(std::log2(cube_size) + 1);
    cmd_buf->addImageBarrier(
        rt_ibl_diffuse_tex_.image,
        er::Helper::getImageAsStore(),
        er::Helper::getImageAsShaderSampler(),
        0, 1, 0, 6);
    cmd_buf->addImageBarrier(
        rt_ibl_specular_tex_.image,
        er::Helper::getImageAsSource(),
        er::Helper::getImageAsShaderSampler(),
        0, num_mips, 0, 6);
    cmd_buf->addImageBarrier(
        rt_ibl_sheen_tex_.image,
        er::Helper::getImageAsSource(),
        er::Helper::getImageAsShaderSampler(),
        0, num_mips, 0, 6);
}

void IblCreator::recreate(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler) {

    createDescriptorSets(
        device,
        descriptor_pool,
        texture_sampler);
}

void IblCreator::destroy(
    const std::shared_ptr<renderer::Device>& device)
{
    device->destroyPipeline(blur_comp_pipeline_);
    device->destroyPipelineLayout(ibl_comp_pipeline_layout_);

    device->destroyPipeline(envmap_pipeline_);
    device->destroyPipeline(lambertian_pipeline_);
    device->destroyPipeline(ggx_pipeline_);
    device->destroyPipeline(charlie_pipeline_);
    device->destroyPipelineLayout(ibl_pipeline_layout_);

    device->destroyDescriptorSetLayout(ibl_desc_set_layout_);
    device->destroyDescriptorSetLayout(ibl_comp_desc_set_layout_);

    rt_envmap_tex_.destroy(device);
    panorama_tex_.destroy(device);
    rt_envmap_tex_.destroy(device);
    tmp_ibl_diffuse_tex_.destroy(device);
    tmp_ibl_specular_tex_.destroy(device);
    tmp_ibl_sheen_tex_.destroy(device);
    rt_ibl_diffuse_tex_.destroy(device);
    rt_ibl_specular_tex_.destroy(device);
    rt_ibl_sheen_tex_.destroy(device);

}

}//namespace scene_rendering
}//namespace engine