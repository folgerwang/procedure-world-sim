#include <fstream>

#include "renderer.h"
#include "renderer_helper.h"

namespace engine {
namespace renderer {
namespace helper {

DescriptorSetLayoutBinding getTextureSamplerDescriptionSetLayoutBinding(
    uint32_t binding, 
    ShaderStageFlags stage_flags/* = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT)*/,
    DescriptorType descript_type/* = DescriptorType::COMBINED_IMAGE_SAMPLER*/) {
    DescriptorSetLayoutBinding texture_binding{};
    texture_binding.binding = binding;
    texture_binding.descriptor_count = 1;
    texture_binding.descriptor_type = descript_type;
    texture_binding.immutable_samplers = nullptr;
    texture_binding.stage_flags = stage_flags;

    return texture_binding;
}

DescriptorSetLayoutBinding getBufferDescriptionSetLayoutBinding(
    uint32_t binding,
    ShaderStageFlags stage_flags/* = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT)*/,
    DescriptorType descript_type/* = DescriptorType::STORAGE_BUFFER*/) {
    DescriptorSetLayoutBinding buffer_binding{};
    buffer_binding.binding = binding;
    buffer_binding.descriptor_count = 1;
    buffer_binding.descriptor_type = descript_type;
    buffer_binding.immutable_samplers = nullptr;
    buffer_binding.stage_flags = stage_flags;

    return buffer_binding;
}

PipelineColorBlendAttachmentState fillPipelineColorBlendAttachmentState(
    ColorComponentFlags color_write_mask/* = SET_FLAG_BIT(ColorComponent, ALL_BITS)*/,
    bool blend_enable/* = false*/,
    BlendFactor src_color_blend_factor/* = BlendFactor::ONE*/,
    BlendFactor dst_color_blend_factor/* = BlendFactor::ZERO*/,
    BlendOp color_blend_op/* = BlendOp::ADD*/,
    BlendFactor src_alpha_blend_factor/* = BlendFactor::ONE*/,
    BlendFactor dst_alpha_blend_factor/* = BlendFactor::ZERO*/,
    BlendOp alpha_blend_op/* = BlendOp::ADD*/) {
    PipelineColorBlendAttachmentState color_blend_attachment{};
    color_blend_attachment.color_write_mask = color_write_mask;
    color_blend_attachment.blend_enable = blend_enable;
    color_blend_attachment.src_color_blend_factor = src_color_blend_factor; // Optional
    color_blend_attachment.dst_color_blend_factor = dst_color_blend_factor; // Optional
    color_blend_attachment.color_blend_op = color_blend_op; // Optional
    color_blend_attachment.src_alpha_blend_factor = src_alpha_blend_factor; // Optional
    color_blend_attachment.dst_alpha_blend_factor = dst_alpha_blend_factor; // Optional
    color_blend_attachment.alpha_blend_op = alpha_blend_op; // Optional

    return color_blend_attachment;
}

PipelineColorBlendStateCreateInfo fillPipelineColorBlendStateCreateInfo(
    const std::vector<PipelineColorBlendAttachmentState>& color_blend_attachments,
    bool logic_op_enable/* = false*/,
    LogicOp logic_op/* = LogicOp::NO_OP*/,
    glm::vec4 blend_constants/* = glm::vec4(0.0f)*/) {
    PipelineColorBlendStateCreateInfo color_blending{};
    color_blending.logic_op_enable = logic_op_enable;
    color_blending.logic_op = logic_op; // Optional
    color_blending.attachment_count = static_cast<uint32_t>(color_blend_attachments.size());
    color_blending.attachments = color_blend_attachments.data();
    color_blending.blend_constants = blend_constants; // Optional

    return color_blending;
}

PipelineRasterizationStateCreateInfo fillPipelineRasterizationStateCreateInfo(
    bool depth_clamp_enable/* = false*/,
    bool rasterizer_discard_enable/* = false*/,
    PolygonMode polygon_mode/* = PolygonMode::FILL*/,
    CullModeFlags cull_mode/* = SET_FLAG_BIT(CullMode, BACK_BIT)*/,
    FrontFace front_face/* = FrontFace::COUNTER_CLOCKWISE*/,
    bool  depth_bias_enable/* = false*/,
    float depth_bias_constant_factor/* = 0.0f*/,
    float depth_bias_clamp/* = 0.0f*/,
    float depth_bias_slope_factor/* = 0.0f*/,
    float line_width/* = 1.0f*/) {
    PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.depth_clamp_enable = depth_clamp_enable;
    rasterizer.rasterizer_discard_enable = rasterizer_discard_enable;
    rasterizer.polygon_mode = polygon_mode;
    rasterizer.line_width = line_width;
    rasterizer.cull_mode = cull_mode;
    rasterizer.front_face = front_face;
    rasterizer.depth_bias_enable = depth_bias_enable;
    rasterizer.depth_bias_constant_factor = depth_bias_constant_factor; // Optional
    rasterizer.depth_bias_clamp = depth_bias_clamp; // Optional
    rasterizer.depth_bias_slope_factor = depth_bias_slope_factor; // Optional

    return rasterizer;
}

PipelineMultisampleStateCreateInfo fillPipelineMultisampleStateCreateInfo(
    SampleCountFlagBits rasterization_samples/* = SampleCountFlagBits::SC_1_BIT*/,
    bool sample_shading_enable/* = false*/,
    float min_sample_shading/* = 1.0f*/,
    const SampleMask* sample_mask/* = nullptr*/,
    bool alpha_to_coverage_enable/* = false*/,
    bool alpha_to_one_enable/* = false*/) {
    PipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sample_shading_enable = sample_shading_enable;
    multisampling.rasterization_samples = rasterization_samples;
    multisampling.min_sample_shading = min_sample_shading; // Optional
    multisampling.sample_mask = sample_mask; // Optional
    multisampling.alpha_to_coverage_enable = alpha_to_coverage_enable; // Optional
    multisampling.alpha_to_one_enable = alpha_to_one_enable; // Optional

    return multisampling;
}

StencilOpState fillStencilInfo(
    StencilOp fail_op/* = StencilOp::KEEP*/,
    StencilOp pass_op/* = StencilOp::KEEP*/,
    StencilOp depth_fail_op/* = StencilOp::KEEP*/,
    CompareOp compare_op/* = CompareOp::NEVER*/,
    uint32_t  compare_mask/* = 0xff*/,
    uint32_t  write_mask/* = 0xff*/,
    uint32_t  reference/* = 0x00*/) {
    StencilOpState stencil_op_state;
    stencil_op_state.fail_op = fail_op;
    stencil_op_state.pass_op = pass_op;
    stencil_op_state.depth_fail_op = depth_fail_op;
    stencil_op_state.compare_op = compare_op;
    stencil_op_state.compare_mask = compare_mask;
    stencil_op_state.write_mask = write_mask;
    stencil_op_state.reference = reference;

    return stencil_op_state;
}

PipelineDepthStencilStateCreateInfo fillPipelineDepthStencilStateCreateInfo(
    bool depth_test_enable/* = true*/,
    bool depth_write_enable/* = true*/,
    CompareOp depth_compare_op/* = CompareOp::LESS*/,
    bool depth_bounds_test_enable/* = false*/,
    float min_depth_bounds/* = 0.0f*/,
    float max_depth_bounds/* = 1.0f*/,
    bool stencil_test_enable/* = false*/,
    StencilOpState front/* = {}*/,
    StencilOpState back/* = {}*/) {
    PipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.depth_test_enable = depth_test_enable;
    depth_stencil.depth_write_enable = depth_write_enable;
    depth_stencil.depth_compare_op = depth_compare_op;
    depth_stencil.depth_bounds_test_enable = depth_bounds_test_enable;
    depth_stencil.min_depth_bounds = min_depth_bounds; // Optional
    depth_stencil.max_depth_bounds = max_depth_bounds; // Optional
    depth_stencil.stencil_test_enable = stencil_test_enable;
    depth_stencil.front = front; // Optional
    depth_stencil.back = back; // Optional

    return depth_stencil;
}

} // namespace helper
} // namespace renderer
} // namespace engine
