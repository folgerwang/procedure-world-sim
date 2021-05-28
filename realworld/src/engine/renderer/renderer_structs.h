#pragma once
#include <memory>
#include <vector>
#include <optional>
#include <functional>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "renderer_definition.h"

template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace engine {
namespace renderer {
class Instance;
class Device;
class Sampler;
class ImageView;
class Buffer;
class DescriptorSet;
class CommandBuffer;
class Image;
class Pipeline;
class RenderPass;
class Framebuffer;
class PipelineLayout;
class DescriptorSetLayout;
class DescriptorPool;
class DeviceMemory;
class PhysicalDevice;
class ShaderModule;
class Swapchain;
class Surface;
class CommandPool;
class Queue;
class Semaphore;
class Fence;
struct ImageResourceInfo;

struct MemoryRequirements {
    uint64_t        size;
    uint64_t        alignment;
    uint32_t        memory_type_bits;
};

struct BufferCopyInfo {
    uint64_t        src_offset;
    uint64_t        dst_offset;
    uint64_t        size;
};

struct ImageSubresourceLayers {
    ImageAspectFlags      aspect_mask;
    uint32_t              mip_level;
    uint32_t              base_array_layer;
    uint32_t              layer_count;
};

struct BufferImageCopyInfo {
    uint64_t        buffer_offset;
    uint32_t        buffer_row_length;
    uint32_t        buffer_image_height;
    ImageSubresourceLayers    image_subresource;
    glm::ivec3      image_offset;
    glm::uvec3      image_extent;
};

union ClearColorValue {
    float       float32[4];
    int32_t     int32[4];
    uint32_t    uint32[4];
};

struct ClearDepthStencilValue {
    float       depth;
    uint32_t    stencil;
};

union ClearValue {
    ClearColorValue           color;
    ClearDepthStencilValue    depth_stencil;
};

struct VertexInputBindingDescription {
    uint32_t            binding;
    uint32_t            stride;
    VertexInputRate     input_rate;
};

struct VertexInputAttributeDescription {
    uint32_t            buffer_view;
    uint32_t            binding;
    uint32_t            location;
    Format              format;
    uint64_t            offset;
    uint64_t            buffer_offset;
};

struct IndexInputBindingDescription {
    uint32_t            binding;
    uint64_t            offset;
    uint64_t            index_count;
    IndexType           index_type;
};

struct ImageResourceInfo {
    ImageLayout             image_layout = ImageLayout::UNDEFINED;
    AccessFlags             access_flags = 0;
    PipelineStageFlags      stage_flags = 0;;
};

struct BufferResourceInfo {
    AccessFlags             access_flags = 0;
    PipelineStageFlags      stage_flags = 0;
};

struct PipelineInputAssemblyStateCreateInfo {
    PrimitiveTopology                  topology;
    bool                               restart_enable;
};

struct PipelineColorBlendAttachmentState {
    bool                     blend_enable;
    BlendFactor              src_color_blend_factor;
    BlendFactor              dst_color_blend_factor;
    BlendOp                  color_blend_op;
    BlendFactor              src_alpha_blend_factor;
    BlendFactor              dst_alpha_blend_factor;
    BlendOp                  alpha_blend_op;
    ColorComponentFlags      color_write_mask;
};

struct PipelineColorBlendStateCreateInfo {
    bool                                          logic_op_enable;
    LogicOp                                       logic_op;
    uint32_t                                      attachment_count;
    const PipelineColorBlendAttachmentState* attachments;
    glm::vec4                                     blend_constants;
};

struct PipelineRasterizationStateCreateInfo {
    bool                                       depth_clamp_enable;
    bool                                       rasterizer_discard_enable;
    PolygonMode                                polygon_mode;
    CullModeFlags                              cull_mode;
    FrontFace                                  front_face;
    bool                                       depth_bias_enable;
    float                                      depth_bias_constant_factor;
    float                                      depth_bias_clamp;
    float                                      depth_bias_slope_factor;
    float                                      line_width;
};

struct PipelineMultisampleStateCreateInfo {
    SampleCountFlagBits         rasterization_samples;
    bool                        sample_shading_enable;
    float                       min_sample_shading;
    const SampleMask* sample_mask;
    bool                        alpha_to_coverage_enable;
    bool                        alpha_to_one_enable;
};

struct StencilOpState {
    StencilOp       fail_op;
    StencilOp       pass_op;
    StencilOp       depth_fail_op;
    CompareOp       compare_op;
    uint32_t        compare_mask;
    uint32_t        write_mask;
    uint32_t        reference;
};

struct PipelineDepthStencilStateCreateInfo {
    bool                        depth_test_enable;
    bool                        depth_write_enable;
    CompareOp                   depth_compare_op;
    bool                        depth_bounds_test_enable;
    bool                        stencil_test_enable;
    StencilOpState              front;
    StencilOpState              back;
    float                       min_depth_bounds;
    float                       max_depth_bounds;
};

struct AttachmentDescription {
    AttachmentDescriptionFlags    flags;
    Format                        format;
    SampleCountFlagBits           samples;
    AttachmentLoadOp              load_op;
    AttachmentStoreOp             store_op;
    AttachmentLoadOp              stencil_load_op;
    AttachmentStoreOp             stencil_store_op;
    ImageLayout                   initial_layout;
    ImageLayout                   final_layout;
};

struct AttachmentReference {
    AttachmentReference() :
        attachment_(0),
        layout_(ImageLayout::UNDEFINED) {}
    AttachmentReference(uint32_t attachment, ImageLayout layout) :
        attachment_(attachment),
        layout_(layout) {}
    uint32_t                      attachment_;
    ImageLayout                   layout_;
};

struct SubpassDescription {
    SubpassDescriptionFlags         flags;
    PipelineBindPoint               pipeline_bind_point;
    std::vector<AttachmentReference> input_attachments;
    std::vector<AttachmentReference> color_attachments;
    std::vector<AttachmentReference> resolve_attachments;
    std::vector<AttachmentReference> depth_stencil_attachment;
    uint32_t                        preserve_attachment_count;
    const uint32_t* preserve_attachments;
};

struct SubpassDependency {
    uint32_t                src_subpass;
    uint32_t                dst_subpass;
    PipelineStageFlags      src_stage_mask;
    PipelineStageFlags      dst_stage_mask;
    AccessFlags             src_access_mask;
    AccessFlags             dst_access_mask;
    DependencyFlags         dependency_flags;
};

struct DescriptorSetLayoutBinding {
    uint32_t              binding;
    DescriptorType        descriptor_type;
    uint32_t              descriptor_count;
    ShaderStageFlags      stage_flags;
    std::shared_ptr<Sampler> immutable_samplers;
};

struct PushConstantRange {
    ShaderStageFlags      stage_flags;
    uint32_t              offset;
    uint32_t              size;
};

struct TextureDescriptor {
    uint32_t binding;
    const std::shared_ptr<Sampler>& sampler = nullptr;
    const std::shared_ptr<ImageView>& texture = nullptr;
    const std::shared_ptr<DescriptorSet>& desc_set = nullptr;
    DescriptorType desc_type = DescriptorType::COMBINED_IMAGE_SAMPLER;
    ImageLayout image_layout = ImageLayout::SHADER_READ_ONLY_OPTIMAL;
};

struct BufferDescriptor {
    uint32_t binding;
    uint32_t offset;
    uint32_t range;
    const std::shared_ptr<Buffer>& buffer = nullptr;
    const std::shared_ptr<DescriptorSet>& desc_set = nullptr;
    DescriptorType desc_type = DescriptorType::UNIFORM_BUFFER;
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics_family_;
    std::optional<uint32_t> present_family_;

    bool isComplete() const {
        return graphics_family_.has_value() && present_family_.has_value();
    }
};

struct DeviceInfo {
    std::shared_ptr<Device>             device;
    std::shared_ptr<Queue>              cmd_queue;
    std::shared_ptr<CommandPool>        cmd_pool;
};

struct TextureInfo {
    bool                               linear = true;
    std::shared_ptr<Image>             image;
    std::shared_ptr<DeviceMemory>      memory;
    std::shared_ptr<ImageView>         view;
    std::vector<std::vector<std::shared_ptr<ImageView>>> surface_views;
    std::vector<std::shared_ptr<Framebuffer>> framebuffers;

    void destroy(const std::shared_ptr<Device>& device);
};

struct BufferInfo {
    std::shared_ptr<Buffer>             buffer;
    std::shared_ptr<DeviceMemory>       memory;

    void destroy(const std::shared_ptr<Device>& device);
};

//indexed
struct DrawIndexedIndirectCommand {
    uint32_t    index_count;
    uint32_t    instance_count;
    uint32_t    first_index;
    int32_t     vertex_offset;
    uint32_t    first_instance;
};

//non indexed
struct DrawIndirectCommand {
    uint32_t    vertex_count;
    uint32_t    instance_count;
    uint32_t    first_vertex;
    uint32_t    first_instance;
};

struct GraphicPipelineInfo {
    std::shared_ptr<PipelineColorBlendStateCreateInfo> blend_state_info;
    std::shared_ptr<PipelineRasterizationStateCreateInfo> rasterization_info;
    std::shared_ptr<PipelineMultisampleStateCreateInfo> ms_info;
    std::shared_ptr<PipelineDepthStencilStateCreateInfo> depth_stencil_info;
};

typedef std::vector<std::shared_ptr<PhysicalDevice>> PhysicalDeviceList;
typedef std::vector<std::shared_ptr<DescriptorSetLayout>> DescriptorSetLayoutList;
typedef std::vector<std::shared_ptr<DescriptorSet>> DescriptorSetList;
typedef std::vector<std::shared_ptr<ShaderModule>> ShaderModuleList;
} // namespace renderer
} // namespace engine