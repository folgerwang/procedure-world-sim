#pragma once
#include <vulkan/vulkan.h>
#include "../command_buffer.h"

namespace engine {
namespace renderer {
namespace vk {

class VulkanCommandBuffer : public CommandBuffer {
    VkCommandBuffer    cmd_buf_;
public:
    VkCommandBuffer get() { return cmd_buf_; }
    void set(const VkCommandBuffer& cmd_buf) { cmd_buf_ = cmd_buf; }

    virtual void beginCommandBuffer(CommandBufferUsageFlags flags) final;
    virtual void endCommandBuffer() final;
    virtual void copyBuffer(std::shared_ptr<Buffer> src_buf, std::shared_ptr<Buffer> dst_buf, std::vector<BufferCopyInfo> copy_regions) final;
    virtual void copyBufferToImage(
        std::shared_ptr<Buffer> src_buf,
        std::shared_ptr<Image> dst_image,
        std::vector<BufferImageCopyInfo> copy_regions,
        renderer::ImageLayout layout) final;
    virtual void bindPipeline(PipelineBindPoint bind, std::shared_ptr< Pipeline> pipeline) final;
    virtual void bindVertexBuffers(uint32_t first_bind, const std::vector<std::shared_ptr<renderer::Buffer>>& vertex_buffers, std::vector<uint64_t> offsets) final;
    virtual void bindIndexBuffer(std::shared_ptr<Buffer> index_buffer, uint64_t offset, IndexType index_type) final;
    virtual void bindDescriptorSets(
        PipelineBindPoint bind_point,
        const std::shared_ptr<PipelineLayout>& pipeline_layout,
        const DescriptorSetList& desc_sets) final;
    virtual void pushConstants(
        ShaderStageFlags stages,
        const std::shared_ptr<PipelineLayout>& pipeline_layout,
        const void* data,
        uint32_t size,
        uint32_t offset = 0) final;
    virtual void draw(uint32_t vertex_count, 
        uint32_t instance_count = 1, 
        uint32_t first_vertex = 0, 
        uint32_t first_instance = 0) final;
    virtual void drawIndexed(
        uint32_t index_count, 
        uint32_t instance_count = 1, 
        uint32_t first_index = 0, 
        uint32_t vertex_offset = 0, 
        uint32_t first_instance = 0) final;
    virtual void drawIndexedIndirect(
        const renderer::BufferInfo& indirect_draw_cmd_buf,
        uint32_t buffer_offset = 0,
        uint32_t draw_count = 1,
        uint32_t stride = sizeof(DrawIndexedIndirectCommand)) final;
    virtual void drawIndirect(
        const renderer::BufferInfo& indirect_draw_cmd_buf,
        uint32_t buffer_offset = 0,
        uint32_t draw_count = 1,
        uint32_t stride = sizeof(DrawIndirectCommand)) final;
    virtual void dispatch(
        uint32_t group_count_x, 
        uint32_t group_count_y, 
        uint32_t group_count_z = 1) final;
    virtual void beginRenderPass(
        std::shared_ptr<RenderPass> render_pass,
        std::shared_ptr<Framebuffer> frame_buffer,
        const glm::uvec2& extent,
        const std::vector<ClearValue>& clear_values) final;
    virtual void endRenderPass() final;
    virtual void reset(uint32_t flags) final;
    virtual void addImageBarrier(
        const std::shared_ptr<Image>& image,
        const ImageResourceInfo& src_info,
        const ImageResourceInfo& dst_info,
        uint32_t base_mip = 0,
        uint32_t mip_count = 1,
        uint32_t base_layer = 0,
        uint32_t layer_count = 1) final;
    virtual void addBufferBarrier(
        const std::shared_ptr<Buffer>& buffer,
        const BufferResourceInfo& src_info,
        const BufferResourceInfo& dst_info,
        uint32_t size = 0,
        uint32_t offset = 0) final;
    //virtual void pipelineBarrier() final;
};

} // namespace vk
} // namespace renderer
} // namespace engine