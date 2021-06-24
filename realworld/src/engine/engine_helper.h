#pragma once
#include <vector>
#include "engine/renderer/renderer.h"

namespace engine {
namespace helper {

std::vector<uint64_t> readFile(
    const std::string& file_name,
    uint64_t& file_size);

void createTextureImage(
    const renderer::DeviceInfo& device_info,
    const std::string& file_name,
    renderer::Format format,
    renderer::TextureInfo& texture);

void loadMtx2Texture(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::RenderPass>& cubemap_render_pass,
    const std::string& input_filename,
    renderer::TextureInfo& texture);

} // namespace helper
} // namespace engine
