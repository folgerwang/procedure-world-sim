#include <stdexcept>
#include <iostream>
#include <vector>
#include <map>
#include <array>
#include <optional>
#include <set>
#include <limits>
#include <algorithm>
#include <fstream>
#include <chrono>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"
#include "tiny_mtx2.h"

#include "application.h"

namespace {
constexpr int kWindowSizeX = 1920;
constexpr int kWindowSizeY = 1080;

static uint32_t max_vertex_input_attribute_offset = 0;

struct SkyBoxVertex {
    glm::vec3 pos;

    static std::vector<work::renderer::VertexInputBindingDescription> getBindingDescription() {
        std::vector<work::renderer::VertexInputBindingDescription> binding_description(1);
        binding_description[0].binding = 0;
        binding_description[0].stride = sizeof(SkyBoxVertex);
        binding_description[0].input_rate = work::renderer::VertexInputRate::VERTEX;
        return binding_description;
    }

    static std::vector<work::renderer::VertexInputAttributeDescription> getAttributeDescriptions() {
        std::vector<work::renderer::VertexInputAttributeDescription> attribute_descriptions(1);
        attribute_descriptions[0].binding = 0;
        attribute_descriptions[0].location = 0;
        attribute_descriptions[0].format = work::renderer::Format::R32G32B32_SFLOAT;
        attribute_descriptions[0].offset = offsetof(SkyBoxVertex, pos);
        return attribute_descriptions;
    }
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics_family_;
    std::optional<uint32_t> present_family_;

    bool isComplete() {
        return graphics_family_.has_value() && present_family_.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities_;
    std::vector<VkSurfaceFormatKHR> formats_;
    std::vector<VkPresentModeKHR> present_modes_;
};

const std::vector<const char*> validation_layers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> device_extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enable_validation_layers = false;
#else
const bool enable_validation_layers = true;
#endif

VkFormat toVkFormat(work::renderer::Format format) {
    if (format == work::renderer::Format::R4G4_UNORM_PACK8) return VK_FORMAT_R4G4_UNORM_PACK8;
    if (format == work::renderer::Format::R4G4B4A4_UNORM_PACK16) return VK_FORMAT_R4G4B4A4_UNORM_PACK16;
    if (format == work::renderer::Format::B4G4R4A4_UNORM_PACK16) return VK_FORMAT_B4G4R4A4_UNORM_PACK16;
    if (format == work::renderer::Format::R5G6B5_UNORM_PACK16) return VK_FORMAT_R5G6B5_UNORM_PACK16;
    if (format == work::renderer::Format::B5G6R5_UNORM_PACK16) return VK_FORMAT_B5G6R5_UNORM_PACK16;
    if (format == work::renderer::Format::R5G5B5A1_UNORM_PACK16) return VK_FORMAT_R5G5B5A1_UNORM_PACK16;
    if (format == work::renderer::Format::B5G5R5A1_UNORM_PACK16) return VK_FORMAT_B5G5R5A1_UNORM_PACK16;
    if (format == work::renderer::Format::A1R5G5B5_UNORM_PACK16) return VK_FORMAT_A1R5G5B5_UNORM_PACK16;
    if (format == work::renderer::Format::R8_UNORM) return VK_FORMAT_R8_UNORM;
    if (format == work::renderer::Format::R8_SNORM) return VK_FORMAT_R8_SNORM;
    if (format == work::renderer::Format::R8_USCALED) return VK_FORMAT_R8_USCALED;
    if (format == work::renderer::Format::R8_SSCALED) return VK_FORMAT_R8_SSCALED;
    if (format == work::renderer::Format::R8_UINT) return VK_FORMAT_R8_UINT;
    if (format == work::renderer::Format::R8_SINT) return VK_FORMAT_R8_SINT;
    if (format == work::renderer::Format::R8_SRGB) return VK_FORMAT_R8_SRGB;
    if (format == work::renderer::Format::R8G8_UNORM) return VK_FORMAT_R8G8_UNORM;
    if (format == work::renderer::Format::R8G8_SNORM) return VK_FORMAT_R8G8_SNORM;
    if (format == work::renderer::Format::R8G8_USCALED) return VK_FORMAT_R8G8_USCALED;
    if (format == work::renderer::Format::R8G8_SSCALED) return VK_FORMAT_R8G8_SSCALED;
    if (format == work::renderer::Format::R8G8_UINT) return VK_FORMAT_R8G8_UINT;
    if (format == work::renderer::Format::R8G8_SINT) return VK_FORMAT_R8G8_SINT;
    if (format == work::renderer::Format::R8G8_SRGB) return VK_FORMAT_R8G8_SRGB;
    if (format == work::renderer::Format::R8G8B8_UNORM) return VK_FORMAT_R8G8B8_UNORM;
    if (format == work::renderer::Format::R8G8B8_SNORM) return VK_FORMAT_R8G8B8_SNORM;
    if (format == work::renderer::Format::R8G8B8_USCALED) return VK_FORMAT_R8G8B8_USCALED;
    if (format == work::renderer::Format::R8G8B8_SSCALED) return VK_FORMAT_R8G8B8_SSCALED;
    if (format == work::renderer::Format::R8G8B8_UINT) return VK_FORMAT_R8G8B8_UINT;
    if (format == work::renderer::Format::R8G8B8_SINT) return VK_FORMAT_R8G8B8_SINT;
    if (format == work::renderer::Format::R8G8B8_SRGB) return VK_FORMAT_R8G8B8_SRGB;
    if (format == work::renderer::Format::B8G8R8_UNORM) return VK_FORMAT_B8G8R8_UNORM;
    if (format == work::renderer::Format::B8G8R8_SNORM) return VK_FORMAT_B8G8R8_SNORM;
    if (format == work::renderer::Format::B8G8R8_USCALED) return VK_FORMAT_B8G8R8_USCALED;
    if (format == work::renderer::Format::B8G8R8_SSCALED) return VK_FORMAT_B8G8R8_SSCALED;
    if (format == work::renderer::Format::B8G8R8_UINT) return VK_FORMAT_B8G8R8_UINT;
    if (format == work::renderer::Format::B8G8R8_SINT) return VK_FORMAT_B8G8R8_SINT;
    if (format == work::renderer::Format::B8G8R8_SRGB) return VK_FORMAT_B8G8R8_SRGB;
    if (format == work::renderer::Format::R8G8B8A8_UNORM) return VK_FORMAT_R8G8B8A8_UNORM;
    if (format == work::renderer::Format::R8G8B8A8_SNORM) return VK_FORMAT_R8G8B8A8_SNORM;
    if (format == work::renderer::Format::R8G8B8A8_USCALED) return VK_FORMAT_R8G8B8A8_USCALED;
    if (format == work::renderer::Format::R8G8B8A8_SSCALED) return VK_FORMAT_R8G8B8A8_SSCALED;
    if (format == work::renderer::Format::R8G8B8A8_UINT) return VK_FORMAT_R8G8B8A8_UINT;
    if (format == work::renderer::Format::R8G8B8A8_SINT) return VK_FORMAT_R8G8B8A8_SINT;
    if (format == work::renderer::Format::R8G8B8A8_SRGB) return VK_FORMAT_R8G8B8A8_SRGB;
    if (format == work::renderer::Format::B8G8R8A8_UNORM) return VK_FORMAT_B8G8R8A8_UNORM;
    if (format == work::renderer::Format::B8G8R8A8_SNORM) return VK_FORMAT_B8G8R8A8_SNORM;
    if (format == work::renderer::Format::B8G8R8A8_USCALED) return VK_FORMAT_B8G8R8A8_USCALED;
    if (format == work::renderer::Format::B8G8R8A8_SSCALED) return VK_FORMAT_B8G8R8A8_SSCALED;
    if (format == work::renderer::Format::B8G8R8A8_UINT) return VK_FORMAT_B8G8R8A8_UINT;
    if (format == work::renderer::Format::B8G8R8A8_SINT) return VK_FORMAT_B8G8R8A8_SINT;
    if (format == work::renderer::Format::B8G8R8A8_SRGB) return VK_FORMAT_B8G8R8A8_SRGB;
    if (format == work::renderer::Format::A8B8G8R8_UNORM_PACK32) return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
    if (format == work::renderer::Format::A8B8G8R8_SNORM_PACK32) return VK_FORMAT_A8B8G8R8_SNORM_PACK32;
    if (format == work::renderer::Format::A8B8G8R8_USCALED_PACK32) return VK_FORMAT_A8B8G8R8_USCALED_PACK32;
    if (format == work::renderer::Format::A8B8G8R8_SSCALED_PACK32) return VK_FORMAT_A8B8G8R8_SSCALED_PACK32;
    if (format == work::renderer::Format::A8B8G8R8_UINT_PACK32) return VK_FORMAT_A8B8G8R8_UINT_PACK32;
    if (format == work::renderer::Format::A8B8G8R8_SINT_PACK32) return VK_FORMAT_A8B8G8R8_SINT_PACK32;
    if (format == work::renderer::Format::A8B8G8R8_SRGB_PACK32) return VK_FORMAT_A8B8G8R8_SRGB_PACK32;
    if (format == work::renderer::Format::A2R10G10B10_UNORM_PACK32) return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
    if (format == work::renderer::Format::A2R10G10B10_SNORM_PACK32) return VK_FORMAT_A2R10G10B10_SNORM_PACK32;
    if (format == work::renderer::Format::A2R10G10B10_USCALED_PACK32) return VK_FORMAT_A2R10G10B10_USCALED_PACK32;
    if (format == work::renderer::Format::A2R10G10B10_SSCALED_PACK32) return VK_FORMAT_A2R10G10B10_SSCALED_PACK32;
    if (format == work::renderer::Format::A2R10G10B10_UINT_PACK32) return VK_FORMAT_A2R10G10B10_UINT_PACK32;
    if (format == work::renderer::Format::A2R10G10B10_SINT_PACK32) return VK_FORMAT_A2R10G10B10_SINT_PACK32;
    if (format == work::renderer::Format::A2B10G10R10_UNORM_PACK32) return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    if (format == work::renderer::Format::A2B10G10R10_SNORM_PACK32) return VK_FORMAT_A2B10G10R10_SNORM_PACK32;
    if (format == work::renderer::Format::A2B10G10R10_USCALED_PACK32) return VK_FORMAT_A2B10G10R10_USCALED_PACK32;
    if (format == work::renderer::Format::A2B10G10R10_SSCALED_PACK32) return VK_FORMAT_A2B10G10R10_SSCALED_PACK32;
    if (format == work::renderer::Format::A2B10G10R10_UINT_PACK32) return VK_FORMAT_A2B10G10R10_UINT_PACK32;
    if (format == work::renderer::Format::A2B10G10R10_SINT_PACK32) return VK_FORMAT_A2B10G10R10_SINT_PACK32;
    if (format == work::renderer::Format::R16_UNORM) return VK_FORMAT_R16_UNORM;
    if (format == work::renderer::Format::R16_SNORM) return VK_FORMAT_R16_SNORM;
    if (format == work::renderer::Format::R16_USCALED) return VK_FORMAT_R16_USCALED;
    if (format == work::renderer::Format::R16_SSCALED) return VK_FORMAT_R16_SSCALED;
    if (format == work::renderer::Format::R16_UINT) return VK_FORMAT_R16_UINT;
    if (format == work::renderer::Format::R16_SINT) return VK_FORMAT_R16_SINT;
    if (format == work::renderer::Format::R16_SFLOAT) return VK_FORMAT_R16_SFLOAT;
    if (format == work::renderer::Format::R16G16_UNORM) return VK_FORMAT_R16G16_UNORM;
    if (format == work::renderer::Format::R16G16_SNORM) return VK_FORMAT_R16G16_SNORM;
    if (format == work::renderer::Format::R16G16_USCALED) return VK_FORMAT_R16G16_USCALED;
    if (format == work::renderer::Format::R16G16_SSCALED) return VK_FORMAT_R16G16_SSCALED;
    if (format == work::renderer::Format::R16G16_UINT) return VK_FORMAT_R16G16_UINT;
    if (format == work::renderer::Format::R16G16_SINT) return VK_FORMAT_R16G16_SINT;
    if (format == work::renderer::Format::R16G16_SFLOAT) return VK_FORMAT_R16G16_SFLOAT;
    if (format == work::renderer::Format::R16G16B16_UNORM) return VK_FORMAT_R16G16B16_UNORM;
    if (format == work::renderer::Format::R16G16B16_SNORM) return VK_FORMAT_R16G16B16_SNORM;
    if (format == work::renderer::Format::R16G16B16_USCALED) return VK_FORMAT_R16G16B16_USCALED;
    if (format == work::renderer::Format::R16G16B16_SSCALED) return VK_FORMAT_R16G16B16_SSCALED;
    if (format == work::renderer::Format::R16G16B16_UINT) return VK_FORMAT_R16G16B16_UINT;
    if (format == work::renderer::Format::R16G16B16_SINT) return VK_FORMAT_R16G16B16_SINT;
    if (format == work::renderer::Format::R16G16B16_SFLOAT) return VK_FORMAT_R16G16B16_SFLOAT;
    if (format == work::renderer::Format::R16G16B16A16_UNORM) return VK_FORMAT_R16G16B16A16_UNORM;
    if (format == work::renderer::Format::R16G16B16A16_SNORM) return VK_FORMAT_R16G16B16A16_SNORM;
    if (format == work::renderer::Format::R16G16B16A16_USCALED) return VK_FORMAT_R16G16B16A16_USCALED;
    if (format == work::renderer::Format::R16G16B16A16_SSCALED) return VK_FORMAT_R16G16B16A16_SSCALED;
    if (format == work::renderer::Format::R16G16B16A16_UINT) return VK_FORMAT_R16G16B16A16_UINT;
    if (format == work::renderer::Format::R16G16B16A16_SINT) return VK_FORMAT_R16G16B16A16_SINT;
    if (format == work::renderer::Format::R16G16B16A16_SFLOAT) return VK_FORMAT_R16G16B16A16_SFLOAT;
    if (format == work::renderer::Format::R32_UINT) return VK_FORMAT_R32_UINT;
    if (format == work::renderer::Format::R32_SINT) return VK_FORMAT_R32_SINT;
    if (format == work::renderer::Format::R32_SFLOAT) return VK_FORMAT_R32_SFLOAT;
    if (format == work::renderer::Format::R32G32_UINT) return VK_FORMAT_R32G32_UINT;
    if (format == work::renderer::Format::R32G32_SINT) return VK_FORMAT_R32G32_SINT;
    if (format == work::renderer::Format::R32G32_SFLOAT) return VK_FORMAT_R32G32_SFLOAT;
    if (format == work::renderer::Format::R32G32B32_UINT) return VK_FORMAT_R32G32B32_UINT;
    if (format == work::renderer::Format::R32G32B32_SINT) return VK_FORMAT_R32G32B32_SINT;
    if (format == work::renderer::Format::R32G32B32_SFLOAT) return VK_FORMAT_R32G32B32_SFLOAT;
    if (format == work::renderer::Format::R32G32B32A32_UINT) return VK_FORMAT_R32G32B32A32_UINT;
    if (format == work::renderer::Format::R32G32B32A32_SINT) return VK_FORMAT_R32G32B32A32_SINT;
    if (format == work::renderer::Format::R32G32B32A32_SFLOAT) return VK_FORMAT_R32G32B32A32_SFLOAT;
    if (format == work::renderer::Format::R64_UINT) return VK_FORMAT_R64_UINT;
    if (format == work::renderer::Format::R64_SINT) return VK_FORMAT_R64_SINT;
    if (format == work::renderer::Format::R64_SFLOAT) return VK_FORMAT_R64_SFLOAT;
    if (format == work::renderer::Format::R64G64_UINT) return VK_FORMAT_R64G64_UINT;
    if (format == work::renderer::Format::R64G64_SINT) return VK_FORMAT_R64G64_SINT;
    if (format == work::renderer::Format::R64G64_SFLOAT) return VK_FORMAT_R64G64_SFLOAT;
    if (format == work::renderer::Format::R64G64B64_UINT) return VK_FORMAT_R64G64B64_UINT;
    if (format == work::renderer::Format::R64G64B64_SINT) return VK_FORMAT_R64G64B64_SINT;
    if (format == work::renderer::Format::R64G64B64_SFLOAT) return VK_FORMAT_R64G64B64_SFLOAT;
    if (format == work::renderer::Format::R64G64B64A64_UINT) return VK_FORMAT_R64G64B64A64_UINT;
    if (format == work::renderer::Format::R64G64B64A64_SINT) return VK_FORMAT_R64G64B64A64_SINT;
    if (format == work::renderer::Format::R64G64B64A64_SFLOAT) return VK_FORMAT_R64G64B64A64_SFLOAT;
    if (format == work::renderer::Format::B10G11R11_UFLOAT_PACK32) return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    if (format == work::renderer::Format::E5B9G9R9_UFLOAT_PACK32) return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    if (format == work::renderer::Format::D16_UNORM) return VK_FORMAT_D16_UNORM;
    if (format == work::renderer::Format::X8_D24_UNORM_PACK32) return VK_FORMAT_X8_D24_UNORM_PACK32;
    if (format == work::renderer::Format::D32_SFLOAT) return VK_FORMAT_D32_SFLOAT;
    if (format == work::renderer::Format::S8_UINT) return VK_FORMAT_S8_UINT;
    if (format == work::renderer::Format::D16_UNORM_S8_UINT) return VK_FORMAT_D16_UNORM_S8_UINT;
    if (format == work::renderer::Format::D24_UNORM_S8_UINT) return VK_FORMAT_D24_UNORM_S8_UINT;
    if (format == work::renderer::Format::D32_SFLOAT_S8_UINT) return VK_FORMAT_D32_SFLOAT_S8_UINT;
    if (format == work::renderer::Format::BC1_RGB_UNORM_BLOCK) return VK_FORMAT_BC1_RGB_UNORM_BLOCK;
    if (format == work::renderer::Format::BC1_RGB_SRGB_BLOCK) return VK_FORMAT_BC1_RGB_SRGB_BLOCK;
    if (format == work::renderer::Format::BC1_RGBA_UNORM_BLOCK) return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    if (format == work::renderer::Format::BC1_RGBA_SRGB_BLOCK) return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
    if (format == work::renderer::Format::BC2_UNORM_BLOCK) return VK_FORMAT_BC2_UNORM_BLOCK;
    if (format == work::renderer::Format::BC2_SRGB_BLOCK) return VK_FORMAT_BC2_SRGB_BLOCK;
    if (format == work::renderer::Format::BC3_UNORM_BLOCK) return VK_FORMAT_BC3_UNORM_BLOCK;
    if (format == work::renderer::Format::BC3_SRGB_BLOCK) return VK_FORMAT_BC3_SRGB_BLOCK;
    if (format == work::renderer::Format::BC4_UNORM_BLOCK) return VK_FORMAT_BC4_UNORM_BLOCK;
    if (format == work::renderer::Format::BC4_SNORM_BLOCK) return VK_FORMAT_BC4_SNORM_BLOCK;
    if (format == work::renderer::Format::BC5_UNORM_BLOCK) return VK_FORMAT_BC5_UNORM_BLOCK;
    if (format == work::renderer::Format::BC5_SNORM_BLOCK) return VK_FORMAT_BC5_SNORM_BLOCK;
    if (format == work::renderer::Format::BC6H_UFLOAT_BLOCK) return VK_FORMAT_BC6H_UFLOAT_BLOCK;
    if (format == work::renderer::Format::BC6H_SFLOAT_BLOCK) return VK_FORMAT_BC6H_SFLOAT_BLOCK;
    if (format == work::renderer::Format::BC7_UNORM_BLOCK) return VK_FORMAT_BC7_UNORM_BLOCK;
    if (format == work::renderer::Format::BC7_SRGB_BLOCK) return VK_FORMAT_BC7_SRGB_BLOCK;
    if (format == work::renderer::Format::ETC2_R8G8B8_UNORM_BLOCK) return VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
    if (format == work::renderer::Format::ETC2_R8G8B8_SRGB_BLOCK) return VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK;
    if (format == work::renderer::Format::ETC2_R8G8B8A1_UNORM_BLOCK) return VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK;
    if (format == work::renderer::Format::ETC2_R8G8B8A1_SRGB_BLOCK) return VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK;
    if (format == work::renderer::Format::ETC2_R8G8B8A8_UNORM_BLOCK) return VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK;
    if (format == work::renderer::Format::ETC2_R8G8B8A8_SRGB_BLOCK) return VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK;
    if (format == work::renderer::Format::EAC_R11_UNORM_BLOCK) return VK_FORMAT_EAC_R11_UNORM_BLOCK;
    if (format == work::renderer::Format::EAC_R11_SNORM_BLOCK) return VK_FORMAT_EAC_R11_SNORM_BLOCK;
    if (format == work::renderer::Format::EAC_R11G11_UNORM_BLOCK) return VK_FORMAT_EAC_R11G11_UNORM_BLOCK;
    if (format == work::renderer::Format::EAC_R11G11_SNORM_BLOCK) return VK_FORMAT_EAC_R11G11_SNORM_BLOCK;
    if (format == work::renderer::Format::ASTC_4x4_UNORM_BLOCK) return VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
    if (format == work::renderer::Format::ASTC_4x4_SRGB_BLOCK) return VK_FORMAT_ASTC_4x4_SRGB_BLOCK;
    if (format == work::renderer::Format::ASTC_5x4_UNORM_BLOCK) return VK_FORMAT_ASTC_5x4_UNORM_BLOCK;
    if (format == work::renderer::Format::ASTC_5x4_SRGB_BLOCK) return VK_FORMAT_ASTC_5x4_SRGB_BLOCK;
    if (format == work::renderer::Format::ASTC_5x5_UNORM_BLOCK) return VK_FORMAT_ASTC_5x5_UNORM_BLOCK;
    if (format == work::renderer::Format::ASTC_5x5_SRGB_BLOCK) return VK_FORMAT_ASTC_5x5_SRGB_BLOCK;
    if (format == work::renderer::Format::ASTC_6x5_UNORM_BLOCK) return VK_FORMAT_ASTC_6x5_UNORM_BLOCK;
    if (format == work::renderer::Format::ASTC_6x5_SRGB_BLOCK) return VK_FORMAT_ASTC_6x5_SRGB_BLOCK;
    if (format == work::renderer::Format::ASTC_6x6_UNORM_BLOCK) return VK_FORMAT_ASTC_6x6_UNORM_BLOCK;
    if (format == work::renderer::Format::ASTC_6x6_SRGB_BLOCK) return VK_FORMAT_ASTC_6x6_SRGB_BLOCK;
    if (format == work::renderer::Format::ASTC_8x5_UNORM_BLOCK) return VK_FORMAT_ASTC_8x5_UNORM_BLOCK;
    if (format == work::renderer::Format::ASTC_8x5_SRGB_BLOCK) return VK_FORMAT_ASTC_8x5_SRGB_BLOCK;
    if (format == work::renderer::Format::ASTC_8x6_UNORM_BLOCK) return VK_FORMAT_ASTC_8x6_UNORM_BLOCK;
    if (format == work::renderer::Format::ASTC_8x6_SRGB_BLOCK) return VK_FORMAT_ASTC_8x6_SRGB_BLOCK;
    if (format == work::renderer::Format::ASTC_8x8_UNORM_BLOCK) return VK_FORMAT_ASTC_8x8_UNORM_BLOCK;
    if (format == work::renderer::Format::ASTC_8x8_SRGB_BLOCK) return VK_FORMAT_ASTC_8x8_SRGB_BLOCK;
    if (format == work::renderer::Format::ASTC_10x5_UNORM_BLOCK) return VK_FORMAT_ASTC_10x5_UNORM_BLOCK;
    if (format == work::renderer::Format::ASTC_10x5_SRGB_BLOCK) return VK_FORMAT_ASTC_10x5_SRGB_BLOCK;
    if (format == work::renderer::Format::ASTC_10x6_UNORM_BLOCK) return VK_FORMAT_ASTC_10x6_UNORM_BLOCK;
    if (format == work::renderer::Format::ASTC_10x6_SRGB_BLOCK) return VK_FORMAT_ASTC_10x6_SRGB_BLOCK;
    if (format == work::renderer::Format::ASTC_10x8_UNORM_BLOCK) return VK_FORMAT_ASTC_10x8_UNORM_BLOCK;
    if (format == work::renderer::Format::ASTC_10x8_SRGB_BLOCK) return VK_FORMAT_ASTC_10x8_SRGB_BLOCK;
    if (format == work::renderer::Format::ASTC_10x10_UNORM_BLOCK) return VK_FORMAT_ASTC_10x10_UNORM_BLOCK;
    if (format == work::renderer::Format::ASTC_10x10_SRGB_BLOCK) return VK_FORMAT_ASTC_10x10_SRGB_BLOCK;
    if (format == work::renderer::Format::ASTC_12x10_UNORM_BLOCK) return VK_FORMAT_ASTC_12x10_UNORM_BLOCK;
    if (format == work::renderer::Format::ASTC_12x10_SRGB_BLOCK) return VK_FORMAT_ASTC_12x10_SRGB_BLOCK;
    if (format == work::renderer::Format::ASTC_12x12_UNORM_BLOCK) return VK_FORMAT_ASTC_12x12_UNORM_BLOCK;
    if (format == work::renderer::Format::ASTC_12x12_SRGB_BLOCK) return VK_FORMAT_ASTC_12x12_SRGB_BLOCK;
    if (format == work::renderer::Format::G8B8G8R8_422_UNORM) return VK_FORMAT_G8B8G8R8_422_UNORM;
    if (format == work::renderer::Format::B8G8R8G8_422_UNORM) return VK_FORMAT_B8G8R8G8_422_UNORM;
    if (format == work::renderer::Format::G8_B8_R8_3PLANE_420_UNORM) return VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
    if (format == work::renderer::Format::G8_B8R8_2PLANE_420_UNORM) return VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
    if (format == work::renderer::Format::G8_B8_R8_3PLANE_422_UNORM) return VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM;
    if (format == work::renderer::Format::G8_B8R8_2PLANE_422_UNORM) return VK_FORMAT_G8_B8R8_2PLANE_422_UNORM;
    if (format == work::renderer::Format::G8_B8_R8_3PLANE_444_UNORM) return VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM;
    if (format == work::renderer::Format::R10X6_UNORM_PACK16) return VK_FORMAT_R10X6_UNORM_PACK16;
    if (format == work::renderer::Format::R10X6G10X6_UNORM_2PACK16) return VK_FORMAT_R10X6G10X6_UNORM_2PACK16;
    if (format == work::renderer::Format::G10X6B10X6G10X6R10X6_422_UNORM_4PACK16) return VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16;
    if (format == work::renderer::Format::B10X6G10X6R10X6G10X6_422_UNORM_4PACK16) return VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16;
    if (format == work::renderer::Format::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16) return VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16;
    if (format == work::renderer::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16) return VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16;
    if (format == work::renderer::Format::G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16) return VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16;
    if (format == work::renderer::Format::G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16) return VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16;
    if (format == work::renderer::Format::G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16) return VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16;
    if (format == work::renderer::Format::R12X4_UNORM_PACK16) return VK_FORMAT_R12X4_UNORM_PACK16;
    if (format == work::renderer::Format::R12X4G12X4_UNORM_2PACK16) return VK_FORMAT_R12X4G12X4_UNORM_2PACK16;
    if (format == work::renderer::Format::R12X4G12X4B12X4A12X4_UNORM_4PACK16) return VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16;
    if (format == work::renderer::Format::G12X4B12X4G12X4R12X4_422_UNORM_4PACK16) return VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16;
    if (format == work::renderer::Format::B12X4G12X4R12X4G12X4_422_UNORM_4PACK16) return VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16;
    if (format == work::renderer::Format::G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16) return VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16;
    if (format == work::renderer::Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16) return VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16;
    if (format == work::renderer::Format::G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16) return VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16;
    if (format == work::renderer::Format::G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16) return VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16;
    if (format == work::renderer::Format::G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16) return VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16;
    if (format == work::renderer::Format::G16B16G16R16_422_UNORM) return VK_FORMAT_G16B16G16R16_422_UNORM;
    if (format == work::renderer::Format::B16G16R16G16_422_UNORM) return VK_FORMAT_B16G16R16G16_422_UNORM;
    if (format == work::renderer::Format::G16_B16_R16_3PLANE_420_UNORM) return VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM;
    if (format == work::renderer::Format::G16_B16R16_2PLANE_420_UNORM) return VK_FORMAT_G16_B16R16_2PLANE_420_UNORM;
    if (format == work::renderer::Format::G16_B16_R16_3PLANE_422_UNORM) return VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM;
    if (format == work::renderer::Format::G16_B16R16_2PLANE_422_UNORM) return VK_FORMAT_G16_B16R16_2PLANE_422_UNORM;
    if (format == work::renderer::Format::G16_B16_R16_3PLANE_444_UNORM) return VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM;
    if (format == work::renderer::Format::PVRTC1_2BPP_UNORM_BLOCK_IMG) return VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG;
    if (format == work::renderer::Format::PVRTC1_4BPP_UNORM_BLOCK_IMG) return VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG;
    if (format == work::renderer::Format::PVRTC2_2BPP_UNORM_BLOCK_IMG) return VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG;
    if (format == work::renderer::Format::PVRTC2_4BPP_UNORM_BLOCK_IMG) return VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG;
    if (format == work::renderer::Format::PVRTC1_2BPP_SRGB_BLOCK_IMG) return VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG;
    if (format == work::renderer::Format::PVRTC1_4BPP_SRGB_BLOCK_IMG) return VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG;
    if (format == work::renderer::Format::PVRTC2_2BPP_SRGB_BLOCK_IMG) return VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG;
    if (format == work::renderer::Format::PVRTC2_4BPP_SRGB_BLOCK_IMG) return VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG;
    if (format == work::renderer::Format::ASTC_4x4_SFLOAT_BLOCK_EXT) return VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK_EXT;
    if (format == work::renderer::Format::ASTC_5x4_SFLOAT_BLOCK_EXT) return VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK_EXT;
    if (format == work::renderer::Format::ASTC_5x5_SFLOAT_BLOCK_EXT) return VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK_EXT;
    if (format == work::renderer::Format::ASTC_6x5_SFLOAT_BLOCK_EXT) return VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK_EXT;
    if (format == work::renderer::Format::ASTC_6x6_SFLOAT_BLOCK_EXT) return VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK_EXT;
    if (format == work::renderer::Format::ASTC_8x5_SFLOAT_BLOCK_EXT) return VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK_EXT;
    if (format == work::renderer::Format::ASTC_8x6_SFLOAT_BLOCK_EXT) return VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK_EXT;
    if (format == work::renderer::Format::ASTC_10x5_SFLOAT_BLOCK_EXT) return VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK_EXT;
    if (format == work::renderer::Format::ASTC_10x6_SFLOAT_BLOCK_EXT) return VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK_EXT;
    if (format == work::renderer::Format::ASTC_10x8_SFLOAT_BLOCK_EXT) return VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK_EXT;
    if (format == work::renderer::Format::ASTC_10x10_SFLOAT_BLOCK_EXT) return VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK_EXT;
    if (format == work::renderer::Format::ASTC_12x10_SFLOAT_BLOCK_EXT) return VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK_EXT;
    if (format == work::renderer::Format::ASTC_12x12_SFLOAT_BLOCK_EXT) return VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK_EXT;
    return VK_FORMAT_UNDEFINED;
}

work::renderer::Format fromVkFormat(VkFormat format) {
    if (format == VK_FORMAT_R4G4_UNORM_PACK8) return work::renderer::Format::R4G4_UNORM_PACK8;
    if (format == VK_FORMAT_R4G4B4A4_UNORM_PACK16) return work::renderer::Format::R4G4B4A4_UNORM_PACK16;
    if (format == VK_FORMAT_B4G4R4A4_UNORM_PACK16) return work::renderer::Format::B4G4R4A4_UNORM_PACK16;
    if (format == VK_FORMAT_R5G6B5_UNORM_PACK16) return work::renderer::Format::R5G6B5_UNORM_PACK16;
    if (format == VK_FORMAT_B5G6R5_UNORM_PACK16) return work::renderer::Format::B5G6R5_UNORM_PACK16;
    if (format == VK_FORMAT_R5G5B5A1_UNORM_PACK16) return work::renderer::Format::R5G5B5A1_UNORM_PACK16;
    if (format == VK_FORMAT_B5G5R5A1_UNORM_PACK16) return work::renderer::Format::B5G5R5A1_UNORM_PACK16;
    if (format == VK_FORMAT_A1R5G5B5_UNORM_PACK16) return work::renderer::Format::A1R5G5B5_UNORM_PACK16;
    if (format == VK_FORMAT_R8_UNORM) return work::renderer::Format::R8_UNORM;
    if (format == VK_FORMAT_R8_SNORM) return work::renderer::Format::R8_SNORM;
    if (format == VK_FORMAT_R8_USCALED) return work::renderer::Format::R8_USCALED;
    if (format == VK_FORMAT_R8_SSCALED) return work::renderer::Format::R8_SSCALED;
    if (format == VK_FORMAT_R8_UINT) return work::renderer::Format::R8_UINT;
    if (format == VK_FORMAT_R8_SINT) return work::renderer::Format::R8_SINT;
    if (format == VK_FORMAT_R8_SRGB) return work::renderer::Format::R8_SRGB;
    if (format == VK_FORMAT_R8G8_UNORM) return work::renderer::Format::R8G8_UNORM;
    if (format == VK_FORMAT_R8G8_SNORM) return work::renderer::Format::R8G8_SNORM;
    if (format == VK_FORMAT_R8G8_USCALED) return work::renderer::Format::R8G8_USCALED;
    if (format == VK_FORMAT_R8G8_SSCALED) return work::renderer::Format::R8G8_SSCALED;
    if (format == VK_FORMAT_R8G8_UINT) return work::renderer::Format::R8G8_UINT;
    if (format == VK_FORMAT_R8G8_SINT) return work::renderer::Format::R8G8_SINT;
    if (format == VK_FORMAT_R8G8_SRGB) return work::renderer::Format::R8G8_SRGB;
    if (format == VK_FORMAT_R8G8B8_UNORM) return work::renderer::Format::R8G8B8_UNORM;
    if (format == VK_FORMAT_R8G8B8_SNORM) return work::renderer::Format::R8G8B8_SNORM;
    if (format == VK_FORMAT_R8G8B8_USCALED) return work::renderer::Format::R8G8B8_USCALED;
    if (format == VK_FORMAT_R8G8B8_SSCALED) return work::renderer::Format::R8G8B8_SSCALED;
    if (format == VK_FORMAT_R8G8B8_UINT) return work::renderer::Format::R8G8B8_UINT;
    if (format == VK_FORMAT_R8G8B8_SINT) return work::renderer::Format::R8G8B8_SINT;
    if (format == VK_FORMAT_R8G8B8_SRGB) return work::renderer::Format::R8G8B8_SRGB;
    if (format == VK_FORMAT_B8G8R8_UNORM) return work::renderer::Format::B8G8R8_UNORM;
    if (format == VK_FORMAT_B8G8R8_SNORM) return work::renderer::Format::B8G8R8_SNORM;
    if (format == VK_FORMAT_B8G8R8_USCALED) return work::renderer::Format::B8G8R8_USCALED;
    if (format == VK_FORMAT_B8G8R8_SSCALED) return work::renderer::Format::B8G8R8_SSCALED;
    if (format == VK_FORMAT_B8G8R8_UINT) return work::renderer::Format::B8G8R8_UINT;
    if (format == VK_FORMAT_B8G8R8_SINT) return work::renderer::Format::B8G8R8_SINT;
    if (format == VK_FORMAT_B8G8R8_SRGB) return work::renderer::Format::B8G8R8_SRGB;
    if (format == VK_FORMAT_R8G8B8A8_UNORM) return work::renderer::Format::R8G8B8A8_UNORM;
    if (format == VK_FORMAT_R8G8B8A8_SNORM) return work::renderer::Format::R8G8B8A8_SNORM;
    if (format == VK_FORMAT_R8G8B8A8_USCALED) return work::renderer::Format::R8G8B8A8_USCALED;
    if (format == VK_FORMAT_R8G8B8A8_SSCALED) return work::renderer::Format::R8G8B8A8_SSCALED;
    if (format == VK_FORMAT_R8G8B8A8_UINT) return work::renderer::Format::R8G8B8A8_UINT;
    if (format == VK_FORMAT_R8G8B8A8_SINT) return work::renderer::Format::R8G8B8A8_SINT;
    if (format == VK_FORMAT_R8G8B8A8_SRGB) return work::renderer::Format::R8G8B8A8_SRGB;
    if (format == VK_FORMAT_B8G8R8A8_UNORM) return work::renderer::Format::B8G8R8A8_UNORM;
    if (format == VK_FORMAT_B8G8R8A8_SNORM) return work::renderer::Format::B8G8R8A8_SNORM;
    if (format == VK_FORMAT_B8G8R8A8_USCALED) return work::renderer::Format::B8G8R8A8_USCALED;
    if (format == VK_FORMAT_B8G8R8A8_SSCALED) return work::renderer::Format::B8G8R8A8_SSCALED;
    if (format == VK_FORMAT_B8G8R8A8_UINT) return work::renderer::Format::B8G8R8A8_UINT;
    if (format == VK_FORMAT_B8G8R8A8_SINT) return work::renderer::Format::B8G8R8A8_SINT;
    if (format == VK_FORMAT_B8G8R8A8_SRGB) return work::renderer::Format::B8G8R8A8_SRGB;
    if (format == VK_FORMAT_A8B8G8R8_UNORM_PACK32) return work::renderer::Format::A8B8G8R8_UNORM_PACK32;
    if (format == VK_FORMAT_A8B8G8R8_SNORM_PACK32) return work::renderer::Format::A8B8G8R8_SNORM_PACK32;
    if (format == VK_FORMAT_A8B8G8R8_USCALED_PACK32) return work::renderer::Format::A8B8G8R8_USCALED_PACK32;
    if (format == VK_FORMAT_A8B8G8R8_SSCALED_PACK32) return work::renderer::Format::A8B8G8R8_SSCALED_PACK32;
    if (format == VK_FORMAT_A8B8G8R8_UINT_PACK32) return work::renderer::Format::A8B8G8R8_UINT_PACK32;
    if (format == VK_FORMAT_A8B8G8R8_SINT_PACK32) return work::renderer::Format::A8B8G8R8_SINT_PACK32;
    if (format == VK_FORMAT_A8B8G8R8_SRGB_PACK32) return work::renderer::Format::A8B8G8R8_SRGB_PACK32;
    if (format == VK_FORMAT_A2R10G10B10_UNORM_PACK32) return work::renderer::Format::A2R10G10B10_UNORM_PACK32;
    if (format == VK_FORMAT_A2R10G10B10_SNORM_PACK32) return work::renderer::Format::A2R10G10B10_SNORM_PACK32;
    if (format == VK_FORMAT_A2R10G10B10_USCALED_PACK32) return work::renderer::Format::A2R10G10B10_USCALED_PACK32;
    if (format == VK_FORMAT_A2R10G10B10_SSCALED_PACK32) return work::renderer::Format::A2R10G10B10_SSCALED_PACK32;
    if (format == VK_FORMAT_A2R10G10B10_UINT_PACK32) return work::renderer::Format::A2R10G10B10_UINT_PACK32;
    if (format == VK_FORMAT_A2R10G10B10_SINT_PACK32) return work::renderer::Format::A2R10G10B10_SINT_PACK32;
    if (format == VK_FORMAT_A2B10G10R10_UNORM_PACK32) return work::renderer::Format::A2B10G10R10_UNORM_PACK32;
    if (format == VK_FORMAT_A2B10G10R10_SNORM_PACK32) return work::renderer::Format::A2B10G10R10_SNORM_PACK32;
    if (format == VK_FORMAT_A2B10G10R10_USCALED_PACK32) return work::renderer::Format::A2B10G10R10_USCALED_PACK32;
    if (format == VK_FORMAT_A2B10G10R10_SSCALED_PACK32) return work::renderer::Format::A2B10G10R10_SSCALED_PACK32;
    if (format == VK_FORMAT_A2B10G10R10_UINT_PACK32) return work::renderer::Format::A2B10G10R10_UINT_PACK32;
    if (format == VK_FORMAT_A2B10G10R10_SINT_PACK32) return work::renderer::Format::A2B10G10R10_SINT_PACK32;
    if (format == VK_FORMAT_R16_UNORM) return work::renderer::Format::R16_UNORM;
    if (format == VK_FORMAT_R16_SNORM) return work::renderer::Format::R16_SNORM;
    if (format == VK_FORMAT_R16_USCALED) return work::renderer::Format::R16_USCALED;
    if (format == VK_FORMAT_R16_SSCALED) return work::renderer::Format::R16_SSCALED;
    if (format == VK_FORMAT_R16_UINT) return work::renderer::Format::R16_UINT;
    if (format == VK_FORMAT_R16_SINT) return work::renderer::Format::R16_SINT;
    if (format == VK_FORMAT_R16_SFLOAT) return work::renderer::Format::R16_SFLOAT;
    if (format == VK_FORMAT_R16G16_UNORM) return work::renderer::Format::R16G16_UNORM;
    if (format == VK_FORMAT_R16G16_SNORM) return work::renderer::Format::R16G16_SNORM;
    if (format == VK_FORMAT_R16G16_USCALED) return work::renderer::Format::R16G16_USCALED;
    if (format == VK_FORMAT_R16G16_SSCALED) return work::renderer::Format::R16G16_SSCALED;
    if (format == VK_FORMAT_R16G16_UINT) return work::renderer::Format::R16G16_UINT;
    if (format == VK_FORMAT_R16G16_SINT) return work::renderer::Format::R16G16_SINT;
    if (format == VK_FORMAT_R16G16_SFLOAT) return work::renderer::Format::R16G16_SFLOAT;
    if (format == VK_FORMAT_R16G16B16_UNORM) return work::renderer::Format::R16G16B16_UNORM;
    if (format == VK_FORMAT_R16G16B16_SNORM) return work::renderer::Format::R16G16B16_SNORM;
    if (format == VK_FORMAT_R16G16B16_USCALED) return work::renderer::Format::R16G16B16_USCALED;
    if (format == VK_FORMAT_R16G16B16_SSCALED) return work::renderer::Format::R16G16B16_SSCALED;
    if (format == VK_FORMAT_R16G16B16_UINT) return work::renderer::Format::R16G16B16_UINT;
    if (format == VK_FORMAT_R16G16B16_SINT) return work::renderer::Format::R16G16B16_SINT;
    if (format == VK_FORMAT_R16G16B16_SFLOAT) return work::renderer::Format::R16G16B16_SFLOAT;
    if (format == VK_FORMAT_R16G16B16A16_UNORM) return work::renderer::Format::R16G16B16A16_UNORM;
    if (format == VK_FORMAT_R16G16B16A16_SNORM) return work::renderer::Format::R16G16B16A16_SNORM;
    if (format == VK_FORMAT_R16G16B16A16_USCALED) return work::renderer::Format::R16G16B16A16_USCALED;
    if (format == VK_FORMAT_R16G16B16A16_SSCALED) return work::renderer::Format::R16G16B16A16_SSCALED;
    if (format == VK_FORMAT_R16G16B16A16_UINT) return work::renderer::Format::R16G16B16A16_UINT;
    if (format == VK_FORMAT_R16G16B16A16_SINT) return work::renderer::Format::R16G16B16A16_SINT;
    if (format == VK_FORMAT_R16G16B16A16_SFLOAT) return work::renderer::Format::R16G16B16A16_SFLOAT;
    if (format == VK_FORMAT_R32_UINT) return work::renderer::Format::R32_UINT;
    if (format == VK_FORMAT_R32_SINT) return work::renderer::Format::R32_SINT;
    if (format == VK_FORMAT_R32_SFLOAT) return work::renderer::Format::R32_SFLOAT;
    if (format == VK_FORMAT_R32G32_UINT) return work::renderer::Format::R32G32_UINT;
    if (format == VK_FORMAT_R32G32_SINT) return work::renderer::Format::R32G32_SINT;
    if (format == VK_FORMAT_R32G32_SFLOAT) return work::renderer::Format::R32G32_SFLOAT;
    if (format == VK_FORMAT_R32G32B32_UINT) return work::renderer::Format::R32G32B32_UINT;
    if (format == VK_FORMAT_R32G32B32_SINT) return work::renderer::Format::R32G32B32_SINT;
    if (format == VK_FORMAT_R32G32B32_SFLOAT) return work::renderer::Format::R32G32B32_SFLOAT;
    if (format == VK_FORMAT_R32G32B32A32_UINT) return work::renderer::Format::R32G32B32A32_UINT;
    if (format == VK_FORMAT_R32G32B32A32_SINT) return work::renderer::Format::R32G32B32A32_SINT;
    if (format == VK_FORMAT_R32G32B32A32_SFLOAT) return work::renderer::Format::R32G32B32A32_SFLOAT;
    if (format == VK_FORMAT_R64_UINT) return work::renderer::Format::R64_UINT;
    if (format == VK_FORMAT_R64_SINT) return work::renderer::Format::R64_SINT;
    if (format == VK_FORMAT_R64_SFLOAT) return work::renderer::Format::R64_SFLOAT;
    if (format == VK_FORMAT_R64G64_UINT) return work::renderer::Format::R64G64_UINT;
    if (format == VK_FORMAT_R64G64_SINT) return work::renderer::Format::R64G64_SINT;
    if (format == VK_FORMAT_R64G64_SFLOAT) return work::renderer::Format::R64G64_SFLOAT;
    if (format == VK_FORMAT_R64G64B64_UINT) return work::renderer::Format::R64G64B64_UINT;
    if (format == VK_FORMAT_R64G64B64_SINT) return work::renderer::Format::R64G64B64_SINT;
    if (format == VK_FORMAT_R64G64B64_SFLOAT) return work::renderer::Format::R64G64B64_SFLOAT;
    if (format == VK_FORMAT_R64G64B64A64_UINT) return work::renderer::Format::R64G64B64A64_UINT;
    if (format == VK_FORMAT_R64G64B64A64_SINT) return work::renderer::Format::R64G64B64A64_SINT;
    if (format == VK_FORMAT_R64G64B64A64_SFLOAT) return work::renderer::Format::R64G64B64A64_SFLOAT;
    if (format == VK_FORMAT_B10G11R11_UFLOAT_PACK32) return work::renderer::Format::B10G11R11_UFLOAT_PACK32;
    if (format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32) return work::renderer::Format::E5B9G9R9_UFLOAT_PACK32;
    if (format == VK_FORMAT_D16_UNORM) return work::renderer::Format::D16_UNORM;
    if (format == VK_FORMAT_X8_D24_UNORM_PACK32) return work::renderer::Format::X8_D24_UNORM_PACK32;
    if (format == VK_FORMAT_D32_SFLOAT) return work::renderer::Format::D32_SFLOAT;
    if (format == VK_FORMAT_S8_UINT) return work::renderer::Format::S8_UINT;
    if (format == VK_FORMAT_D16_UNORM_S8_UINT) return work::renderer::Format::D16_UNORM_S8_UINT;
    if (format == VK_FORMAT_D24_UNORM_S8_UINT) return work::renderer::Format::D24_UNORM_S8_UINT;
    if (format == VK_FORMAT_D32_SFLOAT_S8_UINT) return work::renderer::Format::D32_SFLOAT_S8_UINT;
    if (format == VK_FORMAT_BC1_RGB_UNORM_BLOCK) return work::renderer::Format::BC1_RGB_UNORM_BLOCK;
    if (format == VK_FORMAT_BC1_RGB_SRGB_BLOCK) return work::renderer::Format::BC1_RGB_SRGB_BLOCK;
    if (format == VK_FORMAT_BC1_RGBA_UNORM_BLOCK) return work::renderer::Format::BC1_RGBA_UNORM_BLOCK;
    if (format == VK_FORMAT_BC1_RGBA_SRGB_BLOCK) return work::renderer::Format::BC1_RGBA_SRGB_BLOCK;
    if (format == VK_FORMAT_BC2_UNORM_BLOCK) return work::renderer::Format::BC2_UNORM_BLOCK;
    if (format == VK_FORMAT_BC2_SRGB_BLOCK) return work::renderer::Format::BC2_SRGB_BLOCK;
    if (format == VK_FORMAT_BC3_UNORM_BLOCK) return work::renderer::Format::BC3_UNORM_BLOCK;
    if (format == VK_FORMAT_BC3_SRGB_BLOCK) return work::renderer::Format::BC3_SRGB_BLOCK;
    if (format == VK_FORMAT_BC4_UNORM_BLOCK) return work::renderer::Format::BC4_UNORM_BLOCK;
    if (format == VK_FORMAT_BC4_SNORM_BLOCK) return work::renderer::Format::BC4_SNORM_BLOCK;
    if (format == VK_FORMAT_BC5_UNORM_BLOCK) return work::renderer::Format::BC5_UNORM_BLOCK;
    if (format == VK_FORMAT_BC5_SNORM_BLOCK) return work::renderer::Format::BC5_SNORM_BLOCK;
    if (format == VK_FORMAT_BC6H_UFLOAT_BLOCK) return work::renderer::Format::BC6H_UFLOAT_BLOCK;
    if (format == VK_FORMAT_BC6H_SFLOAT_BLOCK) return work::renderer::Format::BC6H_SFLOAT_BLOCK;
    if (format == VK_FORMAT_BC7_UNORM_BLOCK) return work::renderer::Format::BC7_UNORM_BLOCK;
    if (format == VK_FORMAT_BC7_SRGB_BLOCK) return work::renderer::Format::BC7_SRGB_BLOCK;
    if (format == VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK) return work::renderer::Format::ETC2_R8G8B8_UNORM_BLOCK;
    if (format == VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK) return work::renderer::Format::ETC2_R8G8B8_SRGB_BLOCK;
    if (format == VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK) return work::renderer::Format::ETC2_R8G8B8A1_UNORM_BLOCK;
    if (format == VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK) return work::renderer::Format::ETC2_R8G8B8A1_SRGB_BLOCK;
    if (format == VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK) return work::renderer::Format::ETC2_R8G8B8A8_UNORM_BLOCK;
    if (format == VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK) return work::renderer::Format::ETC2_R8G8B8A8_SRGB_BLOCK;
    if (format == VK_FORMAT_EAC_R11_UNORM_BLOCK) return work::renderer::Format::EAC_R11_UNORM_BLOCK;
    if (format == VK_FORMAT_EAC_R11_SNORM_BLOCK) return work::renderer::Format::EAC_R11_SNORM_BLOCK;
    if (format == VK_FORMAT_EAC_R11G11_UNORM_BLOCK) return work::renderer::Format::EAC_R11G11_UNORM_BLOCK;
    if (format == VK_FORMAT_EAC_R11G11_SNORM_BLOCK) return work::renderer::Format::EAC_R11G11_SNORM_BLOCK;
    if (format == VK_FORMAT_ASTC_4x4_UNORM_BLOCK) return work::renderer::Format::ASTC_4x4_UNORM_BLOCK;
    if (format == VK_FORMAT_ASTC_4x4_SRGB_BLOCK) return work::renderer::Format::ASTC_4x4_SRGB_BLOCK;
    if (format == VK_FORMAT_ASTC_5x4_UNORM_BLOCK) return work::renderer::Format::ASTC_5x4_UNORM_BLOCK;
    if (format == VK_FORMAT_ASTC_5x4_SRGB_BLOCK) return work::renderer::Format::ASTC_5x4_SRGB_BLOCK;
    if (format == VK_FORMAT_ASTC_5x5_UNORM_BLOCK) return work::renderer::Format::ASTC_5x5_UNORM_BLOCK;
    if (format == VK_FORMAT_ASTC_5x5_SRGB_BLOCK) return work::renderer::Format::ASTC_5x5_SRGB_BLOCK;
    if (format == VK_FORMAT_ASTC_6x5_UNORM_BLOCK) return work::renderer::Format::ASTC_6x5_UNORM_BLOCK;
    if (format == VK_FORMAT_ASTC_6x5_SRGB_BLOCK) return work::renderer::Format::ASTC_6x5_SRGB_BLOCK;
    if (format == VK_FORMAT_ASTC_6x6_UNORM_BLOCK) return work::renderer::Format::ASTC_6x6_UNORM_BLOCK;
    if (format == VK_FORMAT_ASTC_6x6_SRGB_BLOCK) return work::renderer::Format::ASTC_6x6_SRGB_BLOCK;
    if (format == VK_FORMAT_ASTC_8x5_UNORM_BLOCK) return work::renderer::Format::ASTC_8x5_UNORM_BLOCK;
    if (format == VK_FORMAT_ASTC_8x5_SRGB_BLOCK) return work::renderer::Format::ASTC_8x5_SRGB_BLOCK;
    if (format == VK_FORMAT_ASTC_8x6_UNORM_BLOCK) return work::renderer::Format::ASTC_8x6_UNORM_BLOCK;
    if (format == VK_FORMAT_ASTC_8x6_SRGB_BLOCK) return work::renderer::Format::ASTC_8x6_SRGB_BLOCK;
    if (format == VK_FORMAT_ASTC_8x8_UNORM_BLOCK) return work::renderer::Format::ASTC_8x8_UNORM_BLOCK;
    if (format == VK_FORMAT_ASTC_8x8_SRGB_BLOCK) return work::renderer::Format::ASTC_8x8_SRGB_BLOCK;
    if (format == VK_FORMAT_ASTC_10x5_UNORM_BLOCK) return work::renderer::Format::ASTC_10x5_UNORM_BLOCK;
    if (format == VK_FORMAT_ASTC_10x5_SRGB_BLOCK) return work::renderer::Format::ASTC_10x5_SRGB_BLOCK;
    if (format == VK_FORMAT_ASTC_10x6_UNORM_BLOCK) return work::renderer::Format::ASTC_10x6_UNORM_BLOCK;
    if (format == VK_FORMAT_ASTC_10x6_SRGB_BLOCK) return work::renderer::Format::ASTC_10x6_SRGB_BLOCK;
    if (format == VK_FORMAT_ASTC_10x8_UNORM_BLOCK) return work::renderer::Format::ASTC_10x8_UNORM_BLOCK;
    if (format == VK_FORMAT_ASTC_10x8_SRGB_BLOCK) return work::renderer::Format::ASTC_10x8_SRGB_BLOCK;
    if (format == VK_FORMAT_ASTC_10x10_UNORM_BLOCK) return work::renderer::Format::ASTC_10x10_UNORM_BLOCK;
    if (format == VK_FORMAT_ASTC_10x10_SRGB_BLOCK) return work::renderer::Format::ASTC_10x10_SRGB_BLOCK;
    if (format == VK_FORMAT_ASTC_12x10_UNORM_BLOCK) return work::renderer::Format::ASTC_12x10_UNORM_BLOCK;
    if (format == VK_FORMAT_ASTC_12x10_SRGB_BLOCK) return work::renderer::Format::ASTC_12x10_SRGB_BLOCK;
    if (format == VK_FORMAT_ASTC_12x12_UNORM_BLOCK) return work::renderer::Format::ASTC_12x12_UNORM_BLOCK;
    if (format == VK_FORMAT_ASTC_12x12_SRGB_BLOCK) return work::renderer::Format::ASTC_12x12_SRGB_BLOCK;
    if (format == VK_FORMAT_G8B8G8R8_422_UNORM) return work::renderer::Format::G8B8G8R8_422_UNORM;
    if (format == VK_FORMAT_B8G8R8G8_422_UNORM) return work::renderer::Format::B8G8R8G8_422_UNORM;
    if (format == VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM) return work::renderer::Format::G8_B8_R8_3PLANE_420_UNORM;
    if (format == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM) return work::renderer::Format::G8_B8R8_2PLANE_420_UNORM;
    if (format == VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM) return work::renderer::Format::G8_B8_R8_3PLANE_422_UNORM;
    if (format == VK_FORMAT_G8_B8R8_2PLANE_422_UNORM) return work::renderer::Format::G8_B8R8_2PLANE_422_UNORM;
    if (format == VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM) return work::renderer::Format::G8_B8_R8_3PLANE_444_UNORM;
    if (format == VK_FORMAT_R10X6_UNORM_PACK16) return work::renderer::Format::R10X6_UNORM_PACK16;
    if (format == VK_FORMAT_R10X6G10X6_UNORM_2PACK16) return work::renderer::Format::R10X6G10X6_UNORM_2PACK16;
    if (format == VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16) return work::renderer::Format::G10X6B10X6G10X6R10X6_422_UNORM_4PACK16;
    if (format == VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16) return work::renderer::Format::B10X6G10X6R10X6G10X6_422_UNORM_4PACK16;
    if (format == VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16) return work::renderer::Format::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16;
    if (format == VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16) return work::renderer::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16;
    if (format == VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16) return work::renderer::Format::G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16;
    if (format == VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16) return work::renderer::Format::G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16;
    if (format == VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16) return work::renderer::Format::G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16;
    if (format == VK_FORMAT_R12X4_UNORM_PACK16) return work::renderer::Format::R12X4_UNORM_PACK16;
    if (format == VK_FORMAT_R12X4G12X4_UNORM_2PACK16) return work::renderer::Format::R12X4G12X4_UNORM_2PACK16;
    if (format == VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16) return work::renderer::Format::R12X4G12X4B12X4A12X4_UNORM_4PACK16;
    if (format == VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16) return work::renderer::Format::G12X4B12X4G12X4R12X4_422_UNORM_4PACK16;
    if (format == VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16) return work::renderer::Format::B12X4G12X4R12X4G12X4_422_UNORM_4PACK16;
    if (format == VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16) return work::renderer::Format::G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16;
    if (format == VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16) return work::renderer::Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16;
    if (format == VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16) return work::renderer::Format::G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16;
    if (format == VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16) return work::renderer::Format::G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16;
    if (format == VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16) return work::renderer::Format::G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16;
    if (format == VK_FORMAT_G16B16G16R16_422_UNORM) return work::renderer::Format::G16B16G16R16_422_UNORM;
    if (format == VK_FORMAT_B16G16R16G16_422_UNORM) return work::renderer::Format::B16G16R16G16_422_UNORM;
    if (format == VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM) return work::renderer::Format::G16_B16_R16_3PLANE_420_UNORM;
    if (format == VK_FORMAT_G16_B16R16_2PLANE_420_UNORM) return work::renderer::Format::G16_B16R16_2PLANE_420_UNORM;
    if (format == VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM) return work::renderer::Format::G16_B16_R16_3PLANE_422_UNORM;
    if (format == VK_FORMAT_G16_B16R16_2PLANE_422_UNORM) return work::renderer::Format::G16_B16R16_2PLANE_422_UNORM;
    if (format == VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM) return work::renderer::Format::G16_B16_R16_3PLANE_444_UNORM;
    if (format == VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG) return work::renderer::Format::PVRTC1_2BPP_UNORM_BLOCK_IMG;
    if (format == VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG) return work::renderer::Format::PVRTC1_4BPP_UNORM_BLOCK_IMG;
    if (format == VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG) return work::renderer::Format::PVRTC2_2BPP_UNORM_BLOCK_IMG;
    if (format == VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG) return work::renderer::Format::PVRTC2_4BPP_UNORM_BLOCK_IMG;
    if (format == VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG) return work::renderer::Format::PVRTC1_2BPP_SRGB_BLOCK_IMG;
    if (format == VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG) return work::renderer::Format::PVRTC1_4BPP_SRGB_BLOCK_IMG;
    if (format == VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG) return work::renderer::Format::PVRTC2_2BPP_SRGB_BLOCK_IMG;
    if (format == VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG) return work::renderer::Format::PVRTC2_4BPP_SRGB_BLOCK_IMG;
    if (format == VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK_EXT) return work::renderer::Format::ASTC_4x4_SFLOAT_BLOCK_EXT;
    if (format == VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK_EXT) return work::renderer::Format::ASTC_5x4_SFLOAT_BLOCK_EXT;
    if (format == VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK_EXT) return work::renderer::Format::ASTC_5x5_SFLOAT_BLOCK_EXT;
    if (format == VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK_EXT) return work::renderer::Format::ASTC_6x5_SFLOAT_BLOCK_EXT;
    if (format == VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK_EXT) return work::renderer::Format::ASTC_6x6_SFLOAT_BLOCK_EXT;
    if (format == VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK_EXT) return work::renderer::Format::ASTC_8x5_SFLOAT_BLOCK_EXT;
    if (format == VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK_EXT) return work::renderer::Format::ASTC_8x6_SFLOAT_BLOCK_EXT;
    if (format == VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK_EXT) return work::renderer::Format::ASTC_10x5_SFLOAT_BLOCK_EXT;
    if (format == VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK_EXT) return work::renderer::Format::ASTC_10x6_SFLOAT_BLOCK_EXT;
    if (format == VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK_EXT) return work::renderer::Format::ASTC_10x8_SFLOAT_BLOCK_EXT;
    if (format == VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK_EXT) return work::renderer::Format::ASTC_10x10_SFLOAT_BLOCK_EXT;
    if (format == VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK_EXT) return work::renderer::Format::ASTC_12x10_SFLOAT_BLOCK_EXT;
    if (format == VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK_EXT) return work::renderer::Format::ASTC_12x12_SFLOAT_BLOCK_EXT;
    return work::renderer::Format::UNDEFINED;
}

VkBufferUsageFlags toVkBufferUsageFlags(work::renderer::BufferUsageFlags usage) {
    VkBufferUsageFlags result = 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::BufferUsageFlagBits::TRANSFER_SRC_BIT)) ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::BufferUsageFlagBits::TRANSFER_DST_BIT)) ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::BufferUsageFlagBits::UNIFORM_TEXEL_BUFFER_BIT)) ? VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::BufferUsageFlagBits::STORAGE_TEXEL_BUFFER_BIT)) ? VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::BufferUsageFlagBits::UNIFORM_BUFFER_BIT)) ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::BufferUsageFlagBits::STORAGE_BUFFER_BIT)) ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::BufferUsageFlagBits::INDEX_BUFFER_BIT)) ? VK_BUFFER_USAGE_INDEX_BUFFER_BIT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::BufferUsageFlagBits::VERTEX_BUFFER_BIT)) ? VK_BUFFER_USAGE_VERTEX_BUFFER_BIT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::BufferUsageFlagBits::INDIRECT_BUFFER_BIT)) ? VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::BufferUsageFlagBits::SHADER_DEVICE_ADDRESS_BIT)) ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::BufferUsageFlagBits::TRANSFORM_FEEDBACK_BUFFER_BIT_EXT)) ? VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::BufferUsageFlagBits::TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT)) ? VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::BufferUsageFlagBits::CONDITIONAL_RENDERING_BIT_EXT)) ? VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::BufferUsageFlagBits::RAY_TRACING_BIT_KHR)) ? VK_BUFFER_USAGE_RAY_TRACING_BIT_KHR : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::BufferUsageFlagBits::RAY_TRACING_BIT_NV)) ? VK_BUFFER_USAGE_RAY_TRACING_BIT_NV : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::BufferUsageFlagBits::SHADER_DEVICE_ADDRESS_BIT_EXT)) ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::BufferUsageFlagBits::SHADER_DEVICE_ADDRESS_BIT_KHR)) ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR : 0;

    return result;
}

VkImageUsageFlags toVkImageUsageFlags(work::renderer::ImageUsageFlags usage) {
    VkImageUsageFlags result = 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::ImageUsageFlagBits::TRANSFER_SRC_BIT)) ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::ImageUsageFlagBits::TRANSFER_DST_BIT)) ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::ImageUsageFlagBits::SAMPLED_BIT)) ? VK_IMAGE_USAGE_SAMPLED_BIT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::ImageUsageFlagBits::STORAGE_BIT)) ? VK_IMAGE_USAGE_STORAGE_BIT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::ImageUsageFlagBits::COLOR_ATTACHMENT_BIT)) ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT_BIT)) ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::ImageUsageFlagBits::TRANSIENT_ATTACHMENT_BIT)) ? VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::ImageUsageFlagBits::INPUT_ATTACHMENT_BIT)) ? VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::ImageUsageFlagBits::SHADING_RATE_IMAGE_BIT_NV)) ? VK_IMAGE_USAGE_SHADING_RATE_IMAGE_BIT_NV : 0;
    result |= (usage & static_cast<uint32_t>(work::renderer::ImageUsageFlagBits::FRAGMENT_DENSITY_MAP_BIT_EXT)) ? VK_IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT : 0;

    return result;
}

VkImageCreateFlags toVkImageCreateFlags(work::renderer::ImageCreateFlags flags) {
    VkImageCreateFlags result = 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageCreateFlagBits::SPARSE_BINDING_BIT)) ? VK_IMAGE_CREATE_SPARSE_BINDING_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageCreateFlagBits::SPARSE_RESIDENCY_BIT)) ? VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageCreateFlagBits::SPARSE_ALIASED_BIT)) ? VK_IMAGE_CREATE_SPARSE_ALIASED_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageCreateFlagBits::MUTABLE_FORMAT_BIT)) ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageCreateFlagBits::CUBE_COMPATIBLE_BIT)) ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageCreateFlagBits::ALIAS_BIT)) ? VK_IMAGE_CREATE_ALIAS_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageCreateFlagBits::SPLIT_INSTANCE_BIND_REGIONS_BIT)) ? VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageCreateFlagBits::TWO_D_ARRAY_COMPATIBLE_BIT)) ? VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageCreateFlagBits::BLOCK_TEXEL_VIEW_COMPATIBLE_BIT)) ? VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageCreateFlagBits::EXTENDED_USAGE_BIT)) ? VK_IMAGE_CREATE_EXTENDED_USAGE_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageCreateFlagBits::PROTECTED_BIT)) ? VK_IMAGE_CREATE_PROTECTED_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageCreateFlagBits::DISJOINT_BIT)) ? VK_IMAGE_CREATE_DISJOINT_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageCreateFlagBits::CORNER_SAMPLED_BIT_NV)) ? VK_IMAGE_CREATE_CORNER_SAMPLED_BIT_NV : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageCreateFlagBits::SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT)) ? VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageCreateFlagBits::SUBSAMPLED_BIT_EXT)) ? VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT : 0;

    return result;
}

VkImageTiling toVkImageTiling(work::renderer::ImageTiling tiling) {
    if (tiling == work::renderer::ImageTiling::LINEAR) return VK_IMAGE_TILING_LINEAR;
    else if (tiling == work::renderer::ImageTiling::DRM_FORMAT_MODIFIER_EXT) return VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
    else return VK_IMAGE_TILING_OPTIMAL;
}

VkImageLayout toVkImageLayout(work::renderer::ImageLayout layout) {
    if (layout == work::renderer::ImageLayout::GENERAL) return VK_IMAGE_LAYOUT_GENERAL;
    else if (layout == work::renderer::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    else if (layout == work::renderer::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    else if (layout == work::renderer::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL) return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    else if (layout == work::renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL) return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    else if (layout == work::renderer::ImageLayout::TRANSFER_SRC_OPTIMAL) return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    else if (layout == work::renderer::ImageLayout::TRANSFER_DST_OPTIMAL) return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    else if (layout == work::renderer::ImageLayout::PREINITIALIZED) return VK_IMAGE_LAYOUT_PREINITIALIZED;
    else if (layout == work::renderer::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL) return VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL;
    else if (layout == work::renderer::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL) return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL;
    else if (layout == work::renderer::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL) return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    else if (layout == work::renderer::ImageLayout::DEPTH_READ_ONLY_OPTIMAL) return VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
    else if (layout == work::renderer::ImageLayout::STENCIL_ATTACHMENT_OPTIMAL) return VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL;
    else if (layout == work::renderer::ImageLayout::STENCIL_READ_ONLY_OPTIMAL) return VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL;
    else if (layout == work::renderer::ImageLayout::PRESENT_SRC_KHR) return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    else if (layout == work::renderer::ImageLayout::SHARED_PRESENT_KHR) return VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR;
    else if (layout == work::renderer::ImageLayout::SHADING_RATE_OPTIMAL_NV) return VK_IMAGE_LAYOUT_SHADING_RATE_OPTIMAL_NV;
    else if (layout == work::renderer::ImageLayout::FRAGMENT_DENSITY_MAP_OPTIMAL_EXT) return VK_IMAGE_LAYOUT_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT;
    return VK_IMAGE_LAYOUT_UNDEFINED;
}

VkShaderStageFlagBits toVkShaderStageFlagBits(work::renderer::ShaderStageFlagBits stage) {
    if (stage == work::renderer::ShaderStageFlagBits::VERTEX_BIT) return VK_SHADER_STAGE_VERTEX_BIT;
    else if (stage == work::renderer::ShaderStageFlagBits::TESSELLATION_CONTROL_BIT) return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    else if (stage == work::renderer::ShaderStageFlagBits::TESSELLATION_EVALUATION_BIT) return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    else if (stage == work::renderer::ShaderStageFlagBits::GEOMETRY_BIT) return VK_SHADER_STAGE_GEOMETRY_BIT;
    else if (stage == work::renderer::ShaderStageFlagBits::FRAGMENT_BIT) return VK_SHADER_STAGE_FRAGMENT_BIT;
    else if (stage == work::renderer::ShaderStageFlagBits::COMPUTE_BIT) return VK_SHADER_STAGE_COMPUTE_BIT;
    else if (stage == work::renderer::ShaderStageFlagBits::ALL_GRAPHICS) return VK_SHADER_STAGE_ALL_GRAPHICS;
    else if (stage == work::renderer::ShaderStageFlagBits::ALL) return VK_SHADER_STAGE_ALL;
    else if (stage == work::renderer::ShaderStageFlagBits::RAYGEN_BIT_KHR) return VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    else if (stage == work::renderer::ShaderStageFlagBits::ANY_HIT_BIT_KHR) return VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    else if (stage == work::renderer::ShaderStageFlagBits::CLOSEST_HIT_BIT_KHR) return VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    else if (stage == work::renderer::ShaderStageFlagBits::MISS_BIT_KHR) return VK_SHADER_STAGE_MISS_BIT_KHR;
    else if (stage == work::renderer::ShaderStageFlagBits::INTERSECTION_BIT_KHR) return VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    else if (stage == work::renderer::ShaderStageFlagBits::CALLABLE_BIT_KHR) return VK_SHADER_STAGE_CALLABLE_BIT_KHR;
    else if (stage == work::renderer::ShaderStageFlagBits::TASK_BIT_NV) return VK_SHADER_STAGE_TASK_BIT_NV;
    else if (stage == work::renderer::ShaderStageFlagBits::MESH_BIT_NV) return VK_SHADER_STAGE_MESH_BIT_NV;
    else if (stage == work::renderer::ShaderStageFlagBits::MESH_BIT_NV) return VK_SHADER_STAGE_MESH_BIT_NV;
    return VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM;
};

VkMemoryPropertyFlags toVkMemoryPropertyFlags(work::renderer::MemoryPropertyFlags properties) {
    VkMemoryPropertyFlags result = 0;
    result |= (properties & static_cast<uint32_t>(work::renderer::MemoryPropertyFlagBits::DEVICE_LOCAL_BIT)) ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : 0;
    result |= (properties & static_cast<uint32_t>(work::renderer::MemoryPropertyFlagBits::HOST_VISIBLE_BIT)) ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT : 0;
    result |= (properties & static_cast<uint32_t>(work::renderer::MemoryPropertyFlagBits::HOST_COHERENT_BIT)) ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : 0;
    result |= (properties & static_cast<uint32_t>(work::renderer::MemoryPropertyFlagBits::HOST_CACHED_BIT)) ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0;
    result |= (properties & static_cast<uint32_t>(work::renderer::MemoryPropertyFlagBits::LAZILY_ALLOCATED_BIT)) ? VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT : 0;
    result |= (properties & static_cast<uint32_t>(work::renderer::MemoryPropertyFlagBits::PROTECTED_BIT)) ? VK_MEMORY_PROPERTY_PROTECTED_BIT : 0;
    result |= (properties & static_cast<uint32_t>(work::renderer::MemoryPropertyFlagBits::DEVICE_COHERENT_BIT_AMD)) ? VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD : 0;
    result |= (properties & static_cast<uint32_t>(work::renderer::MemoryPropertyFlagBits::DEVICE_UNCACHED_BIT_AMD)) ? VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD : 0;

    return result;
}

VkCommandBufferUsageFlags toCommandBufferUsageFlags(work::renderer::CommandBufferUsageFlags flags) {
    VkCommandBufferUsageFlags result = 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::CommandBufferUsageFlagBits::ONE_TIME_SUBMIT_BIT)) ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::CommandBufferUsageFlagBits::RENDER_PASS_CONTINUE_BIT)) ? VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::CommandBufferUsageFlagBits::SIMULTANEOUS_USE_BIT)) ? VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT : 0;

    return result;
}

VkImageType toVkImageType(work::renderer::ImageType view_type) {
    if (view_type == work::renderer::ImageType::TYPE_1D) return VK_IMAGE_TYPE_1D;
    else if (view_type == work::renderer::ImageType::TYPE_2D) return VK_IMAGE_TYPE_2D;
    return VK_IMAGE_TYPE_3D;
}

VkImageViewType toVkImageViewType(work::renderer::ImageViewType view_type) {
    if (view_type == work::renderer::ImageViewType::VIEW_1D) return VK_IMAGE_VIEW_TYPE_1D;
    else if (view_type == work::renderer::ImageViewType::VIEW_2D) return VK_IMAGE_VIEW_TYPE_2D;
    else if (view_type == work::renderer::ImageViewType::VIEW_3D) return VK_IMAGE_VIEW_TYPE_3D;
    else if (view_type == work::renderer::ImageViewType::VIEW_CUBE) return VK_IMAGE_VIEW_TYPE_CUBE;
    else if (view_type == work::renderer::ImageViewType::VIEW_1D_ARRAY) return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    else if (view_type == work::renderer::ImageViewType::VIEW_2D_ARRAY) return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    else if (view_type == work::renderer::ImageViewType::VIEW_CUBE_ARRAY) return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    return VK_IMAGE_VIEW_TYPE_2D;
}

VkImageAspectFlags toVkImageAspectFlags(work::renderer::ImageAspectFlags flags) {
    VkImageAspectFlags result = 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageAspectFlagBits::COLOR_BIT)) ? VK_IMAGE_ASPECT_COLOR_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageAspectFlagBits::DEPTH_BIT)) ? VK_IMAGE_ASPECT_DEPTH_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageAspectFlagBits::STENCIL_BIT)) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageAspectFlagBits::METADATA_BIT)) ? VK_IMAGE_ASPECT_METADATA_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageAspectFlagBits::PLANE_0_BIT)) ? VK_IMAGE_ASPECT_PLANE_0_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageAspectFlagBits::PLANE_1_BIT)) ? VK_IMAGE_ASPECT_PLANE_1_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageAspectFlagBits::PLANE_2_BIT)) ? VK_IMAGE_ASPECT_PLANE_2_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageAspectFlagBits::MEMORY_PLANE_0_BIT_EXT)) ? VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageAspectFlagBits::MEMORY_PLANE_1_BIT_EXT)) ? VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageAspectFlagBits::MEMORY_PLANE_2_BIT_EXT)) ? VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::ImageAspectFlagBits::MEMORY_PLANE_3_BIT_EXT)) ? VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT : 0;
    return result;
}

VkPipelineBindPoint toVkPipelineBindPoint(work::renderer::PipelineBindPoint bind) {
    if (bind == work::renderer::PipelineBindPoint::COMPUTE) return VK_PIPELINE_BIND_POINT_COMPUTE;
    else if (bind == work::renderer::PipelineBindPoint::RAY_TRACING) return VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
    return VK_PIPELINE_BIND_POINT_GRAPHICS;
}

VkCommandPoolCreateFlags toVkCommandPoolCreateFlags(work::renderer::CommandPoolCreateFlags flags) {
    VkCommandPoolCreateFlags result = 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::CommandPoolCreateFlagBits::TRANSIENT_BIT)) ? VK_COMMAND_POOL_CREATE_TRANSIENT_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::CommandPoolCreateFlagBits::RESET_COMMAND_BUFFER_BIT)) ? VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT : 0;
    result |= (flags & static_cast<uint32_t>(work::renderer::CommandPoolCreateFlagBits::PROTECTED_BIT)) ? VK_COMMAND_POOL_CREATE_PROTECTED_BIT : 0;
    return result;
}

VkFilter toVkFilter(work::renderer::Filter filter) {
    if (filter == work::renderer::Filter::NEAREST) return VK_FILTER_NEAREST;
    else if (filter == work::renderer::Filter::CUBIC_IMG) return VK_FILTER_CUBIC_IMG;
    return VK_FILTER_LINEAR;
}

VkSamplerAddressMode toVkSamplerAddressMode(work::renderer::SamplerAddressMode mode) {
    if (mode == work::renderer::SamplerAddressMode::MIRRORED_REPEAT) return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    else if (mode == work::renderer::SamplerAddressMode::CLAMP_TO_EDGE) return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    else if (mode == work::renderer::SamplerAddressMode::CLAMP_TO_BORDER) return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    else if (mode == work::renderer::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE) return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

VkSamplerMipmapMode toVkSamplerMipmapMode(work::renderer::SamplerMipmapMode mode) {
    if (mode == work::renderer::SamplerMipmapMode::NEAREST) return VK_SAMPLER_MIPMAP_MODE_NEAREST;
    return VK_SAMPLER_MIPMAP_MODE_LINEAR;
}

VkColorSpaceKHR toVkColorSpace(work::renderer::ColorSpace color_space) {
    if (color_space == work::renderer::ColorSpace::DISPLAY_P3_NONLINEAR_EXT) return VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT;
    else if (color_space == work::renderer::ColorSpace::DISPLAY_P3_LINEAR_EXT) return VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT;
    else if (color_space == work::renderer::ColorSpace::DCI_P3_NONLINEAR_EXT) return VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT;
    else if (color_space == work::renderer::ColorSpace::BT709_LINEAR_EXT) return VK_COLOR_SPACE_BT709_LINEAR_EXT;
    else if (color_space == work::renderer::ColorSpace::BT709_NONLINEAR_EXT) return VK_COLOR_SPACE_BT709_NONLINEAR_EXT;
    else if (color_space == work::renderer::ColorSpace::BT2020_LINEAR_EXT) return VK_COLOR_SPACE_BT2020_LINEAR_EXT;
    else if (color_space == work::renderer::ColorSpace::HDR10_ST2084_EXT) return VK_COLOR_SPACE_HDR10_ST2084_EXT;
    else if (color_space == work::renderer::ColorSpace::DOLBYVISION_EXT) return VK_COLOR_SPACE_DOLBYVISION_EXT;
    else if (color_space == work::renderer::ColorSpace::HDR10_HLG_EXT) return VK_COLOR_SPACE_HDR10_HLG_EXT;
    else if (color_space == work::renderer::ColorSpace::ADOBERGB_LINEAR_EXT) return VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT;
    else if (color_space == work::renderer::ColorSpace::ADOBERGB_NONLINEAR_EXT) return VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT;
    else if (color_space == work::renderer::ColorSpace::PASS_THROUGH_EXT) return VK_COLOR_SPACE_PASS_THROUGH_EXT;
    else if (color_space == work::renderer::ColorSpace::EXTENDED_SRGB_NONLINEAR_EXT) return VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT;
    else if (color_space == work::renderer::ColorSpace::DISPLAY_NATIVE_AMD) return VK_COLOR_SPACE_DISPLAY_NATIVE_AMD;
    return VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
}

work::renderer::ColorSpace fromVkColorSpace(VkColorSpaceKHR color_space) {
    if (color_space == VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT) return work::renderer::ColorSpace::DISPLAY_P3_NONLINEAR_EXT;
    else if (color_space == VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT) return work::renderer::ColorSpace::DISPLAY_P3_LINEAR_EXT;
    else if (color_space == VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT) return work::renderer::ColorSpace::DCI_P3_NONLINEAR_EXT;
    else if (color_space == VK_COLOR_SPACE_BT709_LINEAR_EXT) return work::renderer::ColorSpace::BT709_LINEAR_EXT;
    else if (color_space == VK_COLOR_SPACE_BT709_NONLINEAR_EXT) return work::renderer::ColorSpace::BT709_NONLINEAR_EXT;
    else if (color_space == VK_COLOR_SPACE_BT2020_LINEAR_EXT) return work::renderer::ColorSpace::BT2020_LINEAR_EXT;
    else if (color_space == VK_COLOR_SPACE_HDR10_ST2084_EXT) return work::renderer::ColorSpace::HDR10_ST2084_EXT;
    else if (color_space == VK_COLOR_SPACE_DOLBYVISION_EXT) return work::renderer::ColorSpace::DOLBYVISION_EXT;
    else if (color_space == VK_COLOR_SPACE_HDR10_HLG_EXT) return work::renderer::ColorSpace::HDR10_HLG_EXT;
    else if (color_space == VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT) return work::renderer::ColorSpace::ADOBERGB_LINEAR_EXT;
    else if (color_space == VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT) return work::renderer::ColorSpace::ADOBERGB_NONLINEAR_EXT;
    else if (color_space == VK_COLOR_SPACE_PASS_THROUGH_EXT) return work::renderer::ColorSpace::PASS_THROUGH_EXT;
    else if (color_space == VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT) return work::renderer::ColorSpace::EXTENDED_SRGB_NONLINEAR_EXT;
    else if (color_space == VK_COLOR_SPACE_DISPLAY_NATIVE_AMD) return work::renderer::ColorSpace::DISPLAY_NATIVE_AMD;
    return work::renderer::ColorSpace::SRGB_NONLINEAR_KHR;
}

VkPresentModeKHR toVkPresentMode(work::renderer::PresentMode mode) {
    if (mode == work::renderer::PresentMode::MAILBOX_KHR) return VK_PRESENT_MODE_MAILBOX_KHR;
    else if (mode == work::renderer::PresentMode::FIFO_KHR) return VK_PRESENT_MODE_FIFO_KHR;
    else if (mode == work::renderer::PresentMode::FIFO_RELAXED_KHR) return VK_PRESENT_MODE_FIFO_RELAXED_KHR;
    else if (mode == work::renderer::PresentMode::SHARED_DEMAND_REFRESH_KHR) return VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR;
    else if (mode == work::renderer::PresentMode::SHARED_CONTINUOUS_REFRESH_KHR) return VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR;
    return VK_PRESENT_MODE_IMMEDIATE_KHR;
}

VkSurfaceTransformFlagBitsKHR toVkSurfaceTransformFlags(work::renderer::SurfaceTransformFlagBits flag) {
    if (flag == work::renderer::SurfaceTransformFlagBits::ROTATE_90_BIT_KHR) return VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR;
    else if (flag == work::renderer::SurfaceTransformFlagBits::ROTATE_180_BIT_KHR) return VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR;
    else if (flag == work::renderer::SurfaceTransformFlagBits::ROTATE_270_BIT_KHR) return VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR;
    else if (flag == work::renderer::SurfaceTransformFlagBits::HORIZONTAL_MIRROR_BIT_KHR) return VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_BIT_KHR;
    else if (flag == work::renderer::SurfaceTransformFlagBits::HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR) return VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR;
    else if (flag == work::renderer::SurfaceTransformFlagBits::HORIZONTAL_MIRROR_ROTATE_180_BIT_KHR) return VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_180_BIT_KHR;
    else if (flag == work::renderer::SurfaceTransformFlagBits::HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR) return VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR;
    else if (flag == work::renderer::SurfaceTransformFlagBits::INHERIT_BIT_KHR) return VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR;
    return VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
}

work::renderer::SurfaceTransformFlagBits fromVkSurfaceTransformFlags(VkSurfaceTransformFlagBitsKHR flag) {
    if (flag == VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR) return work::renderer::SurfaceTransformFlagBits::ROTATE_90_BIT_KHR;
    else if (flag == VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR) return work::renderer::SurfaceTransformFlagBits::ROTATE_180_BIT_KHR;
    else if (flag == VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR) return work::renderer::SurfaceTransformFlagBits::ROTATE_270_BIT_KHR;
    else if (flag == VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_BIT_KHR) return work::renderer::SurfaceTransformFlagBits::HORIZONTAL_MIRROR_BIT_KHR;
    else if (flag == VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR) return work::renderer::SurfaceTransformFlagBits::HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR;
    else if (flag == VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_180_BIT_KHR) return work::renderer::SurfaceTransformFlagBits::HORIZONTAL_MIRROR_ROTATE_180_BIT_KHR;
    else if (flag == VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR) return work::renderer::SurfaceTransformFlagBits::HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR;
    else if (flag == VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR) return work::renderer::SurfaceTransformFlagBits::INHERIT_BIT_KHR;
    return work::renderer::SurfaceTransformFlagBits::IDENTITY_BIT_KHR;
}

VkIndexType toVkIndexType(work::renderer::IndexType index_type) {
    if (index_type == work::renderer::IndexType::UINT32) return VK_INDEX_TYPE_UINT32;
    else if (index_type == work::renderer::IndexType::NONE_KHR) return VK_INDEX_TYPE_NONE_KHR;
    else if (index_type == work::renderer::IndexType::UINT8_EXT) return VK_INDEX_TYPE_UINT8_EXT;
    return VK_INDEX_TYPE_UINT16;
}

VkVertexInputRate toVkVertexInputRate(work::renderer::VertexInputRate input_rate) {
    if (input_rate == work::renderer::VertexInputRate::INSTANCE) return VK_VERTEX_INPUT_RATE_INSTANCE;
    return VK_VERTEX_INPUT_RATE_VERTEX;
}

std::vector<VkVertexInputBindingDescription> toVkVertexInputBindingDescription(
    const std::vector<work::renderer::VertexInputBindingDescription>& description) {
    std::vector<VkVertexInputBindingDescription> binding_description(description.size());
    for (int i = 0; i < description.size(); i++) {
        binding_description[i].binding = description[i].binding;
        binding_description[i].stride = description[i].stride;
        binding_description[i].inputRate = toVkVertexInputRate(description[i].input_rate);
    }
    return binding_description;
}

std::vector<VkVertexInputAttributeDescription> toVkVertexInputAttributeDescription(
    const std::vector<work::renderer::VertexInputAttributeDescription>& description) {
    std::vector<VkVertexInputAttributeDescription> result(description.size());
    for (int i = 0; i < description.size(); i++) {
        result[i].binding = description[i].binding;
        result[i].location = description[i].location;
        result[i].offset = static_cast<uint32_t>(description[i].offset);
        result[i].format = toVkFormat(description[i].format);
    }

    return result;
}

VkPrimitiveTopology toVkPrimitiveTopology(work::renderer::PrimitiveTopology primitive) {
    if (primitive == work::renderer::PrimitiveTopology::LINE_LIST) return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    else if (primitive == work::renderer::PrimitiveTopology::LINE_STRIP) return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    else if (primitive == work::renderer::PrimitiveTopology::TRIANGLE_LIST) return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    else if (primitive == work::renderer::PrimitiveTopology::TRIANGLE_STRIP) return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    else if (primitive == work::renderer::PrimitiveTopology::TRIANGLE_FAN) return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    else if (primitive == work::renderer::PrimitiveTopology::LINE_LIST_WITH_ADJACENCY) return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
    else if (primitive == work::renderer::PrimitiveTopology::LINE_STRIP_WITH_ADJACENCY) return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
    else if (primitive == work::renderer::PrimitiveTopology::TRIANGLE_LIST_WITH_ADJACENCY) return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;
    else if (primitive == work::renderer::PrimitiveTopology::TRIANGLE_STRIP_WITH_ADJACENCY) return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
    else if (primitive == work::renderer::PrimitiveTopology::PATCH_LIST) return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
}

static std::string getFilePathExtension(const std::string& file_name) {
    if (file_name.find_last_of(".") != std::string::npos)
        return file_name.substr(file_name.find_last_of(".") + 1);
    return "";
}

bool checkValidationLayerSupport() {
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

    std::vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

    for (const char* layer_name : validation_layers) {
        bool layer_found = false;

        for (const auto& layer_properties : available_layers) {
            if (strcmp(layer_name, layer_properties.layerName) == 0) {
                layer_found = true;
                break;
            }
        }

        if (!layer_found) {
            return false;
        }
    }

    return true;
}

std::vector<const char*> getRequiredExtensions() {
    uint32_t glfw_extensionCount = 0;
    const char** glfw_extensions;
    glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extensionCount);

    std::vector<const char*> extensions(glfw_extensions, glfw_extensions + glfw_extensionCount);

    if (enable_validation_layers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
    void* pUserData) {

    std::cerr << "validation layer: " << p_callback_data->pMessage << std::endl;

    return VK_FALSE;
}

VkResult CreateDebugUtilsMessengerEXT(const VkInstance& instance,
                                      const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator,
                                      VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(const VkInstance& instance,
                                   VkDebugUtilsMessengerEXT debug_messenger,
                                   const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debug_messenger, pAllocator);
    }
}

void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& create_info) {
    create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    create_info.pfnUserCallback = debugCallback;
}

int rateDeviceSuitability(const VkPhysicalDevice& device) {
    VkPhysicalDeviceProperties device_properties;
    VkPhysicalDeviceFeatures device_features;
    vkGetPhysicalDeviceProperties(device, &device_properties);
    vkGetPhysicalDeviceFeatures(device, &device_features);

    int score = 0;

    // Discrete GPUs have a significant performance advantage
    if (device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += 1000;
    }

    // Maximum possible size of textures affects graphics quality
    score += device_properties.limits.maxImageDimension2D;
    max_vertex_input_attribute_offset = device_properties.limits.maxVertexInputAttributeOffset;

    // Application can't function without geometry shaders
    if (!device_features.geometryShader) {
        return 0;
    }

    return score;
}

QueueFamilyIndices findQueueFamilies(
    const std::shared_ptr<VkPhysicalDevice>& device,
    const std::shared_ptr<work::renderer::Surface>& surface) {
    QueueFamilyIndices indices;

    assert(device);
    auto vk_surface = std::reinterpret_pointer_cast<work::renderer::VulkanSurface>(surface);
    assert(vk_surface);

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(*device, &queue_family_count, nullptr);

    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(*device, &queue_family_count, queue_families.data());

    int i = 0;
    for (const auto& queue_family : queue_families) {
        if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphics_family_ = i;
        }

        VkBool32 present_support = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(*device, i, vk_surface->get(), &present_support);

        if (present_support) {
            indices.present_family_ = i;
        }
        if (indices.isComplete()) {
            break;
        }

        i++;
    }

    return indices;
}

bool checkDeviceExtensionSupport(const std::shared_ptr<VkPhysicalDevice>& device) {
    assert(device);
    uint32_t extension_count;
    vkEnumerateDeviceExtensionProperties(*device, nullptr, &extension_count, nullptr);

    std::vector<VkExtensionProperties> available_extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(*device, nullptr, &extension_count, available_extensions.data());

    std::set<std::string> required_extensions(device_extensions.begin(), device_extensions.end());

    for (const auto& extension : available_extensions) {
        required_extensions.erase(extension.extensionName);
    }

    return required_extensions.empty();
}

SwapChainSupportDetails querySwapChainSupport(
    const std::shared_ptr<VkPhysicalDevice>& device, 
    const std::shared_ptr<work::renderer::Surface>& surface) {
    SwapChainSupportDetails details;
    assert(device);

    auto vk_surface = std::reinterpret_pointer_cast<work::renderer::VulkanSurface>(surface);
    assert(vk_surface);

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(*device, vk_surface->get(), &details.capabilities_);

    uint32_t format_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(*device, vk_surface->get(), &format_count, nullptr);

    if (format_count != 0) {
        details.formats_.resize(format_count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(*device, vk_surface->get(), &format_count, details.formats_.data());
    }

    uint32_t present_mode_count;
    vkGetPhysicalDeviceSurfacePresentModesKHR(*device, vk_surface->get(), &present_mode_count, nullptr);

    if (present_mode_count != 0) {
        details.present_modes_.resize(present_mode_count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(*device, vk_surface->get(), &present_mode_count, details.present_modes_.data());
    }

    return details;
}

bool isDeviceSuitable(
    const std::shared_ptr<VkPhysicalDevice>& device, 
    const std::shared_ptr<work::renderer::Surface>& surface) {
    QueueFamilyIndices indices = findQueueFamilies(device, surface);
    
    bool extensions_supported = checkDeviceExtensionSupport(device);

    bool swap_chain_adequate = false;
    if (extensions_supported) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device, surface);
        swap_chain_adequate = !swapChainSupport.formats_.empty() && !swapChainSupport.present_modes_.empty();
    }

    assert(device);
    VkPhysicalDeviceFeatures supported_features;
    vkGetPhysicalDeviceFeatures(*device, &supported_features);

    return indices.isComplete() && extensions_supported && swap_chain_adequate && supported_features.samplerAnisotropy;
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available_formats) {
    for (const auto& available_format : available_formats) {
        if (available_format.format == VK_FORMAT_B8G8R8A8_UNORM &&
            available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return available_format;
        }
    }

    return available_formats[0];
}

work::renderer::PresentMode chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return work::renderer::PresentMode::MAILBOX_KHR;
        }
    }

    return work::renderer::PresentMode::FIFO_KHR;
}

VkExtent2D chooseSwapExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }
    else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
        actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

        return actualExtent;
    }
}

std::vector<uint64_t> readFile(const std::string& file_name, uint64_t& file_size) {
    std::ifstream file(file_name, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        std::string error_message = std::string("failed to open file! :") + file_name;
        throw std::runtime_error(error_message);
    }

    file_size = (uint64_t)file.tellg();
    std::vector<uint64_t> buffer((file_size + sizeof(uint64_t) - 1) / sizeof(uint64_t));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);

    file.close();
    return buffer;
}

static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<work::app::RealWorldApplication*>(glfwGetWindowUserPointer(window));
    app->setFrameBufferResized(true);
}

uint32_t findMemoryType(const VkPhysicalDevice& physical_device, 
                        uint32_t type_filter, 
                        VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void updateBufferMemory(const std::shared_ptr<work::renderer::Device>& device,
                        const std::shared_ptr<work::renderer::DeviceMemory>& memory,
                        uint64_t size,
                        const void* src_data,
                        uint64_t offset = 0) {
    if (device && memory) {
        void* dst_data = device->mapMemory(memory, size, offset);
        assert(dst_data);
        memcpy(dst_data, src_data, size);
        device->unmapMemory(memory);
    }
}

void createBuffer(const std::shared_ptr<work::renderer::Device>& device,
                  const uint64_t& buffer_size, 
                  const work::renderer::BufferUsageFlags& usage,
                  const work::renderer::MemoryPropertyFlags& properties, 
                  std::shared_ptr<work::renderer::Buffer>& buffer, 
                  std::shared_ptr<work::renderer::DeviceMemory>& buffer_memory) {
    buffer = device->createBuffer(buffer_size, usage);
    auto mem_requirements = device->getBufferMemoryRequirements(buffer);
    buffer_memory = device->allocateMemory(mem_requirements.size,
        mem_requirements.memory_type_bits,
        toVkMemoryPropertyFlags(properties));
    device->bindBufferMemory(buffer, buffer_memory);
}

void copyBuffer(const std::shared_ptr<work::renderer::Device>& device,
                const std::shared_ptr<work::renderer::Queue>& cmd_queue,
                const std::shared_ptr<work::renderer::CommandPool>& cmd_pool, 
                const std::shared_ptr<work::renderer::Buffer> src_buffer,
                const std::shared_ptr<work::renderer::Buffer> dst_buffer,
                uint64_t buffer_size) {

    auto command_buffers = device->allocateCommandBuffers(cmd_pool, 1);
    if (command_buffers.size() > 0) {
        auto& cmd_buf = command_buffers[0];
        if (cmd_buf) {
            cmd_buf->beginCommandBuffer(static_cast<work::renderer::CommandBufferUsageFlags>(work::renderer::CommandBufferUsageFlagBits::ONE_TIME_SUBMIT_BIT));

            std::vector<work::renderer::BufferCopyInfo> copy_regions(1);
            copy_regions[0].src_offset = 0; // Optional
            copy_regions[0].dst_offset = 0; // Optional
            copy_regions[0].size = buffer_size;
            cmd_buf->copyBuffer(src_buffer, dst_buffer, copy_regions);

            cmd_buf->endCommandBuffer();
        }

        cmd_queue->submit(command_buffers);
        cmd_queue->waitIdle();
        device->freeCommandBuffers(cmd_pool, command_buffers);
    }
}

void createBufferWithSrcData(const std::shared_ptr<work::renderer::Device>& device,
                          const std::shared_ptr<work::renderer::Queue>& cmd_queue,
                          const std::shared_ptr<work::renderer::CommandPool>& cmd_pool,
                          const work::renderer::BufferUsageFlags& usage,
                          const uint64_t& buffer_size,
                          const void* src_data,
                          std::shared_ptr<work::renderer::Buffer>& buffer,
                          std::shared_ptr<work::renderer::DeviceMemory>& buffer_memory) {
    std::shared_ptr<work::renderer::Buffer> staging_buffer;
    std::shared_ptr<work::renderer::DeviceMemory> staging_buffer_memory;
    createBuffer(device,
        buffer_size,
        static_cast<work::renderer::BufferUsageFlags>(work::renderer::BufferUsageFlagBits::TRANSFER_SRC_BIT),
        static_cast<work::renderer::MemoryPropertyFlags>(work::renderer::MemoryPropertyFlagBits::HOST_VISIBLE_BIT) | 
        static_cast<work::renderer::MemoryPropertyFlags>(work::renderer::MemoryPropertyFlagBits::HOST_COHERENT_BIT),
        staging_buffer,
        staging_buffer_memory);

    updateBufferMemory(device, staging_buffer_memory, buffer_size, src_data);

    createBuffer(device,
        buffer_size,
        static_cast<work::renderer::BufferUsageFlags>(work::renderer::BufferUsageFlagBits::TRANSFER_DST_BIT) | usage,
        static_cast<work::renderer::MemoryPropertyFlags>(work::renderer::MemoryPropertyFlagBits::DEVICE_LOCAL_BIT),
        buffer,
        buffer_memory);

    copyBuffer(device, cmd_queue, cmd_pool, staging_buffer, buffer, buffer_size);

    device->destroyBuffer(staging_buffer);
    device->freeMemory(staging_buffer_memory);
}

void create2DImage(
    const std::shared_ptr<work::renderer::Device>& device, 
    glm::vec2 tex_size,
    const work::renderer::Format& format, 
    const work::renderer::ImageTiling& tiling, 
    const work::renderer::ImageUsageFlags& usage, 
    const work::renderer::MemoryPropertyFlags& properties,
    std::shared_ptr<work::renderer::Image>& image, 
    std::shared_ptr<work::renderer::DeviceMemory>& image_memory) {
    image = device->createImage(work::renderer::ImageType::TYPE_2D, glm::uvec3(tex_size, 1), format, usage, tiling, work::renderer::ImageLayout::UNDEFINED);
    auto mem_requirements = device->getImageMemoryRequirements(image);
    image_memory = device->allocateMemory(mem_requirements.size,
        mem_requirements.memory_type_bits,
        toVkMemoryPropertyFlags(properties));
    device->bindImageMemory(image, image_memory);
}

work::renderer::Format findSupportedFormat(
    const std::shared_ptr<VkPhysicalDevice>& physical_device,
    const std::vector<work::renderer::Format>& candidates,
    const VkImageTiling& tiling,
    const VkFormatFeatureFlags& features) {
    assert(physical_device);
    for (auto format : candidates) {
        VkFormatProperties props;
        auto vk_format = toVkFormat(format);
        vkGetPhysicalDeviceFormatProperties(*physical_device, vk_format, &props);
        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
        }
        else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}

work::renderer::Format findDepthFormat(const std::shared_ptr<VkPhysicalDevice>& physical_device) {
    return findSupportedFormat(physical_device,
        { work::renderer::Format::D32_SFLOAT_S8_UINT, 
          work::renderer::Format::D32_SFLOAT,
          work::renderer::Format::D24_UNORM_S8_UINT,
          work::renderer::Format::D16_UNORM_S8_UINT,
          work::renderer::Format::D16_UNORM },
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
}

bool hasStencilComponent(const work::renderer::Format& format) {
    return format == work::renderer::Format::D32_SFLOAT_S8_UINT || 
           format == work::renderer::Format::D24_UNORM_S8_UINT || 
           format == work::renderer::Format::D16_UNORM_S8_UINT;
}

void transitionImageLayout(const std::shared_ptr<work::renderer::Device>& device,
                           const std::shared_ptr<work::renderer::Queue>& cmd_queue,
                           const std::shared_ptr<work::renderer::CommandPool>& cmd_pool, 
                           const std::shared_ptr<work::renderer::Image>& image, 
                           const work::renderer::Format& format, 
                           const work::renderer::ImageLayout& old_layout,
                           const work::renderer::ImageLayout& new_layout,
                           uint32_t base_mip_idx = 0,
                           uint32_t mip_count = 1,
                           uint32_t base_layer = 0,
                           uint32_t layer_count = 1) {
    assert(device);
    assert(cmd_queue);
    assert(cmd_pool);

    auto command_buffers = device->allocateCommandBuffers(cmd_pool, 1);
    if (command_buffers.size() > 0) {
        auto& cmd_buf = command_buffers[0];
        if (cmd_buf) {
            cmd_buf->beginCommandBuffer(static_cast<work::renderer::CommandBufferUsageFlags>(work::renderer::CommandBufferUsageFlagBits::ONE_TIME_SUBMIT_BIT));

            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = toVkImageLayout(old_layout);
            barrier.newLayout = toVkImageLayout(new_layout);
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

            // todo.
            auto vk_image = std::reinterpret_pointer_cast<work::renderer::VulkanImage>(image);
            barrier.image = vk_image->get();
            if (new_layout == work::renderer::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

                if (hasStencilComponent(format)) {
                    barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
                }
            }
            else {
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            }
            barrier.subresourceRange.baseMipLevel = base_mip_idx;
            barrier.subresourceRange.levelCount = mip_count;
            barrier.subresourceRange.baseArrayLayer = base_layer;
            barrier.subresourceRange.layerCount = layer_count;
            VkPipelineStageFlags source_stage;
            VkPipelineStageFlags destination_stage;

            if (old_layout == work::renderer::ImageLayout::UNDEFINED && new_layout == work::renderer::ImageLayout::TRANSFER_DST_OPTIMAL) {
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

                source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            }
            else if (old_layout == work::renderer::ImageLayout::TRANSFER_DST_OPTIMAL && new_layout == work::renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL) {
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

                source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            }
            else if (old_layout == work::renderer::ImageLayout::UNDEFINED && new_layout == work::renderer::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

                source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                destination_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            }
            else {
                throw std::invalid_argument("unsupported layout transition!");
            }

            // todo.
            //cmd_buf->pipelineBarrier();
            auto vk_cmd_Buf = std::reinterpret_pointer_cast<work::renderer::VulkanCommandBuffer>(cmd_buf);
            assert(vk_cmd_Buf);
            vkCmdPipelineBarrier(
                vk_cmd_Buf->get(),
                source_stage, destination_stage,
                0,
                0, nullptr,
                0, nullptr,
                1, &barrier
            );

            cmd_buf->endCommandBuffer();
        }

        cmd_queue->submit(command_buffers);
        cmd_queue->waitIdle();
        device->freeCommandBuffers(cmd_pool, command_buffers);
    }
}

void copyBufferToImageWithMips(const std::shared_ptr<work::renderer::Device>& device,
    const std::shared_ptr<work::renderer::Queue>& cmd_queue,
    const std::shared_ptr<work::renderer::CommandPool>& cmd_pool,
    const std::shared_ptr<work::renderer::Buffer>& buffer,
    const std::shared_ptr<work::renderer::Image>& image,
    const std::vector<work::renderer::BufferImageCopyInfo>& copy_regions) {
    auto command_buffers = device->allocateCommandBuffers(cmd_pool, 1);
    if (command_buffers.size() > 0) {
        auto& cmd_buf = command_buffers[0];
        if (cmd_buf) {
            cmd_buf->beginCommandBuffer(static_cast<work::renderer::CommandBufferUsageFlags>(work::renderer::CommandBufferUsageFlagBits::ONE_TIME_SUBMIT_BIT));
            cmd_buf->copyBufferToImage(buffer, image, copy_regions, work::renderer::ImageLayout::TRANSFER_DST_OPTIMAL);
            cmd_buf->endCommandBuffer();
        }

        cmd_queue->submit(command_buffers);
        cmd_queue->waitIdle();
        device->freeCommandBuffers(cmd_pool, command_buffers);
    }
}

void copyBufferToImage(const std::shared_ptr<work::renderer::Device>& device,
                       const std::shared_ptr<work::renderer::Queue>& cmd_queue,
                       const std::shared_ptr<work::renderer::CommandPool>& cmd_pool, 
                       const std::shared_ptr<work::renderer::Buffer>& buffer, 
                       const std::shared_ptr<work::renderer::Image>& image, 
                       glm::uvec2 tex_size) {
    std::vector<work::renderer::BufferImageCopyInfo> copy_regions(1);
    auto& region = copy_regions[0];
    region.buffer_offset = 0;
    region.buffer_row_length = 0;
    region.buffer_image_height = 0;

    region.image_subresource.aspect_mask = static_cast<work::renderer::ImageAspectFlags>(work::renderer::ImageAspectFlagBits::COLOR_BIT);
    region.image_subresource.mip_level = 0;
    region.image_subresource.base_array_layer = 0;
    region.image_subresource.layer_count = 1;

    region.image_offset = glm::ivec3(0, 0, 0);
    region.image_extent = glm::uvec3(tex_size, 1);

    copyBufferToImageWithMips(device, cmd_queue, cmd_pool, buffer, image, copy_regions);
}

void create2DTextureImage(
    const std::shared_ptr<work::renderer::Device>& device,
    const std::shared_ptr<work::renderer::Queue>& cmd_queue,
    const std::shared_ptr<work::renderer::CommandPool>& cmd_pool,
    work::renderer::Format format,
    int tex_width,
    int tex_height,
    int tex_channels,
    const void* pixels,
    std::shared_ptr<work::renderer::Image>& texture_image,
    std::shared_ptr<work::renderer::DeviceMemory>& texture_image_memory) {

    VkDeviceSize image_size = static_cast<VkDeviceSize>(tex_width * tex_height * 4);

    std::shared_ptr<work::renderer::Buffer> staging_buffer;
    std::shared_ptr<work::renderer::DeviceMemory> staging_buffer_memory;
    createBuffer(
        device,
        image_size,
        static_cast<work::renderer::BufferUsageFlags>(work::renderer::BufferUsageFlagBits::TRANSFER_SRC_BIT),
        static_cast<work::renderer::MemoryPropertyFlags>(work::renderer::MemoryPropertyFlagBits::HOST_VISIBLE_BIT) |
        static_cast<work::renderer::MemoryPropertyFlags>(work::renderer::MemoryPropertyFlagBits::HOST_COHERENT_BIT),
        staging_buffer,
        staging_buffer_memory);

    updateBufferMemory(device, staging_buffer_memory, image_size, pixels);

    create2DImage(
        device,
        glm::vec2(tex_width, tex_height),
        format,
        work::renderer::ImageTiling::OPTIMAL,
        static_cast<work::renderer::ImageUsageFlags>(work::renderer::ImageUsageFlagBits::TRANSFER_DST_BIT) |
        static_cast<work::renderer::ImageUsageFlags>(work::renderer::ImageUsageFlagBits::SAMPLED_BIT),
        static_cast<work::renderer::MemoryPropertyFlags>(work::renderer::MemoryPropertyFlagBits::DEVICE_LOCAL_BIT),
        texture_image,
        texture_image_memory);

    transitionImageLayout(device, cmd_queue, cmd_pool, texture_image, format, work::renderer::ImageLayout::UNDEFINED, work::renderer::ImageLayout::TRANSFER_DST_OPTIMAL);
    copyBufferToImage(device, cmd_queue, cmd_pool, staging_buffer, texture_image, glm::uvec2(tex_width, tex_height));
    transitionImageLayout(device, cmd_queue, cmd_pool, texture_image, format, work::renderer::ImageLayout::TRANSFER_DST_OPTIMAL, work::renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    device->destroyBuffer(staging_buffer);
    device->freeMemory(staging_buffer_memory);
}

void create2x2Texture(
    const std::shared_ptr<work::renderer::Device>& device,
    const std::shared_ptr<work::renderer::Queue>& cmd_queue,
    const std::shared_ptr<work::renderer::CommandPool>& cmd_pool,
    uint32_t color,
    work::renderer::TextureInfo& texture) {
    auto format = work::renderer::Format::R8G8B8A8_UNORM;
    uint32_t colors[4] = { color };
    create2DTextureImage(device, cmd_queue, cmd_pool, format, 2, 2, 4, colors, texture.image, texture.memory);
    texture.view = device->createImageView(
        texture.image,
        work::renderer::ImageViewType::VIEW_2D,
        format,
        static_cast<work::renderer::ImageAspectFlags>(work::renderer::ImageAspectFlagBits::COLOR_BIT));
}

std::vector<std::shared_ptr<work::renderer::DescriptorSet>> createDescriptorSets(
    std::shared_ptr<work::renderer::Device> device,
    std::shared_ptr<work::renderer::DescriptorPool> descriptor_pool,
    std::shared_ptr<work::renderer::DescriptorSetLayout> descriptor_set_layout,
    uint64_t buffer_count) {
    // todo.
    auto vk_descriptor_pool = std::reinterpret_pointer_cast<work::renderer::VulkanDescriptorPool>(descriptor_pool);
    auto vk_descriptor_set_layout = std::reinterpret_pointer_cast<work::renderer::VulkanDescriptorSetLayout>(descriptor_set_layout);
    std::vector<VkDescriptorSetLayout> layouts(buffer_count, vk_descriptor_set_layout->get());
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = vk_descriptor_pool->get();
    alloc_info.descriptorSetCount = static_cast<uint32_t>(buffer_count);
    alloc_info.pSetLayouts = layouts.data();

    // todo.
    auto vk_device = std::reinterpret_pointer_cast<work::renderer::VulkanDevice>(device);
    assert(vk_device);

    std::vector<VkDescriptorSet> vk_desc_sets;
    vk_desc_sets.resize(buffer_count);
    if (vkAllocateDescriptorSets(vk_device->get(), &alloc_info, vk_desc_sets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    std::vector<std::shared_ptr<work::renderer::DescriptorSet>> desc_sets(vk_desc_sets.size());
    for (uint32_t i = 0; i < buffer_count; i++) {
        auto vk_desc_set = std::make_shared<work::renderer::VulkanDescriptorSet>();
        vk_desc_set->set(vk_desc_sets[i]);
        desc_sets[i] = vk_desc_set;
    }

    return desc_sets;
}

std::shared_ptr<work::renderer::DescriptorSetLayout> createDescriptorSetLayout(
    const std::shared_ptr<work::renderer::Device>& device,
    const VkDescriptorSetLayoutCreateInfo& descriptor_set_layout_info) {
    auto vk_device = std::reinterpret_pointer_cast<work::renderer::VulkanDevice>(device);
    assert(vk_device);

    VkDescriptorSetLayout descriptor_set_layout;
    if (vkCreateDescriptorSetLayout(vk_device->get(), &descriptor_set_layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    auto vk_set_layout = std::make_shared<work::renderer::VulkanDescriptorSetLayout>();
    vk_set_layout->set(descriptor_set_layout);
    return vk_set_layout;
}

std::shared_ptr<work::renderer::PipelineLayout> createPipelineLayout(
    const std::shared_ptr<work::renderer::Device>& device,
    const VkPipelineLayoutCreateInfo& pipeline_layout_info) {
    auto vk_device = std::reinterpret_pointer_cast<work::renderer::VulkanDevice>(device);
    assert(vk_device);

    VkPipelineLayout pipeline_layout;
    if (vkCreatePipelineLayout(vk_device->get(), &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }
    auto vk_pipeline_layout = std::make_shared<work::renderer::VulkanPipelineLayout>();
    vk_pipeline_layout->set(pipeline_layout);

    return vk_pipeline_layout;
}

std::shared_ptr<work::renderer::Pipeline> createPipeline(
    const std::shared_ptr<work::renderer::Device>& device,
    const VkGraphicsPipelineCreateInfo& pipeline_info) {
    auto vk_device = std::reinterpret_pointer_cast<work::renderer::VulkanDevice>(device);
    assert(vk_device);

    VkPipeline graphics_pipeline;
    if (vkCreateGraphicsPipelines(vk_device->get(), VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &graphics_pipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }
    auto vk_pipeline = std::make_shared<work::renderer::VulkanPipeline>();
    vk_pipeline->set(graphics_pipeline);
    return vk_pipeline;
}

std::shared_ptr<work::renderer::RenderPass> createRenderPass(
    const std::shared_ptr<work::renderer::Device>& device,
    const VkRenderPassCreateInfo& render_pass_info) {
    // todo.
    auto vk_device = std::reinterpret_pointer_cast<work::renderer::VulkanDevice>(device);
    assert(vk_device);
    VkRenderPass render_pass;
    if (vkCreateRenderPass(vk_device->get(), &render_pass_info, nullptr, &render_pass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }

    auto vk_render_pass = std::make_shared<work::renderer::VulkanRenderPass>();
    vk_render_pass->set(render_pass);
    return vk_render_pass;
}

void addImageBarrier(const std::shared_ptr<work::renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<work::renderer::Image>& image,
    VkImageLayout src_image_layout,
    VkImageLayout dst_image_layout,
    VkAccessFlags src_access_flags,
    VkAccessFlags dst_access_flags,
    VkPipelineStageFlags src_stage_flags,
    VkPipelineStageFlags dst_stage_flags,
    uint32_t base_mip = 0,
    uint32_t mip_count = 1,
    uint32_t base_layer = 0,
    uint32_t layer_count = 1) {
    auto vk_cmd_buf = std::reinterpret_pointer_cast<work::renderer::VulkanCommandBuffer>(cmd_buf);
    auto vk_image = std::reinterpret_pointer_cast<work::renderer::VulkanImage>(image);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = src_image_layout;
    barrier.newLayout = dst_image_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = vk_image->get();
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = base_mip;
    barrier.subresourceRange.levelCount = mip_count;
    barrier.subresourceRange.baseArrayLayer = base_layer;
    barrier.subresourceRange.layerCount = layer_count;
    barrier.srcAccessMask = src_access_flags;
    barrier.dstAccessMask = dst_access_flags;

    auto vk_cmd_Buf = std::reinterpret_pointer_cast<work::renderer::VulkanCommandBuffer>(cmd_buf);
    assert(vk_cmd_Buf);
    vkCmdPipelineBarrier(
        vk_cmd_Buf->get(),
        src_stage_flags, dst_stage_flags,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );
}

struct ImageResourceInfo{
    VkImageLayout           image_layout;
    VkAccessFlags           access_flags;
    VkPipelineStageFlags    stage_flags;
};

ImageResourceInfo image_source_info = {
    VK_IMAGE_LAYOUT_UNDEFINED,
    VK_ACCESS_SHADER_READ_BIT,
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT};

ImageResourceInfo image_as_color_attachement = {
    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

ImageResourceInfo image_as_store = {
    VK_IMAGE_LAYOUT_GENERAL,
    VK_ACCESS_SHADER_WRITE_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };

ImageResourceInfo image_as_shader_sampler = {
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_ACCESS_SHADER_READ_BIT,
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };

void addImageBarrier(
    const std::shared_ptr<work::renderer::CommandBuffer>& cmd_buf, 
    const std::shared_ptr<work::renderer::Image>& image,
    const ImageResourceInfo& src_info,
    const ImageResourceInfo& dst_info,
    uint32_t base_mip = 0,
    uint32_t mip_count = 1,
    uint32_t base_layer = 0,
    uint32_t layer_count = 1) {
    addImageBarrier(
        cmd_buf,
        image,
        src_info.image_layout,
        dst_info.image_layout,
        src_info.access_flags,
        dst_info.access_flags,
        src_info.stage_flags,
        dst_info.stage_flags,
        base_mip,
        mip_count,
        base_layer,
        layer_count);
}

void generateMipmapLevels(
    const std::shared_ptr<work::renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<work::renderer::Image>& image, 
    uint32_t mip_count, 
    uint32_t width,
    uint32_t height,
    const VkImageLayout& cur_image_layout)
{
    auto vk_cmd_buf = std::reinterpret_pointer_cast<work::renderer::VulkanCommandBuffer>(cmd_buf);
    auto vk_image = std::reinterpret_pointer_cast<work::renderer::VulkanImage>(image);

    ImageResourceInfo src_info = { cur_image_layout, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    ImageResourceInfo src_as_transfer = { VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT };

    {
        VkImageSubresourceRange mipbaseRange{};
        mipbaseRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        mipbaseRange.baseMipLevel = 0u;
        mipbaseRange.levelCount = 1u;
        mipbaseRange.layerCount = 6u;

        addImageBarrier(
            cmd_buf,
            image,
            src_info,
            src_as_transfer,
            0, 1, 0, 6);
    }

    for (uint32_t i = 1; i < mip_count; i++)
    {
        VkImageBlit imageBlit{};

        // Source
        imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlit.srcSubresource.layerCount = 6u;
        imageBlit.srcSubresource.mipLevel = i - 1;
        imageBlit.srcOffsets[1].x = int32_t(width >> (i - 1));
        imageBlit.srcOffsets[1].y = int32_t(height >> (i - 1));
        imageBlit.srcOffsets[1].z = 1;

        // Destination
        imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlit.dstSubresource.layerCount = 6u;
        imageBlit.dstSubresource.mipLevel = i;
        imageBlit.dstOffsets[1].x = int32_t(width >> i);
        imageBlit.dstOffsets[1].y = int32_t(height >> i);
        imageBlit.dstOffsets[1].z = 1;

        VkImageSubresourceRange mipSubRange = {};
        mipSubRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        mipSubRange.baseMipLevel = i;
        mipSubRange.levelCount = 1;
        mipSubRange.layerCount = 6u;

        ImageResourceInfo src_mip_info = { VK_IMAGE_LAYOUT_UNDEFINED, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        ImageResourceInfo dst_transfer_info = { VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT };

        addImageBarrier(
            cmd_buf,
            image,
            src_mip_info,
            dst_transfer_info,
            i, 1, 0, 6);

        vkCmdBlitImage(
            vk_cmd_buf->get(),
            vk_image->get(),
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            vk_image->get(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &imageBlit,
            VK_FILTER_LINEAR);

        addImageBarrier(
            cmd_buf,
            image,
            dst_transfer_info,
            src_as_transfer,
            i, 1, 0, 6);
    }

    {
        addImageBarrier(
            cmd_buf,
            image,
            src_as_transfer,
            image_as_shader_sampler,
            0, mip_count, 0, 6);
    }
}

}

namespace work {
namespace renderer {

std::shared_ptr<Buffer> VulkanDevice::createBuffer(uint64_t buf_size, BufferUsageFlags usage, bool sharing/* = false*/) {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = buf_size;
    buffer_info.usage = toVkBufferUsageFlags(usage);
    buffer_info.sharingMode = sharing ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    auto result = std::make_shared<VulkanBuffer>();
    result->set(buffer);

    return result;
}

std::shared_ptr<Image> VulkanDevice::createImage(
    ImageType image_type,
    glm::uvec3 image_size,
    Format format,
    ImageUsageFlags usage,
    ImageTiling tiling,
    ImageLayout layout,
    ImageCreateFlags flags/* = 0*/,
    bool sharing/* = false*/,
    uint32_t num_samples/* = 1*/,
    uint32_t num_mips/* = 1*/,
    uint32_t num_layers/* = 1*/) {
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = toVkImageType(image_type);
    image_info.extent.width = image_size.x;
    image_info.extent.height = image_size.y;
    image_info.extent.depth = image_size.z;
    image_info.mipLevels = num_mips;
    image_info.arrayLayers = num_layers;
    image_info.format = toVkFormat(format);
    image_info.tiling = toVkImageTiling(tiling);
    image_info.initialLayout = toVkImageLayout(layout);
    image_info.usage = toVkImageUsageFlags(usage);
    image_info.sharingMode = sharing ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    image_info.samples = static_cast<VkSampleCountFlagBits>(num_samples);
    image_info.flags = toVkImageCreateFlags(flags);

    VkImage image;
    if (vkCreateImage(device_, &image_info, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    auto result = std::make_shared<VulkanImage>();
    result->set(image);
    return result;
}

std::shared_ptr<ImageView> VulkanDevice::createImageView(
    std::shared_ptr<Image> image,
    ImageViewType view_type,
    Format format,
    ImageAspectFlags aspect_flags,
    uint32_t base_mip/* = 0*/,
    uint32_t mip_count/* = 1*/,
    uint32_t base_layer/* = 0*/,
    uint32_t layer_count/* = 1*/) {
    auto vk_image = std::reinterpret_pointer_cast<VulkanImage>(image);

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = vk_image->get();
    view_info.viewType = toVkImageViewType(view_type);
    view_info.format = toVkFormat(format);
    view_info.subresourceRange.aspectMask = toVkImageAspectFlags(aspect_flags);
    view_info.subresourceRange.baseMipLevel = base_mip;
    view_info.subresourceRange.levelCount = mip_count;
    view_info.subresourceRange.baseArrayLayer = base_layer;
    view_info.subresourceRange.layerCount = layer_count;

    VkImageView image_view;
    if (vkCreateImageView(device_, &view_info, nullptr, &image_view) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }

    auto result = std::make_shared<VulkanImageView>();
    result->set(image_view);

    return result;
}

std::shared_ptr<Sampler> VulkanDevice::createSampler(Filter filter, SamplerAddressMode address_mode, SamplerMipmapMode mipmap_mode, float anisotropy) {
    auto vk_filter = toVkFilter(filter);
    auto vk_address_mode = toVkSamplerAddressMode(address_mode);
    auto vk_mipmap_mode = toVkSamplerMipmapMode(mipmap_mode);

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = vk_filter;
    sampler_info.minFilter = vk_filter;
    sampler_info.addressModeU = vk_address_mode;
    sampler_info.addressModeV = vk_address_mode;
    sampler_info.addressModeW = vk_address_mode;
    sampler_info.anisotropyEnable = anisotropy > 0 ? VK_TRUE : VK_FALSE;
    sampler_info.maxAnisotropy = anisotropy;
    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
    sampler_info.mipmapMode = vk_mipmap_mode;
    sampler_info.mipLodBias = 0.0f;
    sampler_info.minLod = 0.0f;
    sampler_info.maxLod = 1024.0f;

    VkSampler tex_sampler;
    if (vkCreateSampler(device_, &sampler_info, nullptr, &tex_sampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }

    auto vk_tex_sampler = std::make_shared<VulkanSampler>();
    vk_tex_sampler->set(tex_sampler);
    return vk_tex_sampler;
}

std::shared_ptr<Semaphore> VulkanDevice::createSemaphore() {
    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkSemaphore semaphore;
    if (vkCreateSemaphore(device_, &semaphore_info, nullptr, &semaphore) != VK_SUCCESS) {
        throw std::runtime_error("failed to create semaphore!");
    }

    auto vk_semaphore = std::make_shared<VulkanSemaphore>();
    vk_semaphore->set(semaphore);
    return vk_semaphore;
}

std::shared_ptr<Fence> VulkanDevice::createFence() {
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkFence fence;
    if (vkCreateFence(device_, &fence_info, nullptr, &fence) != VK_SUCCESS) {
        throw std::runtime_error("failed to create fence!");
    }

    auto vk_fence = std::make_shared<VulkanFence>();
    vk_fence->set(fence);
    return vk_fence;
}

std::shared_ptr<ShaderModule> VulkanDevice::createShaderModule(uint64_t size, void* data) {
    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = size;
    create_info.pCode = reinterpret_cast<const uint32_t*>(data);

    VkShaderModule shader_module;
    if (vkCreateShaderModule(device_, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }

    auto vk_shader_module = std::make_shared<VulkanShaderModule>();
    vk_shader_module->set(shader_module);

    return vk_shader_module;
}

std::shared_ptr<CommandPool> VulkanDevice::createCommandPool(uint32_t queue_family_index, CommandPoolCreateFlags flags) {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family_index;
    pool_info.flags = toVkCommandPoolCreateFlags(flags);

    VkCommandPool cmd_pool;
    if (vkCreateCommandPool(device_, &pool_info, nullptr, &cmd_pool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }

    auto vk_cmd_pool = std::make_shared<VulkanCommandPool>();
    vk_cmd_pool->set(cmd_pool);
    return vk_cmd_pool;
}

std::shared_ptr<Queue> VulkanDevice::getDeviceQueue(uint32_t queue_family_index, uint32_t queue_index/* = 0*/) {
    VkQueue queue;
    vkGetDeviceQueue(device_, queue_family_index, queue_index, &queue);
    auto vk_queue = std::make_shared<VulkanQueue>();
    vk_queue->set(queue);
    return vk_queue;
}

std::shared_ptr<Swapchain> VulkanDevice::createSwapchain(
    std::shared_ptr<Surface> surface,
    uint32_t image_count,
    Format format,
    glm::uvec2 buf_size,
    renderer::ColorSpace color_space,
    work::renderer::SurfaceTransformFlagBits transform,
    renderer::PresentMode present_mode,
    std::vector<uint32_t> queue_index) {
    
    auto vk_surface = std::reinterpret_pointer_cast<VulkanSurface>(surface);

    VkSwapchainCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = vk_surface->get();
    create_info.minImageCount = image_count;
    create_info.imageFormat = toVkFormat(format);
    create_info.imageColorSpace = toVkColorSpace(color_space);
    create_info.imageExtent = {buf_size.x, buf_size.y};
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; //VK_IMAGE_USAGE_TRANSFER_DST_BIT

    create_info.imageSharingMode = queue_index.size() > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    create_info.queueFamilyIndexCount = static_cast<uint32_t>(queue_index.size() <= 1 ? 0 : queue_index.size());
    create_info.pQueueFamilyIndices = queue_index.size() <= 1 ? nullptr : queue_index.data();

    create_info.preTransform = toVkSurfaceTransformFlags(transform);
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = toVkPresentMode(present_mode);
    create_info.clipped = VK_TRUE;
    create_info.oldSwapchain = VK_NULL_HANDLE; //need to be handled when resize.

    VkSwapchainKHR swap_chain;
    if (vkCreateSwapchainKHR(device_, &create_info, nullptr, &swap_chain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }

    auto vk_swap_chain = std::make_shared<VulkanSwapchain>();
    vk_swap_chain->set(swap_chain);
    return vk_swap_chain;
}

std::shared_ptr<Framebuffer> VulkanDevice::createFrameBuffer(
    const std::shared_ptr<RenderPass>& render_pass, 
    const std::vector<std::shared_ptr<ImageView>>& attachments,
    const glm::uvec2& extent) {

    std::vector<VkImageView> image_views(attachments.size());
    for (int i = 0; i < attachments.size(); i++) {
        auto vk_image_view = std::reinterpret_pointer_cast<VulkanImageView>(attachments[i]);
        image_views[i] = vk_image_view->get();
    }

    auto vk_render_pass = std::reinterpret_pointer_cast<VulkanRenderPass>(render_pass);
    VkFramebufferCreateInfo framebuffer_info{};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = vk_render_pass->get();
    framebuffer_info.attachmentCount = static_cast<uint32_t>(image_views.size());
    framebuffer_info.pAttachments = image_views.data();
    framebuffer_info.width = extent.x;
    framebuffer_info.height = extent.y;
    framebuffer_info.layers = 1;

    VkFramebuffer frame_buffer;
    if (vkCreateFramebuffer(device_, &framebuffer_info, nullptr, &frame_buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer!");
    }

    auto vk_frame_buffer = std::make_shared<VulkanFramebuffer>();
    vk_frame_buffer->set(frame_buffer);
    return vk_frame_buffer;
}

std::vector<std::shared_ptr<renderer::Image>> VulkanDevice::getSwapchainImages(std::shared_ptr<Swapchain> swap_chain) {
    auto vk_swap_chain = std::reinterpret_pointer_cast<VulkanSwapchain>(swap_chain);
    uint32_t image_count;
    std::vector<VkImage> swap_chain_images;
    vkGetSwapchainImagesKHR(device_, vk_swap_chain->get(), &image_count, nullptr);
    swap_chain_images.resize(image_count);
    vkGetSwapchainImagesKHR(device_, vk_swap_chain->get(), &image_count, swap_chain_images.data());

    std::vector<std::shared_ptr<Image>> vk_swap_chain_images(swap_chain_images.size());
    for (int i = 0; i < swap_chain_images.size(); i++) {
        auto vk_image = std::make_shared<VulkanImage>();
        vk_image->set(swap_chain_images[i]);
        vk_swap_chain_images[i] = vk_image;
    }
    return vk_swap_chain_images;
}

std::shared_ptr<DeviceMemory> VulkanDevice::allocateMemory(uint64_t buf_size, uint32_t memory_type_bits, MemoryPropertyFlags properties) {
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = buf_size;
    alloc_info.memoryTypeIndex = findMemoryType(physical_device_, memory_type_bits, properties);

    VkDeviceMemory memory;
    if (vkAllocateMemory(device_, &alloc_info, nullptr, &memory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    auto result = std::make_shared<VulkanDeviceMemory>();
    result->set(memory);

    return result;
}

MemoryRequirements VulkanDevice::getBufferMemoryRequirements(std::shared_ptr<Buffer> buffer) {
    auto vk_buffer = std::reinterpret_pointer_cast<VulkanBuffer>(buffer);

    MemoryRequirements mem_requirements = {};
    if (vk_buffer) {
        VkMemoryRequirements vk_mem_requirements;
        vkGetBufferMemoryRequirements(device_, vk_buffer->get(), &vk_mem_requirements);
        mem_requirements.size = vk_mem_requirements.size;
        mem_requirements.alignment = vk_mem_requirements.alignment;
        mem_requirements.memory_type_bits = vk_mem_requirements.memoryTypeBits;
    }

    return mem_requirements;
}

MemoryRequirements VulkanDevice::getImageMemoryRequirements(std::shared_ptr<Image> image) {
    auto vk_image = std::reinterpret_pointer_cast<VulkanImage>(image);

    MemoryRequirements mem_requirements = {};
    if (vk_image) {
        VkMemoryRequirements vk_mem_requirements;
        vkGetImageMemoryRequirements(device_, vk_image->get(), &vk_mem_requirements);
        mem_requirements.size = vk_mem_requirements.size;
        mem_requirements.alignment = vk_mem_requirements.alignment;
        mem_requirements.memory_type_bits = vk_mem_requirements.memoryTypeBits;
    }

    return mem_requirements;
}

void VulkanDevice::bindBufferMemory(std::shared_ptr<Buffer> buffer, std::shared_ptr<DeviceMemory> buffer_memory, uint64_t offset/* = 0*/) {
    auto vk_buffer = std::reinterpret_pointer_cast<VulkanBuffer>(buffer);
    auto vk_buffer_memory = std::reinterpret_pointer_cast<VulkanDeviceMemory>(buffer_memory);

    if (vk_buffer && vk_buffer_memory) {
        vkBindBufferMemory(device_, vk_buffer->get(), vk_buffer_memory->get(), offset);
    }
}

void VulkanDevice::bindImageMemory(std::shared_ptr<Image> image, std::shared_ptr<DeviceMemory> image_memory, uint64_t offset/* = 0*/) {
    auto vk_image = std::reinterpret_pointer_cast<VulkanImage>(image);
    auto vk_image_memory = std::reinterpret_pointer_cast<VulkanDeviceMemory>(image_memory);

    if (vk_image && vk_image_memory) {
        vkBindImageMemory(device_, vk_image->get(), vk_image_memory->get(), offset);
    }
}

std::vector<std::shared_ptr<CommandBuffer>> VulkanDevice::allocateCommandBuffers(std::shared_ptr<CommandPool> cmd_pool, uint32_t num_buffers, bool is_primary/* = true*/) {
    auto vk_cmd_pool = std::reinterpret_pointer_cast<VulkanCommandPool>(cmd_pool);
    std::vector<VkCommandBuffer> cmd_bufs(num_buffers);

    if (vk_cmd_pool) {
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = vk_cmd_pool->get();
        alloc_info.level = is_primary ? VK_COMMAND_BUFFER_LEVEL_PRIMARY : VK_COMMAND_BUFFER_LEVEL_SECONDARY;
        alloc_info.commandBufferCount = num_buffers;

        if (vkAllocateCommandBuffers(device_, &alloc_info, cmd_bufs.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    std::vector<std::shared_ptr<CommandBuffer>> result(num_buffers);
    for (uint32_t i = 0; i < num_buffers; i++) {
        auto cmd_buf = std::make_shared<VulkanCommandBuffer>();
        cmd_buf->set(cmd_bufs[i]);
        result[i] = cmd_buf;
    }

    return result;
}

void* VulkanDevice::mapMemory(std::shared_ptr<DeviceMemory> memory, uint64_t size, uint64_t offset /*=0*/) {
    void* data = nullptr;
    auto vk_memory = std::reinterpret_pointer_cast<VulkanDeviceMemory>(memory);
    if (vk_memory) {
        vkMapMemory(device_, vk_memory->get(), offset, size, 0/*reserved*/, &data);
    }

    return data;
}

void VulkanDevice::unmapMemory(std::shared_ptr<DeviceMemory> memory) {
    auto vk_memory = std::reinterpret_pointer_cast<VulkanDeviceMemory>(memory);
    if (vk_memory) {
        vkUnmapMemory(device_, vk_memory->get());
    }
}

void VulkanDevice::destroyCommandPool(std::shared_ptr<CommandPool> cmd_pool) {
    auto vk_cmd_pool = std::reinterpret_pointer_cast<VulkanCommandPool>(cmd_pool);
    if (vk_cmd_pool) {
        vkDestroyCommandPool(device_, vk_cmd_pool->get(), nullptr);
    }
}

void VulkanDevice::destroySwapchain(std::shared_ptr<Swapchain> swapchain) {
    auto vk_swapchain = std::reinterpret_pointer_cast<VulkanSwapchain>(swapchain);
    if (vk_swapchain) {
        vkDestroySwapchainKHR(device_, vk_swapchain->get(), nullptr);
    }
}

void VulkanDevice::destroyDescriptorPool(std::shared_ptr<DescriptorPool> descriptor_pool) {
    auto vk_descriptor_pool = std::reinterpret_pointer_cast<VulkanDescriptorPool>(descriptor_pool);
    if (vk_descriptor_pool) {
        vkDestroyDescriptorPool(device_, vk_descriptor_pool->get(), nullptr);
    }
}

void VulkanDevice::destroyPipeline(std::shared_ptr<Pipeline> pipeline) {
    auto vk_pipeline = std::reinterpret_pointer_cast<VulkanPipeline>(pipeline);
    if (vk_pipeline) {
        vkDestroyPipeline(device_, vk_pipeline->get(), nullptr);
    }
}

void VulkanDevice::destroyPipelineLayout(std::shared_ptr<PipelineLayout> pipeline_layout) {
    auto vk_pipeline_layout = std::reinterpret_pointer_cast<VulkanPipelineLayout>(pipeline_layout);
    if (vk_pipeline_layout) {
        vkDestroyPipelineLayout(device_, vk_pipeline_layout->get(), nullptr);
    }
}

void VulkanDevice::destroyRenderPass(std::shared_ptr<RenderPass> render_pass) {
    auto vk_render_pass = std::reinterpret_pointer_cast<VulkanRenderPass>(render_pass);
    if (vk_render_pass) {
        vkDestroyRenderPass(device_, vk_render_pass->get(), nullptr);
    }
}

void VulkanDevice::destroyFramebuffer(std::shared_ptr<Framebuffer> frame_buffer) {
    auto vk_framebuffer = std::reinterpret_pointer_cast<VulkanFramebuffer>(frame_buffer);
    if (vk_framebuffer) {
        vkDestroyFramebuffer(device_, vk_framebuffer->get(), nullptr);
    }
}

void VulkanDevice::destroyImageView(std::shared_ptr<ImageView> image_view) {
    auto vk_image_view = std::reinterpret_pointer_cast<VulkanImageView>(image_view);
    if (vk_image_view) {
        vkDestroyImageView(device_, vk_image_view->get(), nullptr);
    }
}

void VulkanDevice::destroySampler(std::shared_ptr<Sampler> sampler) {
    auto vk_sampler = std::reinterpret_pointer_cast<VulkanSampler>(sampler);
    if (vk_sampler) {
        vkDestroySampler(device_, vk_sampler->get(), nullptr);
    }
}

void VulkanDevice::destroyImage(std::shared_ptr<Image> image) {
    auto vk_image = std::reinterpret_pointer_cast<VulkanImage>(image);
    if (vk_image) {
        vkDestroyImage(device_, vk_image->get(), nullptr);
    }
}

void VulkanDevice::destroyBuffer(std::shared_ptr<Buffer> buffer) {
    auto vk_buffer = std::reinterpret_pointer_cast<VulkanBuffer>(buffer);
    if (vk_buffer) {
        vkDestroyBuffer(device_, vk_buffer->get(), nullptr);
    }
}

void VulkanDevice::destroySemaphore(std::shared_ptr<Semaphore> semaphore) {
    auto vk_semaphore = std::reinterpret_pointer_cast<VulkanSemaphore>(semaphore);
    if (vk_semaphore) {
        vkDestroySemaphore(device_, vk_semaphore->get(), nullptr);
    }
}

void VulkanDevice::destroyFence(std::shared_ptr<Fence> fence) {
    auto vk_fence = std::reinterpret_pointer_cast<VulkanFence>(fence);
    if (vk_fence) {
        vkDestroyFence(device_, vk_fence->get(), nullptr);
    }
}

void VulkanDevice::destroyDescriptorSetLayout(std::shared_ptr<DescriptorSetLayout> layout) {
    auto vk_layout = std::reinterpret_pointer_cast<VulkanDescriptorSetLayout>(layout);
    if (vk_layout) {
        vkDestroyDescriptorSetLayout(device_, vk_layout->get(), nullptr);
    }
}

void VulkanDevice::destroyShaderModule(std::shared_ptr<ShaderModule> shader_module) {
    auto vk_shader_module = std::reinterpret_pointer_cast<VulkanShaderModule>(shader_module);
    if (vk_shader_module) {
        vkDestroyShaderModule(device_, vk_shader_module->get(), nullptr);
    }
}

void VulkanDevice::freeMemory(std::shared_ptr<DeviceMemory> memory) {
    auto vk_memory = std::reinterpret_pointer_cast<VulkanDeviceMemory>(memory);
    if (vk_memory) {
        vkFreeMemory(device_, vk_memory->get(), nullptr);
    }
}

void VulkanDevice::freeCommandBuffers(std::shared_ptr<CommandPool> cmd_pool, const std::vector<std::shared_ptr<CommandBuffer>>& cmd_bufs) {
    auto vk_cmd_pool = std::reinterpret_pointer_cast<VulkanCommandPool>(cmd_pool);
    if (vk_cmd_pool) {
        std::vector<VkCommandBuffer> vk_cmd_bufs(cmd_bufs.size());
        for (uint32_t i = 0; i < cmd_bufs.size(); i++) {
            auto vk_cmd_buf = std::reinterpret_pointer_cast<VulkanCommandBuffer>(cmd_bufs[i]);
            vk_cmd_bufs[i] = vk_cmd_buf->get();
        }

        vkFreeCommandBuffers(device_, vk_cmd_pool->get(), static_cast<uint32_t>(vk_cmd_bufs.size()), vk_cmd_bufs.data());
    }
}

void VulkanDevice::resetFences(const std::vector<std::shared_ptr<Fence>>& fences) {
    std::vector<VkFence> vk_fences(fences.size());
    for (int i = 0; i < fences.size(); i++) {
        auto vk_fence = std::reinterpret_pointer_cast<VulkanFence>(fences[i]);
        vk_fences[i] = vk_fence->get();
    }
    vkResetFences(device_, static_cast<uint32_t>(vk_fences.size()), vk_fences.data());
}

void VulkanDevice::waitForFences(const std::vector<std::shared_ptr<Fence>>& fences) {
    std::vector<VkFence> vk_fences(fences.size());
    for (int i = 0; i < fences.size(); i++) {
        auto vk_fence = std::reinterpret_pointer_cast<VulkanFence>(fences[i]);
        vk_fences[i] = vk_fence->get();
    }
    vkWaitForFences(device_, static_cast<uint32_t>(vk_fences.size()), vk_fences.data(), VK_TRUE, UINT64_MAX);
}

void VulkanDevice::waitIdle() {
    vkDeviceWaitIdle(device_);
}

void VulkanQueue::submit(const std::vector<std::shared_ptr<CommandBuffer>>& command_buffers) {
    std::vector<VkCommandBuffer> vk_cmd_bufs(command_buffers.size());
    for (int i = 0; i < command_buffers.size(); i++) {
        auto vk_cmd_buf = std::reinterpret_pointer_cast<VulkanCommandBuffer>(command_buffers[i]);
        vk_cmd_bufs[i] = vk_cmd_buf->get();
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = static_cast<uint32_t>(command_buffers.size());
    submit_info.pCommandBuffers = vk_cmd_bufs.data();

    vkQueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE);
}

void VulkanQueue::waitIdle() {
    vkQueueWaitIdle(queue_);
}

void VulkanCommandBuffer::beginCommandBuffer(CommandBufferUsageFlags flags) {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = toCommandBufferUsageFlags(flags);
    begin_info.pInheritanceInfo = nullptr; // Optional

    if (vkBeginCommandBuffer(cmd_buf_, &begin_info) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }
};

void VulkanCommandBuffer::endCommandBuffer() {
    if (vkEndCommandBuffer(cmd_buf_) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }
}

void VulkanCommandBuffer::copyBuffer(std::shared_ptr<Buffer> src_buf, std::shared_ptr<Buffer> dst_buf, std::vector<BufferCopyInfo> copy_regions) {
    std::vector<VkBufferCopy> vk_copy_regions(copy_regions.size());
    for (uint32_t i = 0; i < copy_regions.size(); i++) {
        vk_copy_regions[i].srcOffset = copy_regions[i].src_offset;
        vk_copy_regions[i].dstOffset = copy_regions[i].dst_offset;
        vk_copy_regions[i].size = copy_regions[i].size;
    }
    auto vk_src_buf = std::reinterpret_pointer_cast<VulkanBuffer>(src_buf);
    auto vk_dst_buf = std::reinterpret_pointer_cast<VulkanBuffer>(dst_buf);
    vkCmdCopyBuffer(cmd_buf_, vk_src_buf->get(), vk_dst_buf->get(), static_cast<uint32_t>(vk_copy_regions.size()), vk_copy_regions.data());
}

void VulkanCommandBuffer::copyBufferToImage(
    std::shared_ptr<Buffer> src_buf, 
    std::shared_ptr<Image> dst_image, 
    std::vector<BufferImageCopyInfo> copy_regions, 
    renderer::ImageLayout layout) {
    std::vector<VkBufferImageCopy> vk_copy_regions(copy_regions.size());
    for (uint32_t i = 0; i < copy_regions.size(); i++) {
        vk_copy_regions[i].bufferOffset = copy_regions[i].buffer_offset;
        vk_copy_regions[i].bufferRowLength = copy_regions[i].buffer_row_length;
        vk_copy_regions[i].bufferImageHeight = copy_regions[i].buffer_image_height;
        vk_copy_regions[i].imageSubresource.aspectMask = toVkImageAspectFlags(copy_regions[i].image_subresource.aspect_mask);
        vk_copy_regions[i].imageSubresource.mipLevel = copy_regions[i].image_subresource.mip_level;
        vk_copy_regions[i].imageSubresource.baseArrayLayer = copy_regions[i].image_subresource.base_array_layer;
        vk_copy_regions[i].imageSubresource.layerCount = copy_regions[i].image_subresource.layer_count;
        vk_copy_regions[i].imageOffset = { copy_regions[i].image_offset.x, copy_regions[i].image_offset.y, copy_regions[i].image_offset.z };
        vk_copy_regions[i].imageExtent = { copy_regions[i].image_extent.x, copy_regions[i].image_extent.y, copy_regions[i].image_extent.z };
    }
    auto vk_src_buf = std::reinterpret_pointer_cast<VulkanBuffer>(src_buf);
    auto vk_dst_image = std::reinterpret_pointer_cast<VulkanImage>(dst_image);
    vkCmdCopyBufferToImage(cmd_buf_, vk_src_buf->get(), vk_dst_image->get(), toVkImageLayout(layout), static_cast<uint32_t>(vk_copy_regions.size()), vk_copy_regions.data());
}

void VulkanCommandBuffer::bindPipeline(PipelineBindPoint bind, std::shared_ptr< Pipeline> pipeline) {
    auto vk_pipeline = std::reinterpret_pointer_cast<VulkanPipeline>(pipeline);
    vkCmdBindPipeline(cmd_buf_, toVkPipelineBindPoint(bind), vk_pipeline->get());
}

void VulkanCommandBuffer::bindVertexBuffers(uint32_t first_bind, const std::vector<std::shared_ptr<renderer::Buffer>>& vertex_buffers, std::vector<uint64_t> offsets) {
    std::vector<VkDeviceSize> vk_offsets(vertex_buffers.size());
    std::vector<VkBuffer> vk_vertex_buffers(vertex_buffers.size());

    for (int i = 0; i < vertex_buffers.size(); i++) {
        auto vk_vertex_buffer = std::reinterpret_pointer_cast<VulkanBuffer>(vertex_buffers[i]);
        vk_vertex_buffers[i] = vk_vertex_buffer->get();
        vk_offsets[i] = i < offsets.size() ? offsets[i] : 0;
    }
    vkCmdBindVertexBuffers(cmd_buf_, first_bind, static_cast<uint32_t>(vk_vertex_buffers.size()), vk_vertex_buffers.data(), vk_offsets.data());
}

void VulkanCommandBuffer::bindIndexBuffer(std::shared_ptr<Buffer> index_buffer, uint64_t offset, IndexType index_type) {
    auto vk_index_buffer = std::reinterpret_pointer_cast<VulkanBuffer>(index_buffer);
    vkCmdBindIndexBuffer(cmd_buf_, vk_index_buffer->get(), offset, toVkIndexType(index_type));
}

void VulkanCommandBuffer::drawIndexed(uint32_t index_count, uint32_t instance_count/* = 1*/, uint32_t first_index/* = 0*/, uint32_t vertex_offset/* = 0*/, uint32_t first_instance/* = 0*/) {
    vkCmdDrawIndexed(cmd_buf_, index_count, instance_count, first_index, vertex_offset, first_instance);
}

void VulkanCommandBuffer::beginRenderPass(
    std::shared_ptr<RenderPass> render_pass,
    std::shared_ptr<Framebuffer> frame_buffer,
    const glm::uvec2& extent,
    const std::vector<ClearValue>& clear_values) {
    std::vector<VkClearValue> vk_clear_values(clear_values.size());

    for (int i = 0; i < clear_values.size(); i++) {
        std::memcpy(&vk_clear_values[i].color, &clear_values[i].color, sizeof(VkClearValue));
    }

    auto vk_render_pass = std::reinterpret_pointer_cast<VulkanRenderPass>(render_pass);
    auto vk_frame_buffer = std::reinterpret_pointer_cast<VulkanFramebuffer>(frame_buffer);

    assert(vk_render_pass);
    assert(vk_frame_buffer);
    VkRenderPassBeginInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = vk_render_pass->get();
    render_pass_info.framebuffer = vk_frame_buffer->get();
    render_pass_info.renderArea.offset = { 0, 0 };
    render_pass_info.renderArea.extent = { extent.x, extent.y };
    render_pass_info.clearValueCount = static_cast<uint32_t>(vk_clear_values.size());
    render_pass_info.pClearValues = vk_clear_values.data();

    vkCmdBeginRenderPass(cmd_buf_, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
}

void VulkanCommandBuffer::endRenderPass() {
    vkCmdEndRenderPass(cmd_buf_);
}

void VulkanCommandBuffer::reset(uint32_t flags) {
    vkResetCommandBuffer(cmd_buf_, flags);
}

void TextureInfo::destroy(const std::shared_ptr<Device>& device) {
    device->destroyImage(image);
    device->destroyImageView(view);
    device->freeMemory(memory);

    for (auto& s_views : surface_views) {
        for (auto& s_view : s_views) {
            device->destroyImageView(s_view);
        }
    }
    
    for (auto& framebuffer : framebuffers) {
        device->destroyFramebuffer(framebuffer);
    }
}

void BufferInfo::destroy(const std::shared_ptr<Device>& device) {
    device->destroyBuffer(buffer);
    device->freeMemory(memory);
}

void ObjectData::destroy(const std::shared_ptr<Device>& device) {
    for (auto& texture : textures_) {
        texture.destroy(device);
    }

    for (auto& material : materials_) {
        material.uniform_buffer_.destroy(device);
    }

    for (auto& buffer : buffers_) {
        buffer.destroy(device);
    }
}

}

namespace app {

void RealWorldApplication::run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

void RealWorldApplication::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(kWindowSizeX, kWindowSizeY, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
}

void RealWorldApplication::initVulkan() {
    // the initialization order has to be strict.
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createRenderPass();
    createImageViews();
    createCubemapRenderPass();
    createCubemapFramebuffers();
    createDescriptorSetLayout();
    createCommandPool();
//    loadGltfModel("assets/Avocado.glb");
//    loadGltfModel("assets/BoomBox.glb");
    loadGltfModel("assets/DamagedHelmet.glb");
//    loadGltfModel("assets/Duck.glb");
//    loadGltfModel("assets/MetalRoughSpheres.glb");
//    loadGltfModel("assets/BarramundiFish.glb");
//    loadGltfModel("assets/Lantern.glb");
//    *loadGltfModel("assets/MetalRoughSpheresNoTextures.glb");
//    loadGltfModel("assets/BrainStem.glb"); 
//    *loadGltfModel("assets/AnimatedTriangle.gltf");
    loadMtx2Texture("assets/environments/doge2/lambertian/diffuse.ktx2", ibl_diffuse_tex_);
    loadMtx2Texture("assets/environments/doge2/ggx/specular.ktx2", ibl_specular_tex_);
    loadMtx2Texture("assets/environments/doge2/charlie/sheen.ktx2", ibl_sheen_tex_);
    createGltfPipelineLayout();
    createSkyboxPipelineLayout();
    createCubemapPipelineLayout();
    createCubemapComputePipelineLayout();
    createGraphicsPipeline();
    createComputePipeline();
    createDepthResources();
    createFramebuffers();
    auto format = work::renderer::Format::R8G8B8A8_UNORM;
    createTextureImage("assets/statue.jpg", format, sample_tex_);
    createTextureImage("assets/brdfLUT.png", format, brdf_lut_tex_);
    createTextureImage("assets/lut_ggx.png", format, ggx_lut_tex_);
    createTextureImage("assets/lut_charlie.png", format, charlie_lut_tex_);
    createTextureImage("assets/lut_thin_film.png", format, thin_film_lut_tex_);
    createTextureImage("assets/environments/doge2.hdr", format, panorama_tex_);
    create2x2Texture(device_, graphics_queue_, command_pool_, 0xffffffff, white_tex_);
    create2x2Texture(device_, graphics_queue_, command_pool_, 0xff000000, black_tex_);
    createTextureSampler();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
}

void RealWorldApplication::recreateSwapChain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window_, &width, &height);
        glfwWaitEvents();
    }

    device_->waitIdle();

    cleanupSwapChain();

    createSwapChain();
    createRenderPass();
    createImageViews();
    createGltfPipelineLayout();
    createSkyboxPipelineLayout();
    createCubemapPipelineLayout();
    createCubemapComputePipelineLayout();
    createGraphicsPipeline();
    createComputePipeline();
    createDepthResources();
    createFramebuffers();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
}

void RealWorldApplication::createInstance() {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Real World";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "No Engine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

#if 0
    uint32_t glfw_extensionCount = 0;
    const char** glfw_extensions;

    glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extensionCount);

    create_info.enabledExtensionCount = glfw_extensionCount;
    create_info.ppEnabledExtensionNames = glfw_extensions;
    if (enable_validation_layers) {
        create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
        create_info.ppEnabledLayerNames = validation_layers.data();
    }
    else {
        create_info.enabledLayerCount = 0;
    }
#else
    auto required_extensions = getRequiredExtensions();
    create_info.enabledExtensionCount = static_cast<uint32_t>(required_extensions.size());
    create_info.ppEnabledExtensionNames = required_extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debug_create_info;
    if (enable_validation_layers) {
        create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
        create_info.ppEnabledLayerNames = validation_layers.data();

        populateDebugMessengerCreateInfo(debug_create_info);
        create_info.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debug_create_info;
    }
    else {
        create_info.enabledLayerCount = 0;
        create_info.pNext = nullptr;
    }
#endif
    if (vkCreateInstance(&create_info, nullptr, &vk_instance_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
    }

    uint32_t extension_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> extensions(extension_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extensions.data());

    std::cout << "available extensions:\n";

    for (const auto& extension : extensions) {
        std::cout << '\t' << extension.extensionName << '\n';
    }

    if (enable_validation_layers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }
}

void RealWorldApplication::setupDebugMessenger() {
    if (!enable_validation_layers) return;

    VkDebugUtilsMessengerCreateInfoEXT create_info;
    populateDebugMessengerCreateInfo(create_info);

    if (CreateDebugUtilsMessengerEXT(vk_instance_, &create_info, nullptr, &debug_messenger_) != VK_SUCCESS) {
        throw std::runtime_error("failed to set up debug messenger!");
    }
}

void RealWorldApplication::pickPhysicalDevice() {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(vk_instance_, &device_count, nullptr);
    
    if (device_count == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(vk_instance_, &device_count, devices.data());

#if 0
    // Use an ordered map to automatically sort candidates by increasing score
    std::multimap<int, VkPhysicalDevice> candidates;

    for (const auto& device : devices) {
        int score = rateDeviceSuitability(device);
        candidates.insert(std::make_pair(score, device));
    }

    // Check if the best candidate is suitable at all
    if (candidates.rbegin()->first > 0) {
        vk_physical_device_ = candidates.rbegin()->second;
    }
    else {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
#else
    for (const auto& device : devices) {
        auto device_tmp = std::make_shared<VkPhysicalDevice>(device);
        if (isDeviceSuitable(device_tmp, surface_)) {
            vk_physical_device_ = std::move(device_tmp);
            break;
        }
    }

    if (vk_physical_device_ == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
#endif
}

void RealWorldApplication::createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(vk_physical_device_, surface_);

    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    std::set<uint32_t> unique_queue_families = { indices.graphics_family_.value(), indices.present_family_.value() };

    float queue_priority = 1.0f;
    for (uint32_t queue_family : unique_queue_families) {
        VkDeviceQueueCreateInfo queue_create_info{};
        queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info.queueFamilyIndex = queue_family;
        queue_create_info.queueCount = 1;
        queue_create_info.pQueuePriorities = &queue_priority;
        queue_create_infos.push_back(queue_create_info);
    }

    VkPhysicalDeviceFeatures device_features{};
    device_features.samplerAnisotropy = VK_TRUE;

    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
    create_info.pQueueCreateInfos = queue_create_infos.data();

    create_info.pEnabledFeatures = &device_features;

    create_info.enabledExtensionCount = 0;

    if (enable_validation_layers) {
        create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
        create_info.ppEnabledLayerNames = validation_layers.data();
    }
    else {
        create_info.enabledLayerCount = 0;
    }

    create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    create_info.ppEnabledExtensionNames = device_extensions.data();

    assert(vk_physical_device_);
    VkDevice vk_device;
    if (vkCreateDevice(*vk_physical_device_, &create_info, nullptr, &vk_device) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }

    auto vk_logic_device = std::make_shared<renderer::VulkanDevice>(*vk_physical_device_, vk_device);
    device_ = std::move(vk_logic_device);

    graphics_queue_ = device_->getDeviceQueue(indices.graphics_family_.value());
    present_queue_ = device_->getDeviceQueue(indices.present_family_.value());
}

void RealWorldApplication::createSurface() {
    VkSurfaceKHR surface;
    if (glfwCreateWindowSurface(vk_instance_, window_, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }

    auto vk_surface = std::make_shared<renderer::VulkanSurface>();
    vk_surface->set(surface);
    surface_ = std::move(vk_surface);
}

void RealWorldApplication::createSwapChain() {
    SwapChainSupportDetails swap_chain_support = querySwapChainSupport(vk_physical_device_, surface_);

    VkSurfaceFormatKHR surface_format = chooseSwapSurfaceFormat(swap_chain_support.formats_);
    auto present_mode = chooseSwapPresentMode(swap_chain_support.present_modes_);
    VkExtent2D extent = chooseSwapExtent(window_, swap_chain_support.capabilities_);

    uint32_t image_count = swap_chain_support.capabilities_.minImageCount + 1;
    if (swap_chain_support.capabilities_.maxImageCount > 0 && image_count > swap_chain_support.capabilities_.maxImageCount) {
        image_count = swap_chain_support.capabilities_.maxImageCount;
    }

    QueueFamilyIndices indices = findQueueFamilies(vk_physical_device_, surface_);
    std::vector<uint32_t> queue_index(2);
    queue_index[0] = indices.graphics_family_.value();
    queue_index[1] = indices.present_family_.value();
    std::sort(queue_index.begin(), queue_index.end());
    auto last = std::unique(queue_index.begin(), queue_index.end());
    queue_index.erase(last, queue_index.end());

    swap_chain_image_format_ = fromVkFormat(surface_format.format);
    swap_chain_extent_ = glm::uvec2(extent.width, extent.height);

    swap_chain_ = device_->createSwapchain(surface_,
        image_count, 
        swap_chain_image_format_,
        swap_chain_extent_,
        fromVkColorSpace(surface_format.colorSpace), 
        fromVkSurfaceTransformFlags(swap_chain_support.capabilities_.currentTransform),
        present_mode, 
        queue_index);

    swap_chain_images_ = device_->getSwapchainImages(swap_chain_);
}

void RealWorldApplication::createImageViews() {
    swap_chain_image_views_.resize(swap_chain_images_.size());
    for (uint64_t i_img = 0; i_img < swap_chain_images_.size(); i_img++) {
        swap_chain_image_views_[i_img] = device_->createImageView(
            swap_chain_images_[i_img],
            renderer::ImageViewType::VIEW_2D,
            swap_chain_image_format_,
            static_cast<renderer::ImageAspectFlags>(renderer::ImageAspectFlagBits::COLOR_BIT));
    }
}

void RealWorldApplication::createCubemapFramebuffers() {
    uint32_t num_mips = static_cast<uint32_t>(std::log2(kCubemapSize) + 1);
    std::vector<work::renderer::BufferImageCopyInfo> dump_copies;

    createCubemapTexture(
        kCubemapSize,
        kCubemapSize,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_envmap_tex_);

    createCubemapTexture(
        kCubemapSize,
        kCubemapSize,
        1,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        tmp_ibl_diffuse_tex_);

    createCubemapTexture(
        kCubemapSize,
        kCubemapSize,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        tmp_ibl_specular_tex_);

    createCubemapTexture(
        kCubemapSize,
        kCubemapSize,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        tmp_ibl_sheen_tex_);

    createCubemapTexture(
        kCubemapSize,
        kCubemapSize,
        1,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_ibl_diffuse_tex_);

    createCubemapTexture(
        kCubemapSize,
        kCubemapSize,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_ibl_specular_tex_);

    createCubemapTexture(
        kCubemapSize,
        kCubemapSize,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_ibl_sheen_tex_);
}

VkDescriptorSetLayoutBinding getTextureSamplerDescriptionSetLayoutBinding(
    uint32_t binding, 
    uint32_t stage_flags = VK_SHADER_STAGE_FRAGMENT_BIT,
    VkDescriptorType descript_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
    VkDescriptorSetLayoutBinding texture_binding{};
    texture_binding.binding = binding;
    texture_binding.descriptorCount = 1;
    texture_binding.descriptorType = descript_type;
    texture_binding.pImmutableSamplers = nullptr;
    texture_binding.stageFlags = stage_flags;

    return texture_binding;
}

std::vector<std::shared_ptr<renderer::ShaderModule>> getGltfShaderModules(
    std::shared_ptr<renderer::Device> device, 
    bool has_normals, 
    bool has_tangent, 
    bool has_texcoord_0,
    bool has_skin_set_0)
{
    std::vector<std::shared_ptr<renderer::ShaderModule>> shader_modules(2);
    std::string feature_str = std::string(has_texcoord_0 ? "_TEX" : "") + 
        (has_tangent ? "_TN" : (has_normals ? "_N" : "")) +
        (has_skin_set_0 ? "_SKIN" : "");
    uint64_t vert_code_size, frag_code_size;
    auto vert_shader_code = readFile("src/shaders/base_vert" + feature_str + ".spv", vert_code_size);
    auto frag_shader_code = readFile("src/shaders/base_frag" + feature_str + ".spv", frag_code_size);

    shader_modules[0] = device->createShaderModule(vert_code_size, vert_shader_code.data());
    shader_modules[1] = device->createShaderModule(frag_code_size, frag_shader_code.data());

    return shader_modules;
}

std::vector<std::shared_ptr<renderer::ShaderModule>> getSkyboxShaderModules(
    std::shared_ptr<renderer::Device> device)
{
    uint64_t vert_code_size, frag_code_size;
    std::vector<std::shared_ptr<renderer::ShaderModule>> shader_modules(2);
    auto vert_shader_code = readFile("src/shaders/skybox_vert.spv", vert_code_size);
    auto frag_shader_code = readFile("src/shaders/skybox_frag.spv", frag_code_size);

    shader_modules[0] = device->createShaderModule(vert_code_size, vert_shader_code.data());
    shader_modules[1] = device->createShaderModule(frag_code_size, frag_shader_code.data());

    return shader_modules;
}

std::vector<std::shared_ptr<renderer::ShaderModule>> getIblShaderModules(
    std::shared_ptr<renderer::Device> device)
{
    uint64_t vert_code_size, frag_code_size;
    std::vector<std::shared_ptr<renderer::ShaderModule>> shader_modules;
    shader_modules.reserve(6);
    auto vert_shader_code = readFile("src/shaders/ibl_vert.spv", vert_code_size);
    shader_modules.push_back(device->createShaderModule(vert_code_size, vert_shader_code.data()));
    auto frag_shader_code = readFile("src/shaders/panorama_to_cubemap_frag.spv", frag_code_size);
    shader_modules.push_back(device->createShaderModule(frag_code_size, frag_shader_code.data()));
    auto labertian_frag_shader_code = readFile("src/shaders/ibl_labertian_frag.spv", frag_code_size);
    shader_modules.push_back(device->createShaderModule(frag_code_size, labertian_frag_shader_code.data()));
    auto ggx_frag_shader_code = readFile("src/shaders/ibl_ggx_frag.spv", frag_code_size);
    shader_modules.push_back(device->createShaderModule(frag_code_size, ggx_frag_shader_code.data()));
    auto charlie_frag_shader_code = readFile("src/shaders/ibl_charlie_frag.spv", frag_code_size);
    shader_modules.push_back(device->createShaderModule(frag_code_size, charlie_frag_shader_code.data()));

    return shader_modules;
}

std::vector<std::shared_ptr<renderer::ShaderModule>> getIblComputeShaderModules(
    std::shared_ptr<renderer::Device> device)
{
    uint64_t compute_code_size;
    std::vector<std::shared_ptr<renderer::ShaderModule>> shader_modules;
    shader_modules.reserve(1);
    auto compute_shader_code = readFile("src/shaders/ibl_smooth_comp.spv", compute_code_size);
    shader_modules.push_back(device->createShaderModule(compute_code_size, compute_shader_code.data()));

    return shader_modules;
}

std::vector<VkPipelineShaderStageCreateInfo> getShaderStages(
    std::vector<std::shared_ptr<renderer::ShaderModule>> shader_modules) {
    std::vector<VkPipelineShaderStageCreateInfo> shader_stages(shader_modules.size());

    // todo.
    auto vk_vert_shader_module = std::reinterpret_pointer_cast<renderer::VulkanShaderModule>(shader_modules[0]);
    shader_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shader_stages[0].module = vk_vert_shader_module->get();
    shader_stages[0].pName = "main";

    // todo.
    for (int i = 1; i < shader_modules.size(); i++) {
        auto vk_frag_shader_module = std::reinterpret_pointer_cast<renderer::VulkanShaderModule>(shader_modules[i]);
        shader_stages[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shader_stages[i].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shader_stages[i].module = vk_frag_shader_module->get();
        shader_stages[i].pName = "main";
    }

    return shader_stages;
}

std::vector<VkPipelineShaderStageCreateInfo> getComputeShaderStages(
    std::vector<std::shared_ptr<renderer::ShaderModule>> shader_modules) {
    std::vector<VkPipelineShaderStageCreateInfo> shader_stages(shader_modules.size());

    // todo.
    for (int i = 0; i < shader_modules.size(); i++) {
        auto vk_comp_shader_module = std::reinterpret_pointer_cast<renderer::VulkanShaderModule>(shader_modules[i]);
        shader_stages[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shader_stages[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shader_stages[i].module = vk_comp_shader_module->get();
        shader_stages[i].pName = "main";
    }

    return shader_stages;
}


VkPipelineVertexInputStateCreateInfo fillVkPipelineVertexInputStateCreateInfo(
    const std::vector<VkVertexInputBindingDescription>& binding_descs,
    const std::vector<VkVertexInputAttributeDescription>& attribute_descs) {
    VkPipelineVertexInputStateCreateInfo vertex_input_info{};
    vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_info.vertexBindingDescriptionCount = static_cast<uint32_t>(binding_descs.size());
    vertex_input_info.pVertexBindingDescriptions = binding_descs.data();
    vertex_input_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribute_descs.size());
    vertex_input_info.pVertexAttributeDescriptions = attribute_descs.data();

    return vertex_input_info;
}

VkPipelineInputAssemblyStateCreateInfo fillVkPipelineInputAssemblyStateCreateInfo(std::shared_ptr<renderer::ObjectData> gltf_object) {
    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = toVkPrimitiveTopology(gltf_object->meshes_[0].primitives_[0].topology_info_.topology);
    input_assembly.primitiveRestartEnable = gltf_object->meshes_[0].primitives_[0].topology_info_.restart_enable;

    return input_assembly;
}

VkPipelineRasterizationStateCreateInfo fillVkPipelineRasterizationStateCreateInfo() {
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional
    rasterizer.depthBiasClamp = 0.0f; // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

    return rasterizer;
}

VkPipelineRasterizationStateCreateInfo fillVkPipelineNocullingRasterizationStateCreateInfo() {
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional
    rasterizer.depthBiasClamp = 0.0f; // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

    return rasterizer;
}


VkPipelineMultisampleStateCreateInfo fillVkPipelineMultisampleStateCreateInfo() {
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f; // Optional
    multisampling.pSampleMask = nullptr; // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE; // Optional

    return multisampling;
}

VkViewport fillViewport(const glm::uvec2 size) {
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)size.x;
    viewport.height = (float)size.y;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    return viewport;
}

VkRect2D fillScissor(const glm::uvec2 size) {
    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = { size.x, size.y };

    return scissor;
}

VkPipelineViewportStateCreateInfo fillVkPipelineViewportStateCreateInfo(const VkViewport* viewport, const VkRect2D* scissor) {
    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.pViewports = viewport;
    viewport_state.scissorCount = 1;
    viewport_state.pScissors = scissor;

    return viewport_state;
}

std::vector<VkPipelineColorBlendAttachmentState> fillVkPipelineColorBlendAttachmentState(uint32_t num_attachments) {
    std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachements(num_attachments);
    VkPipelineColorBlendAttachmentState color_blend_attachment{};
    color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_blend_attachment.blendEnable = VK_FALSE;
    color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
    color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
    color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
    color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

    for (uint32_t i = 0; i < num_attachments; i++) {
        color_blend_attachements[i] = color_blend_attachment;
    }

    return color_blend_attachements;
}

VkPipelineColorBlendStateCreateInfo fillVkPipelineColorBlendStateCreateInfo(
    const std::vector<VkPipelineColorBlendAttachmentState>& color_blend_attachments) {
    VkPipelineColorBlendStateCreateInfo color_blending{};
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.logicOp = VK_LOGIC_OP_COPY; // Optional
    color_blending.attachmentCount = static_cast<uint32_t>(color_blend_attachments.size());
    color_blending.pAttachments = color_blend_attachments.data();
    color_blending.blendConstants[0] = 0.0f; // Optional
    color_blending.blendConstants[1] = 0.0f; // Optional
    color_blending.blendConstants[2] = 0.0f; // Optional
    color_blending.blendConstants[3] = 0.0f; // Optional

    return color_blending;
}

VkPipelineDepthStencilStateCreateInfo fillVkPipelineDepthStencilStateCreateInfo()
{
    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable = VK_TRUE;
    depth_stencil.depthWriteEnable = VK_TRUE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.minDepthBounds = 0.0f; // Optional
    depth_stencil.maxDepthBounds = 1.0f; // Optional
    depth_stencil.stencilTestEnable = VK_FALSE;
    //depth_stencil.front = {}; // Optional
    //depth_stencil.back = {}; // Optional

    return depth_stencil;
}

VkPipelineDepthStencilStateCreateInfo fillVkPipelineNoDepthStencilStateCreateInfo()
{
    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable = VK_FALSE;
    depth_stencil.depthWriteEnable = VK_FALSE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.minDepthBounds = 0.0f; // Optional
    depth_stencil.maxDepthBounds = 1.0f; // Optional
    depth_stencil.stencilTestEnable = VK_FALSE;
    //depth_stencil.front = {}; // Optional
    //depth_stencil.back = {}; // Optional

    return depth_stencil;
}

void RealWorldApplication::createGltfPipelineLayout()
{
    VkPushConstantRange push_const_range{};
    push_const_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    push_const_range.offset = 0;
    push_const_range.size = sizeof(ModelParams);

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    // todo.
    std::vector<VkDescriptorSetLayout> vk_layouts;
    vk_layouts.reserve(3);
    vk_layouts.push_back(std::reinterpret_pointer_cast<renderer::VulkanDescriptorSetLayout>(global_tex_desc_set_layout_)->get());
    vk_layouts.push_back(std::reinterpret_pointer_cast<renderer::VulkanDescriptorSetLayout>(desc_set_layout_)->get());
    if (gltf_object_->materials_.size() > 0) {
        vk_layouts.push_back(std::reinterpret_pointer_cast<renderer::VulkanDescriptorSetLayout>(material_tex_desc_set_layout_)->get());
    }
    pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(vk_layouts.size());
    pipeline_layout_info.pSetLayouts = vk_layouts.data();
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_const_range;

    gltf_pipeline_layout_ = ::createPipelineLayout(device_, pipeline_layout_info);
}

void RealWorldApplication::createSkyboxPipelineLayout()
{
    std::vector<VkDescriptorSetLayout> vk_layouts;
    vk_layouts.reserve(3);
    vk_layouts.push_back(std::reinterpret_pointer_cast<renderer::VulkanDescriptorSetLayout>(skybox_desc_set_layout_)->get());
    vk_layouts.push_back(std::reinterpret_pointer_cast<renderer::VulkanDescriptorSetLayout>(desc_set_layout_)->get());

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(vk_layouts.size());
    pipeline_layout_info.pSetLayouts = vk_layouts.data();
    pipeline_layout_info.pushConstantRangeCount = 0;
    pipeline_layout_info.pPushConstantRanges = nullptr;

    skybox_pipeline_layout_ = ::createPipelineLayout(device_, pipeline_layout_info);
}

void RealWorldApplication::createCubemapPipelineLayout()
{
    VkPushConstantRange push_const_range{};
    push_const_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    push_const_range.offset = 0;
    push_const_range.size = sizeof(IblParams);

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    // todo.
    std::vector<VkDescriptorSetLayout> vk_layouts(1);
    vk_layouts[0] = std::reinterpret_pointer_cast<renderer::VulkanDescriptorSetLayout>(ibl_desc_set_layout_)->get();
    pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(vk_layouts.size());
    pipeline_layout_info.pSetLayouts = vk_layouts.data();
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_const_range;

    ibl_pipeline_layout_ = ::createPipelineLayout(device_, pipeline_layout_info);
}

void RealWorldApplication::createCubemapComputePipelineLayout()
{
    VkPushConstantRange push_const_range{};
    push_const_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_const_range.offset = 0;
    push_const_range.size = sizeof(IblComputeParams);

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    // todo.
    std::vector<VkDescriptorSetLayout> vk_layouts(1);
    vk_layouts[0] = std::reinterpret_pointer_cast<renderer::VulkanDescriptorSetLayout>(ibl_comp_desc_set_layout_)->get();
    pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(vk_layouts.size());
    pipeline_layout_info.pSetLayouts = vk_layouts.data();
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_const_range;

    ibl_comp_pipeline_layout_ = ::createPipelineLayout(device_, pipeline_layout_info);
}

void RealWorldApplication::createGraphicsPipeline() {
    auto viewport = fillViewport(swap_chain_extent_);
    auto scissor = fillScissor(swap_chain_extent_);
    auto cubemap_viewport = fillViewport(glm::uvec2(kCubemapSize, kCubemapSize));
    auto cubemap_scissor = fillScissor(glm::uvec2(kCubemapSize, kCubemapSize));
    auto color_blend_attachment = fillVkPipelineColorBlendAttachmentState(1);
    auto cubemap_color_blend_attachment = fillVkPipelineColorBlendAttachmentState(6);

    auto rasterizer = fillVkPipelineRasterizationStateCreateInfo();
    auto ibl_rasterizer = fillVkPipelineNocullingRasterizationStateCreateInfo();
    auto multisampling = fillVkPipelineMultisampleStateCreateInfo();
    auto viewport_state = fillVkPipelineViewportStateCreateInfo(&viewport, &scissor);
    auto cubemap_viewport_state = fillVkPipelineViewportStateCreateInfo(&cubemap_viewport, &cubemap_scissor);
    auto color_blending = fillVkPipelineColorBlendStateCreateInfo(color_blend_attachment);
    auto cubemap_color_blending = fillVkPipelineColorBlendStateCreateInfo(cubemap_color_blend_attachment);
    auto depth_stencil = fillVkPipelineDepthStencilStateCreateInfo();
    auto cubemap_depth_stencil = fillVkPipelineNoDepthStencilStateCreateInfo();

    VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_LINE_WIDTH
    };

    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = 2;
    dynamic_state.pDynamicStates = dynamic_states;

    auto vk_render_pass = std::reinterpret_pointer_cast<renderer::VulkanRenderPass>(render_pass_);
    assert(vk_render_pass);

    VkGraphicsPipelineCreateInfo pipeline_info{};
    {
        auto gltf_binding_descs = toVkVertexInputBindingDescription(gltf_object_->meshes_[0].primitives_[0].binding_descs_);
        auto gltf_attribute_descs = toVkVertexInputAttributeDescription(gltf_object_->meshes_[0].primitives_[0].attribute_descs_);
        auto gltf_vertex_input_info = fillVkPipelineVertexInputStateCreateInfo(gltf_binding_descs, gltf_attribute_descs);
        auto gltf_input_assembly = fillVkPipelineInputAssemblyStateCreateInfo(gltf_object_);

        // todo.
        auto shader_modules = getGltfShaderModules(
            device_,
            gltf_object_->meshes_[0].primitives_[0].has_normal_,
            gltf_object_->meshes_[0].primitives_[0].has_tangent_,
            gltf_object_->meshes_[0].primitives_[0].has_texcoord_0_,
            gltf_object_->meshes_[0].primitives_[0].has_skin_set_0_);
        auto shader_stages = getShaderStages(shader_modules);

        auto vk_gltf_pipeline_layout = std::reinterpret_pointer_cast<renderer::VulkanPipelineLayout>(gltf_pipeline_layout_);
        assert(vk_gltf_pipeline_layout);

        pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_info.stageCount = static_cast<uint32_t>(shader_stages.size());
        pipeline_info.pStages = shader_stages.data();
        pipeline_info.pVertexInputState = &gltf_vertex_input_info;
        pipeline_info.pInputAssemblyState = &gltf_input_assembly;
        pipeline_info.pViewportState = &viewport_state;
        pipeline_info.pRasterizationState = &rasterizer;
        pipeline_info.pMultisampleState = &multisampling;
        pipeline_info.pDepthStencilState = &depth_stencil;
        pipeline_info.pColorBlendState = &color_blending;
        //    pipeline_info.pDynamicState = nullptr; // Optional
        pipeline_info.layout = vk_gltf_pipeline_layout->get();
        pipeline_info.renderPass = vk_render_pass->get();
        pipeline_info.subpass = 0;
        pipeline_info.basePipelineHandle = VK_NULL_HANDLE; // Optional
        pipeline_info.basePipelineIndex = -1; // Optional

        // todo.
        gltf_pipeline_ = ::createPipeline(device_, pipeline_info);

        for (auto& shader_module : shader_modules) {
            device_->destroyShaderModule(shader_module);
        }
    }

    {
        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        input_assembly.primitiveRestartEnable = false;

        auto skybox_binding_descs = toVkVertexInputBindingDescription(SkyBoxVertex::getBindingDescription());
        auto skybox_attrib_descs = toVkVertexInputAttributeDescription(SkyBoxVertex::getAttributeDescriptions());
        auto skybox_vertex_input_info = fillVkPipelineVertexInputStateCreateInfo(skybox_binding_descs, skybox_attrib_descs);

        // todo.
        auto shader_modules = getSkyboxShaderModules(device_);
        auto shader_stages = getShaderStages(shader_modules);

        auto vk_skybox_pipeline_layout = std::reinterpret_pointer_cast<renderer::VulkanPipelineLayout>(skybox_pipeline_layout_);
        assert(vk_skybox_pipeline_layout);

        VkGraphicsPipelineCreateInfo skybox_pipeline_info = pipeline_info;

        skybox_pipeline_info.stageCount = static_cast<uint32_t>(shader_stages.size());
        skybox_pipeline_info.pStages = shader_stages.data();
        skybox_pipeline_info.pVertexInputState = &skybox_vertex_input_info;
        skybox_pipeline_info.pInputAssemblyState = &input_assembly;
        skybox_pipeline_info.layout = vk_skybox_pipeline_layout->get();

        // todo.
        skybox_pipeline_ = ::createPipeline(device_, skybox_pipeline_info);

        for (auto& shader_module : shader_modules) {
            device_->destroyShaderModule(shader_module);
        }
    }

    {
        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        input_assembly.primitiveRestartEnable = false;

        // todo.
        auto ibl_shader_modules = getIblShaderModules(device_);
        auto ibl_shader_stages = getShaderStages(ibl_shader_modules);

        VkPipelineVertexInputStateCreateInfo vertex_input_info{};
        vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        auto vk_ibl_pipeline_layout = std::reinterpret_pointer_cast<renderer::VulkanPipelineLayout>(ibl_pipeline_layout_);
        assert(vk_ibl_pipeline_layout);

        auto vk_cubemap_render_pass = std::reinterpret_pointer_cast<renderer::VulkanRenderPass>(cubemap_render_pass_);
        assert(vk_cubemap_render_pass);

        VkGraphicsPipelineCreateInfo ibl_pipeline_info = pipeline_info;
        ibl_pipeline_info.stageCount = 2;
        ibl_pipeline_info.pStages = ibl_shader_stages.data();
        ibl_pipeline_info.pVertexInputState = &vertex_input_info;
        ibl_pipeline_info.pInputAssemblyState = &input_assembly;
        ibl_pipeline_info.pViewportState = &cubemap_viewport_state;
        ibl_pipeline_info.pDepthStencilState = &cubemap_depth_stencil;
        ibl_pipeline_info.pColorBlendState = &cubemap_color_blending;
        ibl_pipeline_info.pRasterizationState = &ibl_rasterizer;
        ibl_pipeline_info.layout = vk_ibl_pipeline_layout->get();
        ibl_pipeline_info.renderPass = vk_cubemap_render_pass->get();
        ibl_pipeline_info.subpass = 0;
        ibl_pipeline_info.basePipelineHandle = VK_NULL_HANDLE; // Optional
        ibl_pipeline_info.basePipelineIndex = -1; // Optional

        envmap_pipeline_ = ::createPipeline(device_, ibl_pipeline_info);

        std::array<VkPipelineShaderStageCreateInfo, 2> lambertian_stages = { ibl_shader_stages[0], ibl_shader_stages[2] };
        ibl_pipeline_info.pStages = lambertian_stages.data();
        lambertian_pipeline_ = ::createPipeline(device_, ibl_pipeline_info);

        std::array<VkPipelineShaderStageCreateInfo, 2> ggx_stages = { ibl_shader_stages[0], ibl_shader_stages[3] };
        ibl_pipeline_info.pStages = ggx_stages.data();
        ggx_pipeline_ = ::createPipeline(device_, ibl_pipeline_info);

        std::array<VkPipelineShaderStageCreateInfo, 2> charlie_stages = { ibl_shader_stages[0], ibl_shader_stages[4] };
        ibl_pipeline_info.pStages = charlie_stages.data();
        charlie_pipeline_ = ::createPipeline(device_, ibl_pipeline_info);

        for (auto& shader_module : ibl_shader_modules) {
            device_->destroyShaderModule(shader_module);
        }
    }
}

void RealWorldApplication::createComputePipeline()
{
    auto vk_device = std::reinterpret_pointer_cast<work::renderer::VulkanDevice>(device_);
    assert(vk_device);

    auto ibl_compute_shader_modules = getIblComputeShaderModules(device_);
    auto ibl_compute_shader_stages = getComputeShaderStages(ibl_compute_shader_modules);

    auto vk_ibl_comp_pipeline_layout = std::reinterpret_pointer_cast<renderer::VulkanPipelineLayout>(ibl_comp_pipeline_layout_);
    assert(vk_ibl_comp_pipeline_layout);

    // flags = 0, - e.g. disable optimization
    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = ibl_compute_shader_stages[0];
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
    pipeline_info.basePipelineIndex = -1;
    pipeline_info.layout = vk_ibl_comp_pipeline_layout->get();

    VkPipeline compute_pipeline;
    if (vkCreateComputePipelines(vk_device->get(), VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &compute_pipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create compute pipeline!");
    }

    auto vk_blur_pipeline = std::make_shared<work::renderer::VulkanPipeline>();
    vk_blur_pipeline->set(compute_pipeline);

    blur_comp_pipeline_ = std::move(vk_blur_pipeline);

    for (auto& shader_module : ibl_compute_shader_modules) {
        device_->destroyShaderModule(shader_module);
    }
}

void RealWorldApplication::createRenderPass() {
    VkAttachmentDescription color_attachment{};
    color_attachment.format = toVkFormat(swap_chain_image_format_);
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_attachment_ref{};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depth_attachment{};
    depth_attachment.format = toVkFormat(findDepthFormat(vk_physical_device_));
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depth_attachment_ref{};
    depth_attachment_ref.attachment = 1;
    depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;
    subpass.pDepthStencilAttachment = &depth_attachment_ref;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 2> attachments = { color_attachment, depth_attachment };

    VkRenderPassCreateInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = static_cast<uint32_t>(attachments.size());
    render_pass_info.pAttachments = attachments.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = 1;
    render_pass_info.pDependencies = &dependency;

    render_pass_ = ::createRenderPass(device_, render_pass_info);
}

void RealWorldApplication::createCubemapRenderPass() {
    VkAttachmentDescription color_attachment{};
    color_attachment.format = toVkFormat(renderer::Format::R16G16B16A16_SFLOAT);
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    std::vector<VkAttachmentReference> color_attachment_refs(6);
    for (uint32_t i = 0; i < 6; i++) {
        color_attachment_refs[i].attachment = i;
        color_attachment_refs[i].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = static_cast<uint32_t>(color_attachment_refs.size());
    subpass.pColorAttachments = color_attachment_refs.data();
    subpass.pDepthStencilAttachment = nullptr;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    std::vector<VkAttachmentDescription> attachments = {6, color_attachment};

    VkRenderPassCreateInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = static_cast<uint32_t>(attachments.size());
    render_pass_info.pAttachments = attachments.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = 1;
    render_pass_info.pDependencies = &dependency;

    cubemap_render_pass_ = ::createRenderPass(device_, render_pass_info);
}

void RealWorldApplication::createFramebuffers() {
    swap_chain_framebuffers_.resize(swap_chain_image_views_.size());
    for (uint64_t i = 0; i < swap_chain_image_views_.size(); i++) {
        assert(swap_chain_image_views_[i]);
        assert(depth_buffer_.view);
        assert(render_pass_);
        std::vector<std::shared_ptr<renderer::ImageView>> attachments(2);
        attachments[0] = swap_chain_image_views_[i];
        attachments[1] = depth_buffer_.view;

        swap_chain_framebuffers_[i] = device_->createFrameBuffer(render_pass_, attachments, swap_chain_extent_);
    }
}

void RealWorldApplication::createCommandPool() {
    QueueFamilyIndices queue_family_indices = findQueueFamilies(vk_physical_device_, surface_);
    command_pool_ = device_->createCommandPool(
        queue_family_indices.graphics_family_.value(), 
        static_cast<renderer::CommandPoolCreateFlags>(renderer::CommandPoolCreateFlagBits::RESET_COMMAND_BUFFER_BIT));
}

void RealWorldApplication::createCommandBuffers() {
    command_buffers_ = device_->allocateCommandBuffers(command_pool_, static_cast<uint32_t>(swap_chain_framebuffers_.size()));

/*    std::vector<renderer::ClearValue> clear_values;
    clear_values.resize(2);
    clear_values[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
    clear_values[1].depth_stencil = { 1.0f, 0 };

    for (uint64_t i = 0; i < command_buffers_.size(); i++) {
        auto& cmd_buf = command_buffers_[i];

        cmd_buf->beginCommandBuffer(0);
        cmd_buf->beginRenderPass(render_pass_, swap_chain_framebuffers_[i], swap_chain_extent_, clear_values);
        cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, gltf_pipeline_);

        for (const auto& mesh : gltf_object_->meshes_) {
            for (const auto& prim : mesh.primitives_) {
                const auto& attrib_list = prim.attribute_descs_;

                const auto& material = gltf_object_->materials_[prim.material_idx_];

                std::vector<std::shared_ptr<renderer::Buffer>> buffers(attrib_list.size());
                std::vector<uint64_t> offsets(attrib_list.size());
                for (int i_attrib = 0; i_attrib < attrib_list.size(); i_attrib++) {
                    const auto& buffer_view = gltf_object_->buffer_views_[attrib_list[i_attrib].buffer_view];
                    buffers[i_attrib] = gltf_object_->buffers_[buffer_view.buffer_idx].buffer;
                    offsets[i_attrib] = attrib_list[i_attrib].buffer_offset;
                }
                cmd_buf->bindVertexBuffers(0, buffers, offsets);
                const auto& index_buffer_view = gltf_object_->buffer_views_[prim.index_desc_.binding];
                cmd_buf->bindIndexBuffer(gltf_object_->buffers_[index_buffer_view.buffer_idx].buffer,
                    prim.index_desc_.offset + index_buffer_view.offset,
                    prim.index_desc_.index_type);

                // todo.
                auto vk_cmd_buf = std::reinterpret_pointer_cast<renderer::VulkanCommandBuffer>(cmd_buf);
                auto vk_pipeline_layout = std::reinterpret_pointer_cast<renderer::VulkanPipelineLayout>(pipeline_layout_);
                VkDescriptorSet desc_sets[] = {
                    std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(global_tex_desc_set_)->get(),
                    std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(desc_sets_[i])->get(),
                    std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(material.desc_set_)->get() };
                vkCmdBindDescriptorSets(vk_cmd_buf->get(), VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout->get(), 0, 3, desc_sets, 0, nullptr);

                cmd_buf->drawIndexed(static_cast<uint32_t>(prim.index_desc_.index_count));
            }
        }
        cmd_buf->endRenderPass();
        cmd_buf->endCommandBuffer();
    }*/
}

void RealWorldApplication::mainLoop() {
    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();
        drawFrame();
    }

    device_->waitIdle();
}

void RealWorldApplication::drawMesh(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const renderer::MeshInfo& mesh_info,
    const ModelParams& model_params,
    const uint32_t image_index) {
    for (const auto& prim : mesh_info.primitives_) {
        const auto& attrib_list = prim.attribute_descs_;

        std::vector<std::shared_ptr<renderer::Buffer>> buffers(attrib_list.size());
        std::vector<uint64_t> offsets(attrib_list.size());
        for (int i_attrib = 0; i_attrib < attrib_list.size(); i_attrib++) {
            const auto& buffer_view = gltf_object_->buffer_views_[attrib_list[i_attrib].buffer_view];
            buffers[i_attrib] = gltf_object_->buffers_[buffer_view.buffer_idx].buffer;
            offsets[i_attrib] = attrib_list[i_attrib].buffer_offset;
        }
        cmd_buf->bindVertexBuffers(0, buffers, offsets);
        const auto& index_buffer_view = gltf_object_->buffer_views_[prim.index_desc_.binding];
        cmd_buf->bindIndexBuffer(gltf_object_->buffers_[index_buffer_view.buffer_idx].buffer,
            prim.index_desc_.offset + index_buffer_view.offset,
            prim.index_desc_.index_type);

        // todo.
        auto vk_cmd_buf = std::reinterpret_pointer_cast<renderer::VulkanCommandBuffer>(cmd_buf);
        auto vk_gltf_pipeline_layout = std::reinterpret_pointer_cast<renderer::VulkanPipelineLayout>(gltf_pipeline_layout_);

        std::vector<VkDescriptorSet> desc_sets;
        desc_sets.reserve(3);
        desc_sets.push_back(std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(global_tex_desc_set_)->get());
        desc_sets.push_back(std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(desc_sets_[image_index])->get());
        if (prim.material_idx_ >= 0) {
            const auto& material = gltf_object_->materials_[prim.material_idx_];
            desc_sets.push_back(std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(material.desc_set_)->get());
        }
        vkCmdBindDescriptorSets(
            vk_cmd_buf->get(), 
            VK_PIPELINE_BIND_POINT_GRAPHICS, 
            vk_gltf_pipeline_layout->get(), 
            0, 
            static_cast<uint32_t>(desc_sets.size()), 
            desc_sets.data(), 
            0, 
            nullptr);

        vkCmdPushConstants(vk_cmd_buf->get(), vk_gltf_pipeline_layout->get(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ModelParams), &model_params);

        cmd_buf->drawIndexed(static_cast<uint32_t>(prim.index_desc_.index_count));
    }
}

void RealWorldApplication::drawNodes(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    int32_t node_idx,
    const uint32_t image_index,
    const glm::mat4& parent_matrix) {
    if (node_idx >= 0) {
        const auto& node = gltf_object_->nodes_[node_idx];
        auto cur_matrix = parent_matrix;
        if (node.matrix) {
            cur_matrix *= *node.matrix;
        }
        if (node.mesh_idx >= 0) {
            ModelParams model_params{};
            model_params.model_mat = cur_matrix;
            auto invert_mat = inverse(model_params.model_mat);
            model_params.normal_mat = transpose(invert_mat);
            drawMesh(cmd_buf, gltf_object_->meshes_[node.mesh_idx], model_params, image_index);
        }

        for (auto& child_idx : node.child_idx) {
            drawNodes(cmd_buf, child_idx, image_index, cur_matrix);
        }
    }
}

void RealWorldApplication::drawFrame() {
    std::vector<std::shared_ptr<renderer::Fence>> in_flight_fences(1);
    in_flight_fences[0] = in_flight_fences_[current_frame_];
    device_->waitForFences(in_flight_fences);

    // todo
    auto vk_device = std::reinterpret_pointer_cast<renderer::VulkanDevice>(device_);
    auto vk_swap_chain = std::reinterpret_pointer_cast<renderer::VulkanSwapchain>(swap_chain_);
    auto vk_img_available_semphores = std::reinterpret_pointer_cast<renderer::VulkanSemaphore>(image_available_semaphores_[current_frame_]);
    assert(vk_device);

    uint32_t image_index;
    auto result = vkAcquireNextImageKHR(vk_device->get(), vk_swap_chain->get(), UINT64_MAX, vk_img_available_semphores->get(), VK_NULL_HANDLE, &image_index);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain();
        return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    if (images_in_flight_[image_index] != VK_NULL_HANDLE) {
        std::vector<std::shared_ptr<renderer::Fence>> images_in_flight(1);
        images_in_flight[0] = images_in_flight_[image_index];
        device_->waitForFences(images_in_flight);
    }
    // Mark the image as now being in use by this frame
    images_in_flight_[image_index] = in_flight_fences_[current_frame_];

    int32_t root_node = gltf_object_->default_scene_ >= 0 ? gltf_object_->default_scene_ : 0;
    auto min_t = gltf_object_->scenes_[root_node].bbox_min_;
    auto max_t = gltf_object_->scenes_[root_node].bbox_max_;

    auto center = (min_t + max_t) * 0.5f;
    auto extent = (max_t - min_t) * 0.5f;
    float radius = max(max(extent.x, extent.y), extent.z);

    updateViewConstBuffer(image_index, center, radius);

    device_->resetFences(in_flight_fences);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    // todo.
    auto command_buffer = command_buffers_[image_index];
    auto vk_command_buffer = std::reinterpret_pointer_cast<renderer::VulkanCommandBuffer>(command_buffer);
    std::vector<std::shared_ptr<renderer::CommandBuffer>>command_buffers(1, command_buffer);

    std::vector<renderer::ClearValue> clear_values;
    clear_values.resize(2);
    clear_values[0].color = { 50.0f / 255.0f, 50.0f / 255.0f, 50.0f / 255.0f, 1.0f };
    clear_values[1].depth_stencil = { 1.0f, 0 };

    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    auto& cmd_buf = command_buffer;
    auto vk_cmd_buf = std::reinterpret_pointer_cast<renderer::VulkanCommandBuffer>(cmd_buf);

    cmd_buf->reset(0);
    cmd_buf->beginCommandBuffer(static_cast<renderer::CommandBufferUsageFlags>(renderer::CommandBufferUsageFlagBits::ONE_TIME_SUBMIT_BIT));

    // generate envmap cubemap from panorama hdr image.
    {
        addImageBarrier(cmd_buf, rt_envmap_tex_.image, image_source_info, image_as_color_attachement, 0, 1, 0, 6);

        cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, envmap_pipeline_);

        std::vector<renderer::ClearValue> envmap_clear_values;
        cmd_buf->beginRenderPass(cubemap_render_pass_, rt_envmap_tex_.framebuffers[0], glm::uvec2(kCubemapSize, kCubemapSize), envmap_clear_values);

        IblParams ibl_params = {};

        auto vk_ibl_pipeline_layout = std::reinterpret_pointer_cast<renderer::VulkanPipelineLayout>(ibl_pipeline_layout_);
        vkCmdPushConstants(vk_cmd_buf->get(), vk_ibl_pipeline_layout->get(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ibl_params), &ibl_params);

        // todo.
        std::vector<VkDescriptorSet> desc_sets(1);
        desc_sets[0] = std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(envmap_tex_desc_set_)->get();
        vkCmdBindDescriptorSets(
            vk_cmd_buf->get(),
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            vk_ibl_pipeline_layout->get(),
            0,
            static_cast<uint32_t>(desc_sets.size()),
            desc_sets.data(),
            0,
            nullptr);

        vkCmdDraw(vk_cmd_buf->get(), 3, 1u, 0, 0);

        cmd_buf->endRenderPass();

        uint32_t num_mips = static_cast<uint32_t>(std::log2(kCubemapSize) + 1);

        generateMipmapLevels(cmd_buf, rt_envmap_tex_.image, num_mips, kCubemapSize, kCubemapSize, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }

    // generate ibl diffuse texture.
    {
        addImageBarrier(cmd_buf, rt_ibl_diffuse_tex_.image, image_source_info, image_as_color_attachement, 0, 1, 0, 6);

        cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, lambertian_pipeline_);

        std::vector<renderer::ClearValue> envmap_clear_values;
        cmd_buf->beginRenderPass(cubemap_render_pass_, rt_ibl_diffuse_tex_.framebuffers[0], glm::uvec2(kCubemapSize, kCubemapSize), envmap_clear_values);

        IblParams ibl_params = {};
        ibl_params.roughness = 1.0f;
        ibl_params.currentMipLevel = 0;
        ibl_params.width = kCubemapSize;
        ibl_params.lodBias = 0;

        auto vk_ibl_pipeline_layout = std::reinterpret_pointer_cast<renderer::VulkanPipelineLayout>(ibl_pipeline_layout_);
        vkCmdPushConstants(vk_cmd_buf->get(), vk_ibl_pipeline_layout->get(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ibl_params), &ibl_params);

        // todo.
        std::vector<VkDescriptorSet> desc_sets(1);
        desc_sets[0] = std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(ibl_tex_desc_set_)->get();
        vkCmdBindDescriptorSets(
            vk_cmd_buf->get(),
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            vk_ibl_pipeline_layout->get(),
            0,
            static_cast<uint32_t>(desc_sets.size()),
            desc_sets.data(),
            0,
            nullptr);

        vkCmdDraw(vk_cmd_buf->get(), 3, 1u, 0, 0);

        cmd_buf->endRenderPass();

        addImageBarrier(cmd_buf, rt_ibl_diffuse_tex_.image, image_as_color_attachement, image_as_shader_sampler, 0, 1, 0, 6);
    }

    // generate ibl specular texture.
    {
        uint32_t num_mips = static_cast<uint32_t>(std::log2(kCubemapSize) + 1);
        cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, ggx_pipeline_);

        for (int i_mip = num_mips-1; i_mip >= 0; i_mip--) {
            addImageBarrier(cmd_buf, rt_ibl_specular_tex_.image, image_source_info, image_as_color_attachement, i_mip, 1, 0, 6);

            uint32_t width = std::max(static_cast<uint32_t>(kCubemapSize) >> i_mip, 1u);
            uint32_t height = std::max(static_cast<uint32_t>(kCubemapSize) >> i_mip, 1u);

            std::vector<renderer::ClearValue> envmap_clear_values;
            cmd_buf->beginRenderPass(cubemap_render_pass_, rt_ibl_specular_tex_.framebuffers[i_mip], glm::uvec2(width, height), envmap_clear_values);

            IblParams ibl_params = {};
            ibl_params.roughness = static_cast<float>(i_mip) / static_cast<float>(num_mips - 1);
            ibl_params.currentMipLevel = i_mip;
            ibl_params.width = kCubemapSize;
            ibl_params.lodBias = 0;

            auto vk_ibl_pipeline_layout = std::reinterpret_pointer_cast<renderer::VulkanPipelineLayout>(ibl_pipeline_layout_);
            vkCmdPushConstants(vk_cmd_buf->get(), vk_ibl_pipeline_layout->get(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ibl_params), &ibl_params);

            // todo.
            std::vector<VkDescriptorSet> desc_sets(1);
            desc_sets[0] = std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(ibl_tex_desc_set_)->get();
            vkCmdBindDescriptorSets(
                vk_cmd_buf->get(),
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                vk_ibl_pipeline_layout->get(),
                0,
                static_cast<uint32_t>(desc_sets.size()),
                desc_sets.data(),
                0,
                nullptr);

            vkCmdDraw(vk_cmd_buf->get(), 3, 1u, 0, 0);

            cmd_buf->endRenderPass();
        }

        addImageBarrier(cmd_buf, rt_ibl_specular_tex_.image, image_as_color_attachement, image_as_shader_sampler, 0, num_mips, 0, 6);
    }

    // generate ibl sheen texture.
    {
        uint32_t num_mips = static_cast<uint32_t>(std::log2(kCubemapSize) + 1);
        cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, charlie_pipeline_);

        for (int i_mip = num_mips - 1; i_mip >= 0; i_mip--) {
            addImageBarrier(cmd_buf, rt_ibl_sheen_tex_.image, image_source_info, image_as_color_attachement, i_mip, 1, 0, 6);

            uint32_t width = std::max(static_cast<uint32_t>(kCubemapSize) >> i_mip, 1u);
            uint32_t height = std::max(static_cast<uint32_t>(kCubemapSize) >> i_mip, 1u);

            std::vector<renderer::ClearValue> envmap_clear_values;
            cmd_buf->beginRenderPass(cubemap_render_pass_, rt_ibl_sheen_tex_.framebuffers[i_mip], glm::uvec2(width, height), envmap_clear_values);

            IblParams ibl_params = {};
            ibl_params.roughness = static_cast<float>(i_mip) / static_cast<float>(num_mips - 1);
            ibl_params.currentMipLevel = i_mip;
            ibl_params.width = kCubemapSize;
            ibl_params.lodBias = 0;

            auto vk_ibl_pipeline_layout = std::reinterpret_pointer_cast<renderer::VulkanPipelineLayout>(ibl_pipeline_layout_);
            vkCmdPushConstants(vk_cmd_buf->get(), vk_ibl_pipeline_layout->get(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ibl_params), &ibl_params);

            // todo.
            std::vector<VkDescriptorSet> desc_sets(1);
            desc_sets[0] = std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(ibl_tex_desc_set_)->get();
            vkCmdBindDescriptorSets(
                vk_cmd_buf->get(),
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                vk_ibl_pipeline_layout->get(),
                0,
                static_cast<uint32_t>(desc_sets.size()),
                desc_sets.data(),
                0,
                nullptr);

            vkCmdDraw(vk_cmd_buf->get(), 3, 1u, 0, 0);

            cmd_buf->endRenderPass();
        }

        addImageBarrier(cmd_buf, rt_ibl_sheen_tex_.image, image_as_color_attachement, image_as_shader_sampler, 0, num_mips, 0, 6);
    }

    {
        if (0)
        {
            addImageBarrier(cmd_buf, rt_ibl_diffuse_tex_.image, image_source_info, image_as_store, 0, 1, 0, 6);

            cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, blur_comp_pipeline_);
            IblComputeParams ibl_comp_params = {};
            ibl_comp_params.size = glm::ivec4(kCubemapSize, kCubemapSize, 0, 0);

            auto vk_ibl_comp_pipeline_layout = std::reinterpret_pointer_cast<renderer::VulkanPipelineLayout>(ibl_comp_pipeline_layout_);
            vkCmdPushConstants(vk_cmd_buf->get(), vk_ibl_comp_pipeline_layout->get(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ibl_comp_params), &ibl_comp_params);

            // todo.
            std::vector<VkDescriptorSet> desc_sets(1);
            desc_sets[0] = std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(ibl_diffuse_tex_desc_set_)->get();
            vkCmdBindDescriptorSets(
                vk_cmd_buf->get(),
                VK_PIPELINE_BIND_POINT_COMPUTE,
                vk_ibl_comp_pipeline_layout->get(),
                0,
                static_cast<uint32_t>(desc_sets.size()),
                desc_sets.data(),
                0,
                nullptr);

            vkCmdDispatch(vk_cmd_buf->get(), (kCubemapSize + 7) / 8, (kCubemapSize + 7) / 8, 6);

            uint32_t num_mips = static_cast<uint32_t>(std::log2(kCubemapSize) + 1);
            addImageBarrier(cmd_buf, rt_ibl_diffuse_tex_.image, image_as_store, image_as_shader_sampler, 0, 1, 0, 6);
            addImageBarrier(cmd_buf, rt_ibl_specular_tex_.image, image_source_info, image_as_shader_sampler, 0, num_mips, 0, 6);
            addImageBarrier(cmd_buf, rt_ibl_sheen_tex_.image, image_source_info, image_as_shader_sampler, 0, num_mips, 0, 6);
        }
    }

    // render gltf meshes.
    auto model_mat = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    {
        cmd_buf->beginRenderPass(render_pass_, swap_chain_framebuffers_[image_index], swap_chain_extent_, clear_values);
        cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, gltf_pipeline_);

        for (auto node_idx : gltf_object_->scenes_[root_node].nodes_) {
            drawNodes(cmd_buf, node_idx, image_index, model_mat);
        }

        // render skybox.
        {
            cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, skybox_pipeline_);
            std::vector<std::shared_ptr<renderer::Buffer>> buffers(1);
            std::vector<uint64_t> offsets(1);
            buffers[0] = vertex_buffer_.buffer;
            offsets[0] = 0;

            cmd_buf->bindVertexBuffers(0, buffers, offsets);
            cmd_buf->bindIndexBuffer(index_buffer_.buffer, 0, renderer::IndexType::UINT16);

            // todo.
            auto vk_skybox_pipeline_layout = std::reinterpret_pointer_cast<renderer::VulkanPipelineLayout>(skybox_pipeline_layout_);
            std::vector<VkDescriptorSet> desc_sets(2);
            desc_sets[0] = std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(skybox_tex_desc_set_)->get();
            desc_sets[1] = std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(desc_sets_[image_index])->get();
            vkCmdBindDescriptorSets(
                vk_cmd_buf->get(),
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                vk_skybox_pipeline_layout->get(),
                0,
                static_cast<uint32_t>(desc_sets.size()),
                desc_sets.data(),
                0,
                nullptr);

            cmd_buf->drawIndexed(36);
        }

        cmd_buf->endRenderPass();
    }

    cmd_buf->endCommandBuffer();
    auto vk_render_finished_semphores = std::reinterpret_pointer_cast<renderer::VulkanSemaphore>(render_finished_semaphores_[current_frame_]);
    auto vk_in_flight_fence = std::reinterpret_pointer_cast<renderer::VulkanFence>(in_flight_fences_[current_frame_]);
    auto vk_graphic_queue = std::reinterpret_pointer_cast<renderer::VulkanQueue>(graphics_queue_);
    auto vk_present_queue = std::reinterpret_pointer_cast<renderer::VulkanQueue>(present_queue_);

    std::vector<VkCommandBuffer> cmd_bufs(1, vk_command_buffer->get());
    VkSemaphore wait_semaphores[] = { vk_img_available_semphores->get() };
    VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = wait_semaphores;
    submit_info.pWaitDstStageMask = wait_stages;
    submit_info.commandBufferCount = static_cast<uint32_t>(cmd_bufs.size());
    submit_info.pCommandBuffers = cmd_bufs.data();
    VkSemaphore signal_semaphores[] = { vk_render_finished_semphores->get() };
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = signal_semaphores;

    if (vkQueueSubmit(vk_graphic_queue->get(), 1, &submit_info, vk_in_flight_fence->get()) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = signal_semaphores;
    VkSwapchainKHR swapChains[] = { vk_swap_chain->get() };
    present_info.swapchainCount = 1;
    present_info.pSwapchains = swapChains;
    present_info.pImageIndices = &image_index;
    present_info.pResults = nullptr; // Optional

    result = vkQueuePresentKHR(vk_present_queue->get(), &present_info);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebuffer_resized_) {
        framebuffer_resized_ = false;
        recreateSwapChain();
    }
    else if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image!");
    }

    current_frame_ = (current_frame_ + 1) % kMaxFramesInFlight;
}

void RealWorldApplication::createSyncObjects() {
    image_available_semaphores_.resize(kMaxFramesInFlight);
    render_finished_semaphores_.resize(kMaxFramesInFlight);
    in_flight_fences_.resize(kMaxFramesInFlight);
    images_in_flight_.resize(swap_chain_images_.size(), VK_NULL_HANDLE);

    assert(device_);
    for (uint64_t i = 0; i < kMaxFramesInFlight; i++) {
        image_available_semaphores_[i] = device_->createSemaphore();
        render_finished_semaphores_[i] = device_->createSemaphore();
        in_flight_fences_[i] = device_->createFence();
    }
}

void RealWorldApplication::createVertexBuffer() {
    const std::vector<SkyBoxVertex> vertices = {
        {{-1.0f, -1.0f, -1.0f}},
        {{1.0f, -1.0f, -1.0f}},
        {{-1.0f, 1.0f, -1.0f}},
        {{1.0f, 1.0f, -1.0f}},
        {{-1.0f, -1.0f, 1.0f}},
        {{1.0f, -1.0f, 1.0f}},
        {{-1.0f, 1.0f, 1.0f}},
        {{1.0f, 1.0f, 1.0f}},
    };

    VkDeviceSize buffer_size = sizeof(vertices[0]) * vertices.size();

    createBufferWithSrcData(
        device_,
        graphics_queue_,
        command_pool_,
        static_cast<work::renderer::BufferUsageFlags>(work::renderer::BufferUsageFlagBits::VERTEX_BUFFER_BIT),
        buffer_size,
        vertices.data(),
        vertex_buffer_.buffer,
        vertex_buffer_.memory);
}

void RealWorldApplication::createIndexBuffer() {
    const std::vector<uint16_t> indices = {
        0, 1, 2, 2, 1, 3,
        4, 6, 5, 5, 6, 7,
        0, 4, 1, 1, 4, 5,
        2, 3, 6, 6, 3, 7,
        1, 5, 3, 3, 5, 7,
        0, 2, 4, 4, 2, 6};

    VkDeviceSize buffer_size = 
        sizeof(indices[0]) * indices.size();

    createBufferWithSrcData(
        device_,
        graphics_queue_,
        command_pool_,
        static_cast<work::renderer::BufferUsageFlags>(work::renderer::BufferUsageFlagBits::INDEX_BUFFER_BIT),
        buffer_size,
        indices.data(),
        index_buffer_.buffer,
        index_buffer_.memory);
}

void RealWorldApplication::createDescriptorSetLayout() {
    // global texture descriptor set layout.
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.reserve(5);

        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(GGX_LUT_INDEX));
        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(CHARLIE_LUT_INDEX));
        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(LAMBERTIAN_ENV_TEX_INDEX));
        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(GGX_ENV_TEX_INDEX));
        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(CHARLIE_ENV_TEX_INDEX));

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();

        global_tex_desc_set_layout_ = ::createDescriptorSetLayout(device_, layout_info);
    }

    // material texture descriptor set layout.
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.reserve(7);

        VkDescriptorSetLayoutBinding ubo_pbr_layout_binding{};
        ubo_pbr_layout_binding.binding = PBR_CONSTANT_INDEX;
        ubo_pbr_layout_binding.descriptorCount = 1;
        ubo_pbr_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ubo_pbr_layout_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        ubo_pbr_layout_binding.pImmutableSamplers = nullptr; // Optional
        bindings.push_back(ubo_pbr_layout_binding);

        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(BASE_COLOR_TEX_INDEX));
        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(NORMAL_TEX_INDEX));
        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(METAL_ROUGHNESS_TEX_INDEX));
        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(EMISSIVE_TEX_INDEX));
        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(OCCLUSION_TEX_INDEX));
        bindings.push_back(getTextureSamplerDescriptionSetLayoutBinding(THIN_FILM_LUT_INDEX));

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();

        material_tex_desc_set_layout_ = ::createDescriptorSetLayout(device_, layout_info);
    }

    {
        std::vector<VkDescriptorSetLayoutBinding> bindings(1);

        VkDescriptorSetLayoutBinding ubo_layout_binding{};
        ubo_layout_binding.binding = VIEW_CONSTANT_INDEX;
        ubo_layout_binding.descriptorCount = 1;
        ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ubo_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        ubo_layout_binding.pImmutableSamplers = nullptr; // Optional
        bindings[0] = ubo_layout_binding;

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();

        desc_set_layout_ = ::createDescriptorSetLayout(device_, layout_info);
    }

    {
        std::vector<VkDescriptorSetLayoutBinding> bindings(1);

        VkDescriptorSetLayoutBinding ubo_layout_binding{};
        bindings[0] = getTextureSamplerDescriptionSetLayoutBinding(BASE_COLOR_TEX_INDEX);

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();

        skybox_desc_set_layout_ = ::createDescriptorSetLayout(device_, layout_info);
    }

    // ibl texture descriptor set layout.
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings(1);
        bindings[0] = getTextureSamplerDescriptionSetLayoutBinding(PANORAMA_TEX_INDEX);
//        bindings[1] = getTextureSamplerDescriptionSetLayoutBinding(ENVMAP_TEX_INDEX);

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();

        ibl_desc_set_layout_ = ::createDescriptorSetLayout(device_, layout_info);
    }

    // ibl compute texture descriptor set layout.
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings(2);
        bindings[0] = getTextureSamplerDescriptionSetLayoutBinding(SRC_TEX_INDEX, VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        bindings[1] = getTextureSamplerDescriptionSetLayoutBinding(DST_TEX_INDEX, VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();

        ibl_comp_desc_set_layout_ = ::createDescriptorSetLayout(device_, layout_info);
    }
}

void RealWorldApplication::createUniformBuffers() {
    VkDeviceSize buffer_size = sizeof(ViewParams);

    view_const_buffers_.resize(swap_chain_images_.size());

    for (uint64_t i = 0; i < swap_chain_images_.size(); i++) {
        createBuffer(
            device_, 
            buffer_size, 
            static_cast<renderer::BufferUsageFlags>(renderer::BufferUsageFlagBits::UNIFORM_BUFFER_BIT), 
            static_cast<renderer::MemoryPropertyFlags>(renderer::MemoryPropertyFlagBits::HOST_VISIBLE_BIT) | 
            static_cast<renderer::MemoryPropertyFlags>(renderer::MemoryPropertyFlagBits::HOST_COHERENT_BIT), 
            view_const_buffers_[i].buffer,
            view_const_buffers_[i].memory);
    }
}

void RealWorldApplication::updateViewConstBuffer(uint32_t current_image, const glm::vec3& center, float radius) {
    auto eye_pos = glm::vec3(3.0f, -7.0f, 0.0f) * radius;

    ViewParams view_params{};
    view_params.camera_pos = glm::vec4(eye_pos + center, 0);
    view_params.view = glm::lookAt(eye_pos + center, center, glm::vec3(0.0f, 1.0f, 0.0f));
    view_params.proj = glm::perspective(glm::radians(45.0f), swap_chain_extent_.x / (float)swap_chain_extent_.y, 1.0f * radius, 10000.0f);
    view_params.proj[1][1] *= -1;
    view_params.input_features = glm::vec4(gltf_object_->meshes_[0].primitives_[0].has_tangent_ ? FEATURE_INPUT_HAS_TANGENT : 0, 0, 0, 0);

    updateBufferMemory(device_, view_const_buffers_[current_image].memory, sizeof(view_params), &view_params);
}

void RealWorldApplication::createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 3> pool_sizes{};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[0].descriptorCount = 16;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[1].descriptorCount = 64;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[2].descriptorCount = 16;
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());;
    pool_info.pPoolSizes = pool_sizes.data();
    pool_info.maxSets = 64;

    // todo.
    auto vk_device = std::reinterpret_pointer_cast<renderer::VulkanDevice>(device_);
    assert(vk_device);
    VkDescriptorPool descriptor_pool;
    if (vkCreateDescriptorPool(vk_device->get(), &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
    auto vk_descriptor_pool = std::make_shared<renderer::VulkanDescriptorPool>();
    vk_descriptor_pool->set(descriptor_pool);
    descriptor_pool_ = std::move(vk_descriptor_pool);
}

VkWriteDescriptorSet addDescriptWrite(
    const VkDescriptorSet& description_set,
    const VkDescriptorImageInfo& image_info,
    uint32_t binding,
    const VkDescriptorType& desc_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
    VkWriteDescriptorSet result = {};
    result.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    result.dstSet = description_set;
    result.dstBinding = binding;
    result.dstArrayElement = 0;
    result.descriptorType = desc_type;
    result.descriptorCount = 1;
    result.pImageInfo = &image_info;

    return result;
}

std::vector<VkWriteDescriptorSet> RealWorldApplication::addGlobalTextures(
    const std::shared_ptr<renderer::DescriptorSet>& description_set)
{
    auto vk_desc_set = std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(description_set);
    auto vk_sampler = std::reinterpret_pointer_cast<renderer::VulkanSampler>(texture_sampler_);
    std::vector<VkWriteDescriptorSet> descriptor_writes;
    descriptor_writes.reserve(10);

    // ggx_lut.
    static VkDescriptorImageInfo ggx_lut_image_info = {};
    auto vk_ggx_lut = std::reinterpret_pointer_cast<renderer::VulkanImageView>(ggx_lut_tex_.view);
    ggx_lut_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ggx_lut_image_info.imageView = vk_ggx_lut->get();
    ggx_lut_image_info.sampler = vk_sampler->get();
    descriptor_writes.push_back(addDescriptWrite(vk_desc_set->get(), ggx_lut_image_info, GGX_LUT_INDEX));

    // charlie_lut.
    static VkDescriptorImageInfo charlie_lut_image_info = {};
    auto vk_charlie_lut = std::reinterpret_pointer_cast<renderer::VulkanImageView>(charlie_lut_tex_.view);
    charlie_lut_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    charlie_lut_image_info.imageView = vk_charlie_lut->get();
    charlie_lut_image_info.sampler = vk_sampler->get();
    descriptor_writes.push_back(addDescriptWrite(vk_desc_set->get(), charlie_lut_image_info, CHARLIE_LUT_INDEX));

    // labertian tex.
    static VkDescriptorImageInfo ibl_diffuse_image_info = {};
    auto vk_diffuse_tex = std::reinterpret_pointer_cast<renderer::VulkanImageView>(rt_ibl_diffuse_tex_.view);
    ibl_diffuse_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ibl_diffuse_image_info.imageView = vk_diffuse_tex->get();
    ibl_diffuse_image_info.sampler = vk_sampler->get();
    descriptor_writes.push_back(addDescriptWrite(vk_desc_set->get(), ibl_diffuse_image_info, LAMBERTIAN_ENV_TEX_INDEX));

    // specular tex.
    static VkDescriptorImageInfo ibl_specular_image_info = {};
    auto vk_specular_tex = std::reinterpret_pointer_cast<renderer::VulkanImageView>(rt_ibl_specular_tex_.view);
    ibl_specular_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ibl_specular_image_info.imageView = vk_specular_tex->get();
    ibl_specular_image_info.sampler = vk_sampler->get();
    descriptor_writes.push_back(addDescriptWrite(vk_desc_set->get(), ibl_specular_image_info, GGX_ENV_TEX_INDEX));

    // sheen tex.
    static VkDescriptorImageInfo ibl_sheen_image_info = {};
    auto vk_sheen_tex = std::reinterpret_pointer_cast<renderer::VulkanImageView>(rt_ibl_sheen_tex_.view);
    ibl_sheen_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ibl_sheen_image_info.imageView = vk_sheen_tex->get();
    ibl_sheen_image_info.sampler = vk_sampler->get();
    descriptor_writes.push_back(addDescriptWrite(vk_desc_set->get(), ibl_specular_image_info, CHARLIE_ENV_TEX_INDEX));

    return descriptor_writes;
}

std::vector<VkWriteDescriptorSet> RealWorldApplication::addGltfTextures(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const renderer::MaterialInfo& material) {
    auto vk_desc_set = std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(description_set);
    auto vk_sampler = std::reinterpret_pointer_cast<renderer::VulkanSampler>(texture_sampler_);
    std::vector<VkWriteDescriptorSet> descriptor_writes;
    descriptor_writes.reserve(10);

    auto textures = gltf_object_->textures_;

    // base color.
    static VkDescriptorImageInfo base_color_image_info = {};
    auto base_color_tex_view = material.base_color_idx_ < 0 ? black_tex_ : textures[material.base_color_idx_];
    auto vk_base_color_tex = std::reinterpret_pointer_cast<renderer::VulkanImageView>(base_color_tex_view.view);
    base_color_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    base_color_image_info.imageView = vk_base_color_tex->get();
    base_color_image_info.sampler = vk_sampler->get();
    descriptor_writes.push_back(addDescriptWrite(vk_desc_set->get(), base_color_image_info, BASE_COLOR_TEX_INDEX));

    // normal.
    static VkDescriptorImageInfo normal_image_info = {};
    auto normal_tex_view = material.normal_idx_ < 0 ? black_tex_ : textures[material.normal_idx_];
    auto vk_normal_tex = std::reinterpret_pointer_cast<renderer::VulkanImageView>(normal_tex_view.view);
    normal_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    normal_image_info.imageView = vk_normal_tex->get();
    normal_image_info.sampler = vk_sampler->get();
    descriptor_writes.push_back(addDescriptWrite(vk_desc_set->get(), normal_image_info, NORMAL_TEX_INDEX));

    // metallic roughness.
    static VkDescriptorImageInfo metallic_roughness_image_info = {};
    auto metallic_roughness_tex = material.metallic_roughness_idx_ < 0 ? black_tex_ : textures[material.metallic_roughness_idx_];
    auto vk_metallic_roughness_tex_view = std::reinterpret_pointer_cast<renderer::VulkanImageView>(metallic_roughness_tex.view);
    metallic_roughness_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    metallic_roughness_image_info.imageView = vk_metallic_roughness_tex_view->get();
    metallic_roughness_image_info.sampler = vk_sampler->get();
    descriptor_writes.push_back(addDescriptWrite(vk_desc_set->get(), metallic_roughness_image_info, METAL_ROUGHNESS_TEX_INDEX));

    // emisive.
    static VkDescriptorImageInfo emissive_image_info = {};
    auto emissive_tex = material.emissive_idx_ < 0 ? black_tex_ : textures[material.emissive_idx_];
    auto vk_emissive_tex_view = std::reinterpret_pointer_cast<renderer::VulkanImageView>(emissive_tex.view);
    emissive_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    emissive_image_info.imageView = vk_emissive_tex_view->get();
    emissive_image_info.sampler = vk_sampler->get();
    descriptor_writes.push_back(addDescriptWrite(vk_desc_set->get(), emissive_image_info, EMISSIVE_TEX_INDEX));

    // occlusion.
    static VkDescriptorImageInfo occlusion_image_info = {};
    auto occlusion_tex = material.occlusion_idx_ < 0 ? white_tex_ : textures[material.occlusion_idx_];
    auto vk_occlusion_tex_view = std::reinterpret_pointer_cast<renderer::VulkanImageView>(occlusion_tex.view);
    occlusion_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    occlusion_image_info.imageView = vk_occlusion_tex_view->get();
    occlusion_image_info.sampler = vk_sampler->get();
    descriptor_writes.push_back(addDescriptWrite(vk_desc_set->get(), occlusion_image_info, OCCLUSION_TEX_INDEX));

    // thin_film_lut.
    static VkDescriptorImageInfo thin_film_lut_image_info = {};
    auto vk_thin_film_lut = std::reinterpret_pointer_cast<renderer::VulkanImageView>(thin_film_lut_tex_.view);
    thin_film_lut_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    thin_film_lut_image_info.imageView = vk_thin_film_lut->get();
    thin_film_lut_image_info.sampler = vk_sampler->get();
    descriptor_writes.push_back(addDescriptWrite(vk_desc_set->get(), thin_film_lut_image_info, THIN_FILM_LUT_INDEX));

    return descriptor_writes;
}

std::vector<VkWriteDescriptorSet> RealWorldApplication::addSkyboxTextures(
    const std::shared_ptr<renderer::DescriptorSet>& description_set)
{
    auto vk_desc_set = std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(description_set);
    auto vk_sampler = std::reinterpret_pointer_cast<renderer::VulkanSampler>(texture_sampler_);
    std::vector<VkWriteDescriptorSet> descriptor_writes;
    descriptor_writes.reserve(1);

    // envmap texture.
    static VkDescriptorImageInfo envmap_image_info = {};
    auto vk_envmap_tex = std::reinterpret_pointer_cast<renderer::VulkanImageView>(rt_envmap_tex_.view);
    envmap_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    envmap_image_info.imageView = vk_envmap_tex->get();
    envmap_image_info.sampler = vk_sampler->get();
    descriptor_writes.push_back(addDescriptWrite(vk_desc_set->get(), envmap_image_info, BASE_COLOR_TEX_INDEX));

    return descriptor_writes;
}

std::vector<VkWriteDescriptorSet> RealWorldApplication::addPanoramaTextures(
    const std::shared_ptr<renderer::DescriptorSet>& description_set)
{
    auto vk_desc_set = std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(description_set);
    auto vk_sampler = std::reinterpret_pointer_cast<renderer::VulkanSampler>(texture_sampler_);
    std::vector<VkWriteDescriptorSet> descriptor_writes;
    descriptor_writes.reserve(1);

    // envmap texture.
    static VkDescriptorImageInfo panorama_image_info = {};
    auto vk_panorama_tex = std::reinterpret_pointer_cast<renderer::VulkanImageView>(panorama_tex_.view);
    panorama_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    panorama_image_info.imageView = vk_panorama_tex->get();
    panorama_image_info.sampler = vk_sampler->get();

    descriptor_writes.push_back(addDescriptWrite(vk_desc_set->get(), panorama_image_info, PANORAMA_TEX_INDEX));

    return descriptor_writes;
}

std::vector<VkWriteDescriptorSet> RealWorldApplication::addIblTextures(
    const std::shared_ptr<renderer::DescriptorSet>& description_set)
{
    auto vk_desc_set = std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(description_set);
    auto vk_sampler = std::reinterpret_pointer_cast<renderer::VulkanSampler>(texture_sampler_);
    std::vector<VkWriteDescriptorSet> descriptor_writes;
    descriptor_writes.reserve(1);

    static VkDescriptorImageInfo rt_envmap_image_info = {};
    auto vk_rt_envmap_tex = std::reinterpret_pointer_cast<renderer::VulkanImageView>(rt_envmap_tex_.view);
    rt_envmap_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    rt_envmap_image_info.imageView = vk_rt_envmap_tex->get();
    rt_envmap_image_info.sampler = vk_sampler->get();
    descriptor_writes.push_back(addDescriptWrite(vk_desc_set->get(), rt_envmap_image_info, ENVMAP_TEX_INDEX));

    return descriptor_writes;
}

std::vector<VkWriteDescriptorSet> RealWorldApplication::addIblComputeTextures(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const renderer::TextureInfo& src_tex,
    const renderer::TextureInfo& dst_tex)
{
    auto vk_desc_set = std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(description_set);
    auto vk_sampler = std::reinterpret_pointer_cast<renderer::VulkanSampler>(texture_sampler_);
    std::vector<VkWriteDescriptorSet> descriptor_writes;
    descriptor_writes.reserve(2);

    static VkDescriptorImageInfo rt_src_image_info = {};
    auto vk_src_tex = std::reinterpret_pointer_cast<renderer::VulkanImageView>(src_tex.view);
    rt_src_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    rt_src_image_info.imageView = vk_src_tex->get();
    rt_src_image_info.sampler = vk_sampler->get();
    descriptor_writes.push_back(addDescriptWrite(vk_desc_set->get(), rt_src_image_info, SRC_TEX_INDEX, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE));

    static VkDescriptorImageInfo rt_dst_image_info = {};
    auto vk_dst_tex = std::reinterpret_pointer_cast<renderer::VulkanImageView>(dst_tex.view);
    rt_dst_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    rt_dst_image_info.imageView = vk_dst_tex->get();
    rt_dst_image_info.sampler = vk_sampler->get();
    descriptor_writes.push_back(addDescriptWrite(vk_desc_set->get(), rt_dst_image_info, DST_TEX_INDEX, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE));

    return descriptor_writes;
}

void RealWorldApplication::createDescriptorSets() {
    auto vk_device = std::reinterpret_pointer_cast<work::renderer::VulkanDevice>(device_);
    assert(vk_device);

    auto buffer_count = swap_chain_images_.size();

    {
        auto global_desc_sets = ::createDescriptorSets(device_, descriptor_pool_, global_tex_desc_set_layout_, 1);
        global_tex_desc_set_ = std::move(global_desc_sets[0]);

        // create a global ibl texture descriptor set.
        auto global_texture_descs_write = addGlobalTextures(global_tex_desc_set_);
        vkUpdateDescriptorSets(vk_device->get(),
            static_cast<uint32_t>(global_texture_descs_write.size()),
            global_texture_descs_write.data(),
            0,
            nullptr);
    }

    {
        for (uint32_t i_mat = 0; i_mat < gltf_object_->materials_.size(); i_mat++) {
            auto& material = gltf_object_->materials_[i_mat];
            auto material_desc_sets = ::createDescriptorSets(device_, descriptor_pool_, material_tex_desc_set_layout_, 1);
            material.desc_set_ = std::move(material_desc_sets[0]);

            auto vk_desc_set = std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(material.desc_set_);

            VkDescriptorBufferInfo buffer_info{};
            auto vk_buffer = std::reinterpret_pointer_cast<renderer::VulkanBuffer>(material.uniform_buffer_.buffer);
            buffer_info.buffer = vk_buffer->get();
            buffer_info.offset = 0;
            buffer_info.range = sizeof(PbrMaterialParams);

            VkWriteDescriptorSet descriptor_write = {};
            descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_write.dstSet = vk_desc_set->get();
            descriptor_write.dstBinding = PBR_CONSTANT_INDEX;
            descriptor_write.dstArrayElement = 0;
            descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptor_write.descriptorCount = 1;
            descriptor_write.pBufferInfo = &buffer_info;

            // create a global ibl texture descriptor set.
            auto material_descs_write = addGltfTextures(material.desc_set_, material);
            material_descs_write.push_back(descriptor_write);

            vkUpdateDescriptorSets(vk_device->get(),
                static_cast<uint32_t>(material_descs_write.size()),
                material_descs_write.data(),
                0,
                nullptr);
        }
    }

    {
        desc_sets_ = ::createDescriptorSets(device_, descriptor_pool_, desc_set_layout_, buffer_count);
        for (uint64_t i = 0; i < buffer_count; i++) {
            // todo.
            auto vk_desc_set = std::reinterpret_pointer_cast<renderer::VulkanDescriptorSet>(desc_sets_[i]);
            VkDescriptorBufferInfo buffer_info{};
            auto vk_buffer = std::reinterpret_pointer_cast<renderer::VulkanBuffer>(view_const_buffers_[i].buffer);
            buffer_info.buffer = vk_buffer->get();
            buffer_info.offset = 0;
            buffer_info.range = sizeof(ViewParams);

            VkWriteDescriptorSet descriptor_write = {};
            descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_write.dstSet = vk_desc_set->get();
            descriptor_write.dstBinding = VIEW_CONSTANT_INDEX;
            descriptor_write.dstArrayElement = 0;
            descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptor_write.descriptorCount = 1;
            descriptor_write.pBufferInfo = &buffer_info;

            std::vector<VkWriteDescriptorSet> descriptor_writes(1);
            descriptor_writes[0] = descriptor_write;

            vkUpdateDescriptorSets(vk_device->get(), static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, nullptr);
        }
    }

    // skybox
    {
        auto skybox_desc_sets = ::createDescriptorSets(device_, descriptor_pool_, skybox_desc_set_layout_, 1);
        skybox_tex_desc_set_ = std::move(skybox_desc_sets[0]);

        // create a global ibl texture descriptor set.
        auto skybox_texture_descs_write = addSkyboxTextures(skybox_tex_desc_set_);
        vkUpdateDescriptorSets(vk_device->get(),
            static_cast<uint32_t>(skybox_texture_descs_write.size()),
            skybox_texture_descs_write.data(),
            0,
            nullptr);
    }

    // envmap
    {
        // only one descriptor layout.
        envmap_tex_desc_set_ = std::move(::createDescriptorSets(device_, descriptor_pool_, ibl_desc_set_layout_, 1)[0]);

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs_write = addPanoramaTextures(envmap_tex_desc_set_);
        vkUpdateDescriptorSets(vk_device->get(),
            static_cast<uint32_t>(ibl_texture_descs_write.size()),
            ibl_texture_descs_write.data(),
            0,
            nullptr);
    }

    // ibl
    {
        // only one descriptor layout.
        ibl_tex_desc_set_ = std::move(::createDescriptorSets(device_, descriptor_pool_, ibl_desc_set_layout_, 1)[0]);

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs_write = addIblTextures(ibl_tex_desc_set_);
        vkUpdateDescriptorSets(vk_device->get(),
            static_cast<uint32_t>(ibl_texture_descs_write.size()),
            ibl_texture_descs_write.data(),
            0,
            nullptr);
    }

    // ibl diffuse compute
    {
        // only one descriptor layout.
        ibl_diffuse_tex_desc_set_ = std::move(::createDescriptorSets(device_, descriptor_pool_, ibl_comp_desc_set_layout_, 1)[0]);

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs_write = addIblComputeTextures(ibl_diffuse_tex_desc_set_, tmp_ibl_diffuse_tex_, rt_ibl_diffuse_tex_);
        vkUpdateDescriptorSets(vk_device->get(),
            static_cast<uint32_t>(ibl_texture_descs_write.size()),
            ibl_texture_descs_write.data(),
            0,
            nullptr);
    }

    // ibl specular compute
    {
        // only one descriptor layout.
        ibl_specular_tex_desc_set_ = std::move(::createDescriptorSets(device_, descriptor_pool_, ibl_comp_desc_set_layout_, 1)[0]);

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs_write = addIblComputeTextures(ibl_specular_tex_desc_set_, tmp_ibl_specular_tex_, rt_ibl_specular_tex_);
        vkUpdateDescriptorSets(vk_device->get(),
            static_cast<uint32_t>(ibl_texture_descs_write.size()),
            ibl_texture_descs_write.data(),
            0,
            nullptr);
    }

    // ibl sheen compute
    {
        // only one descriptor layout.
        ibl_sheen_tex_desc_set_ = std::move(::createDescriptorSets(device_, descriptor_pool_, ibl_comp_desc_set_layout_, 1)[0]);

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs_write = addIblComputeTextures(ibl_sheen_tex_desc_set_, tmp_ibl_sheen_tex_, rt_ibl_sheen_tex_);
        vkUpdateDescriptorSets(vk_device->get(),
            static_cast<uint32_t>(ibl_texture_descs_write.size()),
            ibl_texture_descs_write.data(),
            0,
            nullptr);
    }
}

void RealWorldApplication::createTextureImage(const std::string& file_name, renderer::Format format, renderer::TextureInfo& texture) {
    int tex_width, tex_height, tex_channels;
    stbi_uc* pixels = stbi_load(file_name.c_str(), &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);

    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }
    create2DTextureImage(device_, graphics_queue_, command_pool_, format, tex_width, tex_height, tex_channels, pixels, texture.image, texture.memory);

    stbi_image_free(pixels);

    texture.view = device_->createImageView(
        texture.image,
        renderer::ImageViewType::VIEW_2D,
        format,
        static_cast<renderer::ImageAspectFlags>(renderer::ImageAspectFlagBits::COLOR_BIT));
}

void RealWorldApplication::createTextureSampler() {
    texture_sampler_ = device_->createSampler(renderer::Filter::LINEAR, renderer::SamplerAddressMode::REPEAT, renderer::SamplerMipmapMode::LINEAR, 16.0f);
}

void RealWorldApplication::createDepthResources() {
    auto depth_format = findDepthFormat(vk_physical_device_);
    create2DImage(
        device_,
        swap_chain_extent_,
        depth_format,
        renderer::ImageTiling::OPTIMAL,
        static_cast<renderer::ImageUsageFlags>(renderer::ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT_BIT),
        static_cast<renderer::MemoryPropertyFlags>(renderer::MemoryPropertyFlagBits::DEVICE_LOCAL_BIT),
        depth_buffer_.image,
        depth_buffer_.memory);

    depth_buffer_.view = 
        device_->createImageView(
            depth_buffer_.image,
            renderer::ImageViewType::VIEW_2D, 
            depth_format, 
            static_cast<renderer::ImageAspectFlags>(renderer::ImageAspectFlagBits::DEPTH_BIT));

    transitionImageLayout(device_, 
                          graphics_queue_, 
                          command_pool_, 
                          depth_buffer_.image, 
                          depth_format, 
                          renderer::ImageLayout::UNDEFINED, 
                          renderer::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
}

void RealWorldApplication::cleanupSwapChain() {
    assert(device_);
    depth_buffer_.destroy(device_);

    for (auto framebuffer : swap_chain_framebuffers_) {
        device_->destroyFramebuffer(framebuffer);
    }

    device_->freeCommandBuffers(command_pool_, command_buffers_);
    device_->destroyPipeline(gltf_pipeline_);
    device_->destroyPipeline(skybox_pipeline_);
    device_->destroyPipeline(envmap_pipeline_);
    device_->destroyPipeline(lambertian_pipeline_);
    device_->destroyPipeline(ggx_pipeline_);
    device_->destroyPipeline(charlie_pipeline_);
    device_->destroyPipeline(blur_comp_pipeline_);
    device_->destroyPipelineLayout(gltf_pipeline_layout_);
    device_->destroyPipelineLayout(skybox_pipeline_layout_);
    device_->destroyPipelineLayout(ibl_pipeline_layout_);
    device_->destroyPipelineLayout(ibl_comp_pipeline_layout_);
    device_->destroyRenderPass(render_pass_);

    for (auto image_view : swap_chain_image_views_) {
        device_->destroyImageView(image_view);
    }

    device_->destroySwapchain(swap_chain_);

    for (auto& buffer : view_const_buffers_) {
        buffer.destroy(device_);
    }

    device_->destroyDescriptorPool(descriptor_pool_);
}

void RealWorldApplication::cleanup() {
    cleanupSwapChain();

    device_->destroyRenderPass(cubemap_render_pass_);

    gltf_object_->destroy(device_);

    assert(device_);
    device_->destroySampler(texture_sampler_);
    sample_tex_.destroy(device_);
    black_tex_.destroy(device_);
    white_tex_.destroy(device_);
    ggx_lut_tex_.destroy(device_);
    brdf_lut_tex_.destroy(device_);
    charlie_lut_tex_.destroy(device_);
    thin_film_lut_tex_.destroy(device_);
    panorama_tex_.destroy(device_);
    ibl_diffuse_tex_.destroy(device_);
    ibl_specular_tex_.destroy(device_);
    ibl_sheen_tex_.destroy(device_);
    rt_envmap_tex_.destroy(device_);
    tmp_ibl_diffuse_tex_.destroy(device_);
    tmp_ibl_specular_tex_.destroy(device_);
    tmp_ibl_sheen_tex_.destroy(device_);
    rt_ibl_diffuse_tex_.destroy(device_);
    rt_ibl_specular_tex_.destroy(device_);
    rt_ibl_sheen_tex_.destroy(device_);
    device_->destroyDescriptorSetLayout(desc_set_layout_);
    device_->destroyDescriptorSetLayout(global_tex_desc_set_layout_);
    device_->destroyDescriptorSetLayout(material_tex_desc_set_layout_);
    device_->destroyDescriptorSetLayout(skybox_desc_set_layout_);
    device_->destroyDescriptorSetLayout(ibl_desc_set_layout_);
    device_->destroyDescriptorSetLayout(ibl_comp_desc_set_layout_);

    vertex_buffer_.destroy(device_);
    index_buffer_.destroy(device_);

    for (uint64_t i = 0; i < kMaxFramesInFlight; i++) {
        device_->destroySemaphore(render_finished_semaphores_[i]);
        device_->destroySemaphore(image_available_semaphores_[i]);
        device_->destroyFence(in_flight_fences_[i]);
    }

    device_->destroyCommandPool(command_pool_);

    // todo.
    auto vk_device = std::reinterpret_pointer_cast<renderer::VulkanDevice>(device_);
    vkDestroyDevice(vk_device->get(), nullptr);

    if (enable_validation_layers) {
        DestroyDebugUtilsMessengerEXT(vk_instance_, debug_messenger_, nullptr);
    }

    // todo.
    auto vk_surface = std::reinterpret_pointer_cast<renderer::VulkanSurface>(surface_);
    vkDestroySurfaceKHR(vk_instance_, vk_surface->get(), nullptr);
    vkDestroyInstance(vk_instance_, nullptr);
    glfwDestroyWindow(window_);
    glfwTerminate();
}

static void setupMeshState(
    const tinygltf::Model& model,
    std::shared_ptr<renderer::Device> device,
    std::shared_ptr<renderer::Queue> cmd_queue,
    std::shared_ptr<renderer::CommandPool> cmd_pool,
    std::shared_ptr<renderer::ObjectData>& gltf_object) {

    // Buffer
    {
        gltf_object->buffers_.resize(model.buffers.size());
        for (size_t i = 0; i < model.buffers.size(); i++) {
            auto buffer = model.buffers[i];
            createBufferWithSrcData(
                device,
                cmd_queue,
                cmd_pool,
                static_cast<renderer::BufferUsageFlags>(renderer::BufferUsageFlagBits::VERTEX_BUFFER_BIT) |
                static_cast<renderer::BufferUsageFlags>(renderer::BufferUsageFlagBits::INDEX_BUFFER_BIT),
                buffer.data.size(),
                buffer.data.data(),
                gltf_object->buffers_[i].buffer,
                gltf_object->buffers_[i].memory);
        }
    }

    // Buffer views.
    {
        auto& buffer_views = gltf_object->buffer_views_;
        buffer_views.resize(model.bufferViews.size());

        for (size_t i = 0; i < model.bufferViews.size(); i++) {
            const tinygltf::BufferView& bufferView = model.bufferViews[i];
            buffer_views[i].buffer_idx = bufferView.buffer;
            buffer_views[i].offset = bufferView.byteOffset;
            buffer_views[i].range = bufferView.byteLength;
            buffer_views[i].stride = bufferView.byteStride;
        }
    }

    // allocate texture memory at first.
    gltf_object->textures_.resize(model.textures.size());

    // Material
    {
        gltf_object->materials_.resize(model.materials.size());
        for (size_t i_mat = 0; i_mat < model.materials.size(); i_mat++) {
            auto& dst_material = gltf_object->materials_[i_mat];
            const auto& src_material = model.materials[i_mat];

            dst_material.base_color_idx_ = src_material.pbrMetallicRoughness.baseColorTexture.index;
            dst_material.normal_idx_ = src_material.normalTexture.index;
            dst_material.metallic_roughness_idx_ = src_material.pbrMetallicRoughness.metallicRoughnessTexture.index;
            dst_material.emissive_idx_ = src_material.emissiveTexture.index;
            dst_material.occlusion_idx_ = src_material.occlusionTexture.index;

            if (dst_material.base_color_idx_ >= 0) {
                gltf_object->textures_[dst_material.base_color_idx_].linear = false;
            }

            if (dst_material.emissive_idx_ >= 0) {
                gltf_object->textures_[dst_material.emissive_idx_].linear = false;
            }

            createBuffer(
                device,
                sizeof(PbrMaterialParams),
                static_cast<renderer::BufferUsageFlags>(renderer::BufferUsageFlagBits::UNIFORM_BUFFER_BIT),
                static_cast<renderer::MemoryPropertyFlags>(renderer::MemoryPropertyFlagBits::HOST_VISIBLE_BIT) |
                static_cast<renderer::MemoryPropertyFlags>(renderer::MemoryPropertyFlagBits::HOST_COHERENT_BIT),
                dst_material.uniform_buffer_.buffer,
                dst_material.uniform_buffer_.memory);

            PbrMaterialParams ubo{};
            ubo.base_color_factor = glm::vec4(
                src_material.pbrMetallicRoughness.baseColorFactor[0],
                src_material.pbrMetallicRoughness.baseColorFactor[1],
                src_material.pbrMetallicRoughness.baseColorFactor[2],
                src_material.pbrMetallicRoughness.baseColorFactor[3]);

            ubo.glossiness_factor = 1.0f;
            ubo.metallic_roughness_specular_factor = 1.0f;
            ubo.metallic_factor = static_cast<float>(src_material.pbrMetallicRoughness.metallicFactor);
            ubo.roughness_factor = static_cast<float>(src_material.pbrMetallicRoughness.roughnessFactor);
            ubo.alpha_cutoff = static_cast<float>(src_material.alphaCutoff);
            ubo.mip_count = 11;
            ubo.normal_scale = static_cast<float>(src_material.normalTexture.scale);
            ubo.occlusion_strength = static_cast<float>(src_material.occlusionTexture.strength);

            ubo.emissive_factor = glm::vec3(
                src_material.emissiveFactor[0],
                src_material.emissiveFactor[1],
                src_material.emissiveFactor[2]);

            ubo.uv_set_flags = glm::vec4(0, 0, 0, 0);
            ubo.exposure = 1.0f;
            ubo.material_features = (src_material.pbrMetallicRoughness.metallicRoughnessTexture.index >= 0 ? FEATURE_HAS_METALLIC_ROUGHNESS_MAP : 0) | FEATURE_MATERIAL_METALLICROUGHNESS;
            ubo.material_features |= (src_material.pbrMetallicRoughness.baseColorTexture.index >= 0 ? FEATURE_HAS_BASE_COLOR_MAP : 0);
            ubo.material_features |= (src_material.emissiveTexture.index >= 0 ? FEATURE_HAS_EMISSIVE_MAP : 0);
            ubo.material_features |= (src_material.occlusionTexture.index >= 0 ? FEATURE_HAS_OCCLUSION_MAP : 0);
            ubo.material_features |= (src_material.normalTexture.index >= 0 ? FEATURE_HAS_NORMAL_MAP : 0);
            ubo.tonemap_type = TONEMAP_DEFAULT;
            ubo.specular_factor = vec3(1.0f, 1.0f, 1.0f);
            ubo.lights[0].type = LightType_Directional;
            ubo.lights[0].color = glm::vec3(1, 0, 0);
            ubo.lights[0].direction = glm::vec3(0, 0, -1);
            ubo.lights[0].intensity = 1.0f;
            ubo.lights[0].position = glm::vec3(0, 0, 0);

            updateBufferMemory(device, dst_material.uniform_buffer_.memory, sizeof(ubo), &ubo);
        }
    }

    // Texture
    {
        for (size_t i_tex = 0; i_tex < model.textures.size(); i_tex++) {
            auto& dst_tex = gltf_object->textures_[i_tex];
            const auto& src_tex = model.textures[i_tex];
            const auto& src_img = model.images[i_tex];
            auto format = renderer::Format::R8G8B8A8_UNORM;
            create2DTextureImage(device,
                cmd_queue,
                cmd_pool,
                format,
                src_img.width,
                src_img.height,
                src_img.component,
                src_img.image.data(),
                dst_tex.image,
                dst_tex.memory);

            dst_tex.view = device->createImageView(
                dst_tex.image,
                renderer::ImageViewType::VIEW_2D,
                format,
                static_cast<renderer::ImageAspectFlags>(renderer::ImageAspectFlagBits::COLOR_BIT));
        }
    }
}

static void setupMesh(
    const tinygltf::Model& model,
    const tinygltf::Mesh& mesh,
    renderer::MeshInfo& mesh_info) {

    for (size_t i = 0; i < mesh.primitives.size(); i++) {
        const tinygltf::Primitive& primitive = mesh.primitives[i];

        renderer::PrimitiveInfo primitive_info;
        primitive_info.topology_info_.restart_enable = false;
        primitive_info.material_idx_ = primitive.material;

        auto& mode = primitive_info.topology_info_.topology;
        mode = renderer::PrimitiveTopology::MAX_ENUM;
        if (primitive.mode == TINYGLTF_MODE_TRIANGLES) {
            mode = renderer::PrimitiveTopology::TRIANGLE_LIST;
        }
        else if (primitive.mode == TINYGLTF_MODE_TRIANGLE_STRIP) {
            mode = renderer::PrimitiveTopology::TRIANGLE_STRIP;
        }
        else if (primitive.mode == TINYGLTF_MODE_TRIANGLE_FAN) {
            mode = renderer::PrimitiveTopology::TRIANGLE_FAN;
        }
        else if (primitive.mode == TINYGLTF_MODE_POINTS) {
            mode = renderer::PrimitiveTopology::POINT_LIST;
        }
        else if (primitive.mode == TINYGLTF_MODE_LINE) {
            mode = renderer::PrimitiveTopology::LINE_LIST;
        }
        else if (primitive.mode == TINYGLTF_MODE_LINE_LOOP) {
            mode = renderer::PrimitiveTopology::LINE_STRIP;
        }
        else {
            assert(0);
        }

        if (primitive.indices < 0) return;

        std::map<std::string, int>::const_iterator it(primitive.attributes.begin());
        std::map<std::string, int>::const_iterator itEnd(primitive.attributes.end());

        uint32_t dst_binding = 0;
        for (; it != itEnd; it++) {
            assert(it->second >= 0);
            const tinygltf::Accessor& accessor = model.accessors[it->second];

            dst_binding = static_cast<uint32_t>(primitive_info.binding_list_.size());
            primitive_info.binding_list_.push_back(accessor.bufferView);

            work::renderer::VertexInputBindingDescription binding = {};
            binding.binding = dst_binding;
            binding.stride = accessor.ByteStride(model.bufferViews[accessor.bufferView]);
            binding.input_rate = renderer::VertexInputRate::VERTEX;
            primitive_info.binding_descs_.push_back(binding);

            work::renderer::VertexInputAttributeDescription attribute = {};
            attribute.buffer_view = accessor.bufferView;
            attribute.binding = dst_binding;
            attribute.offset = 0;
            attribute.buffer_offset = accessor.byteOffset + model.bufferViews[accessor.bufferView].byteOffset;
            if (it->first.compare("POSITION") == 0) {
                attribute.location = VINPUT_POSITION;
                primitive_info.bbox_min_ = glm::vec3(accessor.minValues[0], accessor.minValues[1], accessor.minValues[2]);
                primitive_info.bbox_max_ = glm::vec3(accessor.maxValues[0], accessor.maxValues[1], accessor.maxValues[2]);
                mesh_info.bbox_min_ = min(mesh_info.bbox_min_, primitive_info.bbox_min_);
                mesh_info.bbox_max_ = max(mesh_info.bbox_max_, primitive_info.bbox_max_);
            }
            else if (it->first.compare("TEXCOORD_0") == 0) {
                attribute.location = VINPUT_TEXCOORD0;
                primitive_info.has_texcoord_0_ = true;
            }
            else if (it->first.compare("NORMAL") == 0) {
                attribute.location = VINPUT_NORMAL;
                primitive_info.has_normal_ = true;
            }
            else if (it->first.compare("TANGENT") == 0) {
                attribute.location = VINPUT_TANGENT;
                primitive_info.has_tangent_ = true;
            }
            else if (it->first.compare("TEXCOORD_1") == 0) {
                attribute.location = VINPUT_TEXCOORD1;
            }
            else if (it->first.compare("COLOR") == 0) {
                attribute.location = VINPUT_COLOR;
            }
            else if (it->first.compare("JOINTS_0") == 0) {
                attribute.location = VINPUT_JOINTS_0;
                primitive_info.has_skin_set_0_ = true;
            }
            else if (it->first.compare("WEIGHTS_0") == 0) {
                attribute.location = VINPUT_WEIGHTS_0;
                primitive_info.has_skin_set_0_ = true;
            }
            else {
                // add support here.
                assert(0);
            }

            if (accessor.componentType == TINYGLTF_PARAMETER_TYPE_FLOAT) {
                if (accessor.type == TINYGLTF_TYPE_SCALAR) {
                    attribute.format = work::renderer::Format::R32_SFLOAT;
                }
                else if (accessor.type == TINYGLTF_TYPE_VEC2) {
                    attribute.format = work::renderer::Format::R32G32_SFLOAT;
                }
                else if (accessor.type == TINYGLTF_TYPE_VEC3) {
                    attribute.format = work::renderer::Format::R32G32B32_SFLOAT;
                }
                else if (accessor.type == TINYGLTF_TYPE_VEC4) {
                    attribute.format = work::renderer::Format::R32G32B32A32_SFLOAT;
                }
                else {
                    assert(0);
                }
            }
            else if (accessor.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT) {
                if (accessor.type == TINYGLTF_TYPE_SCALAR) {
                    attribute.format = work::renderer::Format::R16_UINT;
                }
                else if (accessor.type == TINYGLTF_TYPE_VEC2) {
                    attribute.format = work::renderer::Format::R16G16_UINT;
                }
                else if (accessor.type == TINYGLTF_TYPE_VEC3) {
                    attribute.format = work::renderer::Format::R16G16B16_UINT;
                }
                else if (accessor.type == TINYGLTF_TYPE_VEC4) {
                    attribute.format = work::renderer::Format::R16G16B16A16_UINT;
                }
                else {
                    assert(0);
                }

            }
            else {
                // add support here.
                assert(0);
            }
            primitive_info.attribute_descs_.push_back(attribute);
            dst_binding++;
        }

        const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
        primitive_info.index_desc_.binding = indexAccessor.bufferView;
        primitive_info.index_desc_.offset = indexAccessor.byteOffset;
        primitive_info.index_desc_.index_type = indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ? renderer::IndexType::UINT16 : renderer::IndexType::UINT32;
        primitive_info.index_desc_.index_count = indexAccessor.count;

        mesh_info.primitives_.push_back(primitive_info);
    }
}

static void setupMeshes(const tinygltf::Model& model, std::shared_ptr<renderer::ObjectData>& gltf_object) {
    gltf_object->meshes_.resize(model.meshes.size());
    for (int i_mesh = 0; i_mesh < model.meshes.size(); i_mesh++) {
        setupMesh(model, model.meshes[i_mesh], gltf_object->meshes_[i_mesh]);
    }
}

static void setupNode(
    const tinygltf::Model& model, 
    const tinygltf::Node& node,
    renderer::NodeInfo& node_info) {

    bool has_matrix = false;
    glm::mat4 mesh_matrix(1);
    if (node.matrix.size() == 16) {
        // Use 'matrix' attribute
        const auto& m = node.matrix.data();
        mesh_matrix = glm::mat4(m[0], m[1], m[2], m[3], 
                                 m[4], m[5], m[6], m[7],
                                 m[8], m[9], m[10], m[11],
                                 m[12], m[13], m[14], m[15]);
        has_matrix = true;
    }
    else {
        // Assume Trans x Rotate x Scale order
        if (node.scale.size() == 3) {
            mesh_matrix = glm::scale(mesh_matrix, glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
            has_matrix = true;
        }

        if (node.rotation.size() == 4) {
            mesh_matrix = glm::rotate(mesh_matrix, glm::radians(static_cast<float>(node.rotation[0])), glm::vec3(node.rotation[1], node.rotation[2], node.rotation[3]));
            has_matrix = true;
        }

        if (node.translation.size() == 3) {
            mesh_matrix = glm::translate(mesh_matrix, glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
            has_matrix = true;
        }
    }

    if (has_matrix) {
        node_info.matrix = std::make_shared<glm::mat4>(mesh_matrix);
    }

    node_info.mesh_idx = node.mesh;

    // Draw child nodes.
    node_info.child_idx.resize(node.children.size());
    for (size_t i = 0; i < node.children.size(); i++) {
        assert(node.children[i] < model.nodes.size());
        node_info.child_idx[i] = node.children[i];
    }
}

static void setupNodes(const tinygltf::Model& model, std::shared_ptr<renderer::ObjectData>& gltf_object) {
    gltf_object->nodes_.resize(model.nodes.size());
    for (int i_node = 0; i_node < model.nodes.size(); i_node++) {
        setupNode(model, model.nodes[i_node], gltf_object->nodes_[i_node]);
    }
}

static void setupModel(
    const tinygltf::Model& model,
    std::shared_ptr<renderer::ObjectData>& gltf_object) {
    assert(model.scenes.size() > 0);
    gltf_object->default_scene_ = model.defaultScene;
    gltf_object->scenes_.resize(model.scenes.size());
    for (uint32_t i_scene = 0; i_scene < model.scenes.size(); i_scene++) {
        const auto& src_scene = model.scenes[i_scene];
        auto& dst_scene = gltf_object->scenes_[i_scene];

        dst_scene.nodes_.resize(src_scene.nodes.size());
        for (size_t i_node = 0; i_node < src_scene.nodes.size(); i_node++) {
            dst_scene.nodes_[i_node] = src_scene.nodes[i_node];
        }
    }
}

static void transformBbox(
    const glm::mat4& mat, 
    const glm::vec3& bbox_min, 
    const glm::vec3& bbox_max,
    glm::vec3& output_bbox_min,
    glm::vec3& output_bbox_max) {

    glm::vec3 extent = bbox_max - bbox_min;
    glm::vec3 base = glm::vec3(mat * glm::vec4(bbox_min, 1.0f));
    output_bbox_min = base;
    output_bbox_max = base;
    auto mat_1 = glm::mat3(mat);
    glm::vec3 vec_x = mat_1 * glm::vec3(extent.x, 0, 0);
    glm::vec3 vec_y = mat_1 * glm::vec3(0, extent.y, 0);
    glm::vec3 vec_z = mat_1 * glm::vec3(0, 0, extent.z);

    glm::vec3 points[7];
    points[0] = base + vec_x;
    points[1] = base + vec_y;
    points[2] = base + vec_z;
    points[3] = points[0] + vec_y;
    points[4] = points[0] + vec_z;
    points[5] = points[1] + vec_z;
    points[6] = points[3] + vec_z;

    for (int i = 0; i < 7; i++) {
        output_bbox_min = min(output_bbox_min, points[i]);
        output_bbox_max = max(output_bbox_max, points[i]);
    }
}

static void calculateBbox(
    std::shared_ptr<renderer::ObjectData>& gltf_object, 
    int32_t node_idx, 
    const glm::mat4& parent_matrix,
    glm::vec3& output_bbox_min,
    glm::vec3& output_bbox_max) {
    if (node_idx >= 0) {
        const auto& node = gltf_object->nodes_[node_idx];
        auto cur_matrix = parent_matrix;
        if (node.matrix) {
            cur_matrix *= *node.matrix;
        }
        if (node.mesh_idx >= 0) {
            glm::vec3 bbox_min, bbox_max;
            transformBbox(
                cur_matrix,
                gltf_object->meshes_[node.mesh_idx].bbox_min_,
                gltf_object->meshes_[node.mesh_idx].bbox_max_,
                bbox_min,
                bbox_max);
            output_bbox_min = min(output_bbox_min, bbox_min);
            output_bbox_max = max(output_bbox_max, bbox_max);
        }

        for (auto& child_idx : node.child_idx) {
            calculateBbox(gltf_object, child_idx, cur_matrix, output_bbox_min, output_bbox_max);
        }
    }
}

void RealWorldApplication::loadGltfModel(const std::string& input_filename)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    std::string ext = getFilePathExtension(input_filename);

    bool ret = false;
    if (ext.compare("glb") == 0) {
        // assume binary glTF.
        ret =
            loader.LoadBinaryFromFile(&model, &err, &warn, input_filename.c_str());
    }
    else {
        // assume ascii glTF.
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, input_filename.c_str());
    }

    if (!warn.empty()) {
        std::cout << "Warn: " << warn.c_str() << std::endl;
    }

    if (!err.empty()) {
        std::cout << "ERR: " << err.c_str() << std::endl;
    }
    if (!ret) {
        std::cout << "Failed to load .glTF : " << input_filename << std::endl;
        return;
    }

    gltf_object_ = std::make_shared<renderer::ObjectData>();
    gltf_object_->meshes_.reserve(model.meshes.size());

    setupMeshState(model, device_, graphics_queue_, command_pool_, gltf_object_);
    setupMeshes(model, gltf_object_);
    setupNodes(model, gltf_object_);
    setupModel(model, gltf_object_);
    for (auto& root : gltf_object_->scenes_) {
        for (auto& node : root.nodes_) {
            calculateBbox(gltf_object_, root.nodes_[0], glm::mat4(1.0f), root.bbox_min_, root.bbox_max_);
        }
    }
}

void RealWorldApplication::createCubemapTexture(
    uint32_t width, 
    uint32_t height, 
    uint32_t mip_count, 
    renderer::Format format, 
    const std::vector<work::renderer::BufferImageCopyInfo>& copy_regions, 
    renderer::TextureInfo& texture,
    uint64_t buffer_size /*= 0*/,
    void* data /*= nullptr*/)
{
    bool use_as_framebuffer = data == nullptr;
    VkDeviceSize image_size = static_cast<VkDeviceSize>(buffer_size);

    std::shared_ptr<renderer::Buffer> staging_buffer;
    std::shared_ptr<renderer::DeviceMemory> staging_buffer_memory;
    if (data) {
        createBuffer(
            device_,
            image_size,
            static_cast<renderer::BufferUsageFlags>(renderer::BufferUsageFlagBits::TRANSFER_SRC_BIT),
            static_cast<renderer::MemoryPropertyFlags>(renderer::MemoryPropertyFlagBits::HOST_VISIBLE_BIT) |
            static_cast<renderer::MemoryPropertyFlags>(renderer::MemoryPropertyFlagBits::HOST_COHERENT_BIT),
            staging_buffer,
            staging_buffer_memory);

        updateBufferMemory(device_, staging_buffer_memory, buffer_size, data);
    }

    auto image_usage_flags =
        static_cast<renderer::ImageUsageFlags>(renderer::ImageUsageFlagBits::TRANSFER_DST_BIT) |
        static_cast<renderer::ImageUsageFlags>(renderer::ImageUsageFlagBits::SAMPLED_BIT) |
        static_cast<renderer::ImageUsageFlags>(renderer::ImageUsageFlagBits::STORAGE_BIT);

    if (use_as_framebuffer) {
        image_usage_flags |=
            static_cast<renderer::ImageUsageFlags>(renderer::ImageUsageFlagBits::COLOR_ATTACHMENT_BIT) |
            static_cast<renderer::ImageUsageFlags>(renderer::ImageUsageFlagBits::TRANSFER_SRC_BIT);
    }

    texture.image = device_->createImage(
        renderer::ImageType::TYPE_2D,
        glm::uvec3(width, height, 1),
        format,
        image_usage_flags,
        renderer::ImageTiling::OPTIMAL,
        renderer::ImageLayout::UNDEFINED,
        static_cast<uint32_t>(renderer::ImageCreateFlagBits::CUBE_COMPATIBLE_BIT),
        false,
        1,
        mip_count,
        6u);

    auto mem_requirements = device_->getImageMemoryRequirements(texture.image);
    texture.memory = device_->allocateMemory(
        mem_requirements.size,
        mem_requirements.memory_type_bits,
        toVkMemoryPropertyFlags(static_cast<uint32_t>(renderer::MemoryPropertyFlagBits::DEVICE_LOCAL_BIT)));
    device_->bindImageMemory(texture.image, texture.memory);

    if (data) {
        transitionImageLayout(device_, graphics_queue_, command_pool_, texture.image, format, work::renderer::ImageLayout::UNDEFINED, work::renderer::ImageLayout::TRANSFER_DST_OPTIMAL, 0, mip_count, 0, 6);
        copyBufferToImageWithMips(device_, graphics_queue_, command_pool_, staging_buffer, texture.image, copy_regions);
        transitionImageLayout(device_, graphics_queue_, command_pool_, texture.image, format, work::renderer::ImageLayout::TRANSFER_DST_OPTIMAL, work::renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL, 0, mip_count, 0, 6);
    }

    texture.view = device_->createImageView(
        texture.image,
        renderer::ImageViewType::VIEW_CUBE,
        format,
        static_cast<renderer::ImageAspectFlags>(renderer::ImageAspectFlagBits::COLOR_BIT),
        0,
        mip_count,
        0,
        6);

    assert(cubemap_render_pass_);

    if (use_as_framebuffer) {
        texture.surface_views.resize(mip_count);
        texture.framebuffers.resize(mip_count);
        auto w = width;
        auto h = height;

        for (uint32_t i = 0; i < mip_count; ++i)
        {
            texture.surface_views[i].resize(6, VK_NULL_HANDLE); //sides of the cube

            for (uint32_t j = 0; j < 6; j++)
            {
                texture.surface_views[i][j] =
                    device_->createImageView(
                        texture.image,
                        renderer::ImageViewType::VIEW_2D,
                        format,
                        static_cast<renderer::ImageAspectFlags>(renderer::ImageAspectFlagBits::COLOR_BIT),
                        i,
                        1,
                        j,
                        1);
            }

            texture.framebuffers[i] = device_->createFrameBuffer(cubemap_render_pass_, texture.surface_views[i], glm::uvec2(w, h));
            w = std::max(w >> 1, 1u);
            h = std::max(h >> 1, 1u);
        }
    }

    if (data) {
        device_->destroyBuffer(staging_buffer);
        device_->freeMemory(staging_buffer_memory);
    }
}

void RealWorldApplication::loadMtx2Texture(const std::string& input_filename, renderer::TextureInfo& texture) {
    uint64_t buffer_size;
    auto mtx2_data = readFile(input_filename, buffer_size);
    auto src_data = (char*)mtx2_data.data();

    // header block
    Mtx2HeaderBlock* header_block = reinterpret_cast<Mtx2HeaderBlock*>(src_data);
    src_data += sizeof(Mtx2HeaderBlock);

    assert(header_block->format == renderer::Format::R16G16B16A16_SFLOAT);

    // index block
    Mtx2IndexBlock* index_block = reinterpret_cast<Mtx2IndexBlock*>(src_data);
    src_data += sizeof(Mtx2IndexBlock);

    uint32_t width = header_block->pixel_width;
    uint32_t height = header_block->pixel_height;
    // level index block.
    uint32_t num_level_blocks = std::max(1u, header_block->level_count);
    std::vector<work::renderer::BufferImageCopyInfo> copy_regions(num_level_blocks);
    for (uint32_t i_level = 0; i_level < num_level_blocks; i_level++) {
        Mtx2LevelIndexBlock* level_block = reinterpret_cast<Mtx2LevelIndexBlock*>(src_data);

        auto& region = copy_regions[i_level];
        region.buffer_offset = level_block->byte_offset;
        region.buffer_row_length = 0;
        region.buffer_image_height = 0;

        region.image_subresource.aspect_mask = static_cast<work::renderer::ImageAspectFlags>(work::renderer::ImageAspectFlagBits::COLOR_BIT);
        region.image_subresource.mip_level = i_level;
        region.image_subresource.base_array_layer = 0;
        region.image_subresource.layer_count = 6;

        region.image_offset = glm::ivec3(0, 0, 0);
        region.image_extent = glm::uvec3(width, height, 1);
        width = std::max(1u, width / 2);
        height = std::max(1u, height / 2);

        src_data += sizeof(Mtx2LevelIndexBlock);
    }
    
    char* dfd_data_start = (char*)mtx2_data.data() + index_block->dfd_byte_offset;
    uint32_t dfd_total_size = *reinterpret_cast<uint32_t*>(dfd_data_start);
    src_data += sizeof(uint32_t);

    char* kvd_data_start = (char*)mtx2_data.data() + index_block->kvd_byte_offset;
    uint32_t key_value_byte_length = *reinterpret_cast<uint32_t*>(kvd_data_start);
    uint8_t* key_value = reinterpret_cast<uint8_t*>(kvd_data_start + 4);
    for (uint32_t i = 0; i < key_value_byte_length; i++) {
        auto result = key_value[i];
        int hit = 1;
    }

    char* sgd_data_start = nullptr;
    if (index_block->sgd_byte_length > 0) {
        sgd_data_start = (char*)mtx2_data.data() + index_block->sgd_byte_offset;
    }

    createCubemapTexture(
        header_block->pixel_width,
        header_block->pixel_height,
        num_level_blocks,
        header_block->format,
        copy_regions,
        texture,
        buffer_size,
        mtx2_data.data());
}


}//namespace app
}//namespace work
