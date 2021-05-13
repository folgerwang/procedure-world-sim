#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include <optional>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define RENDER_TYPE_CAST(class, name) std::reinterpret_pointer_cast<renderer::Vulkan##class>(name)
#define ADD_FLAG_BIT(type, type1, name) result |= (flags & static_cast<uint32_t>(renderer::type##FlagBits::name)) ? VK_##type1##_##name : 0
#define SELECT_FLAG(type, type1, name) if (flag == renderer::type::name) return VK_##type1##_##name
#define SELECT_FROM_FLAG(type, type1, name) if (flag == VK_##type1##_##name) return renderer::type::name
#define SET_FLAG_BIT(type, name) static_cast<renderer::type##Flags>(renderer::type##FlagBits::name)

namespace work {
namespace renderer {

enum class Format {
    UNDEFINED = 0,
    R4G4_UNORM_PACK8 = 1,
    R4G4B4A4_UNORM_PACK16 = 2,
    B4G4R4A4_UNORM_PACK16 = 3,
    R5G6B5_UNORM_PACK16 = 4,
    B5G6R5_UNORM_PACK16 = 5,
    R5G5B5A1_UNORM_PACK16 = 6,
    B5G5R5A1_UNORM_PACK16 = 7,
    A1R5G5B5_UNORM_PACK16 = 8,
    R8_UNORM = 9,
    R8_SNORM = 10,
    R8_USCALED = 11,
    R8_SSCALED = 12,
    R8_UINT = 13,
    R8_SINT = 14,
    R8_SRGB = 15,
    R8G8_UNORM = 16,
    R8G8_SNORM = 17,
    R8G8_USCALED = 18,
    R8G8_SSCALED = 19,
    R8G8_UINT = 20,
    R8G8_SINT = 21,
    R8G8_SRGB = 22,
    R8G8B8_UNORM = 23,
    R8G8B8_SNORM = 24,
    R8G8B8_USCALED = 25,
    R8G8B8_SSCALED = 26,
    R8G8B8_UINT = 27,
    R8G8B8_SINT = 28,
    R8G8B8_SRGB = 29,
    B8G8R8_UNORM = 30,
    B8G8R8_SNORM = 31,
    B8G8R8_USCALED = 32,
    B8G8R8_SSCALED = 33,
    B8G8R8_UINT = 34,
    B8G8R8_SINT = 35,
    B8G8R8_SRGB = 36,
    R8G8B8A8_UNORM = 37,
    R8G8B8A8_SNORM = 38,
    R8G8B8A8_USCALED = 39,
    R8G8B8A8_SSCALED = 40,
    R8G8B8A8_UINT = 41,
    R8G8B8A8_SINT = 42,
    R8G8B8A8_SRGB = 43,
    B8G8R8A8_UNORM = 44,
    B8G8R8A8_SNORM = 45,
    B8G8R8A8_USCALED = 46,
    B8G8R8A8_SSCALED = 47,
    B8G8R8A8_UINT = 48,
    B8G8R8A8_SINT = 49,
    B8G8R8A8_SRGB = 50,
    A8B8G8R8_UNORM_PACK32 = 51,
    A8B8G8R8_SNORM_PACK32 = 52,
    A8B8G8R8_USCALED_PACK32 = 53,
    A8B8G8R8_SSCALED_PACK32 = 54,
    A8B8G8R8_UINT_PACK32 = 55,
    A8B8G8R8_SINT_PACK32 = 56,
    A8B8G8R8_SRGB_PACK32 = 57,
    A2R10G10B10_UNORM_PACK32 = 58,
    A2R10G10B10_SNORM_PACK32 = 59,
    A2R10G10B10_USCALED_PACK32 = 60,
    A2R10G10B10_SSCALED_PACK32 = 61,
    A2R10G10B10_UINT_PACK32 = 62,
    A2R10G10B10_SINT_PACK32 = 63,
    A2B10G10R10_UNORM_PACK32 = 64,
    A2B10G10R10_SNORM_PACK32 = 65,
    A2B10G10R10_USCALED_PACK32 = 66,
    A2B10G10R10_SSCALED_PACK32 = 67,
    A2B10G10R10_UINT_PACK32 = 68,
    A2B10G10R10_SINT_PACK32 = 69,
    R16_UNORM = 70,
    R16_SNORM = 71,
    R16_USCALED = 72,
    R16_SSCALED = 73,
    R16_UINT = 74,
    R16_SINT = 75,
    R16_SFLOAT = 76,
    R16G16_UNORM = 77,
    R16G16_SNORM = 78,
    R16G16_USCALED = 79,
    R16G16_SSCALED = 80,
    R16G16_UINT = 81,
    R16G16_SINT = 82,
    R16G16_SFLOAT = 83,
    R16G16B16_UNORM = 84,
    R16G16B16_SNORM = 85,
    R16G16B16_USCALED = 86,
    R16G16B16_SSCALED = 87,
    R16G16B16_UINT = 88,
    R16G16B16_SINT = 89,
    R16G16B16_SFLOAT = 90,
    R16G16B16A16_UNORM = 91,
    R16G16B16A16_SNORM = 92,
    R16G16B16A16_USCALED = 93,
    R16G16B16A16_SSCALED = 94,
    R16G16B16A16_UINT = 95,
    R16G16B16A16_SINT = 96,
    R16G16B16A16_SFLOAT = 97,
    R32_UINT = 98,
    R32_SINT = 99,
    R32_SFLOAT = 100,
    R32G32_UINT = 101,
    R32G32_SINT = 102,
    R32G32_SFLOAT = 103,
    R32G32B32_UINT = 104,
    R32G32B32_SINT = 105,
    R32G32B32_SFLOAT = 106,
    R32G32B32A32_UINT = 107,
    R32G32B32A32_SINT = 108,
    R32G32B32A32_SFLOAT = 109,
    R64_UINT = 110,
    R64_SINT = 111,
    R64_SFLOAT = 112,
    R64G64_UINT = 113,
    R64G64_SINT = 114,
    R64G64_SFLOAT = 115,
    R64G64B64_UINT = 116,
    R64G64B64_SINT = 117,
    R64G64B64_SFLOAT = 118,
    R64G64B64A64_UINT = 119,
    R64G64B64A64_SINT = 120,
    R64G64B64A64_SFLOAT = 121,
    B10G11R11_UFLOAT_PACK32 = 122,
    E5B9G9R9_UFLOAT_PACK32 = 123,
    D16_UNORM = 124,
    X8_D24_UNORM_PACK32 = 125,
    D32_SFLOAT = 126,
    S8_UINT = 127,
    D16_UNORM_S8_UINT = 128,
    D24_UNORM_S8_UINT = 129,
    D32_SFLOAT_S8_UINT = 130,
    BC1_RGB_UNORM_BLOCK = 131,
    BC1_RGB_SRGB_BLOCK = 132,
    BC1_RGBA_UNORM_BLOCK = 133,
    BC1_RGBA_SRGB_BLOCK = 134,
    BC2_UNORM_BLOCK = 135,
    BC2_SRGB_BLOCK = 136,
    BC3_UNORM_BLOCK = 137,
    BC3_SRGB_BLOCK = 138,
    BC4_UNORM_BLOCK = 139,
    BC4_SNORM_BLOCK = 140,
    BC5_UNORM_BLOCK = 141,
    BC5_SNORM_BLOCK = 142,
    BC6H_UFLOAT_BLOCK = 143,
    BC6H_SFLOAT_BLOCK = 144,
    BC7_UNORM_BLOCK = 145,
    BC7_SRGB_BLOCK = 146,
    ETC2_R8G8B8_UNORM_BLOCK = 147,
    ETC2_R8G8B8_SRGB_BLOCK = 148,
    ETC2_R8G8B8A1_UNORM_BLOCK = 149,
    ETC2_R8G8B8A1_SRGB_BLOCK = 150,
    ETC2_R8G8B8A8_UNORM_BLOCK = 151,
    ETC2_R8G8B8A8_SRGB_BLOCK = 152,
    EAC_R11_UNORM_BLOCK = 153,
    EAC_R11_SNORM_BLOCK = 154,
    EAC_R11G11_UNORM_BLOCK = 155,
    EAC_R11G11_SNORM_BLOCK = 156,
    ASTC_4x4_UNORM_BLOCK = 157,
    ASTC_4x4_SRGB_BLOCK = 158,
    ASTC_5x4_UNORM_BLOCK = 159,
    ASTC_5x4_SRGB_BLOCK = 160,
    ASTC_5x5_UNORM_BLOCK = 161,
    ASTC_5x5_SRGB_BLOCK = 162,
    ASTC_6x5_UNORM_BLOCK = 163,
    ASTC_6x5_SRGB_BLOCK = 164,
    ASTC_6x6_UNORM_BLOCK = 165,
    ASTC_6x6_SRGB_BLOCK = 166,
    ASTC_8x5_UNORM_BLOCK = 167,
    ASTC_8x5_SRGB_BLOCK = 168,
    ASTC_8x6_UNORM_BLOCK = 169,
    ASTC_8x6_SRGB_BLOCK = 170,
    ASTC_8x8_UNORM_BLOCK = 171,
    ASTC_8x8_SRGB_BLOCK = 172,
    ASTC_10x5_UNORM_BLOCK = 173,
    ASTC_10x5_SRGB_BLOCK = 174,
    ASTC_10x6_UNORM_BLOCK = 175,
    ASTC_10x6_SRGB_BLOCK = 176,
    ASTC_10x8_UNORM_BLOCK = 177,
    ASTC_10x8_SRGB_BLOCK = 178,
    ASTC_10x10_UNORM_BLOCK = 179,
    ASTC_10x10_SRGB_BLOCK = 180,
    ASTC_12x10_UNORM_BLOCK = 181,
    ASTC_12x10_SRGB_BLOCK = 182,
    ASTC_12x12_UNORM_BLOCK = 183,
    ASTC_12x12_SRGB_BLOCK = 184,
    G8B8G8R8_422_UNORM = 1000156000,
    B8G8R8G8_422_UNORM = 1000156001,
    G8_B8_R8_3PLANE_420_UNORM = 1000156002,
    G8_B8R8_2PLANE_420_UNORM = 1000156003,
    G8_B8_R8_3PLANE_422_UNORM = 1000156004,
    G8_B8R8_2PLANE_422_UNORM = 1000156005,
    G8_B8_R8_3PLANE_444_UNORM = 1000156006,
    R10X6_UNORM_PACK16 = 1000156007,
    R10X6G10X6_UNORM_2PACK16 = 1000156008,
    R10X6G10X6B10X6A10X6_UNORM_4PACK16 = 1000156009,
    G10X6B10X6G10X6R10X6_422_UNORM_4PACK16 = 1000156010,
    B10X6G10X6R10X6G10X6_422_UNORM_4PACK16 = 1000156011,
    G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16 = 1000156012,
    G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 = 1000156013,
    G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16 = 1000156014,
    G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16 = 1000156015,
    G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16 = 1000156016,
    R12X4_UNORM_PACK16 = 1000156017,
    R12X4G12X4_UNORM_2PACK16 = 1000156018,
    R12X4G12X4B12X4A12X4_UNORM_4PACK16 = 1000156019,
    G12X4B12X4G12X4R12X4_422_UNORM_4PACK16 = 1000156020,
    B12X4G12X4R12X4G12X4_422_UNORM_4PACK16 = 1000156021,
    G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16 = 1000156022,
    G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16 = 1000156023,
    G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16 = 1000156024,
    G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16 = 1000156025,
    G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16 = 1000156026,
    G16B16G16R16_422_UNORM = 1000156027,
    B16G16R16G16_422_UNORM = 1000156028,
    G16_B16_R16_3PLANE_420_UNORM = 1000156029,
    G16_B16R16_2PLANE_420_UNORM = 1000156030,
    G16_B16_R16_3PLANE_422_UNORM = 1000156031,
    G16_B16R16_2PLANE_422_UNORM = 1000156032,
    G16_B16_R16_3PLANE_444_UNORM = 1000156033,
    PVRTC1_2BPP_UNORM_BLOCK_IMG = 1000054000,
    PVRTC1_4BPP_UNORM_BLOCK_IMG = 1000054001,
    PVRTC2_2BPP_UNORM_BLOCK_IMG = 1000054002,
    PVRTC2_4BPP_UNORM_BLOCK_IMG = 1000054003,
    PVRTC1_2BPP_SRGB_BLOCK_IMG = 1000054004,
    PVRTC1_4BPP_SRGB_BLOCK_IMG = 1000054005,
    PVRTC2_2BPP_SRGB_BLOCK_IMG = 1000054006,
    PVRTC2_4BPP_SRGB_BLOCK_IMG = 1000054007,
    ASTC_4x4_SFLOAT_BLOCK_EXT = 1000066000,
    ASTC_5x4_SFLOAT_BLOCK_EXT = 1000066001,
    ASTC_5x5_SFLOAT_BLOCK_EXT = 1000066002,
    ASTC_6x5_SFLOAT_BLOCK_EXT = 1000066003,
    ASTC_6x6_SFLOAT_BLOCK_EXT = 1000066004,
    ASTC_8x5_SFLOAT_BLOCK_EXT = 1000066005,
    ASTC_8x6_SFLOAT_BLOCK_EXT = 1000066006,
    ASTC_8x8_SFLOAT_BLOCK_EXT = 1000066007,
    ASTC_10x5_SFLOAT_BLOCK_EXT = 1000066008,
    ASTC_10x6_SFLOAT_BLOCK_EXT = 1000066009,
    ASTC_10x8_SFLOAT_BLOCK_EXT = 1000066010,
    ASTC_10x10_SFLOAT_BLOCK_EXT = 1000066011,
    ASTC_12x10_SFLOAT_BLOCK_EXT = 1000066012,
    ASTC_12x12_SFLOAT_BLOCK_EXT = 1000066013,
    G8B8G8R8_422_UNORM_KHR = G8B8G8R8_422_UNORM,
    B8G8R8G8_422_UNORM_KHR = B8G8R8G8_422_UNORM,
    G8_B8_R8_3PLANE_420_UNORM_KHR = G8_B8_R8_3PLANE_420_UNORM,
    G8_B8R8_2PLANE_420_UNORM_KHR = G8_B8R8_2PLANE_420_UNORM,
    G8_B8_R8_3PLANE_422_UNORM_KHR = G8_B8_R8_3PLANE_422_UNORM,
    G8_B8R8_2PLANE_422_UNORM_KHR = G8_B8R8_2PLANE_422_UNORM,
    G8_B8_R8_3PLANE_444_UNORM_KHR = G8_B8_R8_3PLANE_444_UNORM,
    R10X6_UNORM_PACK16_KHR = R10X6_UNORM_PACK16,
    R10X6G10X6_UNORM_2PACK16_KHR = R10X6G10X6_UNORM_2PACK16,
    R10X6G10X6B10X6A10X6_UNORM_4PACK16_KHR = R10X6G10X6B10X6A10X6_UNORM_4PACK16,
    G10X6B10X6G10X6R10X6_422_UNORM_4PACK16_KHR = G10X6B10X6G10X6R10X6_422_UNORM_4PACK16,
    B10X6G10X6R10X6G10X6_422_UNORM_4PACK16_KHR = B10X6G10X6R10X6G10X6_422_UNORM_4PACK16,
    G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16_KHR = G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16,
    G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR = G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
    G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16_KHR = G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16,
    G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16_KHR = G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16,
    G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16_KHR = G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16,
    R12X4_UNORM_PACK16_KHR = R12X4_UNORM_PACK16,
    R12X4G12X4_UNORM_2PACK16_KHR = R12X4G12X4_UNORM_2PACK16,
    R12X4G12X4B12X4A12X4_UNORM_4PACK16_KHR = R12X4G12X4B12X4A12X4_UNORM_4PACK16,
    G12X4B12X4G12X4R12X4_422_UNORM_4PACK16_KHR = G12X4B12X4G12X4R12X4_422_UNORM_4PACK16,
    B12X4G12X4R12X4G12X4_422_UNORM_4PACK16_KHR = B12X4G12X4R12X4G12X4_422_UNORM_4PACK16,
    G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16_KHR = G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16,
    G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR = G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16,
    G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16_KHR = G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16,
    G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16_KHR = G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16,
    G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16_KHR = G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16,
    G16B16G16R16_422_UNORM_KHR = G16B16G16R16_422_UNORM,
    B16G16R16G16_422_UNORM_KHR = B16G16R16G16_422_UNORM,
    G16_B16_R16_3PLANE_420_UNORM_KHR = G16_B16_R16_3PLANE_420_UNORM,
    G16_B16R16_2PLANE_420_UNORM_KHR = G16_B16R16_2PLANE_420_UNORM,
    G16_B16_R16_3PLANE_422_UNORM_KHR = G16_B16_R16_3PLANE_422_UNORM,
    G16_B16R16_2PLANE_422_UNORM_KHR = G16_B16R16_2PLANE_422_UNORM,
    G16_B16_R16_3PLANE_444_UNORM_KHR = G16_B16_R16_3PLANE_444_UNORM,
    MAX_ENUM = 0x7FFFFFFF
};

enum class ShaderStageFlagBits {
    VERTEX_BIT = 0x00000001,
    TESSELLATION_CONTROL_BIT = 0x00000002,
    TESSELLATION_EVALUATION_BIT = 0x00000004,
    GEOMETRY_BIT = 0x00000008,
    FRAGMENT_BIT = 0x00000010,
    COMPUTE_BIT = 0x00000020,
    ALL_GRAPHICS = 0x0000001F,
    ALL = 0x7FFFFFFF,
    RAYGEN_BIT_KHR = 0x00000100,
    ANY_HIT_BIT_KHR = 0x00000200,
    CLOSEST_HIT_BIT_KHR = 0x00000400,
    MISS_BIT_KHR = 0x00000800,
    INTERSECTION_BIT_KHR = 0x00001000,
    CALLABLE_BIT_KHR = 0x00002000,
    TASK_BIT_NV = 0x00000040,
    MESH_BIT_NV = 0x00000080,
    FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
};
typedef uint32_t ShaderStageFlags;

enum class BufferUsageFlagBits {
    TRANSFER_SRC_BIT = 0x00000001,
    TRANSFER_DST_BIT = 0x00000002,
    UNIFORM_TEXEL_BUFFER_BIT = 0x00000004,
    STORAGE_TEXEL_BUFFER_BIT = 0x00000008,
    UNIFORM_BUFFER_BIT = 0x00000010,
    STORAGE_BUFFER_BIT = 0x00000020,
    INDEX_BUFFER_BIT = 0x00000040,
    VERTEX_BUFFER_BIT = 0x00000080,
    INDIRECT_BUFFER_BIT = 0x00000100,
    SHADER_DEVICE_ADDRESS_BIT = 0x00020000,
    TRANSFORM_FEEDBACK_BUFFER_BIT_EXT = 0x00000800,
    TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT = 0x00001000,
    CONDITIONAL_RENDERING_BIT_EXT = 0x00000200,
    RAY_TRACING_BIT_KHR = 0x00000400,
    RAY_TRACING_BIT_NV = RAY_TRACING_BIT_KHR,
    SHADER_DEVICE_ADDRESS_BIT_EXT = SHADER_DEVICE_ADDRESS_BIT,
    SHADER_DEVICE_ADDRESS_BIT_KHR = SHADER_DEVICE_ADDRESS_BIT
};
typedef uint32_t BufferUsageFlags;

enum class MemoryPropertyFlagBits {
    DEVICE_LOCAL_BIT = 0x00000001,
    HOST_VISIBLE_BIT = 0x00000002,
    HOST_COHERENT_BIT = 0x00000004,
    HOST_CACHED_BIT = 0x00000008,
    LAZILY_ALLOCATED_BIT = 0x00000010,
    PROTECTED_BIT = 0x00000020,
    DEVICE_COHERENT_BIT_AMD = 0x00000040,
    DEVICE_UNCACHED_BIT_AMD = 0x00000080,
    FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
};
typedef uint32_t MemoryPropertyFlags;

enum class CommandBufferUsageFlagBits {
    ONE_TIME_SUBMIT_BIT = 0x00000001,
    RENDER_PASS_CONTINUE_BIT = 0x00000002,
    SIMULTANEOUS_USE_BIT = 0x00000004,
    FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
};
typedef uint32_t CommandBufferUsageFlags;

enum class ImageType {
    TYPE_1D = 0,
    TYPE_2D = 1,
    TYPE_3D = 2,
    MAX_ENUM = 0x7FFFFFFF
};

enum class ImageViewType {
    VIEW_1D = 0,
    VIEW_2D = 1,
    VIEW_3D = 2,
    VIEW_CUBE = 3,
    VIEW_1D_ARRAY = 4,
    VIEW_2D_ARRAY = 5,
    VIEW_CUBE_ARRAY = 6,
    VIEW_MAX_ENUM = 0x7FFFFFFF
};

enum class ImageUsageFlagBits {
    TRANSFER_SRC_BIT = 0x00000001,
    TRANSFER_DST_BIT = 0x00000002,
    SAMPLED_BIT = 0x00000004,
    STORAGE_BIT = 0x00000008,
    COLOR_ATTACHMENT_BIT = 0x00000010,
    DEPTH_STENCIL_ATTACHMENT_BIT = 0x00000020,
    TRANSIENT_ATTACHMENT_BIT = 0x00000040,
    INPUT_ATTACHMENT_BIT = 0x00000080,
    SHADING_RATE_IMAGE_BIT_NV = 0x00000100,
    FRAGMENT_DENSITY_MAP_BIT_EXT = 0x00000200,
    FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
};
typedef uint32_t ImageUsageFlags;

enum class ImageTiling {
    OPTIMAL = 0,
    LINEAR = 1,
    DRM_FORMAT_MODIFIER_EXT = 1000158000,
    MAX_ENUM = 0x7FFFFFFF
};

enum class ImageLayout {
    UNDEFINED = 0,
    GENERAL = 1,
    COLOR_ATTACHMENT_OPTIMAL = 2,
    DEPTH_STENCIL_ATTACHMENT_OPTIMAL = 3,
    DEPTH_STENCIL_READ_ONLY_OPTIMAL = 4,
    SHADER_READ_ONLY_OPTIMAL = 5,
    TRANSFER_SRC_OPTIMAL = 6,
    TRANSFER_DST_OPTIMAL = 7,
    PREINITIALIZED = 8,
    DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL = 1000117000,
    DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL = 1000117001,
    DEPTH_ATTACHMENT_OPTIMAL = 1000241000,
    DEPTH_READ_ONLY_OPTIMAL = 1000241001,
    STENCIL_ATTACHMENT_OPTIMAL = 1000241002,
    STENCIL_READ_ONLY_OPTIMAL = 1000241003,
    PRESENT_SRC_KHR = 1000001002,
    SHARED_PRESENT_KHR = 1000111000,
    SHADING_RATE_OPTIMAL_NV = 1000164003,
    FRAGMENT_DENSITY_MAP_OPTIMAL_EXT = 1000218000,
    MAX_ENUM = 0x7FFFFFFF
};

enum class ImageCreateFlagBits {
    SPARSE_BINDING_BIT = 0x00000001,
    SPARSE_RESIDENCY_BIT = 0x00000002,
    SPARSE_ALIASED_BIT = 0x00000004,
    MUTABLE_FORMAT_BIT = 0x00000008,
    CUBE_COMPATIBLE_BIT = 0x00000010,
    ALIAS_BIT = 0x00000400,
    SPLIT_INSTANCE_BIND_REGIONS_BIT = 0x00000040,
    TWO_D_ARRAY_COMPATIBLE_BIT = 0x00000020,
    BLOCK_TEXEL_VIEW_COMPATIBLE_BIT = 0x00000080,
    EXTENDED_USAGE_BIT = 0x00000100,
    PROTECTED_BIT = 0x00000800,
    DISJOINT_BIT = 0x00000200,
    CORNER_SAMPLED_BIT_NV = 0x00002000,
    SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT = 0x00001000,
    SUBSAMPLED_BIT_EXT = 0x00004000,
    FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
};
typedef uint32_t ImageCreateFlags;

enum class ImageAspectFlagBits {
    COLOR_BIT = 0x00000001,
    DEPTH_BIT = 0x00000002,
    STENCIL_BIT = 0x00000004,
    METADATA_BIT = 0x00000008,
    PLANE_0_BIT = 0x00000010,
    PLANE_1_BIT = 0x00000020,
    PLANE_2_BIT = 0x00000040,
    MEMORY_PLANE_0_BIT_EXT = 0x00000080,
    MEMORY_PLANE_1_BIT_EXT = 0x00000100,
    MEMORY_PLANE_2_BIT_EXT = 0x00000200,
    MEMORY_PLANE_3_BIT_EXT = 0x00000400,
    MAX_ENUM = 0x7FFFFFFF
};
typedef uint32_t ImageAspectFlags;

enum class CommandPoolCreateFlagBits {
    TRANSIENT_BIT = 0x00000001,
    RESET_COMMAND_BUFFER_BIT = 0x00000002,
    PROTECTED_BIT = 0x00000004,
    FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
};
typedef uint32_t CommandPoolCreateFlags;

enum class PipelineBindPoint {
    GRAPHICS = 0,
    COMPUTE = 1,
    RAY_TRACING = 1000165000,
    MAX_ENUM = 0x7FFFFFFF
};

enum class Filter {
    NEAREST = 0,
    LINEAR = 1,
    CUBIC_IMG = 1000015000,
    MAX_ENUM = 0x7FFFFFFF
};

enum class DescriptorType {
    SAMPLER = 0,
    COMBINED_IMAGE_SAMPLER = 1,
    SAMPLED_IMAGE = 2,
    STORAGE_IMAGE = 3,
    UNIFORM_TEXEL_BUFFER = 4,
    STORAGE_TEXEL_BUFFER = 5,
    UNIFORM_BUFFER = 6,
    STORAGE_BUFFER = 7,
    UNIFORM_BUFFER_DYNAMIC = 8,
    STORAGE_BUFFER_DYNAMIC = 9,
    INPUT_ATTACHMENT = 10,
    INLINE_UNIFORM_BLOCK_EXT = 1000138000,
    ACCELERATION_STRUCTURE_KHR = 1000165000,
    ACCELERATION_STRUCTURE_NV = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
    MAX_ENUM = 0x7FFFFFFF
};

enum class SamplerAddressMode {
    REPEAT = 0,
    MIRRORED_REPEAT = 1,
    CLAMP_TO_EDGE = 2,
    CLAMP_TO_BORDER = 3,
    MIRROR_CLAMP_TO_EDGE = 4,
    MAX_ENUM = 0x7FFFFFFF
};

enum class SamplerMipmapMode {
    NEAREST = 0,
    LINEAR = 1,
    MAX_ENUM = 0x7FFFFFFF
};

enum class ColorSpace {
    SRGB_NONLINEAR_KHR = 0,
    DISPLAY_P3_NONLINEAR_EXT = 1000104001,
    EXTENDED_SRGB_LINEAR_EXT = 1000104002,
    DISPLAY_P3_LINEAR_EXT = 1000104003,
    DCI_P3_NONLINEAR_EXT = 1000104004,
    BT709_LINEAR_EXT = 1000104005,
    BT709_NONLINEAR_EXT = 1000104006,
    BT2020_LINEAR_EXT = 1000104007,
    HDR10_ST2084_EXT = 1000104008,
    DOLBYVISION_EXT = 1000104009,
    HDR10_HLG_EXT = 1000104010,
    ADOBERGB_LINEAR_EXT = 1000104011,
    ADOBERGB_NONLINEAR_EXT = 1000104012,
    PASS_THROUGH_EXT = 1000104013,
    EXTENDED_SRGB_NONLINEAR_EXT = 1000104014,
    DISPLAY_NATIVE_AMD = 1000213000,
    MAX_ENUM_KHR = 0x7FFFFFFF
};

enum class PresentMode {
    IMMEDIATE_KHR = 0,
    MAILBOX_KHR = 1,
    FIFO_KHR = 2,
    FIFO_RELAXED_KHR = 3,
    SHARED_DEMAND_REFRESH_KHR = 1000111000,
    SHARED_CONTINUOUS_REFRESH_KHR = 1000111001,
    MAX_ENUM_KHR = 0x7FFFFFFF
};

enum class SurfaceTransformFlagBits {
    IDENTITY_BIT_KHR = 0x00000001,
    ROTATE_90_BIT_KHR = 0x00000002,
    ROTATE_180_BIT_KHR = 0x00000004,
    ROTATE_270_BIT_KHR = 0x00000008,
    HORIZONTAL_MIRROR_BIT_KHR = 0x00000010,
    HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR = 0x00000020,
    HORIZONTAL_MIRROR_ROTATE_180_BIT_KHR = 0x00000040,
    HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR = 0x00000080,
    INHERIT_BIT_KHR = 0x00000100,
    FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
};
typedef uint32_t SurfaceTransformFlags;

enum class IndexType {
    UINT16 = 0,
    UINT32 = 1,
    NONE_KHR = 1000165000,
    UINT8_EXT = 1000265000,
    MAX_ENUM = 0x7FFFFFFF
};

enum class VertexInputRate {
    VERTEX = 0,
    INSTANCE = 1,
    VK_VERTEX_INPUT_RATE_MAX_ENUM = 0x7FFFFFFF
};

enum class PrimitiveTopology {
    POINT_LIST = 0,
    LINE_LIST = 1,
    LINE_STRIP = 2,
    TRIANGLE_LIST = 3,
    TRIANGLE_STRIP = 4,
    TRIANGLE_FAN = 5,
    LINE_LIST_WITH_ADJACENCY = 6,
    LINE_STRIP_WITH_ADJACENCY = 7,
    TRIANGLE_LIST_WITH_ADJACENCY = 8,
    TRIANGLE_STRIP_WITH_ADJACENCY = 9,
    PATCH_LIST = 10,
    MAX_ENUM = 0x7FFFFFFF
};

enum class AccessFlagBits {
    INDIRECT_COMMAND_READ_BIT = 0x00000001,
    INDEX_READ_BIT = 0x00000002,
    VERTEX_ATTRIBUTE_READ_BIT = 0x00000004,
    UNIFORM_READ_BIT = 0x00000008,
    INPUT_ATTACHMENT_READ_BIT = 0x00000010,
    SHADER_READ_BIT = 0x00000020,
    SHADER_WRITE_BIT = 0x00000040,
    COLOR_ATTACHMENT_READ_BIT = 0x00000080,
    COLOR_ATTACHMENT_WRITE_BIT = 0x00000100,
    DEPTH_STENCIL_ATTACHMENT_READ_BIT = 0x00000200,
    DEPTH_STENCIL_ATTACHMENT_WRITE_BIT = 0x00000400,
    TRANSFER_READ_BIT = 0x00000800,
    TRANSFER_WRITE_BIT = 0x00001000,
    HOST_READ_BIT = 0x00002000,
    HOST_WRITE_BIT = 0x00004000,
    MEMORY_READ_BIT = 0x00008000,
    MEMORY_WRITE_BIT = 0x00010000,
    TRANSFORM_FEEDBACK_WRITE_BIT_EXT = 0x02000000,
    TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT = 0x04000000,
    TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT = 0x08000000,
    CONDITIONAL_RENDERING_READ_BIT_EXT = 0x00100000,
    COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT = 0x00080000,
    ACCELERATION_STRUCTURE_READ_BIT_KHR = 0x00200000,
    ACCELERATION_STRUCTURE_WRITE_BIT_KHR = 0x00400000,
    SHADING_RATE_IMAGE_READ_BIT_NV = 0x00800000,
    FRAGMENT_DENSITY_MAP_READ_BIT_EXT = 0x01000000,
    COMMAND_PREPROCESS_READ_BIT_NV = 0x00020000,
    COMMAND_PREPROCESS_WRITE_BIT_NV = 0x00040000,
    FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
};
typedef uint32_t AccessFlags;

enum class PipelineStageFlagBits {
    TOP_OF_PIPE_BIT = 0x00000001,
    DRAW_INDIRECT_BIT = 0x00000002,
    VERTEX_INPUT_BIT = 0x00000004,
    VERTEX_SHADER_BIT = 0x00000008,
    TESSELLATION_CONTROL_SHADER_BIT = 0x00000010,
    TESSELLATION_EVALUATION_SHADER_BIT = 0x00000020,
    GEOMETRY_SHADER_BIT = 0x00000040,
    FRAGMENT_SHADER_BIT = 0x00000080,
    EARLY_FRAGMENT_TESTS_BIT = 0x00000100,
    LATE_FRAGMENT_TESTS_BIT = 0x00000200,
    COLOR_ATTACHMENT_OUTPUT_BIT = 0x00000400,
    COMPUTE_SHADER_BIT = 0x00000800,
    TRANSFER_BIT = 0x00001000,
    BOTTOM_OF_PIPE_BIT = 0x00002000,
    HOST_BIT = 0x00004000,
    ALL_GRAPHICS_BIT = 0x00008000,
    ALL_COMMANDS_BIT = 0x00010000,
    TRANSFORM_FEEDBACK_BIT_EXT = 0x01000000,
    CONDITIONAL_RENDERING_BIT_EXT = 0x00040000,
    RAY_TRACING_SHADER_BIT_KHR = 0x00200000,
    ACCELERATION_STRUCTURE_BUILD_BIT_KHR = 0x02000000,
    SHADING_RATE_IMAGE_BIT_NV = 0x00400000,
    TASK_SHADER_BIT_NV = 0x00080000,
    MESH_SHADER_BIT_NV = 0x00100000,
    FRAGMENT_DENSITY_PROCESS_BIT_EXT = 0x00800000,
    COMMAND_PREPROCESS_BIT_NV = 0x00020000,
    FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
};
typedef uint32_t PipelineStageFlags;

enum class BlendFactor {
    ZERO = 0,
    ONE = 1,
    SRC_COLOR = 2,
    ONE_MINUS_SRC_COLOR = 3,
    DST_COLOR = 4,
    ONE_MINUS_DST_COLOR = 5,
    SRC_ALPHA = 6,
    ONE_MINUS_SRC_ALPHA = 7,
    DST_ALPHA = 8,
    ONE_MINUS_DST_ALPHA = 9,
    CONSTANT_COLOR = 10,
    ONE_MINUS_CONSTANT_COLOR = 11,
    CONSTANT_ALPHA = 12,
    ONE_MINUS_CONSTANT_ALPHA = 13,
    SRC_ALPHA_SATURATE = 14,
    SRC1_COLOR = 15,
    ONE_MINUS_SRC1_COLOR = 16,
    SRC1_ALPHA = 17,
    ONE_MINUS_SRC1_ALPHA = 18,
    MAX_ENUM = 0x7FFFFFFF
};

enum class BlendOp {
    ADD = 0,
    SUBTRACT = 1,
    REVERSE_SUBTRACT = 2,
    MIN = 3,
    MAX = 4,
    ZERO_EXT = 1000148000,
    SRC_EXT = 1000148001,
    DST_EXT = 1000148002,
    SRC_OVER_EXT = 1000148003,
    DST_OVER_EXT = 1000148004,
    SRC_IN_EXT = 1000148005,
    DST_IN_EXT = 1000148006,
    SRC_OUT_EXT = 1000148007,
    DST_OUT_EXT = 1000148008,
    SRC_ATOP_EXT = 1000148009,
    DST_ATOP_EXT = 1000148010,
    XOR_EXT = 1000148011,
    MULTIPLY_EXT = 1000148012,
    SCREEN_EXT = 1000148013,
    OVERLAY_EXT = 1000148014,
    DARKEN_EXT = 1000148015,
    LIGHTEN_EXT = 1000148016,
    COLORDODGE_EXT = 1000148017,
    COLORBURN_EXT = 1000148018,
    HARDLIGHT_EXT = 1000148019,
    SOFTLIGHT_EXT = 1000148020,
    DIFFERENCE_EXT = 1000148021,
    EXCLUSION_EXT = 1000148022,
    INVERT_EXT = 1000148023,
    INVERT_RGB_EXT = 1000148024,
    LINEARDODGE_EXT = 1000148025,
    LINEARBURN_EXT = 1000148026,
    VIVIDLIGHT_EXT = 1000148027,
    LINEARLIGHT_EXT = 1000148028,
    PINLIGHT_EXT = 1000148029,
    HARDMIX_EXT = 1000148030,
    HSL_HUE_EXT = 1000148031,
    HSL_SATURATION_EXT = 1000148032,
    HSL_COLOR_EXT = 1000148033,
    HSL_LUMINOSITY_EXT = 1000148034,
    PLUS_EXT = 1000148035,
    PLUS_CLAMPED_EXT = 1000148036,
    PLUS_CLAMPED_ALPHA_EXT = 1000148037,
    PLUS_DARKER_EXT = 1000148038,
    MINUS_EXT = 1000148039,
    MINUS_CLAMPED_EXT = 1000148040,
    CONTRAST_EXT = 1000148041,
    INVERT_OVG_EXT = 1000148042,
    RED_EXT = 1000148043,
    GREEN_EXT = 1000148044,
    BLUE_EXT = 1000148045,
    MAX_ENUM = 0x7FFFFFFF
};

enum class LogicOp {
    CLEAR = 0,
    AND = 1,
    AND_REVERSE = 2,
    COPY = 3,
    AND_INVERTED = 4,
    NO_OP = 5,
    XOR = 6,
    OR = 7,
    NOR = 8,
    EQUIVALENT = 9,
    INVERT = 10,
    OR_REVERSE = 11,
    COPY_INVERTED = 12,
    OR_INVERTED = 13,
    NAND = 14,
    SET = 15,
    MAX_ENUM = 0x7FFFFFFF
};

enum class ColorComponentFlagBits {
    R_BIT = 0x00000001,
    G_BIT = 0x00000002,
    B_BIT = 0x00000004,
    A_BIT = 0x00000008,
    ALL_BITS = 0x0000000F,
    FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
};
typedef uint32_t ColorComponentFlags;

enum class PolygonMode {
    FILL = 0,
    LINE = 1,
    POINT = 2,
    FILL_RECTANGLE_NV = 1000153000,
    MAX_ENUM = 0x7FFFFFFF
};

enum class CullModeFlagBits {
    NONE = 0,
    FRONT_BIT = 0x00000001,
    BACK_BIT = 0x00000002,
    FRONT_AND_BACK = 0x00000003,
    FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
};
typedef uint32_t CullModeFlags;

enum class FrontFace {
    COUNTER_CLOCKWISE = 0,
    CLOCKWISE = 1,
    MAX_ENUM = 0x7FFFFFFF
};

typedef uint32_t SampleMask;
enum class SampleCountFlagBits {
    SC_1_BIT = 0x00000001,
    SC_2_BIT = 0x00000002,
    SC_4_BIT = 0x00000004,
    SC_8_BIT = 0x00000008,
    SC_16_BIT = 0x00000010,
    SC_32_BIT = 0x00000020,
    SC_64_BIT = 0x00000040,
    SC_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
};
typedef uint32_t SampleCountFlags;

enum class CompareOp {
    NEVER = 0,
    LESS = 1,
    EQUAL = 2,
    LESS_OR_EQUAL = 3,
    GREATER = 4,
    NOT_EQUAL = 5,
    GREATER_OR_EQUAL = 6,
    ALWAYS = 7,
    MAX_ENUM = 0x7FFFFFFF
};

enum class StencilOp {
    KEEP = 0,
    ZERO = 1,
    REPLACE = 2,
    INCREMENT_AND_CLAMP = 3,
    DECREMENT_AND_CLAMP = 4,
    INVERT = 5,
    INCREMENT_AND_WRAP = 6,
    DECREMENT_AND_WRAP = 7,
    MAX_ENUM = 0x7FFFFFFF
};

enum class AttachmentLoadOp {
    LOAD = 0,
    CLEAR = 1,
    DONT_CARE = 2,
    MAX_ENUM = 0x7FFFFFFF
};

enum class AttachmentStoreOp {
    STORE = 0,
    DONT_CARE = 1,
    NONE_QCOM = 1000301000,
    MAX_ENUM = 0x7FFFFFFF
};

enum class AttachmentDescriptionFlagBits {
    MAY_ALIAS_BIT = 0x00000001,
    FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
};
typedef uint32_t AttachmentDescriptionFlags;

enum class SubpassDescriptionFlagBits {
    PER_VIEW_ATTRIBUTES_BIT_NVX = 0x00000001,
    PER_VIEW_POSITION_X_ONLY_BIT_NVX = 0x00000002,
    FRAGMENT_REGION_BIT_QCOM = 0x00000004,
    SHADER_RESOLVE_BIT_QCOM = 0x00000008,
    FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
};
typedef uint32_t SubpassDescriptionFlags;

enum class DependencyFlagBits {
    BY_REGION_BIT = 0x00000001,
    DEVICE_GROUP_BIT = 0x00000004,
    VIEW_LOCAL_BIT = 0x00000002,
    VIEW_LOCAL_BIT_KHR = VK_DEPENDENCY_VIEW_LOCAL_BIT,
    DEVICE_GROUP_BIT_KHR = VK_DEPENDENCY_DEVICE_GROUP_BIT,
    FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
};
typedef uint32_t DependencyFlags;

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
    const SampleMask*           sample_mask;
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

class Sampler;
class ImageView;
class Buffer;
class DescriptorSet;
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

class CommandBuffer;
class Image;
class Pipeline;
class RenderPass;
class Framebuffer;
class PipelineLayout;
class DescriptorSetLayout;
class PhysicalDevice;
class ShaderModule;
struct ImageResourceInfo;

typedef std::vector<std::shared_ptr<PhysicalDevice>> PhysicalDeviceList;
typedef std::vector<std::shared_ptr<DescriptorSetLayout>> DescriptorSetLayoutList;
typedef std::vector<std::shared_ptr<DescriptorSet>> DescriptorSetList;
typedef std::vector<std::shared_ptr<ShaderModule>> ShaderModuleList;

class Queue {
public:
    virtual void submit(const std::vector<std::shared_ptr<CommandBuffer>>& command_buffers) = 0;
    virtual void waitIdle() = 0;
};

class CommandPool {
};

class CommandBuffer {
public:
    virtual void beginCommandBuffer(CommandBufferUsageFlags flags) = 0;
    virtual void endCommandBuffer() = 0;
    virtual void copyBuffer(std::shared_ptr<Buffer> src_buf, std::shared_ptr<Buffer> dst_buf, std::vector<BufferCopyInfo> copy_regions) = 0;
    virtual void copyBufferToImage(
        std::shared_ptr<Buffer> src_buf,
        std::shared_ptr<Image> dst_image,
        std::vector<BufferImageCopyInfo> copy_regions,
        renderer::ImageLayout layout) = 0;
    virtual void bindPipeline(PipelineBindPoint bind, std::shared_ptr<Pipeline> pipeline) = 0;
    virtual void bindVertexBuffers(uint32_t first_bind, const std::vector<std::shared_ptr<renderer::Buffer>>& vertex_buffers, std::vector<uint64_t> offsets) = 0;
    virtual void bindIndexBuffer(std::shared_ptr<Buffer> index_buffer, uint64_t offset, IndexType index_type) = 0;
    virtual void bindDescriptorSets(
        PipelineBindPoint bind_point,
        const std::shared_ptr<PipelineLayout>& pipeline_layout,
        const DescriptorSetList& desc_sets) = 0;
    virtual void pushConstants(
        ShaderStageFlags stages,
        const std::shared_ptr<PipelineLayout>& pipeline_layout,
        const void* data,
        uint32_t size,
        uint32_t offset = 0) = 0;
    virtual void draw(uint32_t vertex_count, uint32_t instance_count = 1, uint32_t first_vertex = 0, uint32_t first_instance = 0) = 0;
    virtual void drawIndexed(uint32_t index_count, uint32_t instance_count = 1, uint32_t first_index = 0, uint32_t vertex_offset = 0, uint32_t first_instance = 0) = 0;
    virtual void dispatch(uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z = 1) = 0;
    virtual void beginRenderPass(
        std::shared_ptr<RenderPass> render_pass,
        std::shared_ptr<Framebuffer> frame_buffer,
        const glm::uvec2& extent,
        const std::vector<ClearValue>& clear_values) = 0;
    virtual void endRenderPass() = 0;
    virtual void reset(uint32_t flags) = 0;
    virtual void addImageBarrier(
        const std::shared_ptr<Image>& image,
        const ImageResourceInfo& src_info,
        const ImageResourceInfo& dst_info,
        uint32_t base_mip = 0,
        uint32_t mip_count = 1,
        uint32_t base_layer = 0,
        uint32_t layer_count = 1) = 0;
    //virtual void pipelineBarrier() = 0;
};

class Instance {
};

class PhysicalDevice {
};

class Surface {
};

class Swapchain {
};

class DescriptorPool {
};

class Pipeline {
};

class PipelineLayout {
};

class RenderPass {
};

class Framebuffer {
};

class ImageView {
};

class Sampler {
};

class Image {
};

class Buffer {
};

class Semaphore {
};

class Fence {
};

class DeviceMemory {
};

class DescriptorSetLayout {
};

class DescriptorSet {
};

class ShaderModule {
};

class Device {
public:
    virtual std::shared_ptr<DescriptorPool> createDescriptorPool() = 0;
    virtual void createBuffer(
        const uint64_t& buffer_size,
        const BufferUsageFlags& usage,
        const MemoryPropertyFlags& properties,
        std::shared_ptr<Buffer>& buffer,
        std::shared_ptr<DeviceMemory>& buffer_memory) = 0;
    virtual void updateDescriptorSets(
        const std::vector<TextureDescriptor>& texture_list,
        const std::vector<BufferDescriptor>& buffer_list) = 0;
    virtual DescriptorSetList createDescriptorSets(
        std::shared_ptr<DescriptorPool> descriptor_pool,
        std::shared_ptr<DescriptorSetLayout> descriptor_set_layout,
        uint64_t buffer_count) = 0;
    virtual std::shared_ptr<PipelineLayout> createPipelineLayout(
        const DescriptorSetLayoutList& desc_set_layouts,
        const std::vector<PushConstantRange>& push_const_ranges) = 0;
    virtual std::shared_ptr<DescriptorSetLayout> createDescriptorSetLayout(
        const std::vector<DescriptorSetLayoutBinding>& bindings) = 0;
    virtual std::shared_ptr<RenderPass> createRenderPass(
        const std::vector<AttachmentDescription>& attachments,
        const std::vector<SubpassDescription>& subpasses,
        const std::vector<SubpassDependency>& dependencies) = 0;
    virtual std::shared_ptr<Pipeline> createPipeline(
        const std::shared_ptr<RenderPass>& render_pass,
        const std::shared_ptr<PipelineLayout>& pipeline_layout,
        const std::vector<VertexInputBindingDescription>& binding_descs,
        const std::vector<VertexInputAttributeDescription>& attribute_descs,
        const PipelineInputAssemblyStateCreateInfo& topology_info,
        const PipelineColorBlendStateCreateInfo& blend_state_info,
        const PipelineRasterizationStateCreateInfo& rasterization_info,
        const PipelineMultisampleStateCreateInfo& ms_info,
        const PipelineDepthStencilStateCreateInfo& depth_stencil_info,
        const ShaderModuleList& shader_modules,
        const glm::uvec2& extent) = 0;
    virtual std::shared_ptr<Pipeline> createPipeline(
        const std::shared_ptr<PipelineLayout>& pipeline_layout,
        const std::shared_ptr<renderer::ShaderModule>& shader_module) = 0;
    virtual std::shared_ptr<Swapchain> createSwapchain(
        std::shared_ptr<Surface> surface,
        uint32_t image_count,
        Format format,
        glm::uvec2 buf_size,
        ColorSpace color_space,
        SurfaceTransformFlagBits transform,
        PresentMode present_mode,
        std::vector<uint32_t> queue_index) = 0;
    virtual void updateBufferMemory(
        const std::shared_ptr<DeviceMemory>& memory,
        uint64_t size,
        const void* src_data,
        uint64_t offset = 0) = 0;
    virtual std::vector<std::shared_ptr<renderer::Image>> getSwapchainImages(std::shared_ptr<Swapchain> swap_chain) = 0;
    virtual std::shared_ptr<CommandPool> createCommandPool(uint32_t queue_family_index, CommandPoolCreateFlags flags) = 0;
    virtual std::shared_ptr<Queue> getDeviceQueue(uint32_t queue_family_index, uint32_t queue_index = 0) = 0;
    virtual std::shared_ptr<DeviceMemory> allocateMemory(uint64_t buf_size, uint32_t memory_type_bits, MemoryPropertyFlags properties) = 0;
    virtual MemoryRequirements getBufferMemoryRequirements(std::shared_ptr<Buffer> buffer) = 0;
    virtual MemoryRequirements getImageMemoryRequirements(std::shared_ptr<Image> image) = 0;
    virtual std::shared_ptr<Buffer> createBuffer(uint64_t buf_size, BufferUsageFlags usage, bool sharing = false) = 0;
    virtual std::shared_ptr<Image> createImage(
        ImageType image_type,
        glm::uvec3 image_size,
        Format format,
        ImageUsageFlags usage,
        ImageTiling tiling,
        ImageLayout layout,
        ImageCreateFlags flags = 0,
        bool sharing = false,
        uint32_t num_samples = 1,
        uint32_t num_mips = 1,
        uint32_t num_layers = 1) = 0;
    virtual std::shared_ptr<ShaderModule> createShaderModule(uint64_t size, void* data) = 0;
    virtual std::shared_ptr<ImageView> createImageView(
        std::shared_ptr<Image> image,
        ImageViewType view_type,
        Format format,
        ImageAspectFlags aspect_flags,
        uint32_t base_mip = 0,
        uint32_t mip_count = 1,
        uint32_t base_layer = 0,
        uint32_t layer_count = 1) = 0;
    virtual std::shared_ptr<Framebuffer> createFrameBuffer(
        const std::shared_ptr<RenderPass>& render_pass,
        const std::vector<std::shared_ptr<ImageView>>& attachments,
        const glm::uvec2& extent) = 0;
    virtual std::shared_ptr<Sampler> createSampler(Filter filter, SamplerAddressMode address_mode, SamplerMipmapMode mipmap_mode, float anisotropy) = 0;
    virtual std::shared_ptr<Semaphore> createSemaphore() = 0;
    virtual std::shared_ptr<Fence> createFence() = 0;
    virtual void bindBufferMemory(std::shared_ptr<Buffer> buffer, std::shared_ptr<DeviceMemory> buffer_memory, uint64_t offset = 0) = 0;
    virtual void bindImageMemory(std::shared_ptr<Image> image, std::shared_ptr<DeviceMemory> image_memory, uint64_t offset = 0) = 0;
    virtual std::vector<std::shared_ptr<CommandBuffer>> allocateCommandBuffers(std::shared_ptr<CommandPool> cmd_pool, uint32_t num_buffers, bool is_primary = true) = 0;
    virtual void* mapMemory(std::shared_ptr<DeviceMemory> memory, uint64_t size, uint64_t offset = 0) = 0;
    virtual void unmapMemory(std::shared_ptr<DeviceMemory> memory) = 0;
    virtual void destroyCommandPool(std::shared_ptr<CommandPool> cmd_pool) = 0;
    virtual void destroySwapchain(std::shared_ptr<Swapchain> swapchain) = 0;
    virtual void destroyDescriptorPool(std::shared_ptr<DescriptorPool> descriptor_pool) = 0;
    virtual void destroyPipeline(std::shared_ptr<Pipeline> pipeline) = 0;
    virtual void destroyPipelineLayout(std::shared_ptr<PipelineLayout> pipeline_layout) = 0;
    virtual void destroyRenderPass(std::shared_ptr<RenderPass> render_pass) = 0;
    virtual void destroyFramebuffer(std::shared_ptr<Framebuffer> frame_buffer) = 0;
    virtual void destroyImageView(std::shared_ptr<ImageView> image_view) = 0;
    virtual void destroySampler(std::shared_ptr<Sampler> sampler) = 0;
    virtual void destroyImage(std::shared_ptr<Image> image) = 0;
    virtual void destroyBuffer(std::shared_ptr<Buffer> buffer) = 0;
    virtual void destroySemaphore(std::shared_ptr<Semaphore> semaphore) = 0;
    virtual void destroyFence(std::shared_ptr<Fence> fence) = 0;
    virtual void destroyDescriptorSetLayout(std::shared_ptr<DescriptorSetLayout> layout) = 0;
    virtual void destroyShaderModule(std::shared_ptr<ShaderModule> layout) = 0;
    virtual void freeMemory(std::shared_ptr<DeviceMemory> memory) = 0;
    virtual void freeCommandBuffers(std::shared_ptr<CommandPool> cmd_pool, const std::vector<std::shared_ptr<CommandBuffer>>& cmd_bufs) = 0;
    virtual void resetFences(const std::vector<std::shared_ptr<Fence>>& fences) = 0;
    virtual void waitForFences(const std::vector<std::shared_ptr<Fence>>& fences) = 0;
    virtual void waitIdle() = 0;
};

class VulkanDevice : public Device {
    VkDevice        device_;
    const std::shared_ptr<PhysicalDevice>& physical_device_;
public:
    VulkanDevice(
        const std::shared_ptr<PhysicalDevice>& physical_device,
        const VkDevice& device)
        : physical_device_(physical_device), device_(device) {}
    VkDevice get() { return device_; }
    const std::shared_ptr<PhysicalDevice>& getPhysicalDevice() { return physical_device_; }
    virtual std::shared_ptr<DescriptorPool> createDescriptorPool() final;
    virtual void createBuffer(
        const uint64_t& buffer_size,
        const BufferUsageFlags& usage,
        const MemoryPropertyFlags& properties,
        std::shared_ptr<Buffer>& buffer,
        std::shared_ptr<DeviceMemory>& buffer_memory) final;
    virtual void updateDescriptorSets(
        const std::vector<TextureDescriptor>& texture_list,
        const std::vector<BufferDescriptor>& buffer_list) final;
    virtual DescriptorSetList createDescriptorSets(
        std::shared_ptr<DescriptorPool> descriptor_pool,
        std::shared_ptr<DescriptorSetLayout> descriptor_set_layout,
        uint64_t buffer_count) final;
    virtual std::shared_ptr<PipelineLayout> createPipelineLayout(
        const DescriptorSetLayoutList& desc_set_layouts,
        const std::vector<PushConstantRange>& push_const_ranges) final;
    virtual std::shared_ptr<DescriptorSetLayout> createDescriptorSetLayout(
        const std::vector<DescriptorSetLayoutBinding>& bindings) final;
    virtual std::shared_ptr<RenderPass> createRenderPass(
        const std::vector<AttachmentDescription>& attachments,
        const std::vector<SubpassDescription>& subpasses,
        const std::vector<SubpassDependency>& dependencies) final;
    virtual std::shared_ptr<Pipeline> createPipeline(
        const std::shared_ptr<RenderPass>& render_pass,
        const std::shared_ptr<PipelineLayout>& pipeline_layout,
        const std::vector<VertexInputBindingDescription>& binding_descs,
        const std::vector<VertexInputAttributeDescription>& attribute_descs,
        const PipelineInputAssemblyStateCreateInfo& topology_info,
        const PipelineColorBlendStateCreateInfo& blend_state_info,
        const PipelineRasterizationStateCreateInfo& rasterization_info,
        const PipelineMultisampleStateCreateInfo& ms_info,
        const PipelineDepthStencilStateCreateInfo& depth_stencil_info,
        const ShaderModuleList& shader_modules,
        const glm::uvec2& extent) final;
    virtual std::shared_ptr<Pipeline> createPipeline(
        const std::shared_ptr<PipelineLayout>& pipeline_layout,
        const std::shared_ptr<renderer::ShaderModule>& shader_module) final;
    virtual std::shared_ptr<Swapchain> createSwapchain(
        std::shared_ptr<Surface> surface,
        uint32_t image_count,
        Format format,
        glm::uvec2 buf_size,
        ColorSpace color_space,
        SurfaceTransformFlagBits transform,
        PresentMode present_mode,
        std::vector<uint32_t> queue_index) final;
    virtual void updateBufferMemory(
        const std::shared_ptr<DeviceMemory>& memory,
        uint64_t size,
        const void* src_data,
        uint64_t offset = 0) final;
    virtual std::vector<std::shared_ptr<Image>> getSwapchainImages(std::shared_ptr<Swapchain> swap_chain) final;
    virtual std::shared_ptr<CommandPool> createCommandPool(uint32_t queue_family_index, CommandPoolCreateFlags flags) final;
    virtual std::shared_ptr<Queue> getDeviceQueue(uint32_t queue_family_index, uint32_t queue_index = 0) final;
    virtual std::shared_ptr<DeviceMemory> allocateMemory(uint64_t buf_size, uint32_t memory_type_bits, MemoryPropertyFlags properties) final;
    virtual MemoryRequirements getBufferMemoryRequirements(std::shared_ptr<Buffer> buffer) final;
    virtual MemoryRequirements getImageMemoryRequirements(std::shared_ptr<Image> image) final;
    virtual std::shared_ptr<Buffer> createBuffer(uint64_t buf_size, BufferUsageFlags usage, bool sharing = false) final;
    virtual std::shared_ptr<Image> createImage(
        ImageType image_type,
        glm::uvec3 image_size,
        Format format,
        ImageUsageFlags usage,
        ImageTiling tiling,
        ImageLayout layout,
        ImageCreateFlags flags = 0,
        bool sharing = false,
        uint32_t num_samples = 1,
        uint32_t num_mips = 1,
        uint32_t num_layers = 1) final;
    virtual std::shared_ptr<ShaderModule> createShaderModule(uint64_t size, void* data) final;
    virtual std::shared_ptr<ImageView> createImageView(
        std::shared_ptr<Image> image,
        ImageViewType view_type,
        Format format,
        ImageAspectFlags aspect_flags,
        uint32_t base_mip = 0,
        uint32_t mip_count = 1,
        uint32_t base_layer = 0,
        uint32_t layer_count = 1) final;
    virtual std::shared_ptr<Framebuffer> createFrameBuffer(
        const std::shared_ptr<RenderPass>& render_pass,
        const std::vector<std::shared_ptr<ImageView>>& attachments,
        const glm::uvec2& extent) final;
    virtual std::shared_ptr<Sampler> createSampler(Filter filter, SamplerAddressMode address_mode, SamplerMipmapMode mipmap_mode, float anisotropy) final;
    virtual std::shared_ptr<Semaphore> createSemaphore() final;
    virtual std::shared_ptr<Fence> createFence() final;
    virtual void bindBufferMemory(std::shared_ptr<Buffer> buffer, std::shared_ptr<DeviceMemory> buffer_memory, uint64_t offset = 0) final;
    virtual void bindImageMemory(std::shared_ptr<Image> image, std::shared_ptr<DeviceMemory> image_memory, uint64_t offset = 0) final;
    virtual std::vector<std::shared_ptr<CommandBuffer>> allocateCommandBuffers(std::shared_ptr<CommandPool> cmd_pool, uint32_t num_buffers, bool is_primary = true) final;
    virtual void* mapMemory(std::shared_ptr<DeviceMemory> memory, uint64_t size, uint64_t offset = 0) final;
    virtual void unmapMemory(std::shared_ptr<DeviceMemory> memory) final;
    virtual void destroyCommandPool(std::shared_ptr<CommandPool> cmd_pool) final;
    virtual void destroySwapchain(std::shared_ptr<Swapchain> swapchain) final;
    virtual void destroyDescriptorPool(std::shared_ptr<DescriptorPool> descriptor_pool) final;
    virtual void destroyPipeline(std::shared_ptr<Pipeline> pipeline) final;
    virtual void destroyPipelineLayout(std::shared_ptr<PipelineLayout> pipeline_layout) final;
    virtual void destroyRenderPass(std::shared_ptr<RenderPass> render_pass) final;
    virtual void destroyFramebuffer(std::shared_ptr<Framebuffer> frame_buffer) final;
    virtual void destroyImageView(std::shared_ptr<ImageView> image_view) final;
    virtual void destroySampler(std::shared_ptr<Sampler> sampler) final;
    virtual void destroyImage(std::shared_ptr<Image> image) final;
    virtual void destroyBuffer(std::shared_ptr<Buffer> buffer) final;
    virtual void destroySemaphore(std::shared_ptr<Semaphore> semaphore) final;
    virtual void destroyFence(std::shared_ptr<Fence> fence) final;
    virtual void destroyDescriptorSetLayout(std::shared_ptr<DescriptorSetLayout> layout) final;
    virtual void destroyShaderModule(std::shared_ptr<ShaderModule> layout) final;
    virtual void freeMemory(std::shared_ptr<DeviceMemory> memory) final;
    virtual void freeCommandBuffers(std::shared_ptr<CommandPool> cmd_pool, const std::vector<std::shared_ptr<CommandBuffer>>& cmd_bufs) final;
    virtual void resetFences(const std::vector<std::shared_ptr<Fence>>& fences) final;
    virtual void waitForFences(const std::vector<std::shared_ptr<Fence>>& fences) final;
    virtual void waitIdle() final;
};

class VulkanInstance : public Instance {
    VkInstance    instance_;
    VkDebugUtilsMessengerEXT debug_messenger_;

public:
    VkInstance get() { return instance_; }
    VkDebugUtilsMessengerEXT getDebugMessenger() { return debug_messenger_; }
    void set(const VkInstance& instance) { instance_ = instance; }
    void setDebugMessenger(const VkDebugUtilsMessengerEXT debug_messenger) {
        debug_messenger_ = debug_messenger;
    }

    void destroy();
};

class VulkanPhysicalDevice : public PhysicalDevice {
    VkPhysicalDevice physical_device_;

public:
    VkPhysicalDevice get() { return physical_device_; }
    void set(const VkPhysicalDevice& physical_device) { physical_device_ = physical_device; }
};

class VulkanSurface : public Surface {
    VkSurfaceKHR    surface_;
public:
    VkSurfaceKHR get() { return surface_; }
    void set(const VkSurfaceKHR& surface) { surface_ = surface; }
};

class VulkanQueue : public Queue {
    VkQueue         queue_;
public:
    VkQueue get() { return queue_; }
    void set(const VkQueue& queue) { queue_ = queue; }

    virtual void submit(const std::vector<std::shared_ptr<CommandBuffer>>& command_buffers) final;
    virtual void waitIdle() final;
};

class VulkanCommandPool : public CommandPool {
    VkCommandPool    cmd_pool_;
public:
    VkCommandPool get() { return cmd_pool_; }
    void set(const VkCommandPool& cmd_pool) { cmd_pool_ = cmd_pool; }
};

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
    virtual void draw(uint32_t vertex_count, uint32_t instance_count = 1, uint32_t first_vertex = 0, uint32_t first_instance = 0) final;
    virtual void drawIndexed(uint32_t index_count, uint32_t instance_count = 1, uint32_t first_index = 0, uint32_t vertex_offset = 0, uint32_t first_instance = 0) final;
    virtual void dispatch(uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z = 1) final;
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
    //virtual void pipelineBarrier() final;
};

class VulkanSwapchain : public Swapchain {
    VkSwapchainKHR    swap_chain_;
public:
    VkSwapchainKHR get() { return swap_chain_; }
    void set(VkSwapchainKHR swap_chain) { swap_chain_ = swap_chain; }
};

class VulkanDescriptorPool : public DescriptorPool {
    VkDescriptorPool    descriptor_pool_;
public:
    VkDescriptorPool get() { return descriptor_pool_; }
    void set(const VkDescriptorPool& descriptor_pool) { descriptor_pool_ = descriptor_pool; }
};

class VulkanPipeline : public Pipeline {
    VkPipeline    pipeline_;
public:
    VkPipeline get() { return pipeline_; }
    void set(const VkPipeline& pipeline) { pipeline_ = pipeline; }
};

class VulkanPipelineLayout : public PipelineLayout {
    VkPipelineLayout    pipeline_layout_;
public:
    VkPipelineLayout get() { return pipeline_layout_; }
    void set(const VkPipelineLayout& layout) { pipeline_layout_ = layout; }
};

class VulkanRenderPass : public RenderPass {
    VkRenderPass    render_pass_;
public:
    VkRenderPass get() { return render_pass_; }
    void set(const VkRenderPass& render_pass) { render_pass_ = render_pass; }
};

class VulkanFramebuffer : public Framebuffer {
    VkFramebuffer    frame_buffer_;
public:
    VkFramebuffer get() { return frame_buffer_; }
    void set(const VkFramebuffer frame_buffer) { frame_buffer_ = frame_buffer; }
};

class VulkanImageView : public ImageView {
    VkImageView     image_view_;
public:
    VkImageView get() { return image_view_; }
    void set(const VkImageView& image_view) { image_view_ = image_view; }
};

class VulkanSampler : public Sampler {
    VkSampler       sampler_;
public:
    VkSampler get() { return sampler_; }
    const VkSampler* getPtr()const { return &sampler_; }
    void set(const VkSampler& sampler) { sampler_ = sampler; }
};

class VulkanImage : public Image {
    VkImage         image_;
    ImageLayout     layout_ = ImageLayout::UNDEFINED;
public:
    VkImage get() { return image_; }
    ImageLayout getImageLayout() { return layout_; }
    void set(VkImage image) { image_ = image; }
    void setImageLayout(ImageLayout layout) { layout_ = layout; }
};

class VulkanBuffer : public Buffer {
    VkBuffer         buffer_;
public:
    VkBuffer get() { return buffer_; }
    void set(const VkBuffer& buffer) { buffer_ = buffer; }
};

class VulkanSemaphore : public Semaphore {
    VkSemaphore      semaphore_;
public:
    VkSemaphore get() { return semaphore_; }
    void set(const VkSemaphore& semaphore) { semaphore_ = semaphore; }
};

class VulkanFence : public Fence {
    VkFence          fence_;
public:
    VkFence get() { return fence_; }
    void set(const VkFence& fence) { fence_ = fence; }
};

class VulkanDeviceMemory : public DeviceMemory {
    VkDeviceMemory  memory_;
public:
    VkDeviceMemory get() { return memory_; }
    void set(const VkDeviceMemory& memory) { memory_ = memory; }
};

class VulkanDescriptorSetLayout : public DescriptorSetLayout {
    VkDescriptorSetLayout  layout_;
public:
    VkDescriptorSetLayout get() { return layout_; }
    void set(const VkDescriptorSetLayout layout) { layout_ = layout; }

    void destroy(const std::shared_ptr<Device>& device);
};

class VulkanDescriptorSet : public DescriptorSet {
    VkDescriptorSet  desc_set_;
public:
    VkDescriptorSet get() { return desc_set_; }
    void set(const VkDescriptorSet desc_set) { desc_set_ = desc_set; }

    void destroy(const std::shared_ptr<Device>& device);
};
    

class VulkanShaderModule : public ShaderModule {
    VkShaderModule shader_module_;
public:
    VkShaderModule get() { return shader_module_; }
    void set(const VkShaderModule& shader_module) { shader_module_ = shader_module; }
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

struct DeviceInfo {
    std::shared_ptr<Device>             device;
    std::shared_ptr<Queue>              cmd_queue;
    std::shared_ptr<CommandPool>        cmd_pool;
};

struct SwapChainInfo {
    renderer::Format format;
    glm::uvec2 extent;
    std::shared_ptr<renderer::Swapchain> swap_chain;
    std::vector<std::shared_ptr<renderer::Image>> images;
    std::vector<std::shared_ptr<renderer::ImageView>> image_views;
    std::vector<std::shared_ptr<renderer::Framebuffer>> framebuffers;
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics_family_;
    std::optional<uint32_t> present_family_;

    bool isComplete() const {
        return graphics_family_.has_value() && present_family_.has_value();
    }
};

class Helper {
public:
    static void init(const DeviceInfo& device_info);

    static void destroy(const std::shared_ptr<Device>& device);

    static Format findDepthFormat(const std::shared_ptr<Device>& device);

    static ImageResourceInfo getImageAsSource() { return image_source_info_; }
    static ImageResourceInfo getImageAsColorAttachment() { return image_as_color_attachement_; }
    static ImageResourceInfo getImageAsStore() { return image_as_store_; }
    static ImageResourceInfo getImageAsShaderSampler() { return image_as_shader_sampler_; }

    static std::shared_ptr<Instance> createInstance();

    static std::shared_ptr<Surface> createSurface(
        const std::shared_ptr<Instance>& instance,
        GLFWwindow* window);

    static PhysicalDeviceList collectPhysicalDevices(
        const std::shared_ptr<Instance>& instance);

    static std::shared_ptr<PhysicalDevice> pickPhysicalDevice(
        const PhysicalDeviceList& physical_devices,
        const std::shared_ptr<Surface>& surface);

    static QueueFamilyIndices findQueueFamilies(
        const std::shared_ptr<PhysicalDevice>& physical_device,
        const std::shared_ptr<Surface>& surface);

    static std::shared_ptr<Device> createLogicalDevice(
        const std::shared_ptr<PhysicalDevice>& physical_device,
        const std::shared_ptr<Surface>& surface,
        const QueueFamilyIndices& indices);

    static void createSwapChain(
        GLFWwindow* window,
        const std::shared_ptr<Device>& device,
        const std::shared_ptr<Surface>& surface,
        const QueueFamilyIndices& indices,
        SwapChainInfo& swap_chain_info);

    static void addOneTexture(
        std::vector<TextureDescriptor>& descriptor_writes,
        uint32_t binding,
        const std::shared_ptr<Sampler>& sampler,
        const std::shared_ptr<ImageView>& texture,
        const std::shared_ptr<DescriptorSet>& desc_set,
        DescriptorType desc_type = DescriptorType::COMBINED_IMAGE_SAMPLER,
        ImageLayout image_layout = ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    static void addOneBuffer(
        std::vector<BufferDescriptor>& descriptor_writes,
        uint32_t binding,
        const std::shared_ptr<Buffer>& buffer,
        const std::shared_ptr<DescriptorSet>& desc_set,
        DescriptorType desc_type,
        uint32_t range,
        uint32_t offset = 0);

    static void generateMipmapLevels(
        const std::shared_ptr<CommandBuffer>& cmd_buf,
        const std::shared_ptr<Image>& image,
        uint32_t mip_count,
        uint32_t width,
        uint32_t height,
        const ImageLayout& cur_image_layout);

    static void create2DTextureImage(
        const DeviceInfo& device_info,
        Format format,
        int tex_width,
        int tex_height,
        int tex_channels,
        const void* pixels,
        std::shared_ptr<Image>& texture_image,
        std::shared_ptr<DeviceMemory>& texture_image_memory);

    static void createDepthResources(
        const DeviceInfo& device_info,
        Format depth_format,
        glm::uvec2 size,
        TextureInfo& depth_buffer);

    static void createCubemapTexture(
        const DeviceInfo& device_info,
        const std::shared_ptr<RenderPass>& render_pass,
        uint32_t width,
        uint32_t height,
        uint32_t mip_count,
        Format format,
        const std::vector<BufferImageCopyInfo>& copy_regions,
        TextureInfo& texture,
        uint64_t buffer_size = 0,
        void* data = nullptr);

    static void createBufferWithSrcData(
        const DeviceInfo& device_info,
        const BufferUsageFlags& usage,
        const uint64_t& buffer_size,
        const void* src_data,
        std::shared_ptr<Buffer>& buffer,
        std::shared_ptr<DeviceMemory>& buffer_memory);

    static TextureInfo& getBlackTexture() { return black_tex_; }
    static TextureInfo& getWhiteTexture() { return white_tex_; }

public:
    static TextureInfo black_tex_;
    static TextureInfo white_tex_;
    static ImageResourceInfo image_source_info_;
    static ImageResourceInfo image_as_color_attachement_;
    static ImageResourceInfo image_as_store_;
    static ImageResourceInfo image_as_shader_sampler_;
};


} // namespace renderer
} // namespace work