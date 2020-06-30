#pragma once
#include <memory>
#include <vector>
#include <unordered_map>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

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

    struct PipelineInputAssemblyStateCreateInfo {
        PrimitiveTopology                  topology;
        bool                               restart_enable;
    };

    struct IndexInputBindingDescription {
        uint32_t            binding;
        uint64_t            offset;
        uint64_t            index_count;
        IndexType           index_type;
    };

    class Buffer;
    class CommandBuffer;
    class Image;
    class Pipeline;
    class RenderPass;
    class Framebuffer;

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
        virtual void drawIndexed(uint32_t index_count, uint32_t instance_count = 1, uint32_t first_index = 0, uint32_t vertex_offset = 0, uint32_t first_instance = 0) = 0;
        virtual void beginRenderPass(
            std::shared_ptr<RenderPass> render_pass,
            std::shared_ptr<Framebuffer> frame_buffer,
            const glm::uvec2& extent,
            const std::vector<ClearValue>& clear_values) = 0;
        virtual void endRenderPass() = 0;
        virtual void reset(uint32_t flags) = 0;
        //virtual void pipelineBarrier() = 0;
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
        virtual std::shared_ptr<Swapchain> createSwapchain(
            std::shared_ptr<Surface> surface,
            uint32_t image_count,
            Format format,
            glm::uvec2 buf_size,
            renderer::ColorSpace color_space,
            work::renderer::SurfaceTransformFlagBits transform,
            renderer::PresentMode present_mode,
            std::vector<uint32_t> queue_index) = 0;
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
        const VkPhysicalDevice& physical_device_;
    public:
        VulkanDevice(const VkPhysicalDevice& physical_device, const VkDevice& device)
            : physical_device_(physical_device), device_(device) {}
        VkDevice get() { return device_; }
        virtual std::shared_ptr<Swapchain> createSwapchain(
            std::shared_ptr<Surface> surface,
            uint32_t image_count,
            Format format,
            glm::uvec2 buf_size,
            renderer::ColorSpace color_space,
            work::renderer::SurfaceTransformFlagBits transform,
            renderer::PresentMode present_mode,
            std::vector<uint32_t> queue_index) final;
        virtual std::vector<std::shared_ptr<renderer::Image>> getSwapchainImages(std::shared_ptr<Swapchain> swap_chain) final;
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
        virtual void drawIndexed(uint32_t index_count, uint32_t instance_count = 1, uint32_t first_index = 0, uint32_t vertex_offset = 0, uint32_t first_instance = 0) final;
        virtual void beginRenderPass(
            std::shared_ptr<RenderPass> render_pass,
            std::shared_ptr<Framebuffer> frame_buffer,
            const glm::uvec2& extent,
            const std::vector<ClearValue>& clear_values) final;
        virtual void endRenderPass() final;
        virtual void reset(uint32_t flags) final;

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
        void set(const VkSampler& sampler) { sampler_ = sampler; }
    };

    class VulkanImage : public Image {
        VkImage         image_;
    public:
        VkImage get() { return image_; }
        void set(VkImage image) { image_ = image; }
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
        std::shared_ptr<Image>             image;
        std::shared_ptr<DeviceMemory>      memory;
        std::shared_ptr<ImageView>         view;

        void destroy(const std::shared_ptr<Device>& device);
    };

    struct BufferInfo {
        std::shared_ptr<Buffer>             buffer;
        std::shared_ptr<DeviceMemory>       memory;

        void destroy(const std::shared_ptr<Device>& device);
    };

    struct MaterialInfo {
        int32_t                base_color_idx_ = -1;
        int32_t                normal_idx_ = -1;
        int32_t                metallic_roughness_idx_ = -1;
        int32_t                emissive_idx_ = -1;
        int32_t                occlusion_idx_ = -1;

        BufferInfo             uniform_buffer_;
        std::shared_ptr<DescriptorSet>  desc_set_;
    };

    struct BufferView {
        uint32_t                buffer_idx;
        uint64_t                stride;
        uint64_t                offset;
        uint64_t                range;
    };

    struct PrimitiveInfo {
        std::vector<uint32_t>   binding_list_;
        int32_t                 material_idx_;
        bool                    has_normal_ = false;
        bool                    has_tangent_ = false;
        bool                    has_texcoord_0_ = false;
        bool                    has_skin_set_0_ = false;
        glm::vec3               bbox_min_ = glm::vec3(std::numeric_limits<float>::max());
        glm::vec3               bbox_max_ = glm::vec3(std::numeric_limits<float>::min());
        IndexInputBindingDescription  index_desc_;
        PipelineInputAssemblyStateCreateInfo topology_info_;
        std::vector<VertexInputBindingDescription> binding_descs_;
        std::vector<VertexInputAttributeDescription> attribute_descs_;
    };

    struct MeshInfo {
        std::vector<PrimitiveInfo>  primitives_;
        glm::vec3                   bbox_min_ = glm::vec3(std::numeric_limits<float>::max());
        glm::vec3                   bbox_max_ = glm::vec3(std::numeric_limits<float>::min());
    };

    struct NodeInfo {
        std::vector<int32_t>        child_idx;
        int32_t                     mesh_idx = -1;
        std::shared_ptr<glm::mat4>  matrix;
    };

    struct SceneInfo {
        std::vector<int32_t>        nodes_;
        glm::vec3                   bbox_min_ = glm::vec3(std::numeric_limits<float>::max());
        glm::vec3                   bbox_max_ = glm::vec3(std::numeric_limits<float>::min());
    };

    struct ObjectData {
        int32_t                     default_scene_;
        std::vector<SceneInfo>      scenes_;
        std::vector<NodeInfo>       nodes_;
        std::vector<MeshInfo>       meshes_;
        std::vector<BufferInfo>     buffers_;
        std::vector<BufferView>     buffer_views_;

        std::vector<TextureInfo>    textures_;
        std::vector<MaterialInfo>   materials_;

    public:
        void destroy(const std::shared_ptr<Device>& device);
    };

} // namespace renderer
} // namespace work