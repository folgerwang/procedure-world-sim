#define HAS_BASE_COLOR_MAP          1
#define USE_IBL                     1
#ifdef  HAS_TANGENT 
#define HAS_NORMALS                 1
#endif
//#define USE_PUNCTUAL                1

//#define MATERIAL_UNLIT

// Vertex input attribute location.
#define VINPUT_POSITION             0
#define VINPUT_TEXCOORD0            1
#define VINPUT_NORMAL               2
#define VINPUT_TANGENT              3
#define VINPUT_JOINTS_0             4
#define VINPUT_WEIGHTS_0            5
#define VINPUT_COLOR                6
#define VINPUT_JOINTS_1             7
#define VINPUT_WEIGHTS_1            8
#define VINPUT_TEXCOORD1            9

#define PBR_GLOBAL_PARAMS_SET       0
#define VIEW_PARAMS_SET             1
#define PBR_MATERIAL_PARAMS_SET     2
#define MODEL_PARAMS_SET            3

// MODEL_PARAMS_SET.
#define MODEL_CONSTANT_INDEX        0
// VIEW_PARAMS_SET
#define VIEW_CONSTANT_INDEX         6

// PBR_GLOBAL_PARAMS_SET ibl lighting textures.
#define GGX_LUT_INDEX               0
#define GGX_ENV_TEX_INDEX           1
#define LAMBERTIAN_ENV_TEX_INDEX    2
#define CHARLIE_LUT_INDEX           3
#define CHARLIE_ENV_TEX_INDEX       4
#define THIN_FILM_LUT_INDEX         5

// PBR_MATERIAL_PARAMS_SET
#define PBR_CONSTANT_INDEX          7
#define BASE_COLOR_TEX_INDEX        8
#define NORMAL_TEX_INDEX            (BASE_COLOR_TEX_INDEX + 1)
#define METAL_ROUGHNESS_TEX_INDEX   (BASE_COLOR_TEX_INDEX + 2)
#define EMISSIVE_TEX_INDEX          (BASE_COLOR_TEX_INDEX + 3)
#define OCCLUSION_TEX_INDEX         (BASE_COLOR_TEX_INDEX + 4)

#define FEATURE_MATERIAL_SPECULARGLOSSINESS     0x00000001
#define FEATURE_MATERIAL_METALLICROUGHNESS      0x00000002
#define FEATURE_MATERIAL_SHEEN                  0x00000004
#define FEATURE_MATERIAL_SUBSURFACE             0x00000008
#define FEATURE_MATERIAL_THIN_FILM              0x00000010
#define FEATURE_MATERIAL_CLEARCOAT              0x00000020
#define FEATURE_MATERIAL_TRANSMISSION           0x00000040
#define FEATURE_MATERIAL_ANISOTROPY             0x00000080
#define FEATURE_MATERIAL_IOR                    0x00000100
#define FEATURE_MATERIAL_THICKNESS              0x00000200
#define FEATURE_MATERIAL_ABSORPTION             0x00000400

#define FEATURE_HAS_BASE_COLOR_MAP              0x00010000
#define FEATURE_HAS_NORMAL_MAP                  0x00020000
#define FEATURE_HAS_METALLIC_ROUGHNESS_MAP      0x00040000
#define FEATURE_HAS_EMISSIVE_MAP                0x00080000
#define FEATURE_HAS_OCCLUSION_MAP               0x00100000

#define FEATURE_INPUT_HAS_TANGENT               0x00000001

#define LIGHT_COUNT             4

#define TONEMAP_DEFAULT         0
#define TONEMAP_UNCHARTED       1
#define TONEMAP_HEJLRICHARD     2
#define TONEMAP_ACES            3

#ifdef __cplusplus
#include "glm/glm.hpp"
using namespace glm;
#else
#endif

// KHR_lights_punctual extension.
// see https://github.com/KhronosGroup/glTF/tree/master/extensions/2.0/Khronos/KHR_lights_punctual
struct Light
{
    vec3 direction;
    float range;

    vec3 color;
    float intensity;

    vec3 position;
    float innerConeCos;

    float outerConeCos;
    int type;

    vec2 padding;
};

const int LightType_Directional = 0;
const int LightType_Point = 1;
const int LightType_Spot = 2;

struct ViewParams {
    vec4 camera_pos;
    mat4 view;
    mat4 proj;
    uvec4 input_features;
};

struct ModelParams {
    mat4 model_mat;
    mat3x4 normal_mat;
};

struct PbrMaterialParams {
    vec4    base_color_factor;

    float   glossiness_factor;
    float   metallic_roughness_specular_factor;
    float   metallic_factor;
    float   roughness_factor;

    vec3    specular_factor;
    float   sheen_intensity_factor;

    vec3    sheen_color_factor;
    float   sheen_roughness;

    float   subsurface_scale;
    float   subsurface_distortion;
    float   subsurface_power;
    float   subsurface_thickness_factor;

    vec3    subsurface_color_factor;
    float   thin_film_factor;

    float   thin_film_thickness_maximum;
    float   clearcoat_factor;
    float   clearcoat_roughness_factor;
    float   transmission;

    float   anisotropy;
    float   thickness;
    float   alpha_cutoff;
    float   exposure;

    vec3    absorption_color;
    float   mip_count;

    vec4    ior_f0;

    uvec4   uv_set_flags;

    uint    material_features;
    uint    pad_1;
    float   normal_scale;
    float   occlusion_strength;

    vec3    emissive_factor;
    uint    tonemap_type;

    Light   lights[LIGHT_COUNT];

    mat3    base_color_uv_transform;
    mat3    normal_uv_transform;
    mat3    metallic_roughness_uv_transform;
};
