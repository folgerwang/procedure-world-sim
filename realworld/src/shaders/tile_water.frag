#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"
#include "brdf.glsl.h"
#include "punctual.glsl.h"

#include "ibl.glsl.h"
#include "tile_common.glsl.h"

layout(set = VIEW_PARAMS_SET, binding = VIEW_CONSTANT_INDEX) uniform ViewUniformBufferObject {
    ViewParams view_params;
};

layout(push_constant) uniform TileUniformBufferObject {
    TileParams tile_params;
};

layout(location = 0) in VsPsData {
    vec3 vertex_position;
    vec2 world_map_uv;
} in_data;

layout(location = 0) out vec4 outColor;

vec3  kSunDir = vec3(-0.624695f, 0.468521f, -0.624695f);

layout(set = TILE_PARAMS_SET, binding = SRC_COLOR_TEX_INDEX) uniform sampler2D src_tex;
layout(set = TILE_PARAMS_SET, binding = SRC_DEPTH_TEX_INDEX) uniform sampler2D src_depth;
layout(set = TILE_PARAMS_SET, binding = WATER_NORMAL_BUFFER_INDEX) uniform sampler2D water_normal_tex;

struct MaterialInfo
{
    float perceptualRoughness;      // roughness value, as authored by the model creator (input to shader)
    vec3 f0;                        // full reflectance color (n incidence angle)

    float alphaRoughness;           // roughness mapped to a more linear change in the roughness (proposed by [2])
    vec3 albedoColor;

    vec3 f90;                       // reflectance color at grazing angle
    float metallic;

    vec3 n;
    vec3 baseColor; // getBaseColor()

    float sheenIntensity;
    vec3 sheenColor;
    float sheenRoughness;

    float anisotropy;

    vec3 clearcoatF0;
    vec3 clearcoatF90;
    float clearcoatFactor;
    vec3 clearcoatNormal;
    float clearcoatRoughness;

    float subsurfaceScale;
    float subsurfaceDistortion;
    float subsurfacePower;
    vec3 subsurfaceColor;
    float subsurfaceThickness;

    float thinFilmFactor;
    float thinFilmThickness;

    float thickness;

    vec3 absorption;

    float transmission;
};

void main() {
    vec3 pos = in_data.vertex_position;
    vec3 tnor = terrainNormal(vec2(pos.x, pos.z));

    vec3 water_normal;
    water_normal.xz = texture(water_normal_tex, in_data.world_map_uv).xy;
    water_normal.y = sqrt(1.0f - dot(water_normal.xz, water_normal.xz));
    vec2 screen_uv = gl_FragCoord.xy * tile_params.inv_screen_size;
    float dist_scale = length(vec3((screen_uv * 2.0f - 1.0f) * view_params.depth_params.zw, 1.0f));

    float depth_z = texture(src_depth, screen_uv).r;
    float bg_view_dist = view_params.proj[3].z / (depth_z + view_params.proj[2].z) * dist_scale;

    vec3 view_vec = view_params.camera_pos.xyz - in_data.vertex_position;
    float view_dist = length(view_vec);
    vec3 view = normalize(view_vec);

    vec3 refract_ray = refract(-view, water_normal, 1.33);

    float fade_rate = exp(-max((bg_view_dist - view_dist) / 10.0f, 0));

    vec3 bg_color = texture(src_tex, screen_uv).xyz;

    // bump map
    vec3 normal = water_normal;

    vec3 albedo = vec3(0.10, 0.10, 0.3)*.75f;
    albedo = mix(albedo, bg_color, fade_rate);

    MaterialInfo material_info;
    material_info.baseColor = albedo;

    vec3 f_diffuse = vec3(0);
    vec3 f_specular = vec3(0);

    float ior = 1.5;
    float f0_ior = 0.04;

    material_info.metallic = 0.9f;//material.metallic_factor;
    material_info.perceptualRoughness = 0.2f;//material.roughness_factor;

    // Achromatic f0 based on IOR.
    vec3 f0 = vec3(f0_ior);

    material_info.albedoColor = mix(material_info.baseColor.rgb * (vec3(1.0) - f0),  vec3(0), material_info.metallic);
    material_info.f0 = mix(f0, material_info.baseColor.rgb, material_info.metallic);

    #ifdef USE_IBL
    float mip_count = 10;
    f_specular += getIBLRadianceGGX(normal, view, material_info.perceptualRoughness, material_info.f0, mip_count);
    f_diffuse += getIBLRadianceLambertian(normal, material_info.albedoColor);
    #endif

    vec3 color = f_diffuse + f_specular;
    color = mix(linearTosRGB(color), bg_color, fade_rate);
    outColor = vec4(color, 1.0f);
}