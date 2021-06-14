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
} in_data;

layout(location = 0) out vec4 outColor;

vec3  kSunDir = vec3(-0.624695f, 0.468521f, -0.624695f);

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

    // bump map
    vec4 tt = fbmd_8(pos * 0.3f * vec3(1.0f, 0.2f, 1.0f));
    vec3 normal = normalize(tnor + 0.8f*(1.0f - abs(tnor.y))*0.8f*vec3(tt.y, tt.z, tt.w));

    vec3 albedo = vec3(0.18, 0.11, 0.10)*.75f;
    albedo = 1.0f* mix(albedo, vec3(0.1, 0.1, 0.0)*0.2f, smoothstep(0.7f, 0.9f, normal.y));

    MaterialInfo material_info;
    material_info.baseColor = albedo;

    vec3 view = normalize(view_params.camera_pos.xyz - in_data.vertex_position);

    vec3 f_diffuse = vec3(0);
    vec3 f_specular = vec3(0);

/*    float sha1 = 1.0f;
    float sha2 = 1.0f;

    float dif = clamp(dot(normal, kSunDir), 0.0f, 1.0f);
    dif *= sha1;
#ifndef LOWQUALITY
    dif *= sha2;
#endif

    float bac = clamp(dot(normalize(vec3(-kSunDir.x, 0.0, -kSunDir.z)), normal), 0.0f, 1.0f);
    float foc = clamp((pos.y + 100.0f) / 100.0f, 0.0f, 1.0f);
    float dom = clamp(0.5f + 0.5f*normal.y, 0.0f, 1.0f);
    vec3  lin = 1.0f*0.2f* mix(0.1f* vec3(0.1, 0.2, 0.1), vec3(0.7, 0.9, 1.5)*3.0f, dom)*foc;
    lin += 1.0f*8.5f* vec3(1.0, 0.9, 0.8)*dif;
    lin += 1.0f*0.27f* vec3(1.0)*bac*foc;

    color *= lin;*/

    float ior = 1.5;
    float f0_ior = 0.04;

    material_info.metallic = 0.3f;//material.metallic_factor;
    material_info.perceptualRoughness = 0.8f;//material.roughness_factor;

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
    float alpha = 1.0f;

    #if defined(WATER_PASS)
    color = vec3(0.4f, 0.4f, 0.9f);
    alpha = 0.5;
    #endif

    outColor = vec4(linearTosRGB(color), alpha);
}