#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"
#include "brdf.glsl.h"
#include "punctual.glsl.h"

#include "ibl.glsl.h"

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	GameCameraInfo camera_info;
};

layout(push_constant) uniform TileUniformBufferObject {
    TileParams tile_params;
};

layout(location = 0) in VsPsData {
    vec3 vertex_position;
} in_data;

vec3  kSunDir = vec3(-0.624695f, 0.468521f, -0.624695f);

mat3 m3 = mat3(
    0.00f, 0.80f, 0.60f,
    -0.80f, 0.36f, -0.48f,
    -0.60f, -0.48f, 0.64f);
mat3 m3i = mat3(
    0.00f, -0.80f, -0.60f,
    0.80f, 0.36f, -0.48f,
    0.60f, -0.48f, 0.64f);
mat2 m2 = mat2(
    0.80f, 0.60f,
    -0.60f, 0.80f);
mat2 m2i = mat2(
    0.80f, -0.60f,
    0.60f, 0.80f);

layout(location = 0) out vec4 outColor;

float hash1(vec2 p)
{
    p = 50.0f*fract(p*0.3183099f);
    return fract(p.x*p.y*(p.x + p.y));
}

float hash1(float n)
{
    return fract(n*17.0f*fract(n*0.3183099f));
}

// value noise, and its analytical derivatives
vec4 noised(vec3 x)
{
    vec3 p = floor(x);
    vec3 w = fract(x);
#if 1
    vec3 u = w * w*w*(w*(w*6.0f - 15.0f) + 10.0f);
    vec3 du = 30.0f*w*w*(w*(w - 2.0f) + 1.0f);
#else
    vec3 u = w * w*(3.0f - 2.0f*w);
    vec3 du = 6.0f*w*(1.0f - w);
#endif

    float n = p.x + 317.0f*p.y + 157.0f*p.z;

    float a = hash1(n + 0.0f);
    float b = hash1(n + 1.0f);
    float c = hash1(n + 317.0f);
    float d = hash1(n + 318.0f);
    float e = hash1(n + 157.0f);
    float f = hash1(n + 158.0f);
    float g = hash1(n + 474.0f);
    float h = hash1(n + 475.0f);

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k3 = e - a;
    float k4 = a - b - c + d;
    float k5 = a - c - e + g;
    float k6 = a - b - e + f;
    float k7 = -a + b + c - d + e - f - g + h;

    return vec4(-1.0f + 2.0f*(k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x*u.y + k5 * u.y*u.z + k6 * u.z*u.x + k7 * u.x*u.y*u.z),
        2.0f* du * vec3(k1 + k4 * u.y + k6 * u.z + k7 * u.y*u.z,
            k2 + k5 * u.z + k4 * u.x + k7 * u.z*u.x,
            k3 + k6 * u.x + k5 * u.y + k7 * u.x*u.y));
}

vec3 noised(vec2 x)
{
    vec2 p = floor(x);
    vec2 w = fract(x);
#if 1
    vec2 u = w * w*w*(w*(w*6.0f - 15.0f) + 10.0f);
    vec2 du = 30.0f*w*w*(w*(w - 2.0f) + 1.0f);
#else
    vec2 u = w * w*(3.0f - 2.0f*w);
    vec2 du = 6.0f*w*(1.0f - w);
#endif

    float a = hash1(p + vec2(0, 0));
    float b = hash1(p + vec2(1, 0));
    float c = hash1(p + vec2(0, 1));
    float d = hash1(p + vec2(1, 1));

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k4 = a - b - c + d;

    return vec3(-1.0f + 2.0f*(k0 + k1 * u.x + k2 * u.y + k4 * u.x*u.y),
        2.0f* du * vec2(k1 + k4 * u.y,
            k2 + k4 * u.x));
}

vec4 fbmd_8(vec3 x)
{
    float f = 1.92f;
    float s = 0.5f;
    float a = 0.0f;
    float b = 0.5f;
    vec3  d = vec3(0.0f);
    mat3  m = mat3(
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f);
    for (int i = 0; i < 7; i++)
    {
        vec4 n = noised(x);
        a += b * n.x;          // accumulate values		
        d += b * m*vec3(n.y, n.z, n.w);      // accumulate derivatives
        b *= s;
        x = f * m3* x;
        m = f * m3i*m;
    }
    return vec4(a, d);
}

vec3 fbmd_9(vec2 x)
{
    float f = 1.9f;
    float s = 0.55f;
    float a = 0.0f;
    float b = 0.5f;
    vec2  d = vec2(0.0);
    mat2  m = mat2(1.0, 0.0, 0.0, 1.0);
    for (int i = 0; i < 9; i++)
    {
        vec3 n = noised(x);
        a += b * n.x;          // accumulate values		
        d += b * m*vec2(n.y, n.z);       // accumulate derivatives
        b *= s;
        x = f * m2*x;
        m = f * m2i*m;
    }
    return vec3(a, d);
}

// return smoothstep and its derivative
vec2 smoothstepd(float a, float b, float x)
{
    if (x < a) return vec2(0.0, 0.0);
    if (x > b) return vec2(1.0, 0.0);
    float ir = 1.0f / (b - a);
    x = (x - a)*ir;
    return vec2(x*x*(3.0f - 2.0f*x), 6.0f*x*(1.0f - x)*ir);
}

vec4 terrainMapD(vec2 p)
{
    const float sca = 0.0010f;
    const float amp = 300.0f;
    p *= sca;
    vec3 e = fbmd_9(p + vec2(1.0f, -2.0f));
    vec2 c = smoothstepd(-0.08f, -0.01f, e.x);
    vec2 e_yz = vec2(e.y, e.z);
    e.x = e.x + 0.15f*c.x;
    e_yz = e_yz + 0.15f*c.y* e_yz;
    e.x *= amp;
    e_yz *= amp*sca;
    return vec4(e.x, normalize(vec3(-e_yz.x, 1.0f, -e_yz.y)));
}

vec3 terrainNormal(vec2 pos)
{
#if 1
    vec4 map_d = terrainMapD(pos);
    return vec3(map_d.y, map_d.z, map_d.w);
#else    
    vec2 e = vec2(0.03, 0.0);
    vec2 e_xy = vec2(e.x, e.y);
    vec2 e_yx = vec2(e.y, e.x);
    return normalize(vec3(terrainMap(pos - e_xy).x - terrainMap(pos + e_xy).x,
        2.0*e.x,
        terrainMap(pos - e_yx).x - terrainMap(pos + e_yx).x));
#endif    
}

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

    vec3 view = normalize(camera_info.position.xyz - in_data.vertex_position);

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

    outColor = vec4(color, 1.0);
}