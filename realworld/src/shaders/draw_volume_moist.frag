#version 450
#extension GL_ARB_separate_shader_objects : enable

#define RENDER_SUN 1
#define RENDER_SUNLIGHT_SCATTERING  1
const int iSteps = 16;
const int jSteps = 8;

#include "global_definition.glsl.h"
#include "noise.glsl.h"
#include "weather_common.glsl.h"

#define UX3D_MATH_PI 3.1415926535897932384626433832795
#define UX3D_MATH_INV_PI (1.0 / UX3D_MATH_PI)

layout(push_constant) uniform VolumeMoistUniformBufferObject {
    VolumeMoistrueParams params;
};

layout(set = VIEW_PARAMS_SET, binding = VIEW_CONSTANT_INDEX) uniform ViewUniformBufferObject {
    ViewParams view_params;
};

layout(set = 0, binding = SRC_TEMP_MOISTURE_INDEX) uniform sampler3D src_temp_moist;
layout(set = 0, binding = SRC_DEPTH_TEX_INDEX) uniform sampler2D src_depth;

layout (location = 0) in vec2 in_uv;

// output cubemap faces
layout(location = 0) out vec4 outColor;

// entry point
void main() 
{
    vec2 ss_xy = in_uv * 2.0f - 1.0f;
    vec4 position_ss = vec4(ss_xy, -1.0f * view_params.proj[2][2] + view_params.proj[3][2], 1.0f);
    vec3 view_dir = normalize((view_params.inv_view_proj_relative * position_ss).xyz);
    float view_vec_length = length(view_dir);

    float depth_z = texture(src_depth, in_uv).r;
    float dist_scale = length(vec3(ss_xy * view_params.depth_params.zw, 1.0f));
    float bg_view_dist = view_params.proj[3].z / (depth_z + view_params.proj[2].z) * dist_scale;

    float cast_dist = 10000.0f;
    if (view_dir.y > 0) {
        cast_dist = max(kAirflowMaxHeight - view_params.camera_pos.y, 0.0f) / view_dir.y * view_vec_length;
    }
    else if (view_dir.y < 0) {
        cast_dist = max(view_params.camera_pos.y - kAirflowLowHeight, 0.0f) / abs(view_dir.y) * view_vec_length;
    }

    cast_dist = min(min(cast_dist, bg_view_dist), 10000.0f);
        
    vec2 noise = hash23(view_dir * 1334.0f);
    float dither = (1.0f + (noise.y * 2.0f - 1.0f) * 0.2f);

    vec3 fg_color = vec3(0);
    float final_alpha = 1.0f;
    float last_dist = cast_dist / 32.0f * (32 + noise.x);
    for (int i = 0; i < 32; i++) {
        float cur_dist = cast_dist / 32.0f * (32 - i + noise.x);
        float thickness = max(last_dist - cur_dist, 0.0f);

        vec3 sample_pos = view_params.camera_pos.xyz + cur_dist * view_dir;

        vec3 uvw;
        uvw.z = log2(max((sample_pos.y - kAirflowLowHeight), 0.0f) + 1.0f) /
                log2(kAirflowMaxHeight - kAirflowLowHeight + 1.0f);

        uvw.xy = (sample_pos.xz - params.world_min) * params.inv_world_range;
        vec2 temp_moisture = texture(src_temp_moist, uvw).xy;
        float moist = denormalizeMoisture(temp_moisture.y);

        float cur_alpha = exp2(-thickness * moist * 0.001f * dither);
        fg_color = mix(vec3(1.0f), fg_color, cur_alpha);
        final_alpha *= cur_alpha;

        last_dist = cur_dist;
    }

    outColor = vec4(fg_color, final_alpha);
}
