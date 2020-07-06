#version 450
#extension GL_ARB_separate_shader_objects : enable

#include "global_definition.glsl.h"

layout(set = VIEW_PARAMS_SET, binding = VIEW_CONSTANT_INDEX) uniform ViewUniformBufferObject {
    ViewParams view_params;
};

layout(set = PBR_GLOBAL_PARAMS_SET, binding = BASE_COLOR_TEX_INDEX) uniform sampler2D skybox_tex;

layout(location = 0) in VsPsData {
    vec3 vertex_position;
} in_data;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 view_dir = normalize(in_data.vertex_position - view_params.camera_pos.xyz);
    vec2 uv;
    float yaw = asin(view_dir.y);
    uv.y = (yaw / (PI / 2.0f)) * -0.5 + 0.5;
    uv.x = acos(view_dir.x / cos(yaw));
    if (view_dir.y < 0) {
        uv.x = 2.0f * PI - uv.x;
    }
    uv.x /= (2.0f * PI);
    vec4 rgbe = texture(skybox_tex, uv);
    float scale = exp2(rgbe.w * 255 - 128);
    outColor = vec4(rgbe.xyz * scale, 1.0);
    return;
}