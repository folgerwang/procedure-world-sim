#version 450
#extension GL_ARB_separate_shader_objects : enable

#include "global_definition.glsl.h"

layout(set = VIEW_PARAMS_SET, binding = VIEW_CONSTANT_INDEX) uniform ViewUniformBufferObject {
    ViewParams view_params;
};

layout(set = PBR_GLOBAL_PARAMS_SET, binding = BASE_COLOR_TEX_INDEX) uniform samplerCube skybox_tex;

layout(location = 0) in VsPsData {
    vec3 vertex_position;
} in_data;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 view_dir = normalize(in_data.vertex_position - view_params.camera_pos.xyz);
    outColor = vec4(textureLod(skybox_tex, view_dir, 0).xyz, 1.0);
}