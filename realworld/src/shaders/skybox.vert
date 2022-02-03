#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

layout(location = 0) in vec3 in_position;

layout(set = VIEW_PARAMS_SET, binding = VIEW_CONSTANT_INDEX) uniform ViewUniformBufferObject {
    ViewParams view_params;
};

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	GameCameraInfo camera_info;
};

layout(location = 0) out VsPsData {
    vec3 vertex_position;
} out_data;

void main() {
    vec4 position_ws = vec4(in_position * 4000.0f + view_params.camera_pos.xyz, 1.0);
    gl_Position = view_params.view_proj * position_ws;

    out_data.vertex_position = position_ws.xyz;
}