#version 450 core
#include "global_definition.glsl.h"

layout (points) in;
layout (triangle_strip, max_vertices = 3) out;

layout(location = 0) in VsPsData {
    vec3 position_ws;
} in_data[];

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	GameCameraInfo camera_info;
};

void main() {    
    gl_Position = camera_info.view_proj * vec4(in_data[0].position_ws + vec3(-0.1, -0.0, 0.0), 1.0);
    EmitVertex();

    gl_Position = camera_info.view_proj * vec4(in_data[0].position_ws + vec3(0.1, 0.0, 0.0), 1.0);
    EmitVertex();

    gl_Position = camera_info.view_proj * vec4(in_data[0].position_ws + vec3(0.0, 2.0, 0.0), 1.0);
    EmitVertex();
    
    EndPrimitive();
}  