#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

layout(location = 0) in vec2 height;

layout(set = VIEW_PARAMS_SET, binding = VIEW_CONSTANT_INDEX) uniform ViewUniformBufferObject {
    ViewParams view_params;
};

layout(push_constant) uniform TileUniformBufferObject {
    TileParams tile_params;
};

layout(location = 0) out VsPsData {
    vec3 vertex_position;
} out_data;

void main() {
    uint tile_size = tile_params.segment_count + 1;

    uint col = gl_VertexIndex % tile_size;
    uint row = gl_VertexIndex / tile_size;

    float inv_segment_count = 1.0f / tile_params.segment_count;
    float factor_x = col * inv_segment_count;
    float factor_y = row * inv_segment_count;

    float x = tile_params.min.x + factor_x * (tile_params.max.x - tile_params.min.x);
    float y = tile_params.min.y + factor_y * (tile_params.max.y - tile_params.min.y);

    vec4 position_ws = vec4(x, height.x, y, 1.0);
    gl_Position = view_params.proj * view_params.view * position_ws;

    out_data.vertex_position = position_ws.xyz;
}