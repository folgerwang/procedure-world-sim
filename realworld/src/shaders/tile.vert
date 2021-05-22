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
    uint size_x = tile_params.segment_count.x + 1;
    uint size_y = tile_params.segment_count.y + 1;

    uint col = gl_VertexIndex % size_x;
    uint row = gl_VertexIndex / size_x;

    float factor_x = col / float(tile_params.segment_count.x);
    float factor_y = row / float(tile_params.segment_count.y);

    float x = tile_params.min.x + factor_x * (tile_params.max.x - tile_params.min.x);
    float y = tile_params.min.y + factor_y * (tile_params.max.y - tile_params.min.y);

    vec4 position_ws = vec4(x, height.x, y, 1.0);
    gl_Position = view_params.proj * view_params.view * position_ws;

    out_data.vertex_position = position_ws.xyz;
}