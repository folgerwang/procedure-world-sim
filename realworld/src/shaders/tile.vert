#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

layout(location = 0) in uvec2 layer_heights;     // 18bits fixed for rock level, 14 bits fixed for soil layer.
                                                 // 14bits fixed for water layer, 10 bits fixed for snow layer.
                                                 // 4bits for rock/soil transition, 4bits for humidity.
                                                 // all reserved have 5bits for fraction.

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

    float layer_height = (layer_heights.x & 0x0003ffff) / 32.0f - ROCK_LAYER_BASE;
#if defined(SOIL_PASS) || defined(WATER_PASS) || defined(SNOW_PASS)
    layer_height += (layer_heights.x >> 18) / 32.0f;
#endif
#if defined(WATER_PASS) || defined(SNOW_PASS)
    layer_height += (layer_heights.y & 0x00003fff) / 32.0f;
#endif
#if defined(SNOW_PASS)
    layer_height += ((layer_heights.y >> 14) & 0x000003ff) / 32.0f;
#endif

    vec4 position_ws = vec4(x, layer_height, y, 1.0);
    gl_Position = view_params.proj * view_params.view * position_ws;

    out_data.vertex_position = position_ws.xyz;
}