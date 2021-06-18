#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

layout(location = 0) in float base_height;
layout(location = 1) in vec4 terrain_layers;     // soil, grass, snow, water layer thickness.

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

    float layer_height = base_height.x;
#if defined(SOIL_PASS) || defined(GRASS_LAYER) || defined(SNOW_PASS) || defined(WATER_PASS)
    layer_height += terrain_layers.x * SOIL_LAYER_MAX_THICKNESS;
#endif
#if defined(GRASS_LAYER) || defined(SNOW_PASS) || defined(WATER_PASS)
    layer_height += terrain_layers.y * GRASS_LAYER_MAX_THICKNESS;
#endif
#if defined(SNOW_PASS) || defined(WATER_PASS)
    layer_height += terrain_layers.z * SNOW_LAYER_MAX_THICKNESS;
#endif
#if defined(WATER_PASS)
    layer_height += terrain_layers.w * WATER_LAYER_MAX_THICKNESS;
#endif

    vec4 position_ws = vec4(x, layer_height, y, 1.0);
    gl_Position = view_params.proj * view_params.view * position_ws;

    out_data.vertex_position = position_ws.xyz;
}