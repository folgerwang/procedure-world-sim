#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

layout(set = VIEW_PARAMS_SET, binding = VIEW_CONSTANT_INDEX) uniform ViewUniformBufferObject {
    ViewParams view_params;
};

layout(push_constant) uniform TileUniformBufferObject {
    TileParams tile_params;
};

layout(location = 0) out VsPsData {
    vec3 vertex_position;
    vec2 world_map_uv;
} out_data;

layout(set = TILE_PARAMS_SET, binding = ROCK_LAYER_BUFFER_INDEX) uniform sampler2D rock_layer;
layout(set = TILE_PARAMS_SET, binding = SOIL_WATER_LAYER_BUFFER_INDEX) uniform sampler2D soil_water_layer;
layout(set = TILE_PARAMS_SET, binding = ORTHER_INFO_LAYER_BUFFER_INDEX) uniform sampler2D orther_info_layer;

void main() {
    uint tile_size = tile_params.segment_count + 1;

    uint col = gl_VertexIndex % tile_size;
    uint row = gl_VertexIndex / tile_size;

    float inv_segment_count = 1.0f / tile_params.segment_count;
    float factor_x = col * inv_segment_count;
    float factor_y = row * inv_segment_count;

    float x = tile_params.min.x + factor_x * tile_params.range.x;
    float y = tile_params.min.y + factor_y * tile_params.range.y;

    vec2 world_map_uv = (vec2(x, y) - tile_params.world_min) * tile_params.inv_world_range;

    float layer_height = texture(rock_layer, world_map_uv).x;
#if defined(SOIL_PASS) || defined(WATER_PASS) || defined(SNOW_PASS)
    vec2 soil_water_thickness = texture(soil_water_layer, world_map_uv).xy * SOIL_WATER_LAYER_MAX_THICKNESS;
    layer_height += soil_water_thickness.x;
#endif
#if defined(WATER_PASS) || defined(SNOW_PASS)
    layer_height += soil_water_thickness.y;
#endif
#if defined(SNOW_PASS)
    layer_height += texture(orther_info_layer, world_map_uv).y * SNOW_LAYER_MAX_THICKNESS;
#endif

    vec4 position_ws = vec4(x, layer_height, y, 1.0);
    gl_Position = view_params.view_proj * position_ws;

    out_data.vertex_position = position_ws.xyz;
    out_data.world_map_uv = world_map_uv;
}