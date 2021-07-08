#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

layout(set = VIEW_PARAMS_SET, binding = VIEW_CONSTANT_INDEX) uniform ViewUniformBufferObject {
    ViewParams view_params;
};

layout(push_constant) uniform DebugDrawUniformBufferObject {
    DebugDrawParams params;
};

layout(location = 0) out VsPsData {
    vec4 debug_info;
} out_data;

layout(set = TILE_PARAMS_SET, binding = SRC_VOLUME_TEST_INDEX) uniform sampler3D src_volume;

void main() {
    uint vertex_idx = gl_VertexIndex % 3;
    uint triangle_idx = gl_VertexIndex / 3;

    uint xy_count = params.size.x * params.size.y;
    uint iz = triangle_idx / xy_count;
    uint nxy = triangle_idx % xy_count;
    uint iy = nxy / params.size.x;
    uint ix = nxy % params.size.x;

    vec3 f_xyz = (vec3(ix, iy, iz) + 0.5f) * params.inv_size;

    vec3 sample_pos = f_xyz * params.debug_range + params.debug_min;
    vec4 position_ss = view_params.proj * view_params.view * vec4(sample_pos, 1.0);

    vec3 uvw;
    uvw.z = log2(max((sample_pos.y - kAirflowLowHeight), 0.0f) + 1.0f) /
            log2(kAirflowMaxHeight - kAirflowLowHeight + 1.0f);

    uvw.xy = (sample_pos.xz - params.world_min) * params.inv_world_range;
    out_data.debug_info.x = texture(src_volume, uvw).x;

    vec3 offset = vec3(0);
    if (vertex_idx == 0) {
        offset = vec3(0.002f, 0, 0);
    }
    else if (vertex_idx == 1) {
        offset = vec3(-0.002f, 0, 0);
    }
    else {
        offset = vec3(0, 0.02f, 0);
    }

    sample_pos += offset * position_ss.w;

    vec4 position_ws = vec4(sample_pos, 1.0);
    gl_Position = view_params.proj * view_params.view * position_ws;
}