#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

layout(push_constant) uniform ModelUniformBufferObject {
    ModelParams model_params;
};

layout(set = VIEW_PARAMS_SET, binding = VIEW_CONSTANT_INDEX) uniform ViewUniformBufferObject {
    ViewParams view_params;
};

layout(location = VINPUT_POSITION) in vec3 in_position;

#ifdef HAS_UV_SET0
layout(location = VINPUT_TEXCOORD0) in vec2 in_tex_coord;
#endif

#ifdef HAS_UV_SET1
layout(location = VINPUT_TEXCOORD1) in vec2 in_tex_coord;
#endif

#ifdef HAS_NORMALS
layout(location = VINPUT_NORMAL) in vec3 in_normal;
#ifdef HAS_TANGENT
layout(location = VINPUT_TANGENT) in vec4 in_tangent;
#endif
#endif

#ifdef HAS_VERTEX_COLOR_VEC3
layout(location = VINPUT_COLOR) in vec3 v_color;
#endif

#ifdef HAS_VERTEX_COLOR_VEC4
layout(location = VINPUT_COLOR) in vec4 v_color;
#endif

#ifdef HAS_SKIN_SET_0
layout(location = VINPUT_JOINTS_0) in uvec4 in_joints_0;
layout(location = VINPUT_WEIGHTS_0) in vec4 in_weights_0;
#endif

#ifdef HAS_SKIN_SET_1
layout(location = VINPUT_JOINTS_1) in uvec4 in_joints_1;
layout(location = VINPUT_WEIGHTS_1) in vec4 in_weights_1;
#endif

layout(location = 0) out VsPsData {
    vec3 vertex_position;
    vec4 vertex_tex_coord;
#ifdef HAS_NORMALS
    vec3 vertex_normal;
#ifdef HAS_TANGENT
    vec3 vertex_tangent;
    vec3 vertex_binormal;
#endif
#endif
#ifdef HAS_VERTEX_COLOR_VEC3
    vec3 vertex_color;
#endif
#ifdef HAS_VERTEX_COLOR_VEC4
    vec4 vertex_color;
#endif
} out_data;

#ifdef HAS_NORMALS
vec3 getNormal()
{
    vec3 normal = in_normal;

#ifdef USE_MORPHING
    normal += getTargetNormal();
#endif

#ifdef USE_SKINNING
    normal = mat3(getSkinningNormalMatrix()) * normal;
#endif

    return normalize(normal);
}
#endif

#ifdef HAS_TANGENT
vec3 getTangent()
{
    vec3 tangent = in_tangent.xyz;

#ifdef USE_MORPHING
    tangent += getTargetTangent();
#endif

#ifdef USE_SKINNING
    tangent = mat3(getSkinningMatrix()) * tangent;
#endif

    return normalize(tangent);
}
#endif

void main() {
    vec4 position_ws = model_params.model_mat * vec4(in_position, 1.0);
    gl_Position = view_params.proj * view_params.view * position_ws;
    out_data.vertex_position = position_ws.xyz;
    out_data.vertex_tex_coord = vec4(0);
#ifdef HAS_UV_SET0
    out_data.vertex_tex_coord.xy = in_tex_coord;
#endif
#ifdef HAS_UV_SET1
    out_data.vertex_tex_coord.zw = in_tex_coord;
#endif

#ifdef HAS_NORMALS
    out_data.vertex_normal = normalize(mat3(model_params.normal_mat) * getNormal());
#ifdef HAS_TANGENT
    out_data.vertex_tangent = normalize(mat3(model_params.model_mat) * getTangent());
    out_data.vertex_binormal = cross(out_data.vertex_normal, out_data.vertex_tangent) * in_tangent.w;
#endif
#endif // !HAS_NORMALS


#if defined(HAS_VERTEX_COLOR_VEC3) || defined(HAS_VERTEX_COLOR_VEC4)
    out_data.vertex_color = v_color;
#endif
}