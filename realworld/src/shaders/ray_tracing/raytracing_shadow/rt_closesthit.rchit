#version 460
#include "..\..\global_definition.glsl.h"

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable

layout(location = 0) rayPayloadInEXT vec3 hitValue;
layout(location = 2) rayPayloadEXT bool shadowed;
hitAttributeEXT vec2 attribs;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 2, set = 0) uniform UBO 
{
	mat4 viewInverse;
	mat4 projInverse;
	vec4 lightPos;
} ubo;

layout(binding = 3, set = 0) buffer Vertices { float v[]; } vertices;
layout(binding = 4, set = 0) buffer Indices { uint16_t i[]; } indices;
layout(binding = 5, set = 0) buffer Geometries { VertexBufferInfo info[]; } geometries;

struct Vertex
{
  vec3 pos;
  vec3 normal;
  vec2 uv;
  vec4 color;
  vec4 _pad0;
  vec4 _pad1;
 };

Vertex unpack(uint16_t index, in VertexBufferInfo geom_info)
{
	const uint normal_idx = geom_info.normal_base + index * 3;
	float x = vertices.v[normal_idx + 0];
	float y = vertices.v[normal_idx + 1];
	float z = vertices.v[normal_idx + 2];

	Vertex v;
	v.normal.x = dot(vec4(x, y, z, 0.0f), geom_info.matrix[0]);
	v.normal.y = dot(vec4(x, y, z, 0.0f), geom_info.matrix[1]);
	v.normal.z = dot(vec4(x, y, z, 0.0f), geom_info.matrix[2]);
	v.pos = vec3(0, 0, 0);
	v.color = vec4(0.8, 0.8, 0.8, 1.0);//vec4(d2.x, d2.y, d2.z, 1.0);

	return v;
}

void main()
{
	VertexBufferInfo geom_info = geometries.info[gl_GeometryIndexEXT];
	uint base_idx = geom_info.index_base + 3 * gl_PrimitiveID;

	Vertex v0 = unpack(indices.i[base_idx], geom_info);
	Vertex v1 = unpack(indices.i[base_idx + 1], geom_info);
	Vertex v2 = unpack(indices.i[base_idx + 2], geom_info);

	// Interpolate normal
	const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
	vec3 normal = normalize(v0.normal * barycentricCoords.x + v1.normal * barycentricCoords.y + v2.normal * barycentricCoords.z);

	// Basic lighting
	vec3 lightVector = normalize(ubo.lightPos.xyz);
	float dot_product = max(dot(lightVector, normal), 0.2);
	hitValue = v0.color.rgb * dot_product;
 
	// Shadow casting
	float tmin = 0.001;
	float tmax = 10000.0;
	vec3 origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
	shadowed = true;  
	// Trace shadow ray and offset indices to match shadow hit/miss shader group indices
	traceRayEXT(topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, 1, 0, 1, origin, tmin, lightVector, tmax, 2);
	if (shadowed) {
		hitValue *= 0.3;
	}

#ifdef DEBUG_GEOMETRY_IDX
	if (gl_GeometryIndexEXT < 2) {
		hitValue = vec3(1, 0, 0);
	}
	else if (gl_GeometryIndexEXT < 4) {
		hitValue = vec3(0, 1, 0);
	}
	else if (gl_GeometryIndexEXT < 6) {
		hitValue = vec3(0, 0, 1);
	}
	else if (gl_GeometryIndexEXT < 8) {
		hitValue = vec3(1, 1, 0);
	}
	else if (gl_GeometryIndexEXT < 10) {
		hitValue = vec3(1, 0, 1);
	}
	else if (gl_GeometryIndexEXT < 12) {
		hitValue = vec3(0, 1, 1);
	}
#endif
}
