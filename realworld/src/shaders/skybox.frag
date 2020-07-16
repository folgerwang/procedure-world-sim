#version 450
#extension GL_ARB_separate_shader_objects : enable

#define RENDER_SUN 1
#define RENDER_SUNLIGHT_SCATTERING  1
const int iSteps = 16;
const int jSteps = 8;

#include "global_definition.glsl.h"
#include "sun.glsl.h"
#include "sunlight_scattering.glsl.h"

#if 0 
// Pretty standard way to make a sky. 
vec3 getSky(in vec3 ro, in vec3 rd, vec3 lp, float t){

	float sun = max(dot(rd, normalize(lp - ro)), 0.0); // Sun strength.
	float horiz = pow(1.0-max(rd.y, 0.0), 3.)*.25; // Horizon strength.
	
	// The blueish sky color. Tinging the sky redish around the sun. 		
	vec3 col = mix(vec3(.25, .5, 1)*.8, vec3(.8, .75, .7), sun*.5);//.zyx;
    // Mixing in the sun color near the horizon.
	col = mix(col, vec3(1, .5, .25), horiz);
    
    //vec3 col = mix(vec3(1, .7, .55), vec3(.6, .5, .55), rd.y*.5 + .5);
    
    // Sun. I can thank IQ for this tidbit. Producing the sun with three
    // layers, rather than just the one. Much better.
	col += 0.25*vec3(1, .7, .4)*pow(sun, 5.0);
	col += 0.25*vec3(1, .8, .6)*pow(sun, 64.0);
	col += 0.15*vec3(1, .9, .7)*max(pow(sun, 512.0), .25);
    
    // Add a touch of speckle. For better or worse, I find it breaks the smooth gradient up a little.
    col = clamp(col + hash31(rd)*0.04 - 0.02, 0., 1.);
    
    //return col; // Clear sky day. Much easier. :)
	
	// Clouds. Render some 3D clouds far off in the distance. I've made them sparse and wispy,
    // since we're in the desert, and all that.
    
    // Mapping some 2D clouds to a plane to save some calculations. Raytrace to a plane above, which
    // is pretty simple, but it's good to have Dave's, IQ's, etc, code to refer to as backup.
    
    // Give the direction ray a bit of concavity for some fake global curvature - My own dodgy addition. :)
    //rd = normalize(vec3(rd.xy, sqrt(rd.z*rd.z + dot(rd.xy, rd.xy)*.1) ));
 
    // If we haven't hit anything and are above the horizon point (there for completeness), render the sky.
    
    // Raytrace to a plane above the scene.
    float tt = (1000. - ro.y)/(rd.y + .2);
 
    if(t>=FAR && tt>0.){

        // Trace out a very small number of layers. In fact, there are so few layer that the following
        // is almost pointless, but I've left it in.
        vec4 cl = cloudLayers(ro + rd*tt, rd, lp, FAR*3.);
        vec3 clouds = cl.xyz;

        // Mix in the clouds.
        col = mix( col, vec3(1), clouds); // *clamp(rd.y*4. + .0, 0., 1.)
    }
    
    return col;
}
#endif

layout(set = VIEW_PARAMS_SET, binding = VIEW_CONSTANT_INDEX) uniform ViewUniformBufferObject {
    ViewParams view_params;
};

layout(set = PBR_GLOBAL_PARAMS_SET, binding = BASE_COLOR_TEX_INDEX) uniform samplerCube skybox_tex;

layout(location = 0) in VsPsData {
    vec3 vertex_position;
} in_data;

layout(push_constant) uniform SunSkyUniformBufferObject {
    SunSkyParams sun_sky_params;
};

layout(location = 0) out vec4 outColor;

void main() {
    vec3 view_dir = normalize(in_data.vertex_position - view_params.camera_pos.xyz);

    vec3 color = vec3(0.0, 0.0, 0.0);
    vec3 sun_pos = normalize(sun_sky_params.sun_pos);

#if RENDER_SUNLIGHT_SCATTERING
  color += atmosphere(view_dir,                       // normalized ray direction
                      vec3(0, 6371e3, 0) + view_params.camera_pos.xyz,             // ray origin
                      sun_pos,                        // position of the sun
                      22.0,                           // intensity of the sun
                      6371e3,                         // radius of the planet in meters
                      6471e3,                         // radius of the atmosphere in meters
                      vec3(5.5e-6, 13.0e-6, 22.4e-6), // Rayleigh scattering coefficient
                      21e-6,                          // Mie scattering coefficient
                      8e3,                            // Rayleigh scale height
                      1.2e3,                          // Mie scale height
                      0.758                           // Mie preferred scattering direction
                      );
#endif

#if RENDER_SUN
  color += renderSun(view_dir, sun_pos);
#endif


    outColor = vec4(color/*textureLod(skybox_tex, view_dir, 0).xyz*/, 1.0);
}