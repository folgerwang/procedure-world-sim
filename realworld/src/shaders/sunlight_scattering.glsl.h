#if RENDER_SUNLIGHT_SCATTERING

#define  ATMOSPHERE_USE_LUT     1

#ifndef _SUNLIGHT_SCATTERING_GLSL_H_
#define _SUNLIGHT_SCATTERING_GLSL_H_

layout(set = 0, binding = SRC_SCATTERING_LUT_INDEX) uniform sampler2D sky_scattering_lut;

// Ref : https://github.com/wwwtyro/glsl-atmosphere/blob/master/index.glsl

float rsi(vec3 r0, vec3 rd, float sr) {
  // Simplified ray-sphere intersection that assumes
  // the ray starts inside the sphere and that the
  // sphere is centered at the origin. Always intersects.
  float a = dot(rd, rd);
  float b = 2.0 * dot(rd, r0);
  float c = dot(r0, r0) - (sr * sr);
  return (-b + sqrt((b * b) - 4.0 * a * c)) / (2.0 * a);
}

vec3 atmosphere(
    vec3 r,
    vec3 r0,
    float cast_range,
    vec3 pSun,
    float iSun,
    float rPlanet,
    float rAtmos,
    vec3 kRlh,
    float kMie,
    float shRlh,
    float shMie,
    float g) {

  // Normalize the sun position.
  pSun = normalize(pSun);

  // Calculate the step size of the primary ray.
  float iStepSize = cast_range / float(iSteps);

  // Initialize the primary ray time.
  float iTime = 0.0;

  // Initialize accumulators for Rayleight and Mie scattering.
  vec3 totalRlh = vec3(0, 0, 0);
  vec3 totalMie = vec3(0, 0, 0);

  // Initialize optical depth accumulators for the primary ray.
  float iOdRlh = 0.0;
  float iOdMie = 0.0;

  // Calculate the Rayleigh and Mie phases.
  float mu = dot(r, pSun);
  float mumu = mu * mu;
  float gg = g * g;
  float pRlh = 3.0 / (16.0 * PI) * (1.0 + mumu);
  float pMie = 3.0 / (8.0 * PI) * ((1.0 - gg) * (mumu + 1.0)) /
      (pow(1.0 + gg - 2.0 * mu * g, 1.5) * (2.0 + gg));

  // Sample the primary ray.
  for (int i = 0; i < iSteps; i++) {
    // Calculate the primary ray sample position.
    vec3 iPos = r0 + r * (iTime + iStepSize * 0.5);

    // Calculate the height of the sample.
    float iHeight = length(iPos) - rPlanet;

    // Calculate the optical depth of the Rayleigh and Mie scattering for this
    // step.
    float odStepRlh = exp(-iHeight / shRlh) * iStepSize;
    float odStepMie = exp(-iHeight / shMie) * iStepSize;

    // Accumulate optical depth.
    iOdRlh += odStepRlh;
    iOdMie += odStepMie;

    // Calculate the step size of the secondary ray.
    float jStepSize = rsi(iPos, pSun, rAtmos) / float(jSteps);

    // Initialize the secondary ray time.
    float jTime = 0.0;

    // Initialize optical depth accumulators for the secondary ray.
    float jOdRlh = 0.0;
    float jOdMie = 0.0;

    // Sample the secondary ray.
    for (int j = 0; j < jSteps; j++) {
      // Calculate the secondary ray sample position.
      vec3 jPos = iPos + pSun * (jTime + jStepSize * 0.5);

      // Calculate the height of the sample.
      float jHeight = length(jPos) - rPlanet;

      // Accumulate the optical depth.
      jOdRlh += exp(-jHeight / shRlh) * jStepSize;
      jOdMie += exp(-jHeight / shMie) * jStepSize;

      // Increment the secondary ray time.
      jTime += jStepSize;
    }

    // Calculate attenuation.
    vec3 attn = exp(-(kMie * (iOdMie + jOdMie) + kRlh * (iOdRlh + jOdRlh)));

    // Accumulate scattering.
    totalRlh += odStepRlh * attn;
    totalMie += odStepMie * attn;

    // Increment the primary ray time.
    iTime += iStepSize;
  }

  // Calculate and return the final color.
  return iSun * max(pRlh * kRlh * totalRlh + pMie * kMie * totalMie, 0.0f);
}

vec4 atmosphereLut(
    vec3 r,
    vec3 r0,
    float cast_range,
    vec3 pSun,
    float iSun,
    float rPlanet,
    float rAtmos,
    vec3 kRlh,
    float kMie,
    float shRlh,
    float shMie,
    float g,
    vec2 visi,
    int iSteps) {

    // Normalize the sun position.
    pSun = normalize(pSun);

    // Calculate the step size of the primary ray.
    float log_cast_range = log2(cast_range);
    float step_log_size = log_cast_range / float(iSteps);

    // Initialize the primary ray time.
    float iTime = 0.0;

    // Initialize accumulators for Rayleight and Mie scattering.
    vec3 totalRlh = vec3(0, 0, 0);
    vec3 totalMie = vec3(0, 0, 0);

    // Initialize optical depth accumulators for the primary ray.
    float iOdRlh = 0.0;
    float iOdMie = 0.0;

    // Calculate the Rayleigh and Mie phases.
    float mu = dot(r, pSun);
    float mumu = mu * mu;
    float gg = g * g;
    float pRlh = 3.0 / (16.0 * PI) * (1.0 + mumu);
    float pMie = 3.0 / (8.0 * PI) * ((1.0 - gg) * (mumu + 1.0)) /
        (pow(1.0 + gg - 2.0 * mu * g, 1.5) * (2.0 + gg));

    const float inv_log_atmos_radius = 1.0f / log2(rAtmos + 1.0f);
    float start_dist = 0;
    float end_dist = 0;
    // Sample the primary ray.
    for (int i = 0; i < iSteps; i++) {
        end_dist = exp2(step_log_size * (i + 1));

        // Calculate the primary ray sample position.
        vec3 iPos = r0 + r * ((start_dist + end_dist) * 0.5);

        // Calculate the height of the sample.
        float sqr_sample_dist_to_core = dot(iPos, iPos);
        float sample_dist_to_core = sqrt(sqr_sample_dist_to_core);
        float r_height = max(sample_dist_to_core - rPlanet, 0.0f);
        float sample_dist = end_dist - start_dist;

        // Calculate the optical depth of the Rayleigh and Mie scattering for this
        // step.
        float odStepRlh = exp(-r_height / shRlh) * sample_dist;
        float odStepMie = exp(-r_height / shMie) * sample_dist;

        // Accumulate optical depth.
        iOdRlh += odStepRlh;
        iOdMie += odStepMie;

        float dist_to_line_middle = dot(iPos, pSun);
        float sqr_dist_to_line_middle =
            dist_to_line_middle * dist_to_line_middle;
        float sqr_perpendicular_dist_to_core =
            sqr_sample_dist_to_core - sqr_dist_to_line_middle;
        float perpendicular_dist_to_core =
            sqrt(sqr_perpendicular_dist_to_core);
        float whole_line_length =
            2.0f * sqrt(rAtmos * rAtmos - sqr_perpendicular_dist_to_core);
        float sample_length = rsi(iPos, pSun, rAtmos);

        vec2 lut_uv;
        lut_uv.x = perpendicular_dist_to_core / rAtmos;
        lut_uv.y = sample_length / whole_line_length;

        vec2 lut_value = exp2(texture(sky_scattering_lut, lut_uv).xy);
        float jOdRlh = lut_value.x;
        float jOdMie = lut_value.y;

        // Calculate attenuation.
        vec3 attn = exp(-(kMie * (iOdMie + jOdMie) + kRlh * (iOdRlh + jOdRlh)));

        // Accumulate scattering.
        totalRlh += odStepRlh * attn;
        totalMie += odStepMie * attn;

        start_dist = end_dist;
    }

    // Calculate and return the final color.
    return vec4(iSun * visi.x *
                    max(pRlh * kRlh * totalRlh + pMie * kMie * totalMie, 0.0f),
                visi.y);
}

#endif // _SUNLIGHT_SCATTERING_GLSL_H_

#endif // RENDER_SUNLIGHT_SCATTERING
