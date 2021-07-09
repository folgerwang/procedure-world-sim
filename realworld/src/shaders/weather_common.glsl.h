#define kAirflowMaxHeight                 12000.0f
#define kAirflowLowHeight                 -100.0f

#define kDegreeDecreasePerKm              (10.0f / 1000.0f)
#define kAbsoluteDegreeFactor             273.15f
#define kTemperatureDiffPositiveOffset    273.15f

float getReferenceDegree(float sea_level_temp_c, float altitude) {

    float sea_level_temp = sea_level_temp_c + kAbsoluteDegreeFactor;
    float temperature = sea_level_temp - altitude * kDegreeDecreasePerKm;
    return temperature;
}

float toCelsius(float absolute_degree) {
    return absolute_degree - kAbsoluteDegreeFactor;
}

float getSampleHeight(float uvw_w, float bias, vec2 height_params) {
    return exp2((uvw_w + bias) * height_params.x) + height_params.y;
}

vec2 getPositionWSXy(vec2 uvw_xy, vec2 world_min, vec2 world_range) {
    return uvw_xy * world_range + world_min;
}