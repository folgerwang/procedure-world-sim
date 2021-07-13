#define kAirflowMaxHeight                 12000.0f
#define kAirflowLowHeight                 -100.0f

#define kDegreeDecreasePerKm              (15.0f / 1000.0f)
#define kTemperaturePositiveOffset        78.0f
#define kTemperatureNormalizer            (1.0f / 128.0f)
#define kTemperatureDenormalizer          (1.0f / kTemperatureNormalizer)
#define kMoistureNormalizer               (1.0f / 8.0f)
#define kMoistureDenormalizer             (1.0f / kMoistureNormalizer)

#define kMaxTemperatureAdjustRange        8.0f
#define kMaxMoistureIntensity             2.0f
#define kMaxAirflowStrength               1.0f
#define kAirflowStrengthNormalizeScale    (1.0f / kMaxAirflowStrength)

#define kMaxTempMoistDiff                 4.0f
#define kTempMoistDiffToFloat             (kMaxTempMoistDiff / 65536.0f)

#define kMinSampleHeight                  4.0f

float getReferenceDegree(float sea_level_temp_c, float altitude) {

    float sea_level_temp = sea_level_temp_c;
    float temperature = clamp(sea_level_temp - altitude * kDegreeDecreasePerKm,
                              -kTemperaturePositiveOffset,
                              kTemperatureDenormalizer - kTemperaturePositiveOffset);
    return temperature;
}

float getSampleHeight(float uvw_w, float bias, vec2 height_params) {
    return exp2((uvw_w + bias) * height_params.x) + height_params.y;
}

vec2 getPositionWSXy(vec2 uvw_xy, vec2 world_min, vec2 world_range) {
    return uvw_xy * world_range + world_min;
}

float normalizeTemperature(float temp) {
    return (temp + kTemperaturePositiveOffset) * kTemperatureNormalizer;
}

float denormalizeTemperature(float normalized_temp) {
    return normalized_temp * kTemperatureDenormalizer - kTemperaturePositiveOffset;
}

vec2 denormalizeTemperature(vec2 normalized_temp) {
    return normalized_temp * kTemperatureDenormalizer - kTemperaturePositiveOffset;
}

float normalizeMoisture(float moist) {
    return moist * kMoistureNormalizer;
}

float denormalizeMoisture(float normalized_moist) {
    return normalized_moist * kMoistureDenormalizer;
}

float getNormalizedVectorLength(vec3 dir_vec) {
    return length(dir_vec) * kAirflowStrengthNormalizeScale;
}

float getPackedVectorLength(float packed_w) {
    return packed_w / kAirflowStrengthNormalizeScale;
}

