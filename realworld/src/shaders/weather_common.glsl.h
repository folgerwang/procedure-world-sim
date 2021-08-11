#define kRealWorldAirflowMaxHeight        12000.0f
#define kDegreeDecreasePerKm              (6.5f / 1000.0f)

#define kAirflowMaxHeight                 (kRealWorldAirflowMaxHeight / kDegreeDecreasePerKm / 1000.0f * 6.5f)
#define kAirflowLowHeight                 -600.0f
#define kAirflowHeightRange               (kAirflowMaxHeight - kAirflowLowHeight)

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
#define kAirflowHeightMinMaxRatio         (kAirflowHeightRange / (kAirflowBufferHeight / 2 * kMinSampleHeight) - 1.0f)
#define kAirflowHeightFactorA             ((kAirflowHeightMinMaxRatio - 1.0f) * kAirflowBufferHeight * kMinSampleHeight / 2)
#define kAirflowHeightFactorB             (kMinSampleHeight * kAirflowBufferHeight)
#define kAirflowHeightFactorInvA          (1.0f / kAirflowHeightFactorA)
#define kAirflowHeightFactorBInv2A        (kAirflowHeightFactorB / 2.0f * kAirflowHeightFactorInvA)
#define kAirflowHeightFactorSqrBInv2A     (kAirflowHeightFactorBInv2A * kAirflowHeightFactorBInv2A)

#define kAirflowToHeightParamsX           log2(1.0f + kAirflowHeightRange)
#define kAirflowToHeightParamsY           (-1.0f + kAirflowLowHeight)
#define kAirflowFromHeightParamsX         (1.0f / log2(kAirflowMaxHeight - kAirflowLowHeight + 1.0f))

#define USE_LINEAR_HEIGHT_SAMPLE          1

#define kNodeLeft                         0x00      // -x
#define kNodeRight                        0x01      // +x
#define kNodeBack                         0x02      // -y
#define kNodeFront                        0x03      // +y
#define kNodeBelow                        0x04      // -z
#define kNodeAbove                        0x05      // +z

float calculateAirflowSampleToHeight(float uvw_w) {
    float sample_h = (kAirflowHeightFactorB + kAirflowHeightFactorA * uvw_w) * uvw_w;
    return sample_h;
}

float calculateAirflowSampleToDeltaHeight(float uvw_w) {
    float delta_h = kAirflowHeightFactorB + 2.0f * kAirflowHeightFactorA * uvw_w;
    return delta_h / kAirflowBufferHeight;
}

float calculateAirflowSampleFromHeight(float sample_h) {
    float delta = max(kAirflowHeightFactorSqrBInv2A + sample_h * kAirflowHeightFactorInvA, 0.0f);
    float uvw_w = -kAirflowHeightFactorBInv2A + sqrt(delta);
    return uvw_w;
}

float getReferenceDegree(float sea_level_temp_c, float altitude) {

    float sea_level_temp = sea_level_temp_c;
    float temperature = clamp(sea_level_temp - altitude * kDegreeDecreasePerKm,
                              -kTemperaturePositiveOffset,
                              kTemperatureDenormalizer - kTemperaturePositiveOffset);
    return temperature;
}

float getSampleToHeight(float uvw_w) {
#if USE_LINEAR_HEIGHT_SAMPLE
    return calculateAirflowSampleToHeight(uvw_w);
#else
    return exp2(uvw_w *kAirflowToHeightParamsX) + kAirflowToHeightParamsY;
#endif
}

float getSampleToDeltaHeight(float uvw_w) {
#if USE_LINEAR_HEIGHT_SAMPLE
    return calculateAirflowSampleToDeltaHeight(uvw_w);
#else
    return exp2(uvw_w * kAirflowToHeightParamsX) * log(2.0f) * kAirflowToHeightParamsX / kAirflowBufferHeight;
#endif
}

float getHeightToSample(float sample_h) {
#if USE_LINEAR_HEIGHT_SAMPLE
    return calculateAirflowSampleFromHeight(sample_h);
#else
    return log2(max((sample_h - kAirflowLowHeight), 0.0f) + 1.0f) * kAirflowFromHeightParamsX;
#endif
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

