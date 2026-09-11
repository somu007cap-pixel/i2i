/*
 * sensor_dsp.h - Embedded C Sensor DSP & Ring Buffer Module for Wearable Seizure Detection
 * Target: Ultra-Low-Power Microcontrollers (e.g., ARM Cortex-M4 / Infineon PSoC 6 / STM32)
 * Memory Allocation: Static (Zero Heap / No malloc)
 */

#ifndef SENSOR_DSP_H
#define SENSOR_DSP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define SENSOR_SAMPLING_RATE_HZ 32
#define WINDOW_SECONDS 5
#define WINDOW_SAMPLES (SENSOR_SAMPLING_RATE_HZ * WINDOW_SECONDS) // 160 samples
#define NUM_STANDARD_CHANNELS 4  // ACC_x, ACC_y, ACC_z, TEMP
#define NUM_PRO_CHANNELS 7       // ACC_x, ACC_y, ACC_z, TEMP, BVP, HR, EDA

typedef struct {
    float buffer[WINDOW_SAMPLES][NUM_PRO_CHANNELS];
    uint16_t head_index;
    uint16_t sample_count;
    bool is_full;
} SensorRingBuffer;

typedef struct {
    float mean[NUM_PRO_CHANNELS];
    float std[NUM_PRO_CHANNELS];
} NormalizationParams;

/**
 * Initialize the sensor ring buffer with zero dynamic memory allocation.
 */
void sensor_ring_buffer_init(SensorRingBuffer *rb);

/**
 * Push a new multi-channel sensor frame (sampled @ 32Hz) into the ring buffer.
 */
void sensor_ring_buffer_push(SensorRingBuffer *rb, const float frame[NUM_PRO_CHANNELS]);

/**
 * Extract the continuous 160-sample 5-second window from the ring buffer into an output destination array.
 */
void sensor_ring_buffer_get_window(const SensorRingBuffer *rb, float out_window[WINDOW_SAMPLES][NUM_PRO_CHANNELS]);

/**
 * Perform online Z-score normalization on the window buffer using static normalization parameters:
 * norm_x = (x - mean) / (std + eps)
 */
void sensor_dsp_normalize_window(
    float window[WINDOW_SAMPLES][NUM_PRO_CHANNELS],
    uint8_t num_channels,
    const NormalizationParams *params
);

/**
 * Quantize float32 window buffer to int8 array for TFLite Micro INT8 input tensor:
 * int8_val = clamp(round(float_val / scale) + zero_point, -128, 127)
 */
void sensor_dsp_quantize_int8(
    const float *in_float_window,
    int8_t *out_int8_tensor,
    size_t total_elements,
    float quant_scale,
    int32_t quant_zero_point
);

#ifdef __cplusplus
}
#endif

#endif // SENSOR_DSP_H
