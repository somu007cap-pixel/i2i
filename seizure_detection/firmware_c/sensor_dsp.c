/*
 * sensor_dsp.c - Embedded C Sensor DSP & Ring Buffer Implementation
 * Bare-metal / RTOS safe, zero dynamic heap allocation.
 */

#include "sensor_dsp.h"
#include <math.h>
#include <string.h>

#define EPSILON 1e-6f

void sensor_ring_buffer_init(SensorRingBuffer *rb) {
    if (!rb) return;
    memset(rb->buffer, 0, sizeof(rb->buffer));
    rb->head_index = 0;
    rb->sample_count = 0;
    rb->is_full = false;
}

void sensor_ring_buffer_push(SensorRingBuffer *rb, const float frame[NUM_PRO_CHANNELS]) {
    if (!rb || !frame) return;

    for (int c = 0; c < NUM_PRO_CHANNELS; c++) {
        rb->buffer[rb->head_index][c] = frame[c];
    }

    rb->head_index = (rb->head_index + 1) % WINDOW_SAMPLES;
    if (rb->sample_count < WINDOW_SAMPLES) {
        rb->sample_count++;
    }
    if (rb->sample_count == WINDOW_SAMPLES) {
        rb->is_full = true;
    }
}

void sensor_ring_buffer_get_window(const SensorRingBuffer *rb, float out_window[WINDOW_SAMPLES][NUM_PRO_CHANNELS]) {
    if (!rb || !out_window) return;

    uint16_t start_idx = rb->is_full ? rb->head_index : 0;
    for (uint16_t i = 0; i < rb->sample_count; i++) {
        uint16_t curr_idx = (start_idx + i) % WINDOW_SAMPLES;
        for (int c = 0; c < NUM_PRO_CHANNELS; c++) {
            out_window[i][c] = rb->buffer[curr_idx][c];
        }
    }
}

void sensor_dsp_normalize_window(
    float window[WINDOW_SAMPLES][NUM_PRO_CHANNELS],
    uint8_t num_channels,
    const NormalizationParams *params
) {
    if (!window || !params) return;

    for (uint16_t i = 0; i < WINDOW_SAMPLES; i++) {
        for (uint8_t c = 0; c < num_channels; c++) {
            float std_val = params->std[c] > EPSILON ? params->std[c] : 1.0f;
            window[i][c] = (window[i][c] - params->mean[c]) / std_val;
        }
    }
}

void sensor_dsp_quantize_int8(
    const float *in_float_window,
    int8_t *out_int8_tensor,
    size_t total_elements,
    float quant_scale,
    int32_t quant_zero_point
) {
    if (!in_float_window || !out_int8_tensor || quant_scale <= 0.0f) return;

    for (size_t i = 0; i < total_elements; i++) {
        float val = in_float_window[i];
        int32_t q = (int32_t)lroundf(val / quant_scale) + quant_zero_point;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        out_int8_tensor[i] = (int8_t)q;
    }
}
