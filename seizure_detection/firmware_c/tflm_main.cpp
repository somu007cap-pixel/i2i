/*
 * tflm_main.cpp - Bare-Metal / RTOS TensorFlow Lite for Microcontrollers (TFLM) Execution Harness
 * Demonstrates static memory allocation (zero malloc/heap) and INT8 hardware inference.
 * Target Hardware: ARM Cortex-M4 / Cortex-M55 (Infineon PSoC 6 / STM32 / NXP)
 */

#include <cstdio>
#include <cstdint>
#include "sensor_dsp.h"

// Note: In an MCU build environment, these headers are included from tensorflow/lite/micro/
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// #include "tensorflow/lite/schema/schema_generated.h"

// Mock definitions for standalone compilation & architectural demonstration:
namespace tflite {
    struct Model {
        static const Model* GetModel(const void* ptr) { return reinterpret_cast<const Model*>(ptr); }
    };
    enum TfLiteStatus { kTfLiteOk = 0, kTfLiteError = 1 };
}

// Model weights C-array exported via xxd -i seizure_model_standard.tflite
// Simulated model buffer footprint for architecture proof:
alignas(16) static const uint8_t g_seizure_model_data[288 * 1024] = { 0x20, 0x00, 0x00, 0x00 };
static const size_t g_seizure_model_size = sizeof(g_seizure_model_data);

// Tensor Arena: 64 KB Static Memory Buffer in SRAM (No malloc/free)
constexpr size_t kTensorArenaSize = 64 * 1024;
alignas(16) static uint8_t g_tensor_arena[kTensorArenaSize];

// System State
static SensorRingBuffer g_sensor_ring_buffer;
static NormalizationParams g_norm_params = {
    .mean = {0.0f, 0.0f, 9.81f, 32.0f, 0.0f, 75.0f, 0.5f},
    .std  = {1.5f, 1.5f, 2.0f,   1.5f,  0.2f, 10.0f, 0.3f}
};

/**
 * Firmware Initialization Phase (Executed once at MCU boot)
 */
extern "C" void firmware_seizure_detection_init(void) {
    printf("[FIRMWARE INIT] Initializing Wearable Seizure Detector...\n");
    printf("[FIRMWARE INIT] Model Size: %zu KB in Flash\n", g_seizure_model_size / 1024);
    printf("[FIRMWARE INIT] Tensor Arena: %zu KB Static SRAM allocated\n", kTensorArenaSize / 1024);

    sensor_ring_buffer_init(&g_sensor_ring_buffer);
    printf("[FIRMWARE INIT] Sensor Ring Buffer initialized (160 samples @ 32Hz).\n");
    printf("[FIRMWARE INIT] Ready for real-time sensor sampling.\n");
}

/**
 * Sensor Interrupt Handler Callback (Fired every 31.25 ms @ 32Hz sampling rate)
 */
extern "C" void firmware_on_sensor_sample_isr(const float raw_channels[NUM_PRO_CHANNELS]) {
    // 1. Push incoming sample into circular buffer
    sensor_ring_buffer_push(&g_sensor_ring_buffer, raw_channels);

    // 2. Execute inference only when a full 5-second window (160 samples) is ready
    if (g_sensor_ring_buffer.is_full && (g_sensor_ring_buffer.head_index % SENSOR_SAMPLING_RATE_HZ == 0)) {
        static float window_buf[WINDOW_SAMPLES][NUM_PRO_CHANNELS];
        static int8_t int8_input_tensor[WINDOW_SAMPLES * NUM_STANDARD_CHANNELS];

        // Extract window from ring buffer
        sensor_ring_buffer_get_window(&g_sensor_ring_buffer, window_buf);

        // Normalize window in-place
        sensor_dsp_normalize_window(window_buf, NUM_STANDARD_CHANNELS, &g_norm_params);

        // Quantize normalized Float32 window to INT8 input tensor
        float scale = 0.035f;
        int32_t zero_point = 0;
        sensor_dsp_quantize_int8(
            (const float*)window_buf,
            int8_input_tensor,
            WINDOW_SAMPLES * NUM_STANDARD_CHANNELS,
            scale,
            zero_point
        );

        // Perform TFLite Micro inference invoke (Simulated execution tick)
        // tflite::TfLiteStatus invoke_status = interpreter->Invoke();
        
        // Output event logging
        // printf("[FIRMWARE INFERENCE] TFLM Invoked. Score INT8 -> Probability.\n");
    }
}

int main(void) {
    firmware_seizure_detection_init();

    // Simulate 10 seconds of 32Hz stream (320 ticks)
    float sample[NUM_PRO_CHANNELS] = {0.1f, 0.05f, 9.8f, 32.5f, 0.01f, 72.0f, 0.4f};
    for (int i = 0; i < 320; i++) {
        firmware_on_sensor_sample_isr(sample);
    }
    printf("[FIRMWARE BENCHMARK] Completed simulation run successfully.\n");
    return 0;
}
