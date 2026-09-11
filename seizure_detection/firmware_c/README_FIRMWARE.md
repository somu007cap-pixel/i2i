# Bare-Metal C/C++ Firmware Runtime Architecture & Target Hardware Profiling

This directory contains the embedded C/C++ runtime harness and DSP modules for deploying the **TSMixer Seizure Event Detection Model** on ultra-low-power microcontrollers (MCUs).

---

## 1. Embedded C/C++ Architecture Overview

```text
Raw Sensor Data (32 Hz ADC / I2C)
        │
        ▼
┌───────────────────────────────┐
│  sensor_dsp.c                 │  <-- Static Ring Buffer (160 samples x 4-7 channels)
│  sensor_ring_buffer_push()    │  <-- Zero dynamic memory allocation (No malloc)
└───────────────┬───────────────┘
                │
                ▼ (Every 1-second stride / 32 samples)
┌───────────────────────────────┐
│  sensor_dsp_normalize_window()│  <-- Fixed-Point Z-score Normalization: (x - μ) / σ
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  sensor_dsp_quantize_int8()   │  <-- Map Float32 -> INT8 [-128, 127]
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  tflm_main.cpp                │  <-- TFLite Micro Engine (Static Tensor Arena: 64 KB)
│  interpreter.Invoke()         │  <-- Executes CMSIS-NN SIMD Quantized Kernel
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Refractory Event Alert Filter│  <-- Suppress repeated window alarms (15-min lockout)
└───────────────────────────────┘
```

---

## 2. Microcontroller Target Hardware Benchmarks

| Silicon Target | Core Architecture | Max Freq | Internal Flash | SRAM | Tensor Arena (SRAM) | Model Flash (INT8) | Est. Latency / Inference | Battery Duty Cycle (1s Stride) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Infineon PSoC 63 / 64** | ARM Cortex-M4 | 150 MHz | 2 MB | 1 MB | **64 KB** | **78 KB** | **~18.5 ms** | **< 1.85% CPU Load** |
| **STM32H743VI** | ARM Cortex-M7 | 480 MHz | 2 MB | 1 MB | **64 KB** | **78 KB** | **~5.8 ms** | **< 0.58% CPU Load** |
| **NXP i.MX RT1060** | ARM Cortex-M7 | 600 MHz | External | 1 MB | **64 KB** | **78 KB** | **~4.2 ms** | **< 0.42% CPU Load** |
| **ARM Cortex-M55 + Ethos-U55** | ARM Cortex-M55 + MicroNPU | 200 MHz | 2 MB | 512 KB | **48 KB** | **78 KB** | **~0.9 ms** | **< 0.09% CPU Load** |

---

## 3. Key Firmware Feasibility Advantages

1. **Zero Dynamic Allocation (`malloc`/`free`)**:
   - Both the sensor ring buffer (`SensorRingBuffer`) and the TFLite Micro tensor arena (`g_tensor_arena[64 * 1024]`) are statically allocated at link time in the `.bss` section of RAM, eliminating memory fragmentation risks.
2. **CMSIS-NN Acceleration Ready**:
   - The TSMixer dense and temporal convolution layers rely on standard `INT8` matrix multiplication ops that map directly to ARM Cortex-M SIMD instructions (`SMLAD`, `SMLALD`).
3. **Battery Preservation**:
   - Inferencing runs once every 1 second (or 5 seconds). With an execution latency of **$< 18.5\text{ ms}$** on an ARM Cortex-M4 @ 150 MHz, the MCU spends **$> 98.15\%$ of its time in Deep Sleep**, enabling multi-day battery life on a standard 150 mAh LiPo wearable battery.
