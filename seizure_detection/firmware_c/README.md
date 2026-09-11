# Firmware Deployment Package

This folder contains firmware-facing artifacts for the wearable seizure-event
detector. It is intended for PSoC Edge E84 / Cortex-M-class evaluation and
portfolio demonstration.

## What This Package Claims

- The model has been exported as an INT8 TensorFlow Lite flatbuffer.
- The code shows a static-memory TFLite Micro integration pattern.
- The DSP module shows fixed-size streaming buffers for a 5-second sensor
  inference window.
- The resource profile is a software feasibility estimate.

## What This Package Does Not Claim Yet

- It does not claim measured latency on the PSoC Edge E84 board.
- It does not claim completed clinical deployment.
- It does not claim final TFLite Micro operator compatibility until the model is
  compiled and tested with the selected TFLM/ModusToolbox runtime.

## Files

- `main.cpp`: Static-memory TFLite Micro inference harness.
- `sensor_dsp.h` / `sensor_dsp.c`: Ring-buffer and online normalization helpers.
- `tflite_to_c_array.py`: Converts a `.tflite` file into a C header.
- `edge_target_profile.md`: PSoC Edge E84 and MCU resource-fit notes.
- `model_data.h`: Generated model array, created from the current Pro TFLite file.

## Regenerate Model Header

From the repository root:

```powershell
.\.venv\Scripts\python.exe .\seizure_detection\firmware_c\tflite_to_c_array.py `
  .\seizure_detection\outputs\seizure_model.tflite `
  .\seizure_detection\firmware_c\model_data.h
```

## Current Model Contract

- Primary input: `int8`, shape `[1, 160, 3]`
- Secondary input: `int8`, shape `[1, 160, 4]`
- Output: `int8`, shape `[1, 1]`
- Window length: 5 seconds at 32 Hz
- Current Pro flatbuffer: about 282 KB

## PSoC Edge E84 Note

The PSoC Edge E84 family is relevant because it includes an Arm Cortex-M55 with
Helium DSP support, Ethos-U55 NPU support, and a Cortex-M33 low-power domain.
This package is structured so the same model can first be tested in a desktop
TFLite interpreter, then moved to a TFLite Micro or vendor AI toolchain flow.
