# Edge Target Resource Profile

This profile is for dissertation and portfolio discussion. It is a resource-fit
analysis, not measured board deployment.

## Current Model Artifact

| Item | Value |
| --- | ---: |
| Selected Pro model | `pro_full_bvp_hr_eda_temp` |
| TFLite type | INT8 input / INT8 output |
| TFLite flatbuffer | about 282 KB |
| Tensor proxy from desktop TFLite allocation | about 263 KB |
| PC TFLite p50 latency | about 11 ms per 5-second window |
| Input window | 160 samples, 5 seconds at 32 Hz |

## Candidate MCU Class

| Target | Why It Is Relevant | Fit Notes |
| --- | --- | --- |
| PSoC Edge E84 AI Kit | Arm Cortex-M55 with Helium DSP, Ethos-U55 NPU, and Cortex-M33 low-power domain. The kit also includes external QSPI flash and Octal RAM. | Current flatbuffer and tensor proxy fit comfortably in the kit-class memory envelope. TFLite Micro operator support still needs a build test. |
| PSoC 6 AI Evaluation Kit | Cortex-M4/M0+ class baseline with smaller compute resources. | Useful as a lower-end comparison if operator support and arena memory are acceptable. |
| Cortex-M55 + Ethos-U55 class devices | Natural target for INT8 neural-network execution. | Best match for the model direction, especially after operator-set simplification or vendor compiler conversion. |

## PSoC Edge E84 Evaluation Plan

1. Convert `seizure_model.tflite` to `model_data.h`.
2. Build `main.cpp` with the selected TFLite Micro runtime.
3. Start with the CPU path and static tensor arena.
4. Inspect unsupported operators from the build.
5. Simplify the model graph if required by replacing shape-heavy operations.
6. Measure board latency with GPIO toggling around `interpreter.Invoke()`.
7. Report measured latency only after board execution is confirmed.

## Current Operator Set to Validate

`ADD`, `CONCATENATION`, `DEQUANTIZE`, `EXPAND_DIMS`, `FULLY_CONNECTED`,
`GATHER`, `LOGISTIC`, `MEAN`, `MUL`, `PACK`, `QUANTIZE`, `REDUCE_MAX`,
`RESHAPE`, `SHAPE`, `STRIDED_SLICE`, `TRANSPOSE`.

The shape and transpose operations are the main risk for a small TFLite Micro
resolver. If the board build fails, the correct fix is graph simplification, not
claiming deployment.

## Resume-Safe Wording

"Built an INT8 TFLite deployment package and static-memory TFLite Micro harness
for resource-constrained wearable seizure-event detection; performed software
resource profiling and prepared PSoC Edge E84 board-porting artifacts."
