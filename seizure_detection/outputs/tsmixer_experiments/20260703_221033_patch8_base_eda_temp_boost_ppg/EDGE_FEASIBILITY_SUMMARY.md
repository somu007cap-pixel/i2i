# Edge Feasibility Summary

Scope: software-only feasibility analysis for resource-constrained wearable GTC seizure-event detection. This is not a claim of measured deployment on a physical MCU.

## Source Detection Run

- Detection timestamp: 2026-07-04T04:47:01.345178
- Pipeline version: edge_safe_patched_dual_stream_tsmixer_v15_dsp_session_norm
- Selected sessions: 146
- Requested max sessions: 146
- Window length: 5.0 seconds

## Current Artifacts

| Tier | Selected variant | Keras size | Parameters | Float32 TFLite | Dynamic-range TFLite | Current calibrated TFLite |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Standard | `standard_acc_only` | 2184.6 KB | 155,633 | 661.3 KB | 245.7 KB | 278.3 KB |
| Pro | `pro_full_bvp_hr_eda_temp` | 2184.6 KB | 155,633 | 663.0 KB | 247.4 KB | 282.0 KB |

## Profiling Notes

| Tier | Current calibrated TFLite tensor proxy | PC TFLite median latency | Input/output dtype |
| --- | ---: | ---: | --- |
| Standard | 260.4 KB | 7.3474 ms | int8 -> int8 |
| Pro | 261.0 KB | 7.3522 ms | int8 -> int8 |

The latency values are single-thread local TensorFlow Lite Interpreter measurements on this PC. They are useful for software profiling, but they are not Cortex-M hardware latency measurements.

## Submission-Safe Interpretation

The current work supports compact-model and resource-constrained feasibility discussion. The current calibrated TFLite files expose int8 input and output tensors. They are suitable for software-level full-integer quantization analysis, subject to operator support on the target runtime.

Next edge step: add representative-dataset full INT8 conversion, operator-resolver review, and tighter memory profiling.
