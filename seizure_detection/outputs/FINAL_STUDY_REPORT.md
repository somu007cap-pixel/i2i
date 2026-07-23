# Final Study Report

This report summarizes the study as a patient-held-out feasibility analysis, not a clinically validated detector.

## Submission Verdict

- The current results are suitable for a rigorous prototype submission when framed as weak-label wearable seizure-event detection.
- The strongest defensible claim is not that every Pro run dominates the Standard tier, but that add-on physiological sensing can improve recall, ranking, or event detection under some operating points while preserving edge-feasible model size.
- The remaining limitation is scientific rather than cosmetic: the source labels provide onset timestamps without exact seizure offsets.

## Active Promoted TSMixer Result

| Tier | Variant | AUC | PR-AUC | Recall | Event sensitivity | Alert events | Alert false alerts/hour | Window false alarms/hour |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Standard | `standard_acc_temp` | 0.4259 | 0.0010 | 0.0276 | 0.1667 [0.0301, 0.5635] | 0/6 | 0.82 | 26.33 |
| Pro | `pro_full_bvp_hr_eda_temp` | 0.5485 | 0.0072 | 0.1614 | 0.5000 [0.1876, 0.8124] | 2/6 | 0.91 | 18.97 |

## Active Result Trade-Off

- Pro AUC change: 0.1226; recall change: 0.1338; event-sensitivity change: 0.3333; false-alarms/hour change: -7.36.
- Interpretation: the add-on configuration should be presented as an event-sensitivity and ranking trade-off, not as uniformly better across all metrics.

Raw accuracy is intentionally not used as a primary success metric because the all-normal baseline is very high under rare seizure windows.

## Label Metadata

- Label rows scanned: 373.
- Rows with usable onset-offset fields: 0.
- Source labels are treated as onset-derived weak labels; exact seizure offsets are not invented.

## Event-Level Post-Processing

- Window scores are converted into alert episodes using a validation-selected refractory interval, so repeated high-score windows in the same episode are not counted as separate user-facing alarms.
- Alert-level false alarms/hour is reported separately from raw window false alarms/hour.

## Robustness Groups

| Group | Runs | Pro AUC | Pro recall | Pro event sensitivity | Pro FA/hr | Pro AUC gain | Pro recall gain | Pro event-sensitivity gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| label_policy::booster_tuned_patch8_base_eda_boost_ppg_temp | 3 | 0.5996+-0.0701 | 0.1600+-0.1091 | 0.6111+-0.0962 | 18.44+-19.48 | -0.0116+-0.0214 | 0.0003+-0.0061 | 0.0000+-0.0000 |
| label_policy::patch8_base_temp_boost_ppg_eda | 3 | 0.5480+-0.0495 | 0.1377+-0.0909 | 0.5741+-0.0849 | 16.23+-7.75 | 0.0018+-0.0027 | 0.0023+-0.0075 | 0.0000+-0.0000 |
| seed_variability::booster_tuned_patch8_base_eda_boost_ppg_temp | 3 | 0.5461+-0.0150 | 0.1026+-0.0618 | 0.5000+-0.0000 | 20.40+-13.91 | 0.1044+-0.0311 | 0.0856+-0.0512 | 0.1667+-0.2887 |
| seed_variability::patch8_base_temp_boost_ppg_eda | 3 | 0.5790+-0.0536 | 0.1826+-0.1505 | 0.6667+-0.1667 | 7.96+-2.81 | 0.0103+-0.0106 | -0.0092+-0.0362 | 0.0000+-0.0000 |

Note: label-policy groups can change the number and duration of positive events, so event sensitivity is more comparable than raw detected-event counts across those groups.

## Strongest Baseline Reference

- `cnn_lstm`: AUC 0.6064, recall 0.0828, events 5/6, false alarms/hour 9.46.

## Same Operating-Point Baselines

| Model | Budget FA/hr | Recall | Events | Alert false alerts/hour |
| --- | ---: | ---: | ---: | ---: |
| autoencoder | 1.0 | 0.0149 | 2/6 | 0.22 |
| autoencoder | 5.0 | 0.0552 | 4/6 | 0.95 |
| autoencoder | 10.0 | 0.0807 | 4/6 | 2.00 |
| autoencoder | 25.0 | 0.1656 | 4/6 | 5.48 |
| autoencoder | 50.0 | 0.2208 | 5/6 | 9.55 |
| cnn_lstm | 1.0 | 0.0149 | 3/6 | 0.42 |
| cnn_lstm | 5.0 | 0.0510 | 4/6 | 0.82 |
| cnn_lstm | 10.0 | 0.0828 | 5/6 | 1.09 |
| cnn_lstm | 25.0 | 0.1529 | 5/6 | 1.63 |
| cnn_lstm | 50.0 | 0.2038 | 5/6 | 2.16 |
| compact_transformer | 1.0 | 0.0127 | 2/6 | 0.18 |
| compact_transformer | 5.0 | 0.0297 | 3/6 | 0.49 |
| compact_transformer | 10.0 | 0.0425 | 4/6 | 0.66 |
| compact_transformer | 25.0 | 0.0616 | 4/6 | 1.00 |
| compact_transformer | 50.0 | 0.1019 | 4/6 | 1.34 |
| lstm | 1.0 | 0.0170 | 2/6 | 0.18 |
| lstm | 5.0 | 0.0552 | 3/6 | 0.60 |
| lstm | 10.0 | 0.0913 | 4/6 | 0.96 |
| lstm | 25.0 | 0.1826 | 5/6 | 1.95 |
| lstm | 50.0 | 0.1932 | 5/6 | 2.32 |

## Failure Notes

- Standard missed 5 event(s), across sessions: 1598624269_A01EE3, 1601509460_A02294, 1604525961_A02294, 1610296047_A01EEC, 1617460460_A01EEC
- Pro missed 3 event(s), across sessions: 1601509460_A02294, 1610296047_A01EEC, 1617460460_A01EEC

## Interpretation Guardrails

- Treat this as a feasibility study under weak labels.
- Compare event sensitivity and false alarms/hour before raw accuracy.
- Report TFLite results as software feasibility, not physical MCU deployment.
- Firmware-facing PSoC Edge E84 artifacts are prepared under `seizure_detection/firmware_c/`; measured board latency should be added only after a successful hardware run.

## Firmware Deployment Package

- INT8 TFLite model header: `seizure_detection/firmware_c/model_data.h`.
- Static-memory TFLite Micro harness: `seizure_detection/firmware_c/main.cpp`.
- Streaming DSP ring buffer and online normalization helpers: `seizure_detection/firmware_c/sensor_dsp.c` and `.h`.
- Target profile and PSoC Edge E84 porting plan: `seizure_detection/firmware_c/edge_target_profile.md`.

## Figure Package

- Dissertation figures are generated under `seizure_detection/outputs/dissertation_figures/`.
- Use `README_FIGURES.md` in that folder as the placement guide for report chapters.
