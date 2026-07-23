# Modern Method Coverage

The study evaluates modern lightweight time-series approaches that are appropriate for Empatica wearable data and weak onset-only seizure labels.

## Evaluated Method Families

| Method family | Implementation in this project | Status | Current result |
| --- | --- | --- | --- |
| Patched channel-independent TSMixer / TTM-style mixer | Dual-stream TSMixer Standard/Pro models | evaluated | Standard AUC 0.4259, events 1/6; Pro AUC 0.5485, events 3/6 |
| Event-level alert post-processing | Validation-selected refractory interval over detector scores | evaluated | Standard alert FA/hr 0.82; Pro alert FA/hr 0.91 |
| Patient-held-out weak-label robustness | Repeated seed and label-duration sensitivity studies | evaluated | Aggregated in FINAL_STUDY_REPORT.md |
| Firmware-facing TinyML deployment package | INT8 model header, static TFLite Micro harness, and streaming DSP ring buffer | prepared | Prepared under seizure_detection/firmware_c; board latency not yet claimed |
| Reconstruction-error anomaly detector | autoencoder | evaluated | AUC 0.5791, recall 0.0807, events 4/6, alert FA/hr 2.00 |
| CNN-LSTM temporal feature model | cnn_lstm | evaluated | AUC 0.6064, recall 0.0828, events 5/6, alert FA/hr 1.09 |
| Compact transformer baseline | compact_transformer | evaluated | AUC 0.5565, recall 0.0425, events 4/6, alert FA/hr 0.66 |
| Recurrent sequence model | lstm | evaluated | AUC 0.4896, recall 0.0913, events 4/6, alert FA/hr 0.96 |

## Why This Is Not Just a Single Lucky Run

- The main claim is based on aggregate behavior under patient-held-out evaluation.
- The run includes label-policy sensitivity because exact offsets are not available.
- The detector is compared with LSTM, CNN-LSTM, compact Transformer, and Autoencoder baselines under the same split.
- Event sensitivity and false alarms/hour are primary; raw accuracy is not used as the success claim.
- Firmware-facing artifacts are included for PSoC Edge E84 / Cortex-M-class evaluation without claiming unmeasured board deployment.

## Methods Not Treated as Direct Comparisons

- EEG-specific seizure transformers trained on CHB-MIT or wearable EEG: Those studies use EEG morphology and often more precise event boundaries; this project uses Empatica ACC/PPG/EDA/TEMP with onset-only labels.
- Large time-series foundation models: Large pretrained models are not aligned with the TinyML/edge memory target and would not solve onset-only label uncertainty.

## Reference Anchors

- Tiny Time Mixers, NeurIPS 2024: https://research.ibm.com/publications/tiny-time-mixers-ttms-fast-pre-trained-models-for-enhanced-zerofew-shot-forecasting-of-multivariate-time-series--1 - Supports compact patch-mixer and channel-independent time-series modeling under edge constraints.
- Wearable armband seizure detection with ACC and PPG, 2025: https://www.sciencedirect.com/science/article/pii/S0169260725005048 - Supports multimodal wearable detection and two-step candidate screening/reporting with false alarms.
- Compact transformer false-alarm reduction for wearable seizure detection: https://thorirmar.com/publication/2024-ieee-tbcas/ - Supports compact transformer comparison, while noting that EEG datasets are not directly comparable to Empatica non-EEG weak labels.
- Systematic review of non-invasive wearable seizure detection: https://www.cureepilepsy.org/news/automated-seizure-detection-with-non-invasive-wearable-devices-a-systematic-review-and-meta-analysis/ - Supports reporting sensitivity and false-alarm burden instead of raw accuracy.
