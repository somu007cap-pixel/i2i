# GTC Seizure Event Detection Pipeline v1.0

This release contains a one-click local pipeline for wearable GTC seizure-event
detection using Empatica-derived Mayo recordings. Experimental pre-ictal risk
modeling remains in the repository for future work, but it is not part of the
current mid-semester detection claim.

## Product Configuration

The detection system is evaluated as two product tiers:

- Standard device: ACC plus the selected base optional sensor.
- Pro device: required add-on PPG plus the selected add-on optional sensor.

The selected v1.0 allocation is:

- Base sensors: ACC x/y/z plus TEMP.
- Add-on sensors: PPG-derived BVP and HR, plus EDA.
- IBI is audited but excluded from active model inputs because several IBI
  exports are empty or incomplete.

## Dataset Layout

```text
Mayo_XXXX/
├── Empatica_Chronic/
│   └── {timestamp}_{device}/
│       ├── ACC.csv
│       ├── BVP.csv
│       ├── EDA.csv
│       ├── TEMP.csv
│       ├── HR.csv
│       ├── IBI.csv
│       ├── tags.csv
│       └── info.txt
└── XXXX.txt
```

`XXXX.txt` contains seizure onset timestamps. The pipeline models each onset as
a configurable interval and merges nearby intervals before window labeling.

## One-Click Run

From `C:\I2I`, run:

```bat
SHOWCASE.bat
```

The batch file records the start time, finish time, exit code, and a timestamped
log under `showcase_outputs\logs\`.

Python entry points are also available:

```bat
python seizure_detection\showcase.py
python seizure_detection\detection_upsell.py
python seizure_detection\prediction_horizon.py
```

## Pipeline

1. Data health scan: validates sessions, sampling rates, labels, and available
   sensor exports.
2. Detection preprocessing: resamples signals to 32 Hz, labels windows, applies
   session-level train/validation/test splits, and fits normalization on the
   training split only.
3. Detection modeling: trains compact TSMixer variants for Standard and Pro
   deployment.
4. Detection evaluation: selects product allocation using validation recall at a
   fixed alarm budget and reports test accuracy, ROC-AUC, PR-AUC, recall,
   precision, confusion counts, thresholds, and false alarms/hour.
5. Experimental risk modeling: optionally trains a multi-horizon sequence model
   for 5, 15, and 30 minute pre-ictal risk ranking. This is future-work output,
   not the current detection submission claim.
6. Reporting: writes final JSON summaries, plots, Keras checkpoints, and TFLite
   exports.

## Active Code

```text
seizure_detection/
├── showcase.py
├── showcase_pipeline.py
├── detection_upsell.py
├── prediction_horizon.py
├── phase1_data_health.py
├── phase2_data_generator_fast.py
├── phase2_data_generator.py
├── phase3_tsmixer_model.py
├── phase4_validation.py
├── prediction_data_generator.py
├── prediction_model.py
├── prediction_visualizations.py
├── run_local.py
├── run_with_log.py
├── requirements.txt
└── outputs/
```

## Main Artifacts

```text
seizure_detection/outputs/results.json
seizure_detection/outputs/seizure_model.tflite
seizure_detection/outputs/seizure_model_standard.tflite
prediction_outputs_local/metrics.json
showcase_outputs/RELEASE_REPORT.md
showcase_outputs/detection/
showcase_outputs/prediction/
```

## Validation Scope

The v1.0 results are personalized/session-level validation on the available
Mayo recordings. The pipeline uses real Empatica signals, session-held-out
splits, train-only normalization, exact overlap checks, and no label-derived
synthetic sensor channels.

Patient-independent validation, prospective clinical validation, and deployment
qualification are outside the v1.0 scope. Detection accuracy should be read with
ROC-AUC, recall, and false alarms/hour because seizure windows are rare.
Experimental pre-ictal risk results, when generated, should be reported as
future-work risk ranking because pre-ictal positives are very sparse.

## Deployment Notes

The selected detection models are exported to TensorFlow Lite. The local export
keeps float inputs and outputs while using TensorFlow Lite optimization. Full
integer quantization requires a representative dataset calibration path.
