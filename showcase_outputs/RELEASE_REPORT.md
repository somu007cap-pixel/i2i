# GTC Seizure Event Detection Release Report

Release: v1.0
Generated: 2026-09-01 23:05:23

## Product Configuration

Standard is the low-cost configuration and uses ACC as the required base sensor plus the selected base optional sensor. Pro is cumulative: it keeps the Standard signals and adds a second sensor module with required PPG plus the selected add-on optional sensor. The PPG stream contributes BVP and HR features. IBI is excluded from model inputs because several source files are empty or malformed.

## Detection Result

- Evaluation split: patient-level held-out split; train 121 sessions, validation 11, test 14.
- Selected allocation: base_acc_temp__addon_ppg_eda; Standard optional sensors ['TEMP'], Pro add-on optional sensors ['EDA'].
- Standard model: standard_acc_temp; secondary features ['TEMP']; AUC 0.4929, PR-AUC 0.0012, recall 0.0042, event sensitivity 0.1667, balanced accuracy 0.4943, false alarms/hour 11.24.
- Pro model: pro_full_bvp_hr_eda_temp; secondary features ['BVP', 'HR', 'EDA', 'TEMP']; AUC 0.5275, PR-AUC 0.0015, recall 0.0510, event sensitivity 0.5000, balanced accuracy 0.5148, false alarms/hour 15.33.
- Raw accuracy is diagnostic only: seizure-window prevalence is 0.0012; an all-normal classifier would score 0.9988 accuracy while detecting no events.
- Pro AUC improvement over Standard: 0.0345; recall improvement: 0.0467.
- Pro score fusion: validation-selected add-on model weight 0.1500; remaining weight comes from the Standard detector score.

### Matched False-Alarm Operating Point

- Validation alarm budget: 10.00 false alarms/hour.
- Test recall: Standard 0.0042 -> Pro 0.0510.
- Event sensitivity: Standard 0.1667 -> Pro 0.5000.
- Test false alarms/hour: Standard 11.24 -> Pro 15.33.

### Fixed Alarm-Budget Grid

- 1 false alarms/hour validation budget: recall 0.0000 -> 0.0000; test false alarms/hour 0.00 -> 0.01; event sensitivity 0.0000 -> 0.0000.
- 5 false alarms/hour validation budget: recall 0.0000 -> 0.0297; test false alarms/hour 6.92 -> 6.23; event sensitivity 0.0000 -> 0.5000.
- 10 false alarms/hour validation budget: recall 0.0042 -> 0.0510; test false alarms/hour 11.24 -> 15.33; event sensitivity 0.1667 -> 0.5000.
- 25 false alarms/hour validation budget: recall 0.0467 -> 0.0955; test false alarms/hour 45.10 -> 44.66; event sensitivity 0.3333 -> 0.5000.
- 50 false alarms/hour validation budget: recall 0.1805 -> 0.1911; test false alarms/hour 122.38 -> 84.83; event sensitivity 0.8333 -> 0.8333.

Interpretation: seizure windows are rare, so raw accuracy is not a sufficient success metric. AUC, recall, balanced accuracy, and alarm burden should be read alongside the confusion matrix. Pro ROC-AUC is higher (0.0345 difference). Pro recall is higher (0.0467 difference). Pro false alarms/hour are higher (4.09 difference).

## Experimental Pre-Ictal Risk Result

- Final metrics evaluated on: full validation split (33481 sequences).
- Horizon 5 min: AUC 0.7190; calibrated F1 0.0101; calibrated recall 0.7400; top-5% risk recall 0.0600.
- Horizon 15 min: AUC 0.7385; calibrated F1 0.0248; calibrated recall 0.2727; top-5% risk recall 0.1049.
- Horizon 30 min: AUC 0.7665; calibrated F1 0.0467; calibrated recall 0.2510; top-5% risk recall 0.1293.

This section is future-work output, not part of the current detection claim. It is reported as risk ranking because pre-ictal positives are sparse. AUC, calibrated operating points, and top-risk capture are more informative than fixed 0.5 threshold precision/recall.

## Validation Scope

- Claim scope: rigorous prototype and feasibility study for patient-level seizure event detection; not a clinically validated detector or deployment claim.
- The pipeline uses real Empatica signals, session-held-out splits, train-only normalization, and no synthetic label-derived features.
- The reported scope is personalized/session-level validation on the available Mayo recordings.
- Patient-independent validation, improved event-level seizure sensitivity, prospective clinical evaluation, and physical hardware deployment are outside the v1.0 scope.

## Key Artifacts

- Detection results: seizure_detection/outputs/results.json
- Matched alarm chart: showcase_outputs/detection/detection_matched_alarm_budget.png
- Detection chart: showcase_outputs/detection/detection_comparison.png
- Pro TFLite model: seizure_detection/outputs/seizure_model.tflite
- Standard TFLite model: seizure_detection/outputs/seizure_model_standard.tflite
- Experimental risk results: prediction_outputs_local/metrics.json
- Experimental risk dashboard: showcase_outputs/prediction/executive_dashboard.png
