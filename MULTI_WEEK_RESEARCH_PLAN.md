# Multi-Week Seizure Detection Research Plan

This plan turns the current prototype into a defensible research campaign. A
long runtime is useful only when every run answers a pre-specified question,
uses the same split policy, and leaves enough evidence to reproduce the result.

## Current Evidence

The latest generated detection artifact reports:

- 146 selected sessions, with 121 train, 11 validation, and 14 test sessions.
- 127 train seizure episodes, 7 validation episodes, and only 6 test episodes.
- 5-second windows and 5-second stride at 32 Hz.
- Pro test ROC-AUC approximately 0.53 and PR-AUC approximately 0.0015.
- Pro test window recall approximately 0.051, with 3 of 6 events detected.
- Pro test false alarms approximately 15.3 per hour before refractory post-processing.

These are feasibility results, not evidence for a 90% or clinical-performance
claim. The six test events also make the event estimate very uncertain; the
reported Wilson interval is wide by construction.

## Non-Negotiable Rules

1. Freeze the test sessions before model or threshold selection.
2. Select features, architecture, loss, thresholds, smoothing, and alarm budget
   using validation only.
3. Report event-level sensitivity and false alarms per hour alongside window
   metrics. Accuracy and ROC-AUC cannot be the headline under this imbalance.
4. Keep the 300-second onset-duration assumption as a sensitivity analysis, not
   as ground truth unless exact offsets are verified.
5. Never claim that a different modality dataset validates wearable detection.
   The public lane is a separately labeled external non-EEG study: its restored
   event TSVs are ground truth, while MOV/ECG/EMG are public sensor analogues,
   not a direct substitute for Empatica.
6. Do not promise a target metric before it exists on an untouched evaluation
   split. A result can improve, remain flat, or reveal that the task is harder
   than expected.

## Four-Week Campaign

The campaign begins with a mandatory public-data gate. It records whether
annotations are actually present before any public result can be reported.

### Week 1: Data and label audit

- Reproduce the current baseline from a clean environment.
- Verify every selected session's timestamps, sampling rates, missingness,
  duration, sensor alignment, and label overlap.
- Compare exact-duration parsing with the 120/300/600-second onset policy.
- Create a session-level table with patient, session, episode count, duration,
  sensor availability, usable hours, and exclusion reason.
- Write down the final frozen train/validation/test manifest and its hash.

Exit gate: no unresolved label-parser ambiguity, no patient overlap between
splits, and a reproducible baseline artifact.

### Week 2: Measurement and model ablations

- Establish all-normal, majority, and simple signal-statistic baselines.
- Compare window lengths and strides as a pre-registered grid. A 0.125-second
  window is not an automatic fix; it may discard the physiological context
  needed for a seizure event.
- Compare focal loss, weighted BCE, and sampling while preserving validation
  and test prevalence.
- Compare ACC-only, ACC+EDA/TEMP, and PPG-derived additions.
- Track parameter count, training time, memory peak, calibration, and edge size.

Exit gate: every improvement has an ablation comparison and a plausible
mechanistic explanation, not just a higher single score.

### Week 3: Robustness and uncertainty

- Repeat finalists over pre-specified random seeds.
- Repeat label-duration assumptions and missing-sensor conditions.
- Report patient-bootstrap or patient-level confidence intervals, not only
  window-level intervals.
- Tune the operating point on validation at explicit alarm budgets, then apply
  it once to the frozen test split.
- Inspect every missed event and the highest-scoring false alarms.

Exit gate: report median and range across seeds, event-level uncertainty, and a
failure taxonomy. If the result is unstable, the instability is a finding.

### Week 4: External scope, edge validation, and publication package

- Run the public non-EEG evaluator with the restored per-recording event TSVs.
  Standard uses MOV ACC where available; Pro uses MOV ACC plus ECG and EMG
  where available. Keep this external result separate from Mayo Standard/Pro.
- Require a public record manifest, subject split manifest, label coverage,
  missing-sensor exclusions, and per-recording resumable feature cache before
  reporting external metrics.
- Test exported models for numerical parity with the source model.
- Benchmark latency, memory, model size, input/output dtype, and behavior under
  missing or corrupted channels.
- Produce the technical report, experiment ledger, data card, model card,
  reproducibility instructions, and a claim-to-evidence table.
- Preserve the final test output and sign the release manifest.

Exit gate: a reviewer can trace every headline number to code, data split,
configuration, log, and saved artifact.

## Running The Campaign

List phases:

```powershell
python seizure_detection\research_campaign.py --list
```

Start or resume the full campaign:

```powershell
python seizure_detection\research_campaign.py
```

The controller writes a durable manifest, per-phase logs, and a heartbeat under
`seizure_detection\outputs\research_campaign\`. If the machine stops, rerun
the same command. Completed phases are skipped; a failed phase is recorded and
can be selected explicitly with `--phase` after the cause is fixed.

Before starting the campaign, restore the labels without restoring EEG signal:

```powershell
python seizure_detection\restore_public_event_annotations.py --dest C:\I2I\public_benchmark_data
python seizure_detection\download_public_benchmark.py --verify-only --dest C:\I2I\public_benchmark_data
```

The verify-only command must report `supervised_external_eval_ready: true`.

This is a several-week research schedule, not a guarantee that the current
model will reach a particular AUC. The value is the accumulated, auditable
evidence and the ability to explain negative results honestly.

## Required Final Tables

- Dataset inventory and exclusion reasons.
- Label-policy sensitivity.
- Window/stride ablation.
- Loss and sampling ablation.
- Sensor ablation.
- Seed robustness with patient-level intervals.
- Validation operating-point table at fixed alarm budgets.
- Frozen-test event table with event IDs, detection status, alert latency, and
  false-alert burden.
- Edge deployment and source/TFLite parity.

## Claim Discipline

Use language such as “in this held-out prototype evaluation” and “exploratory
pre-ictal ranking” until the evidence supports something stronger. Do not turn
an expected result, a validation result, or a cross-modality benchmark into a
clinical claim.
