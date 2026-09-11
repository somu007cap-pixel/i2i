# Review Implementation Matrix

This file is the acceptance checklist for the attached external review. A
phase is not considered evidence merely because its process exited zero; the
required artifact must exist and the quality audit must pass.

| Review item | Implementation | Required artifact | Current state |
|---|---|---|---|
| Validation threshold at fixed alarm budget | Validation-only threshold and smoothing selection in the detection runner | Per-experiment `results.json` with validation policy and frozen test metrics | Implemented in runner; verify in fresh run |
| Event-level post-processing | Causal smoothing, refractory/event alert scoring, and validation-selected operating point | Event metrics and alert policy in experiment results | Implemented; must be compared across fresh experiments |
| Standard/Pro feature ablation | ACC-only, ACC+TEMP, ACC+EDA, ACC+EDA+TEMP, and Pro physiological variants | Per-variant results and paired comparison | Scheduled in focus/robust matrices |
| Weak-label handling | 120/300/600-second label sensitivity runs | Label sensitivity log and results | Scheduled; zero-skip policy enforced |
| Patient-held-out robustness | Seed and label-policy repetitions | Robustness results with seed and split metadata | Scheduled; no resume allowed |
| Lightweight method comparison | Existing TSMixer variants are compared under one protocol | Method comparison table | Partial: no independent attention/TTM baseline is active yet |
| Public external metrics | Restored event TSVs joined to non-EEG MOV/ECG/EMG | Trained public Standard/Pro metrics | Not complete; current public phase is a mapping/split gate only |
| Claim scope | README, plan, and quality audit prohibit clinical/generalization claims | `quality_audit.json` | Implemented |

## Release rule

The study is **not submission-ready** until every row has a concrete artifact,
including trained public external metrics. A completed campaign means the
scheduled computation finished; it does not override this release rule.
