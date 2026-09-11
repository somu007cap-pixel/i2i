# Wearable Seizure Detection Study

This repository contains one clearly scoped study with two wearable solutions:

- **Standard:** movement ACC with the predefined low-complexity secondary sensor configuration.
- **Pro:** Standard movement features plus the predefined richer physiological configuration.

Neither solution uses EEG. Mayo Empatica recordings are the primary development
and frozen-test study. The restored SeizeIT2 public event annotations are used
in a separate, subject-held-out external lane with retained non-EEG MOV, ECG,
and EMG signals. Public results are not direct validation of the Empatica stack.

## Canonical Run

From `C:\I2I`:

```bat
RUN_MULTI_WEEK_CAMPAIGN.bat
```

The launcher restores only public `*_events.tsv` labels, never EEG EDFs, then
runs the resumable research phases. Logs, heartbeats, manifests, and reports
are written under `seizure_detection\outputs\research_campaign\` and
`public_benchmark_outputs\`.

## Evidence Rules

- Patient/subject separation is mandatory; windows from one person never cross splits.
- Thresholds and model selection use validation only; the test set is frozen.
- Event sensitivity and false alarms per hour accompany ROC-AUC and PR-AUC.
- Public labels must be present and mapped before external evaluation can pass.
- No clinical, regulatory, or high-performance claim is made without adequate held-out evidence.

See `MULTI_WEEK_RESEARCH_PLAN.md` for the long-run protocol and
`public_benchmark_outputs\PUBLIC_NON_EEG_EVALUATION.md` for the public-data gate.
