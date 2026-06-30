# I2I GTC Seizure Event Detection Release v1.0

This workspace contains the local release package for Empatica-based GTC
seizure-event detection. Pre-ictal risk modeling code is retained as future
work, but the current submission-facing claim is detection-focused.

## Run

```bat
SHOWCASE.bat
```

The one-click run executes detection and, unless disabled, the experimental
pre-ictal risk pipeline. It records start/finish times and writes final
artifacts under `showcase_outputs\`, `seizure_detection\outputs\`, and
`prediction_outputs_local\`.

## Active Folders

```text
Mayo_1110/ ... Mayo_1965/   Raw Mayo Empatica recordings
SeizureTimesOnly/           Additional onset-label source used by the scanner
seizure_detection/          Active pipeline code
seizure_detection/outputs/  Detection models and metrics
prediction_outputs_local/   Experimental pre-ictal risk outputs
showcase_outputs/           Release plots and summary report
.venv/                      Local Python environment
```

## Main Report

Open `showcase_outputs\RELEASE_REPORT.md` after a run for the concise release
report.

## Validation Scope

The v1.0 detection claim is personalized/session-level validation on the
available Mayo recordings. The pipeline uses real Empatica signals,
session-held-out splits, train-only normalization, and no label-derived
synthetic sensor channels. It is not a clinical deployment claim.
