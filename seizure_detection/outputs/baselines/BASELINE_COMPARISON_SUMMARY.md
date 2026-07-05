# Baseline Comparison Summary

Scope: same session-level data split as the detection pipeline. Metrics are computed on held-out test sessions using thresholds selected on validation.

| Baseline | Parameters | ROC-AUC | PR-AUC | Recall | Event sensitivity | False alarms/hour |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lstm | 5,937 | 0.4965 | 0.0057 | 0.1189 | 4/6 | 4.96 |
| cnn_lstm | 13,521 | 0.5727 | 0.0016 | 0.0573 | 4/6 | 24.65 |
| compact_transformer | 9,617 | 0.5936 | 0.0017 | 0.0340 | 5/6 | 8.79 |
| autoencoder | 3,583 | 0.5706 | 0.0053 | 0.0998 | 4/6 | 5.41 |

These baselines are comparison references, not product claims. Raw accuracy is not used as the main success metric because seizure windows are rare.
