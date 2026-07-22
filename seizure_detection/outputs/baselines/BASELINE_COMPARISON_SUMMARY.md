# Baseline Comparison Summary

Scope: same patient-level held-out split as the detection pipeline. Metrics are computed on held-out test sessions using thresholds selected on validation.

| Baseline | Parameters | ROC-AUC | PR-AUC | Recall | Event sensitivity | Alert event sensitivity | False alerts/hour | Window false alarms/hour |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lstm | 5,937 | 0.4896 | 0.0058 | 0.0913 | 4/6 | 1/6 | 0.96 | 4.18 |
| cnn_lstm | 13,521 | 0.6064 | 0.0030 | 0.0828 | 5/6 | 2/6 | 1.09 | 9.46 |
| compact_transformer | 9,617 | 0.5565 | 0.0026 | 0.0425 | 4/6 | 2/6 | 0.66 | 4.88 |
| autoencoder | 3,583 | 0.5791 | 0.0050 | 0.0807 | 4/6 | 3/6 | 2.00 | 4.83 |

## Same Operating-Point Grid

Each baseline uses thresholds selected only on validation for the listed false-alarm/hour budget and is then evaluated on held-out test sessions.

| Baseline | Budget FA/hr | ROC-AUC | Recall | Events | Alert false alerts/hour |
| --- | ---: | ---: | ---: | ---: | ---: |
| lstm | 1.0 | 0.4896 | 0.0170 | 2/6 | 0.18 |
| lstm | 5.0 | 0.4896 | 0.0552 | 3/6 | 0.60 |
| lstm | 10.0 | 0.4896 | 0.0913 | 4/6 | 0.96 |
| lstm | 25.0 | 0.4896 | 0.1826 | 5/6 | 1.95 |
| lstm | 50.0 | 0.4896 | 0.1932 | 5/6 | 2.32 |
| cnn_lstm | 1.0 | 0.6064 | 0.0149 | 3/6 | 0.42 |
| cnn_lstm | 5.0 | 0.6064 | 0.0510 | 4/6 | 0.82 |
| cnn_lstm | 10.0 | 0.6064 | 0.0828 | 5/6 | 1.09 |
| cnn_lstm | 25.0 | 0.6064 | 0.1529 | 5/6 | 1.63 |
| cnn_lstm | 50.0 | 0.6064 | 0.2038 | 5/6 | 2.16 |
| compact_transformer | 1.0 | 0.5565 | 0.0127 | 2/6 | 0.18 |
| compact_transformer | 5.0 | 0.5565 | 0.0297 | 3/6 | 0.49 |
| compact_transformer | 10.0 | 0.5565 | 0.0425 | 4/6 | 0.66 |
| compact_transformer | 25.0 | 0.5565 | 0.0616 | 4/6 | 1.00 |
| compact_transformer | 50.0 | 0.5565 | 0.1019 | 4/6 | 1.34 |
| autoencoder | 1.0 | 0.5791 | 0.0149 | 2/6 | 0.22 |
| autoencoder | 5.0 | 0.5791 | 0.0552 | 4/6 | 0.95 |
| autoencoder | 10.0 | 0.5791 | 0.0807 | 4/6 | 2.00 |
| autoencoder | 25.0 | 0.5791 | 0.1656 | 4/6 | 5.48 |
| autoencoder | 50.0 | 0.5791 | 0.2208 | 5/6 | 9.55 |

These baselines are comparison references, not product claims. Raw accuracy is not used as the main success metric because seizure windows are rare.
