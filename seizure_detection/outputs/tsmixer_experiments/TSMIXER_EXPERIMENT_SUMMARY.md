# TSMixer Experiment Summary

## Latest Marathon Run

Ranking in this section first requires a credible Standard tier, then favors Pro event non-regression, event gain, recall gain, AUC gain, and lower Pro false alarms.

| Experiment | Standard AUC | Standard recall | Standard events | Standard FA/hr | Pro AUC | Pro recall | Pro events | Pro FA/hr | AUC gain | Recall gain | Event gain | Params | TFLite KB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| robust_seed123_patch8_base_temp_boost_ppg_eda | 0.6254 | 0.2930 | 4/6 | 8.49 | 0.6390 | 0.3057 | 4/6 | 10.14 | 0.0136 | 0.0127 | +0 | 155,729 | 282.1 |
| robust_seed7_patch8_base_temp_boost_ppg_eda | 0.5373 | 0.2166 | 5/6 | 7.86 | 0.5358 | 0.2272 | 5/6 | 8.96 | -0.0015 | 0.0106 | +0 | 155,729 | 282.1 |
| robust_label120s_patch8_base_temp_boost_ppg_eda | 0.4971 | 0.0342 | 5/9 | 10.84 | 0.4967 | 0.0427 | 5/9 | 11.11 | -0.0004 | 0.0085 | +0 | 155,729 | 282.1 |
| robust_label300s_patch8_base_temp_boost_ppg_eda | 0.5470 | 0.1423 | 3/6 | 14.76 | 0.5518 | 0.1465 | 3/6 | 12.43 | 0.0048 | 0.0042 | +0 | 155,729 | 282.1 |
| robust_label300s_booster_tuned_patch8_base_eda_boost_ppg_temp | 0.6370 | 0.2824 | 3/6 | 7.56 | 0.6389 | 0.2803 | 3/6 | 7.63 | 0.0020 | -0.0021 | +0 | 155,729 | 282.1 |
| robust_label120s_booster_tuned_patch8_base_eda_boost_ppg_temp | 0.5550 | 0.1368 | 6/9 | 6.75 | 0.5187 | 0.1325 | 6/9 | 6.76 | -0.0363 | -0.0043 | +0 | 155,729 | 282.1 |
| robust_label600s_patch8_base_temp_boost_ppg_eda | 0.5945 | 0.2298 | 4/6 | 29.50 | 0.5955 | 0.2238 | 4/6 | 25.15 | 0.0010 | -0.0060 | +0 | 155,729 | 282.1 |
| robust_seed7_booster_tuned_patch8_base_eda_boost_ppg_temp | 0.4375 | 0.0170 | 4/6 | 7.53 | 0.5597 | 0.1083 | 3/6 | 7.25 | 0.1222 | 0.0913 | -1 | 155,729 | 282.1 |
| robust_seed42_booster_tuned_patch8_base_eda_boost_ppg_temp | 0.4259 | 0.0276 | 1/6 | 26.33 | 0.5485 | 0.1614 | 3/6 | 18.97 | 0.1226 | 0.1338 | +2 | 155,729 | 282.1 |
| robust_seed123_booster_tuned_patch8_base_eda_boost_ppg_temp | 0.4615 | 0.0064 | 1/6 | 34.65 | 0.5300 | 0.0382 | 3/6 | 34.97 | 0.0685 | 0.0318 | +2 | 155,729 | 282.1 |
| robust_label600s_booster_tuned_patch8_base_eda_boost_ppg_temp | 0.6416 | 0.0602 | 4/6 | 39.94 | 0.6411 | 0.0674 | 4/6 | 40.93 | -0.0005 | 0.0072 | +0 | 155,729 | 282.1 |
| robust_seed42_patch8_base_temp_boost_ppg_eda | 0.5432 | 0.0658 | 3/6 | 37.95 | 0.5621 | 0.0149 | 3/6 | 4.78 | 0.0189 | -0.0510 | +0 | 155,729 | 282.1 |

## Full Archive

Archive ranking uses Pro AUC first, then PR-AUC, recall, and lower false alarms.

| Experiment | Standard AUC | Standard recall | Standard events | Standard FA/hr | Pro AUC | Pro recall | Pro events | Pro FA/hr | AUC gain | Recall gain | Event gain | Params | TFLite KB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| robust_label600s_booster_tuned_patch8_base_eda_boost_ppg_temp | 0.6416 | 0.0602 | 4/6 | 39.94 | 0.6411 | 0.0674 | 4/6 | 40.93 | -0.0005 | 0.0072 | +0 | 155,729 | 282.1 |
| robust_seed123_patch8_base_temp_boost_ppg_eda | 0.6254 | 0.2930 | 4/6 | 8.49 | 0.6390 | 0.3057 | 4/6 | 10.14 | 0.0136 | 0.0127 | +0 | 155,729 | 282.1 |
| robust_label300s_booster_tuned_patch8_base_eda_boost_ppg_temp | 0.6370 | 0.2824 | 3/6 | 7.56 | 0.6389 | 0.2803 | 3/6 | 7.63 | 0.0020 | -0.0021 | +0 | 155,729 | 282.1 |
| robust_label600s_patch8_base_temp_boost_ppg_eda | 0.5945 | 0.2298 | 4/6 | 29.50 | 0.5955 | 0.2238 | 4/6 | 25.15 | 0.0010 | -0.0060 | +0 | 155,729 | 282.1 |
| robust_seed42_patch8_base_temp_boost_ppg_eda | 0.5432 | 0.0658 | 3/6 | 37.95 | 0.5621 | 0.0149 | 3/6 | 4.78 | 0.0189 | -0.0510 | +0 | 155,729 | 282.1 |
| robust_seed7_booster_tuned_patch8_base_eda_boost_ppg_temp | 0.4375 | 0.0170 | 4/6 | 7.53 | 0.5597 | 0.1083 | 3/6 | 7.25 | 0.1222 | 0.0913 | -1 | 155,729 | 282.1 |
| robust_label300s_patch8_base_temp_boost_ppg_eda | 0.5470 | 0.1423 | 3/6 | 14.76 | 0.5518 | 0.1465 | 3/6 | 12.43 | 0.0048 | 0.0042 | +0 | 155,729 | 282.1 |
| robust_seed42_booster_tuned_patch8_base_eda_boost_ppg_temp | 0.4259 | 0.0276 | 1/6 | 26.33 | 0.5485 | 0.1614 | 3/6 | 18.97 | 0.1226 | 0.1338 | +2 | 155,729 | 282.1 |
| robust_seed7_patch8_base_temp_boost_ppg_eda | 0.5373 | 0.2166 | 5/6 | 7.86 | 0.5358 | 0.2272 | 5/6 | 8.96 | -0.0015 | 0.0106 | +0 | 155,729 | 282.1 |
| robust_seed123_booster_tuned_patch8_base_eda_boost_ppg_temp | 0.4615 | 0.0064 | 1/6 | 34.65 | 0.5300 | 0.0382 | 3/6 | 34.97 | 0.0685 | 0.0318 | +2 | 155,729 | 282.1 |
| robust_label120s_booster_tuned_patch8_base_eda_boost_ppg_temp | 0.5550 | 0.1368 | 6/9 | 6.75 | 0.5187 | 0.1325 | 6/9 | 6.76 | -0.0363 | -0.0043 | +0 | 155,729 | 282.1 |
| robust_label120s_patch8_base_temp_boost_ppg_eda | 0.4971 | 0.0342 | 5/9 | 10.84 | 0.4967 | 0.0427 | 5/9 | 11.11 | -0.0004 | 0.0085 | +0 | 155,729 | 282.1 |
