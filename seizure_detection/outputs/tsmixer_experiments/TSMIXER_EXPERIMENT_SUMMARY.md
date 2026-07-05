# TSMixer Experiment Summary

## Latest Marathon Run

Ranking in this section favors the product claim: event gain, recall gain, AUC gain, and lower Pro false alarms.

| Experiment | Standard AUC | Standard recall | Standard events | Standard FA/hr | Pro AUC | Pro recall | Pro events | Pro FA/hr | AUC gain | Recall gain | Event gain | Params | TFLite KB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| booster_tuned_patch8_base_temp_boost_ppg_eda | 0.3535 | 0.0573 | 1/6 | 31.72 | 0.5122 | 0.0722 | 4/6 | 18.60 | 0.1587 | 0.0149 | +3 | 155,729 | 282.1 |
| booster_tuned_patch8_base_eda_boost_ppg_temp | 0.4409 | 0.0255 | 5/6 | 22.66 | 0.4431 | 0.0318 | 6/6 | 22.76 | 0.0022 | 0.0064 | +1 | 155,729 | 282.1 |
| patch8_base_eda_temp_boost_ppg | 0.4603 | 0.0255 | 5/6 | 13.54 | 0.4973 | 0.0297 | 5/6 | 16.51 | 0.0371 | 0.0042 | +0 | 155,633 | 282.0 |
| sswce_patch8_base_eda_boost_ppg_temp | 0.5650 | 0.2357 | 4/6 | 38.42 | 0.4810 | 0.1805 | 4/6 | 35.21 | -0.0840 | -0.0552 | +0 | 155,729 | 282.1 |
| patch8_base_temp_boost_ppg_eda | 0.3783 | 0.0212 | 5/6 | 77.18 | 0.4035 | 0.0234 | 4/6 | 92.49 | 0.0253 | 0.0021 | -1 | 155,729 | 282.1 |

## Full Archive

Archive ranking uses Pro AUC first, then PR-AUC, recall, and lower false alarms.

| Experiment | Standard AUC | Standard recall | Standard events | Standard FA/hr | Pro AUC | Pro recall | Pro events | Pro FA/hr | AUC gain | Recall gain | Event gain | Params | TFLite KB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| booster_tuned_patch8_base_temp_boost_ppg_eda | 0.3535 | 0.0573 | 1/6 | 31.72 | 0.5122 | 0.0722 | 4/6 | 18.60 | 0.1587 | 0.0149 | +3 | 155,729 | 282.1 |
| patch8_base_eda_temp_boost_ppg | 0.4603 | 0.0255 | 5/6 | 13.54 | 0.4973 | 0.0297 | 5/6 | 16.51 | 0.0371 | 0.0042 | +0 | 155,633 | 282.0 |
| sswce_patch8_base_eda_boost_ppg_temp | 0.5650 | 0.2357 | 4/6 | 38.42 | 0.4810 | 0.1805 | 4/6 | 35.21 | -0.0840 | -0.0552 | +0 | 155,729 | 282.1 |
| booster_tuned_patch8_base_eda_boost_ppg_temp | 0.4409 | 0.0255 | 5/6 | 22.66 | 0.4431 | 0.0318 | 6/6 | 22.76 | 0.0022 | 0.0064 | +1 | 155,729 | 282.1 |
| patch8_base_temp_boost_ppg_eda | 0.3783 | 0.0212 | 5/6 | 77.18 | 0.4035 | 0.0234 | 4/6 | 92.49 | 0.0253 | 0.0021 | -1 | 155,729 | 282.1 |
