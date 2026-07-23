# Submission Readiness Audit

Overall status: **PASS**

| Check | Status | Detail |
| --- | --- | --- |
| allocation_selected_on_validation | pass | selection_split=validation |
| checkpoint_provenance_disclosed | pass | all variants retrained from scratch |
| claim_scope_disclosed | pass | rigorous prototype and feasibility study for patient-level seizure event detection; not a clinically validated detector or deployment claim |
| detection_status | pass | status=completed |
| evaluation_scope | pass | patient-level generalization; held-out patients are reserved for validation/test while train patients remain unseen in those splits |
| event_level_disclosed | pass | standard=1/6, pro=3/6 |
| label_policy_documented | pass | status=pass |
| midsem_detection_scope | pass | no forbidden future-work or generated-text wording in docx |
| midsem_tflite_caveat | pass | TFLite limitation caveat present |
| operating_point_documented | pass | status=pass |
| pro_event_sensitivity_gain | pass | standard_event_sensitivity=0.1667, pro_event_sensitivity=0.5000, gain=0.3333 |
| pro_signal_gain | pass | auc_gain=0.1226, recall_gain=0.1338, event_sensitivity_gain=0.3333, false_alarm_delta=-7.36/hour |
| pro_tflite_io_dtype_disclosed | pass | inputs=['int8'], outputs=['int8'] |
| pro_tflite_size | pass | 282.1 KB |
| raw_accuracy_not_primary | pass | all-normal baseline=0.9988, standard_accuracy=0.9623, pro_accuracy=0.9727 |
| science_product_split_present | pass | best_science_model/best_product_model present |
| split_strategy | pass | patient-level held-out split |
| standard_tflite_io_dtype_disclosed | pass | inputs=['int8'], outputs=['int8'] |
| standard_tflite_size | pass | 281.9 KB |
| uncertainty_estimates | pass | status=pass |
