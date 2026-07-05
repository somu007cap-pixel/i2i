# Submission Readiness Audit

Overall status: **WARN**

| Check | Status | Detail |
| --- | --- | --- |
| pro_tflite_io_dtype_disclosed | warn | inputs=['int8'], outputs=['int8'] |
| standard_tflite_io_dtype_disclosed | warn | inputs=['int8'], outputs=['int8'] |
| allocation_selected_on_validation | pass | selection_split=validation |
| checkpoint_provenance_disclosed | pass | all variants retrained from scratch |
| claim_scope_disclosed | pass | rigorous prototype and feasibility study for patient-level seizure event detection; not a clinically validated detector or deployment claim |
| detection_status | pass | status=completed |
| evaluation_scope | pass | patient-level generalization; held-out patients are reserved for validation/test while train patients remain unseen in those splits |
| event_level_disclosed | pass | standard=1/6, pro=4/6 |
| midsem_detection_scope | pass | no forbidden future-work or generated-text wording in docx |
| midsem_tflite_caveat | pass | TFLite limitation caveat present |
| pro_signal_gain | pass | auc_gain=0.1587, recall_gain=0.0149 |
| pro_tflite_size | pass | 282.1 KB |
| raw_accuracy_not_primary | pass | all-normal baseline=0.9988, standard_accuracy=0.9548, pro_accuracy=0.9731 |
| science_product_split_present | pass | best_science_model/best_product_model present |
| split_strategy | pass | patient-level held-out split |
| standard_tflite_size | pass | 278.7 KB |
