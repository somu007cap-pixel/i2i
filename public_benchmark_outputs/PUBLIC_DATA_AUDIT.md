# Public Data Audit

- Status: **PASS**
- Root: `C:\I2I\public_benchmark_data`
- EDF files: 8152
- Subjects: 125
- Modalities: `{"ecg": 2800, "emg": 2801, "mov": 2551}`
- Event TSV files: 2850
- Annotation rows: 3659
- Event types: `{"bckg": 2312, "impd": 464, "sz_foc_a_m_automatisms": 4, "sz_foc_a_m_hyperkinetic": 158, "sz_foc_a_nm": 66, "sz_foc_a_nm_behavior": 38, "sz_foc_a_um": 50, "sz_foc_f2b": 55, "sz_foc_ia": 35, "sz_foc_ia_m_automatisms": 62, "sz_foc_ia_m_clonic": 1, "sz_foc_ia_m_hyperkinetic": 100, "sz_foc_ia_m_tonic": 24, "sz_foc_ia_nm": 160, "sz_foc_ia_nm_behavior": 2, "sz_foc_ia_um": 7, "sz_foc_ua_m": 4, "sz_foc_ua_m_automatisms": 2, "sz_foc_ua_m_hyperkinetic": 9, "sz_foc_ua_m_tonic": 2, "sz_foc_ua_nm": 40, "sz_foc_ua_nm_behavior": 17, "sz_foc_ua_um": 41, "sz_uo_m_hyperkinetic": 3, "sz_uo_m_myoclonic": 1, "sz_uo_m_tonicMyio": 1, "sz_uo_nm": 1}`
- Event files mapped to retained non-EEG recordings: 2847
- Expected modalities absent: `["eeg"]`
- Label status: per-recording event annotations found

EEG signal is optional for this project; event TSV labels must be present, parsed, and included in a subject-held-out protocol.
