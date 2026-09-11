"""
Phase 2 data generator compatibility module.

Imports from this module resolve to the real-signal preprocessing path, which
uses ACC for the base device and BVP, HR, EDA, and TEMP for the add-on tier.
"""

from phase2_data_generator_fast import (
    SeizureDataGen,
    SeizureDataGenFast,
    SensorData,
    fast_create_labels,
    fast_create_windows,
    fast_downsample_2x,
    fast_read_sensor_file,
    fast_resample,
    fast_upsample_8x,
    get_session_paths,
    get_session_paths_smart,
    load_single_session,
    read_hr_file,
)
