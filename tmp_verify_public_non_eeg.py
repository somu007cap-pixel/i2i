import numpy as np
import sys
sys.path.insert(0, r'c:\I2I')
import seizure_detection.public_non_eeg_evaluator as e


def fake_features(path, starts, window):
    modality = path.split('_')[-1].lower()
    dims = {'mov': 18, 'ecg': 3, 'emg': 6}[modality]
    return np.ones((len(starts), dims), dtype=np.float32)


e._features = fake_features
records = [
    {'signals': {'mov': 'mov_1', 'ecg': 'ecg_1'}, 'seizure_intervals': [(0, 10)], 'subject': 's1'},
    {'signals': {'mov': 'mov_2'}, 'seizure_intervals': [(0, 10)], 'subject': 's2'},
]
X, y, groups = e._window_dataset(records, 'pro', 1000)
print('X_shape', X.shape)
print('y_len', len(y))
print('groups', sorted(set(groups)))
