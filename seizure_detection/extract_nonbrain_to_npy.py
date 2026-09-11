# extract_nonbrain_to_npy.py
"""Utility to read .edf files under a root folder, drop all EEG channels, and write the remaining (ECG, EMG, motion) data as compact NumPy ``.npy`` files.

Usage::
    python extract_nonbrain_to_npy.py <root_dir>

* ``<root_dir>`` – folder that contains the downloaded .edf files (e.g. C:\\I2I\\public_benchmark_data).

The script walks the directory recursively, loads each .edf with MNE, removes any channel whose name starts with "EEG" (case‑insensitive), and saves the remaining data as a NumPy array ``<basename>_nonbrain.npy`` together with a tiny JSON side‑car ``<basename>_nonbrain_info.json`` containing channel names and sampling frequency.
"""

import argparse
import json
from pathlib import Path

import mne
import numpy as np


def process_one_edf(edf_path: Path) -> None:
    """Load an EDF, drop EEG channels, and write the remaining data as .npy + meta JSON.

    Parameters
    ----------
    edf_path: Path
        Full path to the .edf file.
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    # Keep channels that do NOT start with "EEG"
    keep_idxs = [i for i, name in enumerate(raw.ch_names) if not name.lower().startswith("eeg")]
    if not keep_idxs:
        print(f"[skip] No non‑EEG channels in {edf_path.name}")
        return
    data, _ = raw.get_data(picks=keep_idxs, return_times=True)
    ch_names = [raw.ch_names[i] for i in keep_idxs]
    sfreq = raw.info["sfreq"]
    # Output filenames in the same folder as the source .edf
    out_npy = edf_path.with_name(edf_path.stem + "_nonbrain.npy")
    out_json = edf_path.with_name(edf_path.stem + "_nonbrain_info.json")
    np.save(out_npy, data.astype(np.float32))
    meta = {
        "channel_names": ch_names,
        "sampling_frequency": sfreq,
        "original_edf": str(edf_path),
    }
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[saved] {out_npy.name} ({data.shape[0]} ch, {data.shape[1]} samples)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create non‑EEG NumPy files alongside each .edf in a dataset")
    parser.add_argument("root_dir", type=str, help="Root folder that contains the .edf files (e.g. C:\\I2I\\public_benchmark_data)")
    args = parser.parse_args()
    root = Path(args.root_dir).expanduser().resolve()
    edf_files = list(root.rglob("*.edf"))
    if not edf_files:
        print(f"[error] No .edf files found under {root}")
        return
    print(f"[info] Found {len(edf_files)} .edf files – processing…")
    for edf in edf_files:
        try:
            process_one_edf(edf)
        except Exception as exc:
            print(f"[failed] {edf}: {exc}")
    print("[done] All files processed.")

if __name__ == "__main__":
    main()
