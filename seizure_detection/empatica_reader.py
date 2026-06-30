"""Robust Empatica CSV parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EmpaticaSensorData:
    timestamps: np.ndarray
    values: np.ndarray
    sample_rate: float
    start_time: float


def read_empatica_sensor_csv(filepath: str) -> Optional[EmpaticaSensorData]:
    """Read an Empatica sensor CSV with tolerant handling of blank rows."""
    path = Path(filepath)
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            first_line = handle.readline().strip()
            second_line = handle.readline().strip()
        if not first_line or not second_line:
            return None

        start_time = float(first_line.split(",")[0].strip())
        sample_rate = float(second_line.split(",")[0].strip())

        frame = pd.read_csv(
            path,
            header=None,
            skiprows=2,
            sep=",",
            engine="python",
            skip_blank_lines=True,
            na_values=["", " ", "NA", "NaN", "nan"],
        )
        if frame.empty:
            return EmpaticaSensorData(
                timestamps=np.empty(0, dtype=np.float64),
                values=np.empty((0, 1), dtype=np.float32),
                sample_rate=sample_rate,
                start_time=start_time,
            )

        frame = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        frame = frame.dropna(axis=1, how="all")
        if frame.empty:
            values = np.empty((0, 1), dtype=np.float32)
        else:
            frame = frame.ffill().bfill().fillna(0.0)
            values = frame.to_numpy(dtype=np.float32, copy=True)
            if values.ndim == 1:
                values = values.reshape(-1, 1)

        timestamps = start_time + np.arange(len(values), dtype=np.float64) / sample_rate
        return EmpaticaSensorData(
            timestamps=timestamps,
            values=values,
            sample_rate=sample_rate,
            start_time=start_time,
        )
    except Exception as exc:
        print(f"Error reading {filepath}: {exc}", flush=True)
        return None


def read_empatica_sensor_array(filepath: str):
    """Compatibility helper returning start time, rate, and values."""
    data = read_empatica_sensor_csv(filepath)
    if data is None:
        return None, None, None
    return data.start_time, data.sample_rate, data.values
