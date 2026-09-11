"""Smoke/benchmark harness for the six wearable-only modern baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from modern_methods import build_all


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="seizure_detection/outputs/modern_methods")
    args = parser.parse_args()
    torch.manual_seed(42)
    x = torch.randn(8, 160, 3)  # ACC only smoke input; no EEG.
    secondary = torch.randn(8, 160, 4)  # optional wearable secondary channels.
    report = {"scope": "wearable-only; no EEG", "methods": {}}
    for name, model in build_all().items():
        model.eval()
        with torch.no_grad():
            output = model(x, secondary)
        params = sum(p.numel() for p in model.parameters())
        report["methods"][name] = {"status": "pass", "output_shape": list(output.shape), "parameters": params}
    path = Path(args.output)
    path.mkdir(parents=True, exist_ok=True)
    (path / "modern_method_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
