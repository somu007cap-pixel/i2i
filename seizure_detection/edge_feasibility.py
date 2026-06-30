import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel("ERROR")

from phase3_tsmixer_model import (
    ChannelIndependentTSMixerBlock,
    ChannelIndependentTimeMixer,
    ChannelPatchEmbedding,
    GatedBoosterFusion,
    GatedTimeMixerBlock,
    MLPBlock,
    ModalityDropout,
    SecondaryFeatureSelector,
    SummaryStats,
    TSMixerBlock,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "seizure_detection" / "outputs"
EDGE_DIR = OUTPUT_DIR / "edge_feasibility"
RESULTS_PATH = OUTPUT_DIR / "results.json"
SUMMARY_PATH = EDGE_DIR / "EDGE_FEASIBILITY_SUMMARY.md"


def load_detection_results() -> Dict:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Detection results not found: {RESULTS_PATH}")
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def load_model(variant_name: str) -> keras.Model:
    path = OUTPUT_DIR / f"{variant_name}.keras"
    if not path.exists():
        raise FileNotFoundError(f"Keras checkpoint not found: {path}")
    return keras.models.load_model(
        path,
        custom_objects={
            "MLPBlock": MLPBlock,
            "TSMixerBlock": TSMixerBlock,
            "ChannelIndependentTimeMixer": ChannelIndependentTimeMixer,
            "ChannelIndependentTSMixerBlock": ChannelIndependentTSMixerBlock,
            "ChannelPatchEmbedding": ChannelPatchEmbedding,
            "GatedTimeMixerBlock": GatedTimeMixerBlock,
            "GatedBoosterFusion": GatedBoosterFusion,
            "ModalityDropout": ModalityDropout,
            "SecondaryFeatureSelector": SecondaryFeatureSelector,
            "SummaryStats": SummaryStats,
        },
        compile=False,
    )


def export_tflite(model: keras.Model, path: Path, mode: str) -> Path:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if mode == "dynamic_range":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif mode != "float32":
        raise ValueError(f"Unsupported export mode: {mode}")
    model_bytes = converter.convert()
    path.write_bytes(model_bytes)
    return path


def dtype_size(dtype) -> int:
    return int(np.dtype(dtype).itemsize)


def tensor_bytes(details: List[Dict]) -> int:
    total = 0
    for item in details:
        shape = item.get("shape")
        dtype = item.get("dtype")
        if shape is None or dtype is None:
            continue
        dims = [int(dim) for dim in shape]
        if any(dim < 0 for dim in dims):
            continue
        count = int(np.prod(dims)) if dims else 1
        total += count * dtype_size(dtype)
    return int(total)


def make_input(detail: Dict, rng: np.random.Generator) -> np.ndarray:
    shape = [int(dim) if int(dim) > 0 else 1 for dim in detail["shape"]]
    dtype = detail["dtype"]
    if np.issubdtype(dtype, np.floating):
        return np.zeros(shape, dtype=dtype)
    quant_scale, quant_zero = detail.get("quantization", (0.0, 0))
    if quant_scale and quant_scale > 0:
        info = np.iinfo(dtype)
        return np.full(shape, np.clip(quant_zero, info.min, info.max), dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def benchmark_tflite(path: Path, runs: int = 200, warmup: int = 20) -> Dict:
    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        num_threads=1,
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    rng = np.random.default_rng(42)
    inputs = [make_input(detail, rng) for detail in input_details]
    for detail, value in zip(input_details, inputs):
        interpreter.set_tensor(detail["index"], value)
    for _ in range(warmup):
        interpreter.invoke()
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        interpreter.invoke()
        timings.append((time.perf_counter() - start) * 1000.0)
    timings = np.asarray(timings, dtype=np.float64)
    return {
        "runs": int(runs),
        "warmup_runs": int(warmup),
        "latency_ms_mean": float(np.mean(timings)),
        "latency_ms_p50": float(np.percentile(timings, 50)),
        "latency_ms_p95": float(np.percentile(timings, 95)),
        "latency_ms_min": float(np.min(timings)),
        "latency_ms_max": float(np.max(timings)),
        "benchmark_note": (
            "Single-thread local TFLite Interpreter latency on this PC. "
            "This is a software proxy, not a Cortex-M latency measurement."
        ),
    }


def benchmark_tflite_subprocess(path: Path, runs: int = 200, warmup: int = 20) -> Dict:
    timeout_s = int(os.environ.get("EDGE_BENCHMARK_TIMEOUT_SEC", "120"))
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--benchmark-one",
        str(path),
        "--runs",
        str(runs),
        "--warmup",
        str(warmup),
    ]
    env = os.environ.copy()
    env["EDGE_BENCHMARK_CHILD"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {
            "benchmark_status": "failed",
            "benchmark_error": f"Timed out after {timeout_s} seconds.",
        }
    if completed.returncode != 0:
        return {
            "benchmark_status": "failed",
            "benchmark_exit_code": int(completed.returncode),
            "benchmark_error": (completed.stderr or completed.stdout)[-2000:],
        }
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {
            "benchmark_status": "failed",
            "benchmark_error": f"Could not parse benchmark output: {exc}",
            "benchmark_output_tail": completed.stdout[-2000:],
        }
    payload["benchmark_status"] = "completed"
    return payload


def inspect_tflite(
    path: Path,
    benchmark: bool = True,
    benchmark_runs: int = 200,
    benchmark_warmup: int = 20,
) -> Dict:
    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        num_threads=1,
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    tensors = interpreter.get_tensor_details()
    ops = interpreter._get_ops_details()
    op_counts = Counter(op["op_name"] for op in ops)
    payload = {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "size_kb": float(path.stat().st_size / 1024.0),
        "inputs": [
            {
                "name": item["name"],
                "shape": [int(dim) for dim in item["shape"]],
                "dtype": str(np.dtype(item["dtype"])),
                "quantization": tuple(float(x) for x in item.get("quantization", (0, 0))),
            }
            for item in inputs
        ],
        "outputs": [
            {
                "name": item["name"],
                "shape": [int(dim) for dim in item["shape"]],
                "dtype": str(np.dtype(item["dtype"])),
                "quantization": tuple(float(x) for x in item.get("quantization", (0, 0))),
            }
            for item in outputs
        ],
        "op_count": int(len(ops)),
        "op_histogram": dict(sorted(op_counts.items())),
        "estimated_tensor_arena_proxy_bytes": tensor_bytes(tensors),
        "estimated_tensor_arena_proxy_kb": float(tensor_bytes(tensors) / 1024.0),
    }
    if benchmark:
        payload["latency_proxy"] = benchmark_tflite_subprocess(
            path,
            runs=benchmark_runs,
            warmup=benchmark_warmup,
        )
    return payload


def keras_summary(model: keras.Model, path: Path) -> Dict:
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "size_kb": float(path.stat().st_size / 1024.0),
        "parameters": int(model.count_params()),
        "trainable_parameters": int(
            np.sum([np.prod(v.shape) for v in model.trainable_weights])
        ),
        "non_trainable_parameters": int(
            np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
        ),
    }


def deployment_notes(tflite_payload: Dict) -> List[str]:
    notes = []
    input_dtypes = {item["dtype"] for item in tflite_payload["inputs"]}
    output_dtypes = {item["dtype"] for item in tflite_payload["outputs"]}
    ops = set(tflite_payload["op_histogram"])
    if input_dtypes != {"int8"} or output_dtypes != {"int8"}:
        notes.append(
            "Model is not a full-integer INT8 TFLite Micro artifact because input/output tensors are not int8."
        )
    complex_ops = sorted(ops.intersection({"RSQRT", "SQUARED_DIFFERENCE", "TRANSPOSE", "MEAN"}))
    if complex_ops:
        notes.append(
            "Operator set contains normalization or reshape-style ops that need TFLite Micro resolver support: "
            + ", ".join(complex_ops)
            + "."
        )
    notes.append(
        "Flash proxy is the flatbuffer size. RAM proxy is the sum of allocated TFLite tensor buffers after allocation."
    )
    notes.append(
        "The result supports software-only edge-feasibility discussion, not a claim of measured hardware deployment."
    )
    return notes


def write_markdown_summary(report: Dict) -> None:
    calibrated_dtypes = []
    for payload in report["models"].values():
        calibrated = payload["tflite"]["calibrated_existing"]
        calibrated_dtypes.append(
            (
                {item["dtype"] for item in calibrated["inputs"]},
                {item["dtype"] for item in calibrated["outputs"]},
            )
        )
    all_int8 = all(inputs == {"int8"} and outputs == {"int8"} for inputs, outputs in calibrated_dtypes)
    if all_int8:
        tflite_status = (
            "The current calibrated TFLite files expose int8 input and output tensors. "
            "They are suitable for software-level full-integer quantization analysis, "
            "subject to operator support on the target runtime."
        )
    else:
        tflite_status = (
            "At least one calibrated TFLite file does not expose int8 input and output tensors, "
            "so the report should not claim completed full-integer TFLite Micro deployment for every tier."
        )
    lines = [
        "# Edge Feasibility Summary",
        "",
        "Scope: software-only feasibility analysis for resource-constrained wearable GTC seizure-event detection. "
        "This is not a claim of measured deployment on a physical MCU.",
        "",
        "## Source Detection Run",
        "",
        f"- Detection timestamp: {report['source_detection_run'].get('timestamp', 'unknown')}",
        f"- Pipeline version: {report['source_detection_run'].get('pipeline_version', 'unknown')}",
        f"- Selected sessions: {report['source_detection_run'].get('selected_session_count', 'unknown')}",
        f"- Requested max sessions: {report['source_detection_run'].get('max_sessions_requested', 'unknown')}",
        f"- Window length: {report['source_detection_run'].get('window_seconds', 'unknown')} seconds",
        "",
        "## Current Artifacts",
        "",
        "| Tier | Selected variant | Keras size | Parameters | Float32 TFLite | Dynamic-range TFLite | Current calibrated TFLite |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tier, payload in report["models"].items():
        tflite = payload["tflite"]
        lines.append(
            "| {tier} | `{variant}` | {keras_size:.1f} KB | {params:,} | "
            "{float_size:.1f} KB | {dynamic_size:.1f} KB | {cal_size:.1f} KB |".format(
                tier=tier.title(),
                variant=payload["variant"],
                keras_size=payload["keras"]["size_kb"],
                params=payload["keras"]["parameters"],
                float_size=tflite["float32"]["size_kb"],
                dynamic_size=tflite["dynamic_range"]["size_kb"],
                cal_size=tflite["calibrated_existing"]["size_kb"],
            )
        )
    lines += [
        "",
        "## Profiling Notes",
        "",
        "| Tier | Current calibrated TFLite tensor proxy | PC TFLite median latency | Input/output dtype |",
        "| --- | ---: | ---: | --- |",
    ]
    for tier, payload in report["models"].items():
        calibrated = payload["tflite"]["calibrated_existing"]
        input_dtypes = sorted({item["dtype"] for item in calibrated["inputs"]})
        output_dtypes = sorted({item["dtype"] for item in calibrated["outputs"]})
        latency_proxy = calibrated.get("latency_proxy", {})
        latency = latency_proxy.get("latency_ms_p50")
        if latency_proxy.get("benchmark_status") == "failed":
            latency_text = "FAILED"
        elif latency is not None:
            latency_text = f"{latency:.4f} ms"
        else:
            latency_text = "not run"
        lines.append(
            "| {tier} | {tensor:.1f} KB | {latency} | {inputs} -> {outputs} |".format(
                tier=tier.title(),
                tensor=calibrated["estimated_tensor_arena_proxy_kb"],
                latency=latency_text,
                inputs=", ".join(input_dtypes),
                outputs=", ".join(output_dtypes),
            )
        )
    lines += [
        "",
        "The latency values are single-thread local TensorFlow Lite Interpreter measurements on this PC. "
        "They are useful for software profiling, but they are not Cortex-M hardware latency measurements.",
        "",
        "## Submission-Safe Interpretation",
        "",
        "The current work supports compact-model and resource-constrained feasibility discussion. "
        + tflite_status,
        "",
        "Next edge step: add representative-dataset full INT8 conversion, operator-resolver review, and tighter memory profiling.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    EDGE_DIR.mkdir(parents=True, exist_ok=True)
    results = load_detection_results()
    run_config = results.get("run_config", {})
    variants = {
        "standard": results["standard_mode"]["variant_name"],
        "pro": results["pro_mode"]["variant_name"],
    }
    existing_tflite = {
        "standard": OUTPUT_DIR / "seizure_model_standard.tflite",
        "pro": OUTPUT_DIR / "seizure_model.tflite",
    }
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scope": (
            "Software-only feasibility analysis for resource-constrained wearable "
            "GTC seizure-event detection. No physical MCU deployment is claimed."
        ),
        "source_detection_run": {
            "timestamp": results.get("timestamp"),
            "pipeline_version": results.get("pipeline_version"),
            "status": results.get("status"),
            "selected_session_count": run_config.get("selected_session_count"),
            "max_sessions_requested": run_config.get("max_sessions_requested"),
            "window_seconds": run_config.get("window_seconds"),
            "stride_seconds": run_config.get("stride_seconds"),
        },
        "selected_variants": variants,
        "models": {},
    }
    for tier, variant in variants.items():
        print(f"\nAnalyzing {tier}: {variant}")
        model = load_model(variant)
        keras_path = OUTPUT_DIR / f"{variant}.keras"
        tier_dir = EDGE_DIR / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        float_path = export_tflite(model, tier_dir / f"{variant}_float32.tflite", "float32")
        dynamic_path = export_tflite(
            model, tier_dir / f"{variant}_dynamic_range.tflite", "dynamic_range"
        )
        calibrated_path = tier_dir / f"{variant}_calibrated_existing.tflite"
        shutil.copy2(existing_tflite[tier], calibrated_path)
        benchmark = os.environ.get("EDGE_BENCHMARK", "1") == "1"
        benchmark_runs = int(os.environ.get("EDGE_BENCHMARK_RUNS", "200"))
        benchmark_warmup = int(os.environ.get("EDGE_BENCHMARK_WARMUP", "20"))
        tflite_reports = {
            "float32": inspect_tflite(
                float_path,
                benchmark=benchmark,
                benchmark_runs=benchmark_runs,
                benchmark_warmup=benchmark_warmup,
            ),
            "dynamic_range": inspect_tflite(
                dynamic_path,
                benchmark=benchmark,
                benchmark_runs=benchmark_runs,
                benchmark_warmup=benchmark_warmup,
            ),
            "calibrated_existing": inspect_tflite(
                calibrated_path,
                benchmark=benchmark,
                benchmark_runs=benchmark_runs,
                benchmark_warmup=benchmark_warmup,
            ),
        }
        report["models"][tier] = {
            "variant": variant,
            "keras": keras_summary(model, keras_path),
            "tflite": tflite_reports,
            "deployment_notes": deployment_notes(tflite_reports["calibrated_existing"]),
        }
        for name, payload in tflite_reports.items():
            latency_proxy = payload.get("latency_proxy", {})
            if latency_proxy.get("benchmark_status") == "failed":
                latency_text = ", latency benchmark FAILED"
            elif latency_proxy.get("benchmark_status") == "completed":
                latency_text = f", p50={latency_proxy['latency_ms_p50']:.4f} ms"
            else:
                latency_text = ", latency benchmark not run"
            print(
                f"  {name}: {payload['size_kb']:.1f} KB, "
                f"ops={payload['op_count']}, "
                f"tensor_proxy={payload['estimated_tensor_arena_proxy_kb']:.1f} KB"
                + latency_text
            )
    report_path = EDGE_DIR / "edge_feasibility_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_summary(report)
    print(f"\nSaved edge feasibility report: {report_path}")
    print(f"Saved edge feasibility summary: {SUMMARY_PATH}")
    if os.environ.get("EDGE_FAIL_ON_BENCHMARK_FAILURE", "1") == "1":
        failures = []
        for tier, payload in report["models"].items():
            calibrated = payload["tflite"]["calibrated_existing"]
            latency_proxy = calibrated.get("latency_proxy", {})
            if latency_proxy.get("benchmark_status") == "failed":
                failures.append(f"{tier}: latency benchmark failed")
        if failures:
            raise RuntimeError("; ".join(failures))
    if os.environ.get("EDGE_REQUIRE_INT8", "1") == "1":
        non_int8 = []
        for tier, payload in report["models"].items():
            calibrated = payload["tflite"]["calibrated_existing"]
            input_dtypes = {item["dtype"] for item in calibrated["inputs"]}
            output_dtypes = {item["dtype"] for item in calibrated["outputs"]}
            if input_dtypes != {"int8"} or output_dtypes != {"int8"}:
                non_int8.append(f"{tier}: {sorted(input_dtypes)} -> {sorted(output_dtypes)}")
        if non_int8:
            raise RuntimeError("Calibrated TFLite is not full INT8: " + "; ".join(non_int8))


def benchmark_one_cli() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-one", required=True)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    payload = benchmark_tflite(Path(args.benchmark_one), runs=args.runs, warmup=args.warmup)
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    if "--benchmark-one" in sys.argv:
        raise SystemExit(benchmark_one_cli())
    main()
