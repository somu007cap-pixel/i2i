"""Convert a TFLite flatbuffer into a C header for firmware builds."""

from __future__ import annotations

import argparse
from pathlib import Path


def convert(source: Path, target: Path, symbol: str = "g_seizure_model_data") -> None:
    data = source.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#ifndef SEIZURE_MODEL_DATA_H_",
        "#define SEIZURE_MODEL_DATA_H_",
        "",
        "#include <cstdint>",
        "",
        f"alignas(16) const unsigned char {symbol}[] = {{",
    ]
    for i in range(0, len(data), 12):
        chunk = ", ".join(f"0x{byte:02x}" for byte in data[i : i + 12])
        lines.append(f"  {chunk},")
    lines += [
        "};",
        f"const unsigned int {symbol}_len = {len(data)};",
        "",
        "#endif",
        "",
    ]
    target.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--symbol", default="g_seizure_model_data")
    args = parser.parse_args()
    convert(args.source, args.target, args.symbol)
    print(args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
