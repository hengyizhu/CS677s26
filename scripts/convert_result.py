#!/usr/bin/env python3
"""Convert detection result image formats (e.g., PPM -> PNG)."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert result image format")
    p.add_argument("--input", default="detection_result.ppm")
    p.add_argument("--output", default="detection_result.png")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(in_path) as img:
        img.save(out_path)

    print(f"[convert_result] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
