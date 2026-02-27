#!/usr/bin/env python3
"""Prepare training/eval assets for CUDA Viola-Jones pipeline.

Outputs (default in project root):
- faces_u8.bin      : [N_pos][24][24] uint8 grayscale
- demo_input.pgm    : grayscale PGM converted from demo_input.png

Positive samples:
- Download from Kaggle via kagglehub (or reuse local folder).

"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterable, List

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".ppm", ".tif", ".tiff"}


def log(msg: str) -> None:
    print(f"[prepare_data] {msg}")


def ensure_python_deps(need_numpy: bool, need_pillow: bool) -> None:
    missing = []
    if need_numpy and np is None:
        missing.append("numpy")
    if need_pillow and Image is None:
        missing.append("Pillow")
    if missing:
        raise RuntimeError(
            "Missing Python deps: "
            + ", ".join(missing)
            + ". Run: python3 -m pip install -r scripts/requirements.txt"
        )


def iter_image_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def iter_npy_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.npy"):
        if p.is_file():
            yield p


def to_square_gray_patch(image: Image.Image, win: int) -> np.ndarray:
    gray = image.convert("L")
    w, h = gray.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    crop = gray.crop((left, top, left + side, top + side))
    patch = crop.resize((win, win), Image.Resampling.BILINEAR)
    arr = np.asarray(patch, dtype=np.uint8)
    return arr


def _to_u8_patch(arr: np.ndarray) -> np.ndarray:
    """Convert arbitrary numeric patch to uint8 with robust range handling."""
    if arr.dtype == np.uint8:
        return arr

    a = arr.astype(np.float32, copy=False)
    vmin = float(np.min(a))
    vmax = float(np.max(a))

    # Common normalized ranges used by generative datasets.
    if -1.05 <= vmin and vmax <= 1.05:
        if vmin < 0.0:
            a = (a + 1.0) * 127.5
        else:
            a = a * 255.0
    elif 0.0 <= vmin and vmax <= 255.0:
        pass
    else:
        if vmax - vmin < 1e-12:
            a = np.zeros_like(a, dtype=np.float32)
        else:
            a = (a - vmin) * (255.0 / (vmax - vmin))

    return np.clip(a, 0.0, 255.0).astype(np.uint8)


def _sample_from_npy_file(path: Path, num_pos: int, win: int, rng: random.Random) -> List[np.ndarray]:
    arr = np.load(path, mmap_mode="r")

    if arr.ndim == 3:
        # [N, H, W]
        n = arr.shape[0]
        idxs = list(range(n))
        rng.shuffle(idxs)
        idxs = idxs[: min(num_pos, n)]
        out: List[np.ndarray] = []
        for i in idxs:
            patch = np.asarray(arr[i])
            patch = _to_u8_patch(patch)
            if patch.shape != (win, win):
                patch = np.asarray(
                    Image.fromarray(patch, mode="L").resize((win, win), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
            out.append(patch)
        return out

    if arr.ndim == 4:
        # [N, H, W, C] or [N, C, H, W]
        n = arr.shape[0]
        idxs = list(range(n))
        rng.shuffle(idxs)
        idxs = idxs[: min(num_pos, n)]
        out: List[np.ndarray] = []
        for i in idxs:
            x = np.asarray(arr[i])
            if x.ndim != 3:
                continue
            if x.shape[-1] in (1, 3, 4):
                # HWC
                if x.shape[-1] == 1:
                    gray = x[..., 0]
                else:
                    rgb = x[..., :3].astype(np.float32, copy=False)
                    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            elif x.shape[0] in (1, 3, 4):
                # CHW
                if x.shape[0] == 1:
                    gray = x[0]
                else:
                    c = x[:3].astype(np.float32, copy=False)
                    gray = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            else:
                continue

            patch = _to_u8_patch(np.asarray(gray))
            if patch.shape != (win, win):
                patch = np.asarray(
                    Image.fromarray(patch, mode="L").resize((win, win), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
            out.append(patch)
        return out

    if arr.ndim == 2:
        # [N, D], try square reshape.
        n, d = arr.shape
        side = int(round(d ** 0.5))
        if side * side != d:
            raise RuntimeError(f"Unsupported .npy shape for faces: {arr.shape} in {path}")
        idxs = list(range(n))
        rng.shuffle(idxs)
        idxs = idxs[: min(num_pos, n)]
        out: List[np.ndarray] = []
        for i in idxs:
            patch = np.asarray(arr[i]).reshape(side, side)
            patch = _to_u8_patch(patch)
            if patch.shape != (win, win):
                patch = np.asarray(
                    Image.fromarray(patch, mode="L").resize((win, win), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
            out.append(patch)
        return out

    raise RuntimeError(f"Unsupported .npy face array shape: {arr.shape} in {path}")


def write_patch_bin(path: Path, patches: List[np.ndarray], win: int) -> None:
    if not patches:
        raise RuntimeError(f"No patches to write for {path}")
    arr = np.stack(patches, axis=0)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    if arr.shape[1:] != (win, win):
        raise RuntimeError(f"Unexpected patch shape {arr.shape}, expected [N,{win},{win}]")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def save_preview(path: Path, patches: List[np.ndarray], win: int, cols: int = 20, max_items: int = 400) -> None:
    if not patches:
        return
    items = patches[: min(len(patches), max_items)]
    rows = (len(items) + cols - 1) // cols
    canvas = np.zeros((rows * win, cols * win), dtype=np.uint8)
    for idx, p in enumerate(items):
        r = idx // cols
        c = idx % cols
        canvas[r * win : (r + 1) * win, c * win : (c + 1) * win] = p
    Image.fromarray(canvas, mode="L").save(path)


def resolve_faces_root(kaggle_ref: str, faces_root: str | None) -> Path:
    if faces_root:
        root = Path(faces_root).expanduser().resolve()
        if not root.exists():
            raise RuntimeError(f"faces_root does not exist: {root}")
        return root

    try:
        import kagglehub  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub not installed. Run: pip install kagglehub"
        ) from exc

    log(f"Downloading Kaggle dataset: {kaggle_ref}")
    ds_path = Path(kagglehub.dataset_download(kaggle_ref)).resolve()
    if not ds_path.exists():
        raise RuntimeError(f"kagglehub returned missing path: {ds_path}")
    return ds_path


def build_positive_faces(
    kaggle_ref: str,
    faces_root: str | None,
    num_pos: int,
    win: int,
    rng: random.Random,
) -> List[np.ndarray]:
    root = resolve_faces_root(kaggle_ref, faces_root)
    files = list(iter_image_files(root))
    npy_files = list(iter_npy_files(root))
    if not files and not npy_files:
        raise RuntimeError(f"No image files or .npy files found in positive dataset root: {root}")

    patches: List[np.ndarray] = []
    bad = 0

    # Priority: natural image files first, then .npy fallback/extension.
    if files:
        rng.shuffle(files)
        for p in files:
            if len(patches) >= num_pos:
                break
            try:
                with Image.open(p) as img:
                    patches.append(to_square_gray_patch(img, win))
            except Exception:
                bad += 1

    if len(patches) < num_pos and npy_files:
        rng.shuffle(npy_files)
        for npy in npy_files:
            if len(patches) >= num_pos:
                break
            need = num_pos - len(patches)
            try:
                sampled = _sample_from_npy_file(npy, need, win, rng)
                patches.extend(sampled)
            except Exception:
                bad += 1

    if len(patches) < num_pos:
        raise RuntimeError(
            f"Positive samples not enough: need {num_pos}, got {len(patches)} "
            f"(bad={bad}, image_files={len(files)}, npy_files={len(npy_files)})"
        )

    log(f"Positive samples built: {len(patches)} from {root}")
    return patches


def convert_demo_to_pgm(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise RuntimeError(f"Demo input image not found: {input_path}")
    with Image.open(input_path) as img:
        gray = img.convert("L")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gray.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset files for CUDA Viola-Jones")
    parser.add_argument("--kaggle-ref", default="shivamshinde123/face-images")
    parser.add_argument("--faces-root", default=None, help="Optional local positive dataset root")
    parser.add_argument("--num-pos", type=int, default=6000)
    parser.add_argument("--win", type=int, default=24)
    parser.add_argument("--seed", type=int, default=677)

    parser.add_argument("--out-dir", default="./data_cache")
    parser.add_argument("--cache-dir", default="./data_cache")

    parser.add_argument("--faces-bin", default="faces_u8.bin")
    parser.add_argument("--faces-preview", default="faces_preview.png")

    parser.add_argument("--demo-input", default="demo_input.png")
    parser.add_argument("--demo-output", default="demo_input.pgm")
    parser.add_argument("--reuse-existing", action="store_true", default=True)
    parser.add_argument("--no-reuse-existing", action="store_false", dest="reuse_existing")
    parser.add_argument("--skip-demo-convert", action="store_true")
    parser.add_argument("--skip-pos", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    need_numpy = not args.skip_pos
    need_pillow = (not args.skip_pos) or (not args.skip_demo_convert)
    ensure_python_deps(need_numpy=need_numpy, need_pillow=need_pillow)
    rng = random.Random(args.seed)

    out_dir = Path(args.out_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not args.skip_pos:
            faces_bin = out_dir / args.faces_bin
            faces_preview = out_dir / args.faces_preview
            expected_bytes = args.num_pos * args.win * args.win
            if args.reuse_existing and faces_bin.exists() and faces_bin.stat().st_size == expected_bytes:
                log(f"Reuse existing {faces_bin}")
            else:
                pos = build_positive_faces(
                    kaggle_ref=args.kaggle_ref,
                    faces_root=args.faces_root,
                    num_pos=args.num_pos,
                    win=args.win,
                    rng=rng,
                )
                write_patch_bin(faces_bin, pos, args.win)
                save_preview(faces_preview, pos, args.win)
                log(f"Wrote {faces_bin} ({len(pos)} samples)")
                log(f"Wrote {faces_preview}")

        if not args.skip_demo_convert:
            demo_in = Path(args.demo_input).expanduser().resolve()
            demo_out = out_dir / args.demo_output
            if args.reuse_existing and demo_out.exists():
                log(f"Reuse existing {demo_out}")
            else:
                convert_demo_to_pgm(demo_in, demo_out)
                log(f"Wrote {demo_out}")

    except Exception as exc:
        print(f"[prepare_data][ERROR] {exc}", file=sys.stderr)
        return 1

    log("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
