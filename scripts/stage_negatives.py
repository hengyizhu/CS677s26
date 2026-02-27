#!/usr/bin/env python3
"""Cache COCO large negatives locally and resample per-stage negative bins.

Workflow:
1) cache: download non-person COCO val images into local cache directory.
2) sample: for each stage, randomly crop cached images into 24x24 negatives.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import urllib.error
import urllib.request
import zipfile
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


COCO_ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_VAL_BASE_URL = "http://images.cocodataset.org/val2017"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".ppm", ".tif", ".tiff"}


def log(msg: str) -> None:
    print(f"[stage_negatives] {msg}")


def ensure_python_deps() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if Image is None:
        missing.append("Pillow")
    if missing:
        raise RuntimeError(
            "Missing Python deps: "
            + ", ".join(missing)
            + ". Run: python3 -m pip install -r scripts/requirements.txt"
        )


def ensure_coco_val_annotations(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "annotations_trainval2017.zip"
    json_path = cache_dir / "instances_val2017.json"

    if json_path.exists():
        return json_path

    if not zip_path.exists():
        log(f"Downloading COCO annotations: {COCO_ANN_ZIP_URL}")
        urllib.request.urlretrieve(COCO_ANN_ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        member = "annotations/instances_val2017.json"
        if member not in zf.namelist():
            raise RuntimeError("instances_val2017.json missing in annotation zip")
        with zf.open(member, "r") as src, open(json_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return json_path


def parse_non_person_coco_images(instances_json: Path) -> List[dict]:
    with open(instances_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    categories = {c["id"]: c["name"] for c in meta["categories"]}
    person_id = None
    for cid, name in categories.items():
        if name == "person":
            person_id = cid
            break
    if person_id is None:
        raise RuntimeError("Cannot find COCO person category")

    person_image_ids = set()
    for ann in meta["annotations"]:
        if ann.get("category_id") == person_id:
            person_image_ids.add(ann["image_id"])

    candidates = [img for img in meta["images"] if img["id"] not in person_image_ids]
    if not candidates:
        raise RuntimeError("No non-person images found in COCO val metadata")
    return candidates


def ensure_large_negatives_cache(
    cache_dir: Path,
    images_dir: Path,
    max_images: int,
    seed: int,
) -> None:
    instances_json = ensure_coco_val_annotations(cache_dir)
    candidates = parse_non_person_coco_images(instances_json)
    rng = random.Random(seed)
    rng.shuffle(candidates)

    images_dir.mkdir(parents=True, exist_ok=True)
    present = {p.name for p in images_dir.iterdir() if p.is_file()}
    target = max(1, max_images)

    downloaded = 0
    skipped = 0
    for meta in candidates:
        if len(present) >= target:
            break
        fname = meta.get("file_name")
        if not fname:
            continue
        if fname in present:
            skipped += 1
            continue
        url = f"{COCO_VAL_BASE_URL}/{fname}"
        out = images_dir / fname
        try:
            urllib.request.urlretrieve(url, out)
            present.add(fname)
            downloaded += 1
        except (urllib.error.URLError, TimeoutError, OSError):
            continue

    log(
        "cache ready: "
        f"images={len(present)} target={target} downloaded_now={downloaded} skipped_existing={skipped}"
    )
    if len(present) == 0:
        raise RuntimeError("No cached COCO images available")


def iter_cached_images(images_dir: Path) -> Iterable[Path]:
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def sample_random_patches(gray: Image.Image, win: int, k: int, rng: random.Random) -> List[np.ndarray]:
    w, h = gray.size
    if min(w, h) < win:
        return []

    out: List[np.ndarray] = []
    max_side = min(w, h)
    for _ in range(k):
        crop_side = rng.randint(win, max_side)
        x0 = rng.randint(0, w - crop_side)
        y0 = rng.randint(0, h - crop_side)
        patch = gray.crop((x0, y0, x0 + crop_side, y0 + crop_side))
        patch = patch.resize((win, win), Image.Resampling.BILINEAR)
        out.append(np.asarray(patch, dtype=np.uint8))
    return out


def write_patch_bin(path: Path, patches: List[np.ndarray], win: int) -> None:
    arr = np.stack(patches, axis=0)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    if arr.shape[1:] != (win, win):
        raise RuntimeError(f"Unexpected patch shape {arr.shape}, expected [N,{win},{win}]")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def save_preview(path: Path, patches: List[np.ndarray], win: int, cols: int = 20, max_items: int = 400) -> None:
    items = patches[: min(len(patches), max_items)]
    rows = (len(items) + cols - 1) // cols
    canvas = np.zeros((rows * win, cols * win), dtype=np.uint8)
    for idx, p in enumerate(items):
        r = idx // cols
        c = idx % cols
        canvas[r * win : (r + 1) * win, c * win : (c + 1) * win] = p
    Image.fromarray(canvas, mode="L").save(path)


def build_stage_negatives(
    images_dir: Path,
    num_neg: int,
    win: int,
    patches_per_image: int,
    seed: int,
) -> List[np.ndarray]:
    files = list(iter_cached_images(images_dir))
    if not files:
        raise RuntimeError(f"No cached images found in {images_dir}")

    rng = random.Random(seed)
    rng.shuffle(files)

    # Keep only usable images once; later we can sample the same image many times.
    usable: List[Path] = []
    for img_path in files:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                if min(w, h) >= win:
                    usable.append(img_path)
        except OSError:
            continue

    if not usable:
        raise RuntimeError(f"No usable cached images (min side >= {win}) in {images_dir}")

    patches: List[np.ndarray] = []
    while len(patches) < num_neg:
        img_path = usable[rng.randrange(len(usable))]
        try:
            with Image.open(img_path) as img:
                gray = img.convert("L")
                k = min(num_neg - len(patches), patches_per_image)
                patches.extend(sample_random_patches(gray, win, k, rng))
        except OSError:
            continue

    if len(patches) > num_neg:
        patches = patches[:num_neg]
    return patches


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="COCO local cache + per-stage negative sampler")
    p.add_argument("--cache-dir", default="./data_cache")
    p.add_argument("--images-dir", default="./data_cache/coco_val2017_nonperson")

    p.add_argument("--prepare-cache", action="store_true", help="Download/cache non-person COCO large images")
    p.add_argument("--max-cache-images", type=int, default=12000)
    p.add_argument("--build-pgm-cache", action="store_true", help="Convert cached images to PGM cache for C++ miner")
    p.add_argument("--pgm-dir", default="./data_cache/coco_val2017_nonperson_pgm")

    p.add_argument("--sample-stage", action="store_true", help="Sample a stage negative bin from local cached images")
    p.add_argument("--stage-idx", type=int, default=0)
    p.add_argument("--num-neg", type=int, default=42000)
    p.add_argument("--win", type=int, default=24)
    p.add_argument("--patches-per-image", type=int, default=6)
    p.add_argument("--seed", type=int, default=677)
    p.add_argument("--out-dir", default=".")
    p.add_argument("--reuse-existing", action="store_true", help="Reuse existing stage bin if it matches expected size")
    p.add_argument("--write-preview", action="store_true", help="Also write preview image (disabled by default)")
    return p.parse_args()


def build_pgm_cache(images_dir: Path, pgm_dir: Path, reuse_existing: bool) -> int:
    pgm_dir.mkdir(parents=True, exist_ok=True)
    files = list(iter_cached_images(images_dir))
    converted = 0
    skipped = 0
    bad = 0
    for src in files:
        dst = pgm_dir / (src.stem + ".pgm")
        if reuse_existing and dst.exists():
            skipped += 1
            continue
        try:
            with Image.open(src) as img:
                gray = img.convert("L")
                gray.save(dst)
            converted += 1
        except OSError:
            bad += 1
            continue
    log(f"PGM cache ready: converted={converted} skipped={skipped} bad={bad} dir={pgm_dir}")
    return converted


def main() -> int:
    args = parse_args()
    ensure_python_deps()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.prepare_cache and not args.build_pgm_cache and not args.sample_stage:
        raise RuntimeError("Nothing to do. Use --prepare-cache and/or --build-pgm-cache and/or --sample-stage.")

    if args.prepare_cache:
        ensure_large_negatives_cache(
            cache_dir=cache_dir,
            images_dir=images_dir,
            max_images=args.max_cache_images,
            seed=args.seed,
        )

    if args.build_pgm_cache:
        pgm_dir = Path(args.pgm_dir).expanduser().resolve()
        build_pgm_cache(images_dir=images_dir, pgm_dir=pgm_dir, reuse_existing=args.reuse_existing)

    if args.sample_stage:
        neg_bin = out_dir / f"non_faces_stage{args.stage_idx:02d}.bin"
        neg_preview = out_dir / f"non_faces_stage{args.stage_idx:02d}_preview.png"
        expected_bytes = args.num_neg * args.win * args.win
        if args.reuse_existing and neg_bin.exists() and neg_bin.stat().st_size == expected_bytes:
            log(f"Reuse existing {neg_bin}")
            if args.write_preview and not neg_preview.exists():
                # Rebuild preview on-demand if requested and missing.
                arr = np.fromfile(neg_bin, dtype=np.uint8).reshape(args.num_neg, args.win, args.win)
                save_preview(neg_preview, [arr[i] for i in range(arr.shape[0])], args.win)
                log(f"Wrote {neg_preview}")
            return 0

        stage_seed = args.seed + args.stage_idx * 1000003
        patches = build_stage_negatives(
            images_dir=images_dir,
            num_neg=args.num_neg,
            win=args.win,
            patches_per_image=args.patches_per_image,
            seed=stage_seed,
        )
        write_patch_bin(neg_bin, patches, args.win)
        log(f"Wrote {neg_bin} ({len(patches)} samples)")
        if args.write_preview:
            save_preview(neg_preview, patches, args.win)
            log(f"Wrote {neg_preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
