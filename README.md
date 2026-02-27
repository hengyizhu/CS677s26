# CUDA Viola-Jones Pipeline (Train/Detect Decoupled)

This project implements a CUDA-accelerated Viola-Jones style detector with:
- GPU integral image + squared integral image
- GPU AdaBoost weak learner search
- Cascade training with hard negative mining
- Decoupled model serialization (`.bin`) for fast detect-only runs

## 1) Build

```bash
cmake -S . -B build
cmake --build build -j
```

Binary:
- `./build/cuda_hello`

## 2) Install Python script dependencies

```bash
python3 -m pip install -r scripts/requirements.txt
```

## 3) End-to-end workflow

### Step A: Prepare positive samples and demo image

```bash
python3 scripts/prepare_data.py \
  --num-pos 23042 \
  --out-dir data_cache
```

Outputs (default):
- `data_cache/faces_u8.bin`
- `data_cache/faces_preview.png`
- `data_cache/demo_input.pgm`

### Step B: Build negative image cache and PGM cache

```bash
# Cache large non-person COCO images
python3 scripts/stage_negatives.py \
  --prepare-cache \
  --cache-dir data_cache \
  --images-dir data_cache/coco_val2017_nonperson \
  --max-cache-images 12000

# Convert cached images to PGM for mining
python3 scripts/stage_negatives.py \
  --build-pgm-cache \
  --images-dir data_cache/coco_val2017_nonperson \
  --pgm-dir data_cache/coco_val2017_nonperson_pgm
```

### Step C: Train and save model (no detect in train mode)

```bash
./build/cuda_hello train \
  data_cache/faces_u8.bin \
  10000 \
  250 \
  200000 \
  4 \
  1.25 \
  38 \
  0.99 \
  0.40 \
  24 \
  data_cache \
  data_cache/coco_val2017_nonperson_pgm \
  677 \
  4 \
  data_cache/cascade_model.bin
```

### Step D: Detect from saved model

```bash
./build/cuda_hello detect \
  data_cache/cascade_model.bin \
  data_cache/demo_input.pgm \
  detection_result.ppm
```

Optional detect params:

```bash
./build/cuda_hello detect \
  data_cache/cascade_model.bin \
  data_cache/demo_input.pgm \
  detection_result.ppm \
  1.25 \
  4 \
  24 \
  200000
```

Parameter order:
1. `scaleFactor` (default `1.25`)
2. `minNeighbors` (default `4`)
3. `minObjectSize` (default `24`)
4. `maxDetections` (default `200000`)

### Step E: Convert PPM to PNG (optional)

```bash
python3 scripts/convert_result.py \
  --input detection_result.ppm \
  --output detection_result.png
```

## 4) CLI reference

### Train mode

```bash
./build/cuda_hello train \
  [faces_u8.bin] \
  [numNeg] \
  [maxWeakPerStage] \
  [maxDetections] \
  [minNeighbors] \
  [scaleFactor] \
  [maxStages] \
  [minHitRate] \
  [maxFalseAlarm] \
  [minObjectSize] \
  [stageNegCacheDir] \
  [stageNegImagesDir] \
  [stageNegSeed] \
  [hardNegCandidateMultiplier] \
  <out_model.bin>
```

Defaults:
- `faces_u8.bin = faces_u8.bin`
- `numNeg = 10000 (fixed to paper-style per-stage setting)`
- `maxWeakPerStage = 250`
- `maxDetections = 200000`
- `minNeighbors = 4`
- `scaleFactor = 1.25`
- `maxStages = 38`
- `minHitRate = 0.99`
- `maxFalseAlarm = 0.40`
- `minObjectSize = 24`
- `stageNegImagesDir = data_cache/coco_val2017_nonperson_pgm`
- `stageNegSeed = 677`
- `hardNegCandidateMultiplier = 3`

Notes:
- `stageNegCacheDir` is required in train mode.
- Train mode does not run final image detection anymore.
- Per-stage sample policy: positives `10000` random (or all if pool < `10000`), negatives `10000`.
- Feature budget policy: total weak classifiers capped at `6061`; first five stages are capped as `1, 10, 25, 25, 50`.

### Detect mode

```bash
./build/cuda_hello detect \
  <in_model.bin> \
  <image.pgm> \
  <out.ppm> \
  [scaleFactor] [minNeighbors] [minObjectSize] [maxDetections]
```

## 5) Script reference

### `scripts/prepare_data.py`

Purpose:
- Build positive face patches (`faces_u8.bin`) from Kaggle or local dataset
- Convert one demo image to grayscale PGM

Common:
```bash
python3 scripts/prepare_data.py --num-pos 6000 --out-dir data_cache
```

Important flags:
- `--kaggle-ref`
- `--faces-root`
- `--num-pos`
- `--win`
- `--seed`
- `--skip-pos`
- `--skip-demo-convert`
- `--reuse-existing` / `--no-reuse-existing`

### `scripts/stage_negatives.py`

Purpose:
- Cache COCO non-person images
- Build PGM cache
- Sample per-stage negative patch bins

Examples:
```bash
python3 scripts/stage_negatives.py --prepare-cache
python3 scripts/stage_negatives.py --build-pgm-cache
python3 scripts/stage_negatives.py --sample-stage --stage-idx 0 --num-neg 42000 --out-dir data_cache/stage_negatives
```

Important flags:
- `--cache-dir`
- `--images-dir`
- `--max-cache-images`
- `--pgm-dir`
- `--stage-idx`
- `--num-neg`
- `--win`
- `--patches-per-image`
- `--seed`
- `--out-dir`

### `scripts/convert_result.py`

Purpose:
- Convert result image format (e.g., PPM -> PNG)

Example:
```bash
python3 scripts/convert_result.py --input detection_result.ppm --output detection_result.png
```

### `scripts/validate_vj_consistency.py`

Purpose:
- Validate OpenCV-consistent variance normalization math (`sqsum` unsigned semantics)

Examples:
```bash
python3 scripts/validate_vj_consistency.py --width 1920 --height 1080 --samples 20000
python3 scripts/validate_vj_consistency.py --pgm data_cache/demo_input.pgm --samples 20000
```
