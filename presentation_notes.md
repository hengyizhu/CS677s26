# Final Project Presentation Notes

## Current Project Status

- Project: CUDA-accelerated Viola-Jones style face detector.
- Pipeline implemented:
  - GPU integral image and squared integral image.
  - GPU AdaBoost weak learner search.
  - Cascade training with hard negative mining.
  - Serialized model for detect-only runs.
  - CPU reference paths for training-core and detection comparisons.
- Current model:
  - Window: 24x24.
  - Haar features in model file: 210,400.
  - Weak classifiers: 1,939.
  - Cascade stages: 38.
  - Stage weak counts: 8, 17, 16, 19, 15, 21, 21, 26, 21, 35, 38, 58, 63, 87, 106, 112, 135, 155, 175, 132, 71, 61, 53, 42, 34, 41, 39, 32, 34, 33, 28, 31, 33, 31, 32, 29, 29, 26.

## Benchmark Environment

- GPU: NVIDIA RTX A6000, compute capability 8.6, 49 GB memory.
- Demo image: 406x612 PGM.
- Positive training bin: 23,042 24x24 face patches.
- Negative cache: 2,307 COCO non-person PGM images.

## Correctness And CPU/GPU Comparisons

Training weak learner search, same samples/features/weights:

| Samples | Features | CPU avg ms | GPU avg ms | Speedup | Correctness |
|---:|---:|---:|---:|---:|---|
| 256 | 512 | 41.339 | 0.334 | 123.9x | Same best feature/theta/parity/error |
| 512 | 1024 | 169.941 | 0.654 | 259.8x | Same best feature/theta/parity/error |
| 1024 | 2048 | 749.189 | 1.396 | 536.5x | Same best feature/theta/parity/error |

Detection comparison, same trained cascade and raw ungrouped detections:

| Image | CPU avg ms | GPU avg ms | Speedup | CPU raw detections | GPU raw detections |
|---|---:|---:|---:|---:|---:|
| Demo 406x612 | 965.805 | 30.363 | 31.8x | 137 | 137 |

Final grouped GPU detection:

| Image / setting | Result detections | GPU avg ms | FPS |
|---|---:|---:|---:|
| Demo, scaleFactor=1.25 | 6 | 30.640 | 32.6 |
| Demo, scaleFactor=1.10 | 7 | 72.108 | 13.9 |
| Demo, scaleFactor=1.50 | 2 | 32.663 | 30.6 |
| COCO non-person 586x640 | 0 | 15.179 | 65.9 |
| COCO non-person 640x512 | 0 | 45.712 | 21.9 |

## GPU Implementation Points To Present

- Parallelism:
  - Integral image: rows and columns processed in parallel across samples/columns.
  - Training: independent Haar feature/sample responses; one block per feature and threads across samples.
  - Threshold search: one block per feature scans sorted responses and reduces best split.
  - Detection: one thread per sliding window candidate, with cascade early rejection.
- Memory/resource choices:
  - Integral images stored transposed as `[integral_pixel][sample]` so warp lanes read contiguous samples for the same Haar corner.
  - Haar feature loaded into shared memory in the training response kernel.
  - Detection model packed as 16-byte `float4` stumps.
  - Feature lookup tables avoid repeated decoding of integral-image corner offsets.
  - Pinned host memory used for detection result transfer.
  - CUB segmented radix sort used for per-feature response sorting.
- Resource usage from `cuobjdump --dump-resource-usage`:
  - `EvaluateFeatureResponsesKernel`: 28 registers/thread, 64 B shared memory, 256-thread blocks.
  - `EvaluateAndFindThresholdKernel`: 36 registers/thread, 1396 B shared memory, 256-thread blocks.
  - `DetectCascadeKernel`: 40 registers/thread, 8 B shared memory, 256-thread blocks.
  - On RTX A6000 / sm_86, these custom 256-thread kernels are not occupancy-limited by registers or shared memory; theoretical active blocks are capped mainly by the SM thread limit.
- Note: Nsight Compute hardware counter access is blocked on this machine, so report this as theoretical occupancy/resource analysis unless profiling permissions are enabled.

## What Is Still Weak

- We do not currently have measured timings for multiple historical GPU versions, such as naive layout vs transposed layout. Do not invent these. If asked, frame the comparison as CPU baseline plus current optimized GPU kernels, and mention optimization ablations as future work.
- Detection evaluation is small. Add more images if time allows, especially a few face-containing images and a few non-face images.
- Accuracy metrics are not a full benchmark dataset precision/recall curve. We can honestly present demo correctness plus CPU/GPU output agreement and small false-positive checks.

## Suggested 6.5-Minute Slide Flow

1. Problem and importance: face detection by scanning multi-scale windows with Haar features and a cascade.
2. Why GPU: most compute is independent over features, samples, thresholds, scales, and windows.
3. Algorithm pipeline: data preparation, integral images, AdaBoost weak search, cascade, detection.
4. GPU implementation: thread mapping and memory layout.
5. Optimizations: transposed integral images, shared feature load, CUB segmented sort, compact stumps, early rejection.
6. Results: training-core speedup table and detection speedup table.
7. Evaluation and limitations: setting sensitivity, negative image checks, no full precision/recall yet.
8. Takeaway: same outputs as CPU reference on tested comparisons, much faster search/detection, remaining work is broader accuracy evaluation and ablation study.
