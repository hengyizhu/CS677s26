#!/usr/bin/env bash
set -euo pipefail

# Nsight Systems CLI profiler wrapper for headless servers.
# Stores all artifacts under data_cache/nsys_reports/<timestamp>/ by default.

show_help() {
  cat <<'USAGE'
Usage:
  scripts/nsys_detect_profile.sh [options] -- <detect command and args>

Options:
  -o, --out-dir DIR        Output directory (default: data_cache/nsys_reports/<timestamp>)
  -n, --name NAME          Run name/prefix inside output directory (default: detect_profile)
  --trace LIST             nsys --trace value (default: cuda,nvtx,osrt)
  --sample MODE            nsys --sample value (default: none)
  --cuda-memory-usage VAL  nsys --cuda-memory-usage value (default: true)
  -h, --help               Show this help

Example:
  scripts/nsys_detect_profile.sh -- \
    ./build/cuda_hello detect ./data_cache/cascade_model.bin ./demo_input.pgm /tmp/out.ppm 1.25 4 24 200000 20
USAGE
}

timestamp="$(date +%Y%m%d_%H%M%S)"
out_dir="data_cache/nsys_reports/${timestamp}"
run_name="detect_profile"
trace_opt="cuda,nvtx,osrt"
sample_opt="none"
cuda_mem_opt="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--out-dir)
      out_dir="$2"
      shift 2
      ;;
    -n|--name)
      run_name="$2"
      shift 2
      ;;
    --trace)
      trace_opt="$2"
      shift 2
      ;;
    --sample)
      sample_opt="$2"
      shift 2
      ;;
    --cuda-memory-usage)
      cuda_mem_opt="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help >&2
      exit 1
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "Missing target detect command. Pass it after --" >&2
  show_help >&2
  exit 1
fi

mkdir -p "$out_dir"
out_prefix="${out_dir}/${run_name}"

cmd=("$@")

printf '[NSYS] Output directory: %s\n' "$out_dir"
printf '[NSYS] Command:'
for arg in "${cmd[@]}"; do
  printf ' %q' "$arg"
done
printf '\n'

nsys profile \
  --trace="$trace_opt" \
  --sample="$sample_opt" \
  --cuda-memory-usage="$cuda_mem_opt" \
  --force-overwrite=true \
  -o "$out_prefix" \
  "${cmd[@]}"

rep_file="${out_prefix}.nsys-rep"
qdstrm_file="${out_prefix}.qdstrm"
sqlite_file="${out_prefix}.sqlite"

if [[ ! -f "$rep_file" ]]; then
  importer_bin=""
  if command -v QdstrmImporter >/dev/null 2>&1; then
    importer_bin="$(command -v QdstrmImporter)"
  elif [[ -x /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter ]]; then
    importer_bin="/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter"
  fi

  if [[ -n "$importer_bin" && -f "$qdstrm_file" ]]; then
    printf '[NSYS] Importing qdstrm via %s\n' "$importer_bin"
    "$importer_bin" -i "$qdstrm_file" -o "$sqlite_file"
  else
    echo "[NSYS] .nsys-rep was not generated and no usable importer found." >&2
    echo "[NSYS] qdstrm path: $qdstrm_file" >&2
    exit 2
  fi
fi

api_report="${out_dir}/cuda_api_summary.txt"
kern_report="${out_dir}/gpu_kernel_summary.txt"
osrt_report="${out_dir}/osrt_summary.txt"
all_report="${out_dir}/summary_all.txt"

nsys stats --report cudaapisum --format table "$rep_file" > "$api_report"
nsys stats --report gpukernsum --format table "$rep_file" > "$kern_report"
nsys stats --report osrtsum --format table "$rep_file" > "$osrt_report"

{
  echo "==== CUDA API Summary ===="
  cat "$api_report"
  echo
  echo "==== GPU Kernel Summary ===="
  cat "$kern_report"
  echo
  echo "==== OS Runtime Summary ===="
  cat "$osrt_report"
} > "$all_report"

printf '[NSYS] Done. Main files:\n'
printf '  - %s\n' "$rep_file" "$qdstrm_file" "$sqlite_file" "$api_report" "$kern_report" "$osrt_report" "$all_report"
