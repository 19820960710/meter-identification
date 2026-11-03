#!/usr/bin/env bash
set -euo pipefail
base_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:-$base_dir/models}"
mkdir -p "$out"
base='https://huggingface.co/D-Robotics/InternVL2_5-1B-GGUF-BPU/resolve/main'
wget -c "$base/rdkx5/vit_model_int16_v2.bin" -O "$out/vit_model_int16_v2.bin"
wget -c "$base/Qwen2.5-0.5B-Instruct-Q4_0.gguf" -O "$out/Qwen2.5-0.5B-Instruct-Q4_0.gguf"
ls -lh "$out/vit_model_int16_v2.bin" "$out/Qwen2.5-0.5B-Instruct-Q4_0.gguf"
