#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu_no=0
if (( $# > 0 )) && [[ "$1" != --* ]]; then
  gpu_no=$1
  shift
fi
python_bin=${PYTHON_BIN:-python}

cd "$repo_dir"
CUDA_VISIBLE_DEVICES="$gpu_no" "$python_bin" "$repo_dir/chat_snn_t16.py" "$@"
