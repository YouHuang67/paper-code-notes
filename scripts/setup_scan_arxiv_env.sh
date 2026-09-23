#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
env_dir="$root/envs/scan-arxiv"
req="$root/envs/scan-arxiv-requirements.txt"

if [[ ! -f "$req" ]]; then
  echo "missing $req" >&2
  exit 1
fi

python3 -m venv "$env_dir"
"$env_dir/bin/python" -m pip install --upgrade pip
"$env_dir/bin/python" -m pip install -r "$req"

echo "scan environment ready"
echo "source envs/scan-arxiv/bin/activate"
