#!/usr/bin/env bash
set -o pipefail

cd "$(dirname "$0")"

python -m pytest || exit 1
python -m exp.exp1 || exit 1
python -m exp.exp2 || exit 1
python -m exp.exp3 || exit 1
