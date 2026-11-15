#!/usr/bin/env bash
# Every experiment in the paper, in order. Finished training runs are skipped, so this can be restarted.
# Needs data/UAV-GESTURE (data/get_uavgesture.sh) and, for the last step, data/uavhuman_skeletons.npz
# (data/prepare_uavhuman.py).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m experiments.recognition
python -m experiments.baselines
python -m experiments.dataset
python -m experiments.conditions
python -m experiments.viewpoint
python -m experiments.gating
if [ -f data/uavhuman_skeletons.npz ]; then python -m experiments.cross_dataset; fi
python -m experiments.benchmark
python -m experiments.summary
