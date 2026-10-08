#!/usr/bin/env bash
# Rerun every simulation with the settings of the paper, then draw the panels.
#
#   WORKERS=32 bash scripts/run_all.sh
#
# The H_x cache needs about 16 GB of disk per dataset (64 GB in total).
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS=${WORKERS:-10}
SEED=${SEED:-5}
# one BLAS thread per worker process avoids oversubscribing the CPU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}

for dataset in fashion_mnist mnist aa cim; do
    python scripts/simulate.py --dataset "$dataset" --seed "$SEED" --workers "$WORKERS"
done
python scripts/simulate.py --dataset fashion_mnist --lr 0.8 --steps 10 --report-interval 1 \
    --tag fashion_mnist_large_lr --seed "$SEED" --workers "$WORKERS"
python scripts/reusability.py --seed "$SEED" --workers "$WORKERS"
python scripts/noisy.py --seed "$SEED" --workers "$WORKERS"
python scripts/spectrum_scaling.py --seed "$SEED" --workers "$WORKERS"
python scripts/make_figures.py
