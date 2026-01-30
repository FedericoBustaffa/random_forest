#!/usr/bin/env bash

set -e
trap 'echo "[interrupted]: killing children"; kill 0; exit 1' INT TERM

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 DATASET TREES NODES THREADS"
    exit 1
fi

DATASET=$1
TREES=$2
NODES=$3
THREADS=$4

echo "--- Sequential ---"
srun -N 1 -n 1 -c 1 ./build/rf.out "$TREES" 0 "seq" 1 "$DATASET"

echo "--- OpenMP ---"
srun -N 1 -n 1 -c 32 ./build/rf.out "$TREES" 0 "omp" "$THREADS" "$DATASET"

echo "--- FastFlow ---"
srun -N 1 -n 1 -c 32 ./build/rf.out $e 0 "ff" "$THREADS" "$DATASET"

echo "--- MPI ---"
srun --mpi=pmix -N "$NODES" -n "$NODES" -c 32 \
    ./build/rf.out "$TREES" 0 "mpi" "$THREADS" "$DATASET"

echo "--- Sklearn ---"
python ./test/sklearn_rf.py "$TREE" 0 "$THREADS" "$DATASET"

