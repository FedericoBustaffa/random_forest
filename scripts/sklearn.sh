#!/usr/bin/env bash

set -e
trap 'echo "[interrupted]: killing children"; kill 0; exit 1' INT TERM

LOG_FLAG=""
if [[ "${2:-}" == "log" ]]; then
    LOG_FLAG="--log"
fi


# openmp
for i in 8 16 32 64 128 256; do
    for t in 1 2 4 8 16 32; do
        for j in $(seq 1 5); do
            srun -N 1 -n 1 -c 32 python test/sklearn_rf.py $i 0 $t $1 $LOG_FLAG
        done
    done
done

