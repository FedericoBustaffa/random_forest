#!/usr/bin/env bash

set -e
trap 'echo "[interrupted]: killing children"; kill 0; exit 1' INT TERM

# sequential
for i in 8 16 32 64 128 256; do
    for j in $(seq 1 5); do
        srun -N 1 -n 1 -c 1 ./build/rf.out $i 0 "seq" 1 $1 $2
    done
done

