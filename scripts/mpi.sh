#!/usr/bin/env bash

set -e
trap 'echo "[interrupted]: killing children"; kill 0; exit 1' INT TERM

# mpi
for i in 32 64 128 256; do
    for t in 8 16 32; do
        for n in 2 4 6 8; do
            for j in $(seq 1 5); do
                srun --mpi=pmix -N $n -n $n -c 32 ./build/rf.out $i 0 "mpi" $t $1 $2
            done
        done
    done
done

