#!/usr/bin/env bash

set -e
trap 'echo "[interrupted]: killing children..."; kill 0; exit 1' INT TERM

# sequential
for i in 8 16 32 64 128 256; do
    for j in $(seq 1 5); do
        srun -N 1 -n 1 -c 1 ./build/rf.out $i 0 "seq" 1 1 $1 1
    done
done


# openmp
for i in 8 16 32 64 128 256; do
    for t in 1 2 4 8 16 32; do
        for j in $(seq 1 5); do
            srun -N 1 -n 1 -c 32 ./build/rf.out $i 0 "omp" $t 1 $1 1
        done
    done
done


# fastflow
for i in 8 16 32 64 128 256; do
    for t in 1 2 4 8 16 32; do
        for j in $(seq 1 5); do
            srun -N 1 -n 1 -c 32 ./build/rf.out $i 0 "ff" $t 1 $1 1
        done
    done
done


# mpi
for i in 32 64 128 256; do
    for t in 8 16 32; do
        for n in 2 4 6 8; do
            for j in $(seq 1 5); do
                srun --mpi=pmix -N $n -n $n -c 32 ./build/rf.out $i 0 "mpi" $t $n $1 1
            done
        done
    done
done

python scripts/merge.py cluster
