#!/usr/bin/env bash

set -e
trap 'echo "[interrupted]: killing children"; kill 0; exit 1' INT TERM

# sequential
for i in 8 16 32 64; do
    ./build/rf.out $i 0 "seq" 1 $1 "log"
done


# openmp
for i in 16 32 64; do
    for t in 2 4; do
        OMP_NUM_THREADS=$t ./build/rf.out $i 0 "omp" $t $1 "log"
    done
done


# fastflow
for i in 16 32 64; do
    for t in 2 4; do
        ./build/rf.out $i 0 "ff" $t $1 "log"
    done
done


# mpi
for i in 32 64; do
    for n in 2 4; do
        mpirun -n $n ./build/rf.out $i 0 "mpi" 1 $1 "log"
    done
done

python scripts/merge.py results/local.csv

