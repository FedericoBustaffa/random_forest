#!/usr/bin/env bash


make -j

for i in 64 128 256 512; do
    for j in 1 2 4 8; do
        echo "estimators: ${i}"
        echo "threads: ${j}"
        if [[ j -eq 1 ]]; then
            ./build/rf_bm.out $i 0 "seq" $1
        else
            OMP_NUM_THREADS=$j ./build/rf_bm.out $i 0 "omp" $1
        fi
        echo ""
    done
done

python scripts/merge.py local
