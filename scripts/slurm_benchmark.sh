#!/usr/bin/env bash


for i in 64 128 256 512; do
    for j in 1 2 4 8 16 32; do
        echo "estimators: ${i}"
        echo "threads: ${j}"
        if [[ j -eq 1 ]]; then
            srun -N 1 -n 1 -c $j ./build/rf_bm.out $i 0 "seq" $1
        else
            srun -N 1 -n 1 -c $j ./build/rf_bm.out $i 0 "omp" $1
        fi
        echo ""
    done
done

python scripts/merge.py cluster
