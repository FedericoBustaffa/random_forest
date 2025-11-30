#!/usr/bin/env bash


for i in 64 128 256 512; do
    for j in 1 2 4 8 16; do
        echo "estimators: ${i}"
        echo "threads: ${j}"
        if [[ j -eq 1 ]]; then
            ./build/rf.out $i 0 $1 1
        else
            OMP_NUM_THREADS=$j ./build/rf.out $i 0 $1 1 "omp" $j 1
            ./build/rf.out $i 0 $1 1 "ff" $j 1
        fi
        echo ""
    done
done

python scripts/merge.py local
