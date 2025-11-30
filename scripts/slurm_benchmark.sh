#!/usr/bin/env bash


for i in 64 128 256 512; do
    for j in 1 2 4 8 16 32; do
        echo "estimators: ${i}"
        echo "threads: ${j}"
        if [[ j -eq 1 ]]; then
            srun -N 1 -n 1 -c $j ./build/rf.out $i 0 "seq" $j $1
        else
            srun -N 1 -n 1 -c $j ./build/rf.out $i 0 "omp" $j $1
            srun -N 1 -n 1 -c $j ./build/rf.out $i 0 "ff" $j $1
        fi
        echo ""
    done
done

python scripts/merge.py cluster
