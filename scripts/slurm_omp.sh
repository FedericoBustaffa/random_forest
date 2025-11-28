#!/usr/bin/env bash


srun make clean
srun make -j

for i in 64 128 256 512; do
    for j in 1 2 4; do
        echo "estimators: ${i}"
        echo "threads: ${j}"
        srun -N 1 -n 1 -c $j ./build/random_forest.out $i 0 $1
        echo ""
    done
done

python scripts/merge.py

