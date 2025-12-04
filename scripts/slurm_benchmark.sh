#!/usr/bin/env bash


for i in 32 64 128 256; do
    for j in 1 2 4 8 16 32; do
        if [[ j -eq 1 ]]; then
            srun -N 1 -n 1 -c 32 ./build/rf.out $i 0 $1 1 "seq" $j 1 >> run.log
        else
            srun -N 1 -n 1 -c 32 ./build/rf.out $i 0 $1 1 "omp" $j 1 >> run.log
            srun -N 1 -n 1 -c 32 ./build/rf.out $i 0 $1 1 "ff" $j 1 >> run.log
        fi
    done
done

python scripts/merge.py cluster
