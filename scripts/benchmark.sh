#!/usr/bin/env bash


for i in 32 64 128 256; do
    for j in 1 2 4 8 16; do
        echo "estimators: ${i}"
        echo "threads: ${j}"
        if [[ j -eq 1 ]]; then
            ./build/rf.out $i 0 $1 1 "seq" $j 1
        else
            ./build/rf.out $i 0 $1 1 "omp" $j 1
            echo ""

            ./build/rf.out $i 0 $1 1 "ff" $j 1
        fi
        echo ""
    done
done

python scripts/merge.py local
