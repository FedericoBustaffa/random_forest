#!/usr/bin/env bash


# sequential
for i in 32 64 128 256; do
    for j in $(seq 1 5); do
        ./build/rf.out $i 0 "seq" 1 1 $1 1
    done
done


# openmp
for i in 32 64 128 256; do
    for t in 1 2 4; do
        for j in $(seq 1 5); do
            ./build/rf.out $i 0 "omp" $t 1 $1 1
        done
    done
done


# fastflow
for i in 32 64 128 256; do
    for t in 1 2 4; do
        for j in $(seq 1 5); do
            ./build/rf.out $i 0 "ff" $t 1 $1 1
        done
    done
done


# mpi on my local machines
for i in 32 64 128 256; do
    for t in 1 2 4; do
        for n in 1 2; do
            for j in $(seq 1 5); do
                mpirun -n $n --hostfile hosts.txt --bind-to none --map-by ppr:1:node ./build/rf.out $i 0 "mpi" $t $n $1 1
            done
        done
    done
done


python scripts/merge.py local
