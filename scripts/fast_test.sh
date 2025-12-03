#!/usr/bin/env bash


echo "--- sequential ---"
./build/rf.out $1 0 $2 0 "seq" $3 1
echo ""

echo "--- openmp ---"
./build/rf.out $1 0 $2 0 "omp" $3 1
echo ""

echo "--- fastflow ---"
./build/rf.out $1 0 $2 0 "ff" $3 1
echo ""


echo "--- mpi ---"
mpirun -n $4 ./build/rf.out $1 0 $2 0 "omp" $3 $4
