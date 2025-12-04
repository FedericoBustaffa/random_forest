#!/usr/bin/env bash


echo "--- sequential ---"
./build/rf.out $1 $2 $3 0 "seq" $4 1
echo ""

echo "--- openmp ---"
./build/rf.out $1 $2 $3 0 "omp" $4 1
echo ""

echo "--- fastflow ---"
./build/rf.out $1 $2 $3 0 "ff" $4 1
echo ""


echo "--- mpi ---"
mpirun -n $4 ./build/rf.out $1 $2 $3 0 "omp" $4 $5
