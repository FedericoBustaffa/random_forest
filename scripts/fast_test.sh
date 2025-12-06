#!/usr/bin/env bash


echo "--- sequential ---"
./build/rf.out $1 $2 "seq" $3 $4 $5 0
echo ""

echo "--- openmp ---"
./build/rf.out $1 $2 "omp" $3 $4 $5 0
echo ""

echo "--- fastflow ---"
./build/rf.out $1 $2 "ff" $3 $4 $5 0
echo ""

echo "--- mpi ---"
mpirun -n $4 ./build/rf.out $1 $2 "mpi" $3 $4 $5 0
