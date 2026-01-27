#!/usr/bin/env bash

./scripts/sequential.sh $1 $2
./scripts/openmp.sh $1 $2
./scripts/fastflow.sh $1 $2
./scripts/mpi.sh $1 $2
./scripts/sklearn.sh $1 $2

python scripts/merge.py cluster
