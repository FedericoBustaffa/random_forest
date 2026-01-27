#!/usr/bin/env bash

srun -N 1 -n 1 -c 1 python test/sklearn_dt.py 0 datasets/breast_cancer.csv --log
srun -N 1 -n 1 -c 1 python test/sklearn_dt.py 0 datasets/magic.csv --log
srun -N 1 -n 1 -c 1 python test/sklearn_dt.py 0 datasets/susy20000.csv --log

srun -N 1 -n 1 -c 1 ./build/dt.out 0 datasets/breast_cancer.csv log
srun -N 1 -n 1 -c 1 ./build/dt.out 0 datasets/magic.csv log
srun -N 1 -n 1 -c 1 ./build/dt.out 0 datasets/susy20000.csv log

python scripts/merge.py results/tree.csv

