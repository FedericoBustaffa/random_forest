# Parallel Random Forest

This project implements a **Random Forest classifier** in C++ with multiple
parallel backends:

- **Sequential**
- **OpenMP**
- **FastFlow**
- **MPI**

Additionally, Python scripts allow comparisons with **scikit-learn's
RandomForestClassifier**.

## Building and Usage

To build the project is sufficient to move in the root directory and run

```bash
srun make -j
```

Note that the Makefile assumes fastflow present in the `$HOME` directory.

### Fast Test

After compilation, running a fast test simulation with every backend can be done with

```bash
./scripts/test.sh datasets/magic.csv 128 4 16
```

in this case starting a simulation on forest of 128 trees and requiring 4 nodes
and 16 threads per node on the cluster.

Note that the last simulation is with Scikit-Learn, so in order to worker there
is the need to create a Python virtual environment like this

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

or have every package already installed on the system.

### Specific Simulations

A specific simulation can be executed by running the `./build/rf.out` executable
produced with this usage

```bash
./build/rf.out <estimators> <max_depth> <backend> <threads> <dataset> [log]
```

where you need to specify

- **Number of trees**
- **Max depth** (if zero the tree is unbounded)
- **Backend** choosing from: seq, omp, ff, mpi
- **Number of threads**
- **Dataset** choosing whatever csv file from the `./datasets` directory.
- **Log** will save stats in a json file; leave it blank if not interested.

Of course the command must be precede by `srun` with appropriate arguments to
work correctly.

**Note**: the CSV reader and converter assumes a dataset with column of targets
as the last one.
