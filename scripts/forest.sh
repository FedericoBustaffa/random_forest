#!/usr/bin/env bash


set -e
trap 'echo "[interrupted]: killing children"; kill 0; exit 1' INT TERM

DATASET=$1

TREES=(8 16 32 64 128 256)
THREADS=(2 4 8 16 32)

run_seq() {
    echo "--- Sequential ---"
    for e in "${TREES[@]}"; do
        for j in $(seq 1 5); do
            srun -N 1 -n 1 -c 1 \
                ./build/rf.out $e 0 "seq" 1 "$DATASET" log
        done
    done
}

run_omp() {
    echo "--- OpenMP ---"
    for e in "${TREES[@]}"; do
        for t in "${THREADS[@]}"; do
            for j in $(seq 1 $REPS); do
                srun -N 1 -n 1 -c 32 \
                    ./build/rf.out $e 0 "omp" $t "$DATASET" log
            done
        done
    done
}

run_ff() {
    echo "--- FastFlow ---"
    for e in "${TREES[@]}"; do
        for t in "${THREADS[@]}"; do
            for j in $(seq 1 5); do
                srun -N 1 -n 1 -c 32 \
                    ./build/rf.out $e 0 "ff" $t "$DATASET" log
            done
        done
    done
}

run_mpi() {
    echo "--- MPI ---"
    for e in "${TREES[@]}"; do
        for n in 2 3 4 5 6 7 8; do
            for t in 8 16 32; do
                for j in $(seq 1 5); do
                    srun --mpi=pmix -N $n -n $n -c 32 \
                        ./build/rf.out $e 0 "mpi" $t "$DATASET" log
                done
            done
        done
    done
}


run_sklearn() {
    echo "--- Sklearn ---"
    for e in "${TREES[@]}"; do
        for t in 1 2 4 8 16 32; do
            for j in $(seq 1 5); do
                srun -N 1 -n 1 -c 32 \
                    python ./test/sklearn_rf.py $e 0 $t "$DATASET" --log
            done
        done
    done
}


if [[ $# -lt 1 ]]; then
    echo "Usage: $0 DATASET [seq] [omp] [ff] [mpi] [sklearn]"
    exit 1
fi

shift 1

if [[ $# -eq 0 ]]; then
    # default
    run_seq
    run_omp
    run_ff
    run_mpi
    run_sklearn
else
    for mode in "$@"; do
        case "$mode" in
            seq) run_seq ;;
            omp) run_omp ;;
            ff)  run_ff  ;;
            mpi) run_mpi ;;
            sklearn) run_sklearn ;;
            *)
                echo "Unknown mode: $mode"
                exit 1 ;;
        esac
    done
fi

python scripts/merge.py results/forest.csv
