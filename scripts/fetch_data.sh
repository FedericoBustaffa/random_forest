#!/usr/bin/env bash

set -e

DATA_DIR="datasets"
IRIS_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
IRIS_FILE="${DATA_DIR}/iris.csv"

mkdir -p "${DATA_DIR}"

if [ ! -f "${IRIS_FILE}" ]; then
    curl -L "${IRIS_URL}" -o "${IRIS_FILE}"
    echo "Dataset Iris downloaded"
else
    echo "Dataset Iris already present"
fi
