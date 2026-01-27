import argparse

import numpy as np
import pandas as pd


def shared_mem_speedup(df, backend):
    seq_df = df[df["backend"] == "seq"]
    shm_df = df[df["backend"] == backend]

    df[df["backend"] == backend]["speedup"] = (
        seq_df["train_time"] / shm_df["train_time"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="path of the file")
    args = parser.parse_args()

    df = pd.read_csv(args.filepath)
    df["speedup"] = df["train_time"]
    shared_mem_speedup(df, "seq")
    shared_mem_speedup(df, "omp")
    shared_mem_speedup(df, "ff")
    print(df)
