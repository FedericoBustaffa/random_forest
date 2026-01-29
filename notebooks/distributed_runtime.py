import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("title", type=str, help="title of the plot")
    parser.add_argument("dataset", type=str, help="name of the dataset")
    parser.add_argument("output_file", type=str, help="filepath to save the plot")
    args = parser.parse_args()

    df = pd.read_csv("results/forest.csv")
    INPUT_COLS = ["dataset", "estimators", "max_depth", "backend", "nodes", "threads"]
    OUTPUT_COLS = ["accuracy", "f1", "train_time", "predict_time"]
    df = df.groupby(by=INPUT_COLS, as_index=False)[OUTPUT_COLS].mean()

    df = df[df["dataset"] == args.dataset]
    df = df[df["estimators"] == 128]
    df = df[(df["backend"] == "omp") | (df["backend"] == "mpi")]

    dataset = args.title

    assert isinstance(df, pd.DataFrame)
    threads = df["threads"].unique()
    assert isinstance(threads, np.ndarray)
    threads.sort()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=100)

    ax1.set_title(f"{dataset} Training Time", fontsize=14)
    ax2.set_title(f"{dataset} Prediction Time", fontsize=14)

    greens = plt.cm.Greens(np.linspace(0.4, 0.8, len(threads)))

    for i, t in enumerate(threads):
        df_e = df[df["threads"] == t]
        omp = df_e[(df_e["backend"] == "omp") | (df_e["backend"] == "mpi")]

        assert isinstance(omp, pd.DataFrame)
        omp = omp.sort_values(["nodes"])

        ax1.plot(
            omp["nodes"],
            omp["train_time"] / 1000,
            marker="o",
            label=f"MPI {t} threads",
            color=greens[i],
        )

        ax2.plot(
            omp["nodes"],
            omp["predict_time"],
            marker="o",
            label=f"MPI {t} threads",
            color=greens[i],
        )

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Nodes", fontsize=12)
    ax1.set_ylabel("Time (sec)", fontsize=12)
    nodes = [1, 2, 4, 6, 8]
    ax1.set_xticks(nodes, [str(t) for t in nodes])
    ax1.legend()
    ax1.grid()

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Nodes", fontsize=12)
    ax2.set_ylabel("Time (ms)", fontsize=12)
    ax2.set_xticks(nodes, [str(t) for t in nodes])
    ax2.legend()
    ax2.grid()

    plt.tight_layout()

    # plt.savefig(args.output_file, dpi=300)
    plt.show()
