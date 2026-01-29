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
    df = df[df["estimators"] >= 64]
    df = df[df["backend"] != "mpi"]

    dataset = args.title

    assert isinstance(df, pd.DataFrame)
    estimators = df["estimators"].unique()
    assert isinstance(estimators, np.ndarray)
    estimators.sort()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, dpi=100)

    ax1.set_title(f"{dataset} Training Speedup", fontsize=14)
    ax2.set_title(f"{dataset} Prediction Speedup", fontsize=14)

    blues = plt.cm.Blues(np.linspace(0.4, 0.8, len(estimators)))
    reds = plt.cm.Reds(np.linspace(0.4, 0.8, len(estimators)))
    greens = plt.cm.Greens(np.linspace(0.4, 0.8, len(estimators)))

    for i, e in enumerate(estimators):
        df_e = df[df["estimators"] == e]

        seq = df_e[df_e["backend"] == "seq"]
        omp = df_e[df_e["backend"] == "omp"]
        ff = df_e[df_e["backend"] == "ff"]

        assert isinstance(omp, pd.DataFrame)
        omp = omp.sort_values(["threads"])
        assert isinstance(ff, pd.DataFrame)
        ff = ff.sort_values(["threads"])

        omp_su = [seq["train_time"] / t for t in omp["train_time"]]
        ff_su = [seq["train_time"] / t for t in ff["train_time"]]
        ax1.plot(
            omp["threads"],
            omp_su,
            marker="o",
            label=f"OpenMP {e} estimators",
            color=blues[i],
        )
        ax1.plot(
            ff["threads"],
            ff_su,
            marker="o",
            label=f"FastFlow {e} estimators",
            color=reds[i],
        )

        omp_su = [seq["predict_time"] / t for t in omp["predict_time"]]
        ff_su = [seq["predict_time"] / t for t in ff["predict_time"]]
        ax2.plot(
            omp["threads"],
            omp_su,
            marker="o",
            label=f"OpenMP {e} estimators",
            color=blues[i],
        )
        ax2.plot(
            ff["threads"],
            ff_su,
            marker="o",
            label=f"FastFlow {e} estimators",
            color=reds[i],
        )

    threads = [2, 4, 8, 16, 32]
    ax1.plot(
        threads,
        threads,
        "g--",
        label="Ideal",
    )
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=2)
    ax1.set_xlabel("Threads", fontsize=12)
    ax1.set_ylabel("Speedup", fontsize=12)
    ax1.set_xticks(threads, [str(t) for t in threads])
    ax1.set_yticks(threads, [str(t) for t in threads])
    ax1.legend()
    ax1.grid()

    ax2.plot(
        threads,
        threads,
        "g--",
        label="Ideal",
    )
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=2)
    ax2.set_xlabel("Threads", fontsize=12)
    ax2.set_xticks(threads, [str(t) for t in threads])
    ax2.set_yticks(threads, [str(t) for t in threads])
    ax2.legend()
    ax2.grid()

    plt.tight_layout()

    plt.savefig(args.output_file, dpi=300)
    plt.show()
