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
    df = df[df["backend"] != "mpi"]

    dataset = args.title

    assert isinstance(df, pd.DataFrame)
    estimators = df["estimators"].unique()
    assert isinstance(estimators, np.ndarray)
    estimators.sort()

    assert isinstance(df, pd.DataFrame)
    threads = df["threads"].unique()
    assert isinstance(threads, np.ndarray)
    threads.sort()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True, dpi=100)

    ax1.set_title(f"{dataset} Training Weak Scaling", fontsize=14)
    ax2.set_title(f"{dataset} Prediction Weak Scaling", fontsize=14)

    blues = plt.cm.Blues(np.linspace(0.4, 0.8, len(estimators)))
    reds = plt.cm.Reds(np.linspace(0.4, 0.8, len(estimators)))

    omp_tr = []
    omp_pr = []
    ff_tr = []
    ff_pr = []
    for i, (e, t) in enumerate(zip(estimators, threads)):
        mask = (df["backend"] == "omp") | (df["backend"] == "seq")
        mask = mask & (df["estimators"] == e) & (df["threads"] == t)
        omp_tr.append(df[mask]["train_time"].to_numpy()[0])
        omp_pr.append(df[mask]["predict_time"].to_numpy()[0])

        mask = (df["backend"] == "ff") | (df["backend"] == "seq")
        mask = mask & (df["estimators"] == e) & (df["threads"] == t)
        ff_tr.append(df[mask]["train_time"].to_numpy()[0])
        ff_pr.append(df[mask]["predict_time"].to_numpy()[0])

    omp_tr = np.array(omp_tr)
    omp_tr = [omp_tr[i - 1] / omp_tr[i] for i in range(1, len(omp_tr), 1)]
    omp_tr.insert(0, 1.0)
    ff_tr = np.array(ff_tr)
    ff_tr = [ff_tr[i - 1] / ff_tr[i] for i in range(1, len(ff_tr), 1)]
    ff_tr.insert(0, 1.0)

    omp_pr = np.array(omp_pr)
    omp_pr = [omp_pr[i - 1] / omp_pr[i] for i in range(1, len(omp_pr), 1)]
    omp_pr.insert(0, 1.0)
    ff_pr = np.array(ff_pr)
    ff_pr = [ff_pr[i - 1] / ff_pr[i] for i in range(1, len(ff_pr), 1)]
    ff_pr.insert(0, 1.0)

    ax1.plot(
        threads,
        omp_tr,
        marker="o",
        label=f"OpenMP",
        color=blues[i],
    )
    ax1.plot(
        threads,
        ff_tr,
        marker="o",
        label=f"FastFlow",
        color=reds[i],
    )

    ax2.plot(
        threads,
        omp_pr,
        marker="o",
        label=f"OpenMP",
        color=blues[i],
    )
    ax2.plot(
        threads,
        ff_pr,
        marker="o",
        label=f"FastFlow",
        color=reds[i],
    )

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Threads-Estimators", fontsize=12)
    ax1.set_ylabel("Relative Speedup", fontsize=12)
    threads = [1, 2, 4, 8, 16, 32]
    ax1.set_xticks(threads, [f"{t}-{t * 8}" for t in threads])
    ax1.legend()
    ax1.grid()

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Threads-Estimators", fontsize=12)
    # ax2.set_ylabel("Time (ms)", fontsize=12)
    ax2.set_xticks(threads, [f"{t}-{t * 8}" for t in threads])
    ax2.legend()
    ax2.grid()

    plt.tight_layout()

    plt.savefig(args.output_file, dpi=300)
    plt.show()
