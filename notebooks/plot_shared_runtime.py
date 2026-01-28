import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("results/forest.csv")

    INPUT_COLS = ["dataset", "estimators", "max_depth", "backend", "nodes", "threads"]
    OUTPUT_COLS = ["accuracy", "f1", "train_time", "predict_time"]

    df = df.groupby(by=INPUT_COLS, as_index=False)[OUTPUT_COLS].mean()

    # plot_shared_memory_runtime(df)

    df = df[df["dataset"] == "susy20000"]
    df = df[df["estimators"] >= 64]
    df = df[df["backend"] != "mpi"]

    dataset = "SUSY 20k"

    assert isinstance(df, pd.DataFrame)
    estimators = df["estimators"].unique()
    assert isinstance(estimators, np.ndarray)
    estimators.sort()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=100)

    ax1.set_title(f"{dataset} Training Time", fontsize=14)
    ax2.set_title(f"{dataset} Prediction Time", fontsize=14)

    blues = plt.cm.Blues(np.linspace(0.4, 0.8, len(estimators)))
    reds = plt.cm.Reds(np.linspace(0.4, 0.8, len(estimators)))
    greens = plt.cm.Greens(np.linspace(0.4, 0.8, len(estimators)))

    for i, e in enumerate(estimators):
        df_e = df[df["estimators"] == e]
        omp = df_e[(df_e["backend"] == "omp") | (df_e["backend"] == "seq")]
        ff = df_e[(df_e["backend"] == "ff") | (df_e["backend"] == "seq")]
        jl = df_e[(df_e["backend"] == "joblib")]

        assert isinstance(omp, pd.DataFrame)
        omp = omp.sort_values(["threads"])
        assert isinstance(ff, pd.DataFrame)
        ff = ff.sort_values(["threads"])
        assert isinstance(jl, pd.DataFrame)
        jl = jl.sort_values(["threads"])

        ax1.plot(
            omp["threads"],
            omp["train_time"] / 1000,
            marker="o",
            label=f"OpenMP {e} estimators",
            color=blues[i],
        )
        ax1.plot(
            ff["threads"],
            ff["train_time"] / 1000,
            marker="o",
            label=f"FastFlow {e} estimators",
            color=reds[i],
        )
        # ax1.plot(
        #     omp["threads"],
        #     omp["train_time"] / 1000,
        #     marker="o",
        #     label=f"Sklearn {e} estimators",
        #     color=greens[i],
        # )

        ax2.plot(
            omp["threads"],
            omp["predict_time"],
            marker="o",
            label=f"OpenMP {e} estimators",
            color=blues[i],
        )
        ax2.plot(
            ff["threads"],
            ff["predict_time"],
            marker="o",
            label=f"FastFlow {e} estimators",
            color=reds[i],
        )

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Threads", fontsize=12)
    ax1.set_ylabel("Time (sec)", fontsize=12)
    threads = [1, 2, 4, 8, 16, 32]
    ax1.set_xticks(threads, [str(t) for t in threads])
    ax1.legend()
    ax1.grid()

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Threads", fontsize=12)
    ax2.set_ylabel("Time (ms)", fontsize=12)
    ax2.set_xticks(threads, [str(t) for t in threads])
    ax2.legend()
    ax2.grid()

    plt.tight_layout()

    plt.savefig(f"report/images/{dataset.lower()}_runtime.svg", dpi=300)
    plt.show()
