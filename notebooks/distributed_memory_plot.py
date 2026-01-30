import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def training_runtime(df):
    datasets = ["susy20000", "susy100000"]
    threads = df["threads"].unique().astype(int)
    nodes = df["nodes"].unique().astype(int)
    nodes.sort()

    _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, dpi=200)
    axes[0].set_title(f"SUSY 20k Training Time")
    axes[1].set_title(f"SUSY 100k Training Time")

    purples = mpl.colormaps["Purples"](np.linspace(0.4, 0.8, len(threads)))

    for i, t in enumerate(threads):
        for ax, d in zip(axes, datasets):
            mask = (df["threads"] == t) & (df["dataset"] == d)

            ax.plot(
                nodes,
                df[mask]["train_time"] / 1000,
                marker="o",
                color=purples[i],
                label=f"{t} threads",
            )

            ax.grid()
            ax.legend()

    axes[0].set_xlabel("Nodes")
    axes[0].set_ylabel("Time (sec)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(nodes, [str(t) for t in nodes])

    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Nodes")
    axes[1].set_xticks(nodes, [str(t) for t in nodes])

    plt.tight_layout()
    plt.savefig("../report/images/distributed_training_time.svg", dpi=300)
    plt.show()


def prediction_runtime(df):
    datasets = ["susy20000", "susy100000"]
    threads = df["threads"].unique().astype(int)
    nodes = df["nodes"].unique().astype(int)
    nodes.sort()

    _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, dpi=200)
    axes[0].set_title(f"SUSY 20k Prediction Time")
    axes[1].set_title(f"SUSY 100k Prediction Time")
    purples = mpl.colormaps["Purples"](np.linspace(0.4, 0.8, len(threads)))

    for i, t in enumerate(threads):
        for ax, d in zip(axes, datasets):
            mask = (df["threads"] == t) & (df["dataset"] == d)

            ax.plot(
                nodes,
                df[mask]["predict_time"],
                marker="o",
                color=purples[i],
                label=f"{t} threads",
            )

            ax.grid()
            ax.legend()

    axes[0].set_xlabel("Nodes")
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(nodes, [str(t) for t in nodes])

    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Nodes")
    axes[1].set_xlabel("Nodes")
    axes[1].set_xticks(nodes, [str(t) for t in nodes])

    plt.tight_layout()
    plt.savefig("../report/images/distributed_prediction_time.svg", dpi=300)
    plt.show()


def training_speedup(df):
    datasets = ["susy20000", "susy100000"]
    threads = df["threads"].unique().astype(int)
    nodes = df["nodes_mpi"].unique()
    nodes = np.append(nodes, df["nodes_omp"].unique()).astype(int)
    nodes.sort()

    _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, dpi=200)
    axes[0].set_title(f"SUSY 20k Training Speedup")
    axes[1].set_title(f"SUSY 100k Training Speedup")
    purples = mpl.colormaps["Purples"](np.linspace(0.4, 0.8, len(threads)))

    axes[0].plot(nodes, nodes, "g--", label="Ideal")
    axes[1].plot(nodes, nodes, "g--", label="Ideal")

    for i, t in enumerate(threads):
        for ax, d in zip(axes, datasets):
            mask = (df["threads"] == t) & (df["dataset"] == d)

            ax.plot(
                nodes,
                np.insert(df[mask]["train_speedup"], 0, 1.0),
                marker="o",
                color=purples[i],
                label=f"{t} threads",
            )

            ax.grid()
            ax.legend()

    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log", base=2)
    axes[0].set_xlabel("Nodes")
    axes[0].set_ylabel("Speedup")
    axes[0].set_xticks(nodes, [str(t) for t in nodes])
    axes[0].set_yticks(nodes, [str(t) for t in nodes])

    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log", base=2)
    axes[1].set_xlabel("Nodes")
    axes[1].set_xticks(nodes, [str(t) for t in nodes])
    axes[1].set_yticks(nodes, [str(t) for t in nodes])

    plt.tight_layout()
    plt.savefig("../report/images/distributed_training_speedup.svg", dpi=300)
    plt.show()


def prediction_speedup(df):
    datasets = ["susy20000", "susy100000"]
    threads = df["threads"].unique().astype(int)
    nodes = df["nodes_mpi"].unique()
    nodes = np.append(nodes, df["nodes_omp"].unique()).astype(int)
    nodes.sort()

    _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, dpi=200)
    axes[0].set_title(f"SUSY 20k Prediction Speedup")
    axes[1].set_title(f"SUSY 100k Prediction Speedup")
    purples = mpl.colormaps["Purples"](np.linspace(0.4, 0.8, len(threads)))

    axes[0].plot(nodes, nodes, "g--", label="Ideal")
    axes[1].plot(nodes, nodes, "g--", label="Ideal")

    for i, t in enumerate(threads):
        for ax, d in zip(axes, datasets):
            mask = (df["threads"] == t) & (df["dataset"] == d)

            ax.plot(
                nodes,
                np.insert(df[mask]["predict_speedup"], 0, 1.0),
                marker="o",
                color=purples[i],
                label=f"{t} threads",
            )

            ax.grid()
            ax.legend()

    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log", base=2)
    axes[0].set_xlabel("Nodes")
    axes[0].set_ylabel("Speedup")
    axes[0].set_xticks(nodes, [str(t) for t in nodes])
    # axes[0].set_yticks(nodes, [str(t) for t in nodes])

    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log", base=2)
    axes[1].set_xlabel("Nodes")
    axes[1].set_xticks(nodes, [str(t) for t in nodes])
    # axes[1].set_yticks(nodes, [str(t) for t in nodes])

    plt.tight_layout()
    plt.savefig("../report/images/distributed_prediction_speedup.svg", dpi=300)
    plt.show()


def prediction_efficiency(df):
    datasets = ["susy20000", "susy100000"]
    threads = df["threads"].unique().astype(int)
    nodes = df["nodes_mpi"].unique()
    nodes = np.append(nodes, df["nodes_omp"].unique()).astype(int)
    nodes.sort()

    _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, dpi=200)
    axes[0].set_title(f"SUSY 20k Prediction Efficiency")
    axes[1].set_title(f"SUSY 100k Prediction Efficiency")
    purples = mpl.colormaps["Purples"](np.linspace(0.4, 0.8, len(threads)))

    for i, t in enumerate(threads):
        for ax, d in zip(axes, datasets):
            mask = (df["threads"] == t) & (df["dataset"] == d)

            ax.plot(
                nodes,
                np.insert(df[mask]["predict_efficiency"], 0, 1.0),
                marker="o",
                color=purples[i],
                label=f"{t} threads",
            )

            ax.grid()
            ax.legend()

    axes[0].set_xlabel("Nodes")
    axes[0].set_ylabel("Efficiency")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(nodes, [str(t) for t in nodes])

    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Nodes")
    axes[1].set_xticks(nodes, [str(t) for t in nodes])

    plt.tight_layout()
    plt.savefig("../report/images/distributed_prediction_efficiency.svg", dpi=300)
    plt.show()


def training_efficiency(df):
    datasets = ["susy20000", "susy100000"]
    threads = df["threads"].unique().astype(int)
    nodes = df["nodes_mpi"].unique()
    nodes = np.append(nodes, df["nodes_omp"].unique()).astype(int)
    nodes.sort()

    _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, dpi=200)
    axes[0].set_title(f"SUSY 20k Training Efficiency")
    axes[1].set_title(f"SUSY 100k Training Efficiency")
    purples = mpl.colormaps["Purples"](np.linspace(0.4, 0.8, len(threads)))

    for i, t in enumerate(threads):
        for ax, d in zip(axes, datasets):
            mask = (df["threads"] == t) & (df["dataset"] == d)

            ax.plot(
                nodes,
                np.insert(df[mask]["train_efficiency"], 0, 1.0),
                marker="o",
                color=purples[i],
                label=f"{t} threads",
            )

            ax.grid()
            ax.legend()

    axes[0].set_xlabel("Nodes")
    axes[0].set_ylabel("Efficiency")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(nodes, [str(t) for t in nodes])

    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Nodes")
    axes[1].set_xticks(nodes, [str(t) for t in nodes])

    plt.tight_layout()
    plt.savefig("../report/images/distributed_training_efficiency.svg", dpi=300)
    plt.show()


def weak_scaling_of(df, dataset, label, field, ax, cmap):
    estimators = df["estimators"].unique().astype(int)
    estimators.sort()

    nodes = df["nodes"].unique().astype(int)
    nodes.sort()

    threads = df["threads"].unique().astype(int)
    threads.sort()
    values = []
    for i, t in enumerate(threads):
        for e, n in zip(estimators, nodes):
            mask = (
                (df["dataset"] == dataset)
                & (df["threads"] == t)
                & (df["estimators"] == e)
                & (df["nodes"] == n)
            )
            values.append(df[mask][field].iloc[0])
        values = [values[i - 1] / values[i] for i in range(1, len(values))]
        values.insert(0, 1)

        ax.plot(
            nodes,
            values,
            marker="o",
            color=cmap[i],
            label=f"{label} {t} threads",
        )
        values.clear()


def weak_scaling(df):
    datasets = ["susy20000", "susy100000"]

    df = df[(df["backend"] == "omp") | (df["backend"] == "mpi")]
    threads = df["threads"].unique().astype(int)
    nodes = df["nodes"].unique().astype(int)
    estimators = df["estimators"].unique().astype(int)

    _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, dpi=200)
    axes[0].set_title(f"SUSY Training Weak Scaling")
    axes[1].set_title(f"SUSY Prediction Weak Scaling")
    purples = mpl.colormaps["Purples"](np.linspace(0.4, 0.8, len(threads)))
    yellows = mpl.colormaps["Greens"](np.linspace(0.4, 0.8, len(threads)))

    weak_scaling_of(df, "susy20000", "SUSY 20k", "train_time", axes[0], purples)
    weak_scaling_of(df, "susy100000", "SUSY 100k", "train_time", axes[0], yellows)
    weak_scaling_of(df, "susy20000", "SUSY 20k", "predict_time", axes[1], purples)
    weak_scaling_of(df, "susy100000", "SUSY 100k", "predict_time", axes[1], yellows)

    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Nodes-Estimators", fontsize=12)
    axes[0].set_ylabel("Relative Speedup", fontsize=12)
    axes[0].set_xticks(nodes, [f"{t}-{e}" for t, e in zip(nodes, estimators)])
    axes[0].legend()
    axes[0].grid()

    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Nodes-Estimators", fontsize=12)
    axes[1].set_xticks(nodes, [f"{t}-{e}" for t, e in zip(nodes, estimators)])
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.savefig("report/images/distributed_weak_scaling.svg")
    plt.show()


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("results/forest.csv")
    df = df[(df["backend"] == "mpi") | (df["backend"] == "omp")]
    df = df[(df["threads"] >= 16)]
    df = df[(df["estimators"] >= 32)]
    df = df[(df["dataset"] == "susy100000") | (df["dataset"] == "susy20000")]
    df = df[
        (df["nodes"] == 1)
        | (df["nodes"] == 2)
        | (df["nodes"] == 4)
        | (df["nodes"] == 8)
    ]
    INPUT_COLS = ["dataset", "estimators", "max_depth", "backend", "nodes", "threads"]
    OUTPUT_COLS = ["train_time", "predict_time"]
    df = df.groupby(by=INPUT_COLS, as_index=False)[OUTPUT_COLS].mean()
    df = df.sort_values(by=["dataset", "nodes", "threads"])

    weak_scaling(df)
