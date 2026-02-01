import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def runtime_2x2(df):
    datasets = ["susy20000", "susy100000"]
    dataset_titles = ["SUSY 20k", "SUSY 100k"]
    metrics = [
        ("train_time", "Training Time (s)", 1000),  # ms → s
        ("predict_time", "Prediction Time (ms)", 1),
    ]

    threads = np.sort(df["threads"].unique().astype(int))
    nodes = np.sort(df["nodes"].unique().astype(int))

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, dpi=200)

    purples = mpl.colormaps["Purples"](np.linspace(0.4, 0.8, len(threads)))

    for row, (metric, ylabel, scale) in enumerate(metrics):
        for col, (dataset, title) in enumerate(zip(datasets, dataset_titles)):
            ax = axes[row, col]
            ax.set_title(f"{title} – {ylabel}")

            for i, t in enumerate(threads):
                mask = (df["threads"] == t) & (df["dataset"] == dataset)

                ax.plot(
                    nodes,
                    df[mask][metric] / scale,
                    marker=".",
                    color=purples[i],
                    label=f"{t} threads",
                )

            ax.grid()
            ax.set_xscale("log", base=2)
            ax.set_xticks(nodes, [str(n) for n in nodes])

    # etichette comuni
    axes[1, 0].set_xlabel("Nodes")
    axes[1, 1].set_xlabel("Nodes")
    axes[0, 0].set_ylabel("Time (s)")
    axes[1, 0].set_ylabel("Time (ms)")

    # legenda globale
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("../report/images/distributed_runtime.svg", dpi=300)
    plt.show()


def training_runtime(df):
    datasets = ["susy20000", "susy100000"]
    threads = df["threads"].unique().astype(int)
    nodes = df["nodes"].unique().astype(int)
    nodes.sort()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), dpi=200)
    axes[0].set_title(f"SUSY 20k Training Time")
    axes[1].set_title(f"SUSY 100k Training Time")

    purples = mpl.colormaps["Purples"](np.linspace(0.4, 0.8, len(threads)))

    for i, t in enumerate(threads):
        for ax, d in zip(axes, datasets):
            mask = (df["threads"] == t) & (df["dataset"] == d)

            ax.plot(
                nodes,
                df[mask]["train_time"] / 1000,
                marker=".",
                color=purples[i],
                label=f"{t} threads",
            )

            ax.grid()

    axes[0].set_xlabel("Nodes")
    axes[0].set_ylabel("Time (sec)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(nodes, [str(t) for t in nodes])

    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Nodes")
    axes[1].set_xticks(nodes, [str(t) for t in nodes])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig("../report/images/distributed_training_time.svg", dpi=300)
    plt.show()


def prediction_runtime(df):
    datasets = ["susy20000", "susy100000"]
    threads = df["threads"].unique().astype(int)
    nodes = df["nodes"].unique().astype(int)
    nodes.sort()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), dpi=200)
    axes[0].set_title(f"SUSY 20k Prediction Time")
    axes[1].set_title(f"SUSY 100k Prediction Time")
    purples = mpl.colormaps["Purples"](np.linspace(0.4, 0.8, len(threads)))

    for i, t in enumerate(threads):
        for ax, d in zip(axes, datasets):
            mask = (df["threads"] == t) & (df["dataset"] == d)

            ax.plot(
                nodes,
                df[mask]["predict_time"],
                marker=".",
                color=purples[i],
                label=f"{t} threads",
            )

            ax.grid()

    axes[0].set_xlabel("Nodes")
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(nodes, [str(t) for t in nodes])

    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Nodes")
    axes[1].set_xlabel("Nodes")
    axes[1].set_xticks(nodes, [str(t) for t in nodes])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig("../report/images/distributed_prediction_time.svg", dpi=300)
    plt.show()


def distributed_speedup(df):
    datasets = {
        "susy20000": "SUSY 20k",
        "susy100000": "SUSY 100k",
    }

    threads = np.sort(df["threads"].unique().astype(int))
    nodes = np.unique(
        np.concatenate([df["nodes_mpi"].values, df["nodes_omp"].values])
    ).astype(int)
    nodes.sort()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), dpi=200)

    purples = mpl.colormaps["Purples"](np.linspace(0.4, 0.85, len(threads)))
    greens = mpl.colormaps["Greens"](np.linspace(0.4, 0.85, len(threads)))

    for ax, metric, title in zip(
        axes,
        ["train_speedup", "predict_speedup"],
        ["Training Speedup", "Prediction Speedup"],
    ):
        # ideal line
        ax.plot(nodes, nodes, "r--", label="Ideal")

        c = 0
        for d, dlabel in datasets.items():
            cmap = purples if d == "susy20000" else greens
            for i, t in enumerate(threads):
                mask = (df["dataset"] == d) & (df["threads"] == t)

                if mask.sum() == 0:
                    continue

                y = np.insert(df[mask][metric].values, 0, 1.0)

                ax.plot(
                    nodes,
                    y,
                    marker=".",
                    color=cmap[i],
                    label=f"{dlabel} – {t} threads",
                )
                c += 1

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(nodes, nodes)
        ax.set_yticks(nodes, nodes)
        ax.set_xlabel("Nodes")
        ax.set_title(title)
        ax.grid()

    axes[0].set_ylabel("Speedup")

    # legenda comune (una sola)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig("../report/images/distributed_speedup.svg", dpi=300)
    plt.show()


def distributed_efficiency(df):
    datasets = {
        "susy20000": "SUSY 20k",
        "susy100000": "SUSY 100k",
    }

    threads = np.sort(df["threads"].unique().astype(int))
    nodes = np.unique(
        np.concatenate([df["nodes_mpi"].values, df["nodes_omp"].values])
    ).astype(int)
    nodes.sort()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), dpi=200)

    purples = mpl.colormaps["Purples"](np.linspace(0.4, 0.85, len(threads)))
    greens = mpl.colormaps["Greens"](np.linspace(0.4, 0.85, len(threads)))

    for ax, metric, title in zip(
        axes,
        ["train_efficiency", "predict_efficiency"],
        ["Training Efficiency", "Prediction Efficiency"],
    ):
        c = 0
        for d, dlabel in datasets.items():
            cmap = purples if d == "susy20000" else greens
            for i, t in enumerate(threads):
                mask = (df["dataset"] == d) & (df["threads"] == t)

                if mask.sum() == 0:
                    continue

                y = np.insert(df[mask][metric].values, 0, 1.0)

                ax.plot(
                    nodes,
                    y,
                    marker=".",
                    color=cmap[i],
                    label=f"{dlabel} – {t} threads",
                )
                c += 1

        ax.set_xscale("log", base=2)
        # ax.set_yscale("log", base=2)
        ax.set_xticks(nodes, nodes)
        # ax.set_yticks(nodes, nodes)
        ax.set_xlabel("Nodes")
        ax.set_title(title)
        ax.grid()

    axes[0].set_ylabel("Efficiency")

    # legenda comune (una sola)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig("../report/images/distributed_efficiency.svg", dpi=300)
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
            marker=".",
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

    _, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True, dpi=200)
    axes[0].set_title(f"Training Weak Scaling")
    axes[1].set_title(f"Prediction Weak Scaling")
    purples = mpl.colormaps["Purples"](np.linspace(0.4, 0.8, len(threads)))
    yellows = mpl.colormaps["Greens"](np.linspace(0.4, 0.8, len(threads)))

    weak_scaling_of(df, "susy20000", "SUSY 20k", "train_time", axes[0], purples)
    weak_scaling_of(df, "susy100000", "SUSY 100k", "train_time", axes[0], yellows)
    weak_scaling_of(df, "susy20000", "SUSY 20k", "predict_time", axes[1], purples)
    weak_scaling_of(df, "susy100000", "SUSY 100k", "predict_time", axes[1], yellows)

    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Nodes-Estimators")
    axes[0].set_ylabel("Relative Speedup")
    axes[0].set_xticks(nodes, [f"{t}-{e}" for t, e in zip(nodes, estimators)])
    axes[0].grid()

    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Nodes-Estimators")
    axes[1].set_xticks(nodes, [f"{t}-{e}" for t, e in zip(nodes, estimators)])
    axes[1].grid()

    # legenda unica di figura
    handles, labels = axes[0].get_legend_handles_labels()
    fig = plt.gcf()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig("report/images/distributed_weak_scaling.svg")
    plt.show()


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("results/forest.csv")
    df = df[(df["backend"] == "mpi") | (df["backend"] == "omp")]
    df = df[(df["threads"] >= 8)]
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
