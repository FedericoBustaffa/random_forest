import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def training_runtime(df):
    datasets = ["susy20000", "susy100000"]
    threads = df["threads"].unique()
    nodes = df["nodes"].unique()
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

    axes[1].set_xlabel("Nodes")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(nodes, [str(t) for t in nodes])

    plt.tight_layout()
    plt.savefig("../report/images/distributed_training_time.svg", dpi=300)
    plt.show()


def prediction_runtime(df):
    datasets = ["susy20000", "susy100000"]
    threads = df["threads"].unique()
    nodes = df["nodes"].unique()
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

    axes[1].set_xlabel("Nodes")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(nodes, [str(t) for t in nodes])

    plt.tight_layout()
    plt.savefig("../report/images/distributed_prediction_time.svg", dpi=300)
    plt.show()


def training_speedup(df):
    datasets = ["susy20000", "susy100000"]
    threads = df["threads"].unique()
    nodes = df["nodes_mpi"].unique()

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
                df[mask]["train_speedup"],
                marker="o",
                color=purples[i],
                label=f"{t} threads",
            )

            ax.grid()
            ax.legend()

    axes[0].set_xlabel("Nodes")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log", base=2)
    axes[0].set_ylabel("Speedup")
    axes[0].set_xticks(nodes, [str(t) for t in nodes])
    axes[0].set_yticks(nodes, [str(t) for t in nodes])

    axes[1].set_xlabel("Nodes")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log", base=2)
    axes[1].set_xticks(nodes, [str(t) for t in nodes])
    axes[1].set_yticks(nodes, [str(t) for t in nodes])

    plt.tight_layout()
    plt.savefig("../report/images/distributed_training_speedup.svg", dpi=300)
    plt.show()


def prediction_speedup(df):
    datasets = ["susy20000", "susy100000"]
    threads = df["threads"].unique()
    nodes = df["nodes_mpi"].unique()

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
                df[mask]["predict_speedup"],
                marker="o",
                color=purples[i],
                label=f"{t} threads",
            )

            ax.grid()
            ax.legend()

    axes[0].set_xlabel("Nodes")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log", base=2)
    axes[0].set_ylabel("Speedup")
    axes[0].set_xticks(nodes, [str(t) for t in nodes])
    # axes[0].set_yticks(nodes, [str(t) for t in nodes])

    axes[1].set_xlabel("Nodes")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log", base=2)
    axes[1].set_xticks(nodes, [str(t) for t in nodes])
    # axes[1].set_yticks(nodes, [str(t) for t in nodes])

    plt.tight_layout()
    plt.savefig("../report/images/distributed_prediction_speedup.svg", dpi=300)
    plt.show()


def prediction_efficiency(df):
    datasets = ["susy20000", "susy100000"]
    threads = df["threads"].unique()
    nodes = df["nodes_mpi"].unique()

    _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, dpi=200)
    axes[0].set_title(f"SUSY 20k Prediction Efficiency")
    axes[1].set_title(f"SUSY 100k Prediction Efficiency")
    purples = mpl.colormaps["Purples"](np.linspace(0.4, 0.8, len(threads)))

    for i, t in enumerate(threads):
        for ax, d in zip(axes, datasets):
            mask = (df["threads"] == t) & (df["dataset"] == d)

            ax.plot(
                nodes,
                df[mask]["predict_efficiency"],
                marker="o",
                color=purples[i],
                label=f"{t} threads",
            )

            ax.grid()
            ax.legend()

    axes[0].set_xlabel("Nodes")
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylabel("Efficiency")
    axes[0].set_xticks(nodes, [str(t) for t in nodes])

    axes[1].set_xlabel("Nodes")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(nodes, [str(t) for t in nodes])

    plt.tight_layout()
    plt.savefig("../report/images/distributed_prediction_efficiency.svg", dpi=300)
    plt.show()


def training_efficiency(df):
    datasets = ["susy20000", "susy100000"]
    threads = df["threads"].unique()
    nodes = df["nodes_mpi"].unique()

    _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, dpi=200)
    axes[0].set_title(f"SUSY 20k Training Efficiency")
    axes[1].set_title(f"SUSY 100k Training Efficiency")
    purples = mpl.colormaps["Purples"](np.linspace(0.4, 0.8, len(threads)))

    for i, t in enumerate(threads):
        for ax, d in zip(axes, datasets):
            mask = (df["threads"] == t) & (df["dataset"] == d)

            ax.plot(
                nodes,
                df[mask]["train_efficiency"],
                marker="o",
                color=purples[i],
                label=f"{t} threads",
            )

            ax.grid()
            ax.legend()

    axes[0].set_xlabel("Nodes")
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylabel("Efficiency")
    axes[0].set_xticks(nodes, [str(t) for t in nodes])

    axes[1].set_xlabel("Nodes")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(nodes, [str(t) for t in nodes])

    plt.tight_layout()
    plt.savefig("../report/images/distributed_training_efficiency.svg", dpi=300)
    plt.show()


def weak_scaling_of(df, estimators, threads, dataset, backend, metric):
    ws = []
    for e, t in zip(estimators, threads):
        mask = (df["backend"] == backend) | (df["backend"] == "seq")
        mask = mask & (df["estimators"] == e)
        mask = mask & (df["threads"] == t)
        mask = mask & (df["dataset"] == dataset)
        ws.append(df[mask][metric].iloc[0])

    ws = [ws[i - 1] / ws[i] for i in range(1, len(estimators), 1)]
    ws.insert(0, 1.0)

    return ws


def weak_scaling(df):
    tmp = df.sort_values(by=["estimators", "threads"])

    estimators = tmp["estimators"].unique()
    threads = tmp["threads"].unique()

    _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, dpi=200)
    axes[0].set_title(f"Training Weak Scaling")
    axes[1].set_title(f"Prediction Weak Scaling")

    blues = mpl.colormaps["Blues"](np.linspace(0.4, 0.8, 2))
    reds = mpl.colormaps["Reds"](np.linspace(0.4, 0.8, 2))

    omp_magic_train = weak_scaling_of(
        df, estimators, threads, "susy100000", "omp", "train_time"
    )
    ff_magic_train = weak_scaling_of(
        df, estimators, threads, "susy100000", "ff", "train_time"
    )
    omp_susy_train = weak_scaling_of(
        df, estimators, threads, "susy20000", "omp", "train_time"
    )
    ff_susy_train = weak_scaling_of(
        df, estimators, threads, "susy20000", "ff", "train_time"
    )

    omp_magic_predict = weak_scaling_of(
        df, estimators, threads, "susy100000", "omp", "predict_time"
    )
    ff_magic_predict = weak_scaling_of(
        df, estimators, threads, "susy100000", "ff", "predict_time"
    )
    omp_susy_predict = weak_scaling_of(
        df, estimators, threads, "susy20000", "omp", "predict_time"
    )
    ff_susy_predict = weak_scaling_of(
        df, estimators, threads, "susy20000", "ff", "predict_time"
    )

    axes[0].plot(
        threads,
        omp_magic_train,
        marker="o",
        label=f"OpenMP susy100000",
        color=blues[0],
    )

    axes[0].plot(
        threads,
        ff_magic_train,
        marker="o",
        label=f"FastFlow susy100000",
        color=reds[0],
    )

    axes[0].plot(
        threads,
        omp_susy_train,
        marker="o",
        label=f"OpenMP SUSY",
        color=blues[1],
    )

    axes[0].plot(
        threads,
        ff_susy_train,
        marker="o",
        label=f"FastFlow SUSY",
        color=reds[1],
    )

    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Threads-Estimators", fontsize=12)
    axes[0].set_ylabel("Relative Speedup", fontsize=12)
    threads = [1, 2, 4, 8, 16, 32]
    axes[0].set_xticks(threads, [f"{t}-{t * 8}" for t in threads])
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(
        threads,
        omp_magic_predict,
        marker="o",
        label=f"OpenMP susy100000",
        color=blues[0],
    )

    axes[1].plot(
        threads,
        ff_magic_predict,
        marker="o",
        label=f"FastFlow susy100000",
        color=reds[0],
    )

    axes[1].plot(
        threads,
        omp_susy_predict,
        marker="o",
        label=f"OpenMP SUSY",
        color=blues[1],
    )

    axes[1].plot(
        threads,
        ff_susy_predict,
        marker="o",
        label=f"FastFlow SUSY",
        color=reds[1],
    )

    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Threads-Estimators", fontsize=12)
    axes[1].set_xticks(threads, [f"{t}-{t * 8}" for t in threads])
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()

    plt.savefig("../report/images/distributed_weak_scaling.svg", dpi=300)
    plt.show()
