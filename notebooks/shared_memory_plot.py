import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def training_runtime(df, min_estimators):
    tmp = df[df["estimators"] >= min_estimators]
    tmp = tmp.sort_values(by=["threads"])

    datasets = ["magic", "susy20000"]
    estimators = tmp["estimators"].unique().astype(int)
    threads = tmp["threads"].unique().astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True, dpi=200)
    axes[0].set_title(f"Magic Training Time")
    axes[1].set_title(f"SUSY 20k Training Time")

    blues = mpl.colormaps["Blues"](np.linspace(0.4, 0.8, len(estimators)))
    reds = mpl.colormaps["Reds"](np.linspace(0.4, 0.8, len(estimators)))

    for i, e in enumerate(estimators):
        for ax, d in zip(axes, datasets):
            mask = (tmp["estimators"] == e) & (tmp["dataset"] == d)
            omp_mask = mask & ((tmp["backend"] == "seq") | (tmp["backend"] == "omp"))

            ax.plot(
                threads,
                tmp[omp_mask]["train_time"] / 1000,
                marker=".",
                color=blues[i],
                label=f"OpenMP {e} trees",
            )

            ff_mask = mask & ((tmp["backend"] == "seq") | (tmp["backend"] == "ff"))
            ax.plot(
                threads,
                tmp[ff_mask]["train_time"] / 1000,
                marker=".",
                color=reds[i],
                label=f"FastFlow {e} trees",
            )

            ax.grid()
            # ax.legend()

    axes[0].set_xlabel("Threads")
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylabel("Time (sec)")
    axes[0].set_xticks(threads, [str(t) for t in threads])

    axes[1].set_xlabel("Threads")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(threads, [str(t) for t in threads])

    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(
    #     handles,
    #     labels,
    #     loc="lower center",
    #     ncol=3,
    #     frameon=False,
    # )

    plt.tight_layout()
    plt.savefig("../report/images/shared_training_time.svg", dpi=300)
    plt.show()


def prediction_runtime(df, min_estimators):
    tmp = df[df["estimators"] >= min_estimators]
    tmp = tmp.sort_values(by=["threads"])

    datasets = ["magic", "susy20000"]
    estimators = tmp["estimators"].unique().astype(int)
    threads = tmp["threads"].unique().astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True, dpi=200)
    axes[0].set_title(f"Magic Prediction Time")
    axes[1].set_title(f"SUSY 20k Prediction Time")

    blues = mpl.colormaps["Blues"](np.linspace(0.4, 0.8, len(estimators)))
    reds = mpl.colormaps["Reds"](np.linspace(0.4, 0.8, len(estimators)))

    for i, e in enumerate(estimators):
        for ax, d in zip(axes, datasets):
            mask = (tmp["estimators"] == e) & (tmp["dataset"] == d)
            omp_mask = mask & ((tmp["backend"] == "seq") | (tmp["backend"] == "omp"))

            ax.plot(
                threads,
                tmp[omp_mask]["predict_time"],
                marker=".",
                color=blues[i],
                label=f"OpenMP {e} trees",
            )

            ff_mask = mask & ((tmp["backend"] == "seq") | (tmp["backend"] == "ff"))
            ax.plot(
                threads,
                tmp[ff_mask]["predict_time"],
                marker=".",
                color=reds[i],
                label=f"FastFlow {e} trees",
            )

            ax.grid()

    axes[0].set_xlabel("Threads")
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_xticks(threads, [str(t) for t in threads])

    axes[1].set_xlabel("Threads")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(threads, [str(t) for t in threads])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig("../report/images/shared_prediction_time.svg", dpi=300)
    plt.show()


def speedup(df, min_estimators):
    tmp = df[df["estimators"] >= min_estimators]
    tmp = tmp.sort_values(by=["threads"])

    datasets = ["magic", "susy20000"]
    dataset_titles = ["Magic", "SUSY 20k"]
    metrics = [
        ("train_speedup", "Training Speedup"),
        ("predict_speedup", "Prediction Speedup"),
    ]

    estimators = tmp["estimators"].unique().astype(int)
    threads = tmp["threads"].unique().astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True, dpi=200)

    blues = mpl.colormaps["Blues"](np.linspace(0.4, 0.8, len(estimators)))
    reds = mpl.colormaps["Reds"](np.linspace(0.4, 0.8, len(estimators)))

    for row, (metric, row_title) in enumerate(metrics):
        for col, (dataset, col_title) in enumerate(zip(datasets, dataset_titles)):
            ax = axes[row, col]

            ax.set_title(f"{col_title} – {row_title}")
            ax.plot(threads, threads, "g--", label="Ideal")

            for i, e in enumerate(estimators):
                mask = (tmp["estimators"] == e) & (tmp["dataset"] == dataset)

                omp_mask = mask & (
                    (tmp["backend"] == "seq") | (tmp["backend"] == "omp")
                )
                ax.plot(
                    threads,
                    tmp[omp_mask][metric],
                    marker=".",
                    color=blues[i],
                    label=f"OpenMP {e} trees",
                )

                ff_mask = mask & ((tmp["backend"] == "seq") | (tmp["backend"] == "ff"))
                ax.plot(
                    threads,
                    tmp[ff_mask][metric],
                    marker=".",
                    color=reds[i],
                    label=f"FastFlow {e} trees",
                )

            ax.grid()
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
            ax.set_xticks(threads, [str(t) for t in threads])
            ax.set_yticks(threads, [str(t) for t in threads])

    # etichette comuni
    axes[1, 0].set_xlabel("Threads")
    axes[1, 1].set_xlabel("Threads")
    axes[0, 0].set_ylabel("Speedup")
    axes[1, 0].set_ylabel("Speedup")

    # legenda globale
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig("../report/images/shared_speedup.svg", dpi=300)
    plt.show()


def training_speedup(df, min_estimators):
    tmp = df[df["estimators"] >= min_estimators]
    tmp = tmp.sort_values(by=["threads"])

    datasets = ["magic", "susy20000"]
    estimators = tmp["estimators"].unique().astype(int)
    threads = tmp["threads"].unique().astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True, dpi=200)
    axes[0].set_title(f"Magic Training Speedup")
    axes[1].set_title(f"SUSY 20k Training Speedup")
    axes[0].plot(threads, threads, "g--", label="Ideal")
    axes[1].plot(threads, threads, "g--", label="Ideal")

    blues = mpl.colormaps["Blues"](np.linspace(0.4, 0.8, len(estimators)))
    reds = mpl.colormaps["Reds"](np.linspace(0.4, 0.8, len(estimators)))

    for i, e in enumerate(estimators):
        for ax, d in zip(axes, datasets):
            mask = (tmp["estimators"] == e) & (tmp["dataset"] == d)
            omp_mask = mask & ((tmp["backend"] == "seq") | (tmp["backend"] == "omp"))

            ax.plot(
                threads,
                tmp[omp_mask]["train_speedup"],
                marker=".",
                color=blues[i],
                label=f"OpenMP {e} trees",
            )

            ff_mask = mask & ((tmp["backend"] == "seq") | (tmp["backend"] == "ff"))
            ax.plot(
                threads,
                tmp[ff_mask]["train_speedup"],
                marker=".",
                color=reds[i],
                label=f"FastFlow {e} trees",
            )

            ax.grid()

    axes[0].set_xlabel("Threads")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log", base=2)
    axes[0].set_ylabel("Speedup")
    axes[0].set_xticks(threads, [str(t) for t in threads])
    axes[0].set_yticks(threads, [str(t) for t in threads])

    axes[1].set_xlabel("Threads")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log", base=2)
    axes[1].set_xticks(threads, [str(t) for t in threads])
    axes[1].set_yticks(threads, [str(t) for t in threads])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig("../report/images/shared_training_speedup.svg", dpi=300)
    plt.show()


def prediction_speedup(df, min_estimators):
    tmp = df[df["estimators"] >= min_estimators]
    tmp = tmp.sort_values(by=["threads"])

    datasets = ["magic", "susy20000"]
    estimators = tmp["estimators"].unique().astype(int)
    threads = tmp["threads"].unique().astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True, dpi=200)
    axes[0].set_title(f"Magic Prediction Speedup")
    axes[1].set_title(f"SUSY 20k Prediction Speedup")
    axes[0].plot(threads, threads, "g--", label="Ideal")
    axes[1].plot(threads, threads, "g--", label="Ideal")

    blues = mpl.colormaps["Blues"](np.linspace(0.4, 0.8, len(estimators)))
    reds = mpl.colormaps["Reds"](np.linspace(0.4, 0.8, len(estimators)))

    for i, e in enumerate(estimators):
        for ax, d in zip(axes, datasets):
            mask = (tmp["estimators"] == e) & (tmp["dataset"] == d)
            omp_mask = mask & ((tmp["backend"] == "seq") | (tmp["backend"] == "omp"))

            ax.plot(
                threads,
                tmp[omp_mask]["predict_speedup"],
                marker=".",
                color=blues[i],
                label=f"OpenMP {e} trees",
            )

            ff_mask = mask & ((tmp["backend"] == "seq") | (tmp["backend"] == "ff"))
            ax.plot(
                threads,
                tmp[ff_mask]["predict_speedup"],
                marker=".",
                color=reds[i],
                label=f"FastFlow {e} trees",
            )

            ax.grid()

    axes[0].set_xlabel("Threads")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log", base=2)
    axes[0].set_ylabel("Speedup")
    axes[0].set_xticks(threads, [str(t) for t in threads])
    axes[0].set_yticks(threads, [str(t) for t in threads])

    axes[1].set_xlabel("Threads")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log", base=2)
    axes[1].set_xticks(threads, [str(t) for t in threads])
    axes[1].set_yticks(threads, [str(t) for t in threads])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
    )

    plt.tight_layout(rect=(0.0, 0.1, 1.0, 1.0))
    plt.savefig("../report/images/shared_prediction_speedup.svg", dpi=300)
    plt.show()


def training_efficiency(df, min_estimators):
    tmp = df[df["estimators"] >= min_estimators]
    tmp = tmp.sort_values(by=["threads"])

    datasets = ["magic", "susy20000"]
    estimators = tmp["estimators"].unique().astype(int)
    threads = tmp["threads"].unique().astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True, dpi=200)
    axes[0].set_title(f"Magic Training Efficiency")
    axes[1].set_title(f"SUSY 20k Training Efficiency")

    blues = mpl.colormaps["Blues"](np.linspace(0.4, 0.8, len(estimators)))
    reds = mpl.colormaps["Reds"](np.linspace(0.4, 0.8, len(estimators)))

    for i, e in enumerate(estimators):
        for ax, d in zip(axes, datasets):
            mask = (tmp["estimators"] == e) & (tmp["dataset"] == d)
            omp_mask = mask & ((tmp["backend"] == "seq") | (tmp["backend"] == "omp"))

            ax.plot(
                threads,
                tmp[omp_mask]["train_efficiency"],
                marker=".",
                color=blues[i],
                label=f"OpenMP {e} trees",
            )

            ff_mask = mask & ((tmp["backend"] == "seq") | (tmp["backend"] == "ff"))
            ax.plot(
                threads,
                tmp[ff_mask]["train_efficiency"],
                marker=".",
                color=reds[i],
                label=f"FastFlow {e} trees",
            )

            ax.grid()

    axes[0].set_xlabel("Threads")
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylabel("Efficiency")
    axes[0].set_xticks(threads, [str(t) for t in threads])

    axes[1].set_xlabel("Threads")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(threads, [str(t) for t in threads])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig("../report/images/shared_training_efficiency.svg", dpi=300)
    plt.show()


def prediction_efficiency(df, min_estimators):
    tmp = df[df["estimators"] >= min_estimators]
    tmp = tmp.sort_values(by=["threads"])

    datasets = ["magic", "susy20000"]
    estimators = tmp["estimators"].unique().astype(int)
    threads = tmp["threads"].unique().astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True, dpi=200)
    axes[0].set_title(f"Magic Prediction Efficiency")
    axes[1].set_title(f"SUSY 20k Prediction Efficiency")

    blues = mpl.colormaps["Blues"](np.linspace(0.4, 0.8, len(estimators)))
    reds = mpl.colormaps["Reds"](np.linspace(0.4, 0.8, len(estimators)))

    for i, e in enumerate(estimators):
        for ax, d in zip(axes, datasets):
            mask = (tmp["estimators"] == e) & (tmp["dataset"] == d)
            omp_mask = mask & ((tmp["backend"] == "seq") | (tmp["backend"] == "omp"))

            ax.plot(
                threads,
                tmp[omp_mask]["predict_efficiency"],
                marker=".",
                color=blues[i],
                label=f"OpenMP {e} trees",
            )

            ff_mask = mask & ((tmp["backend"] == "seq") | (tmp["backend"] == "ff"))
            ax.plot(
                threads,
                tmp[ff_mask]["predict_efficiency"],
                marker=".",
                color=reds[i],
                label=f"FastFlow {e} trees",
            )

            ax.grid()

    axes[0].set_xlabel("Threads")
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylabel("Efficiency")
    axes[0].set_xticks(threads, [str(t) for t in threads])

    axes[1].set_xlabel("Threads")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(threads, [str(t) for t in threads])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig("../report/images/shared_prediction_efficiency.svg", dpi=300)
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

    estimators = tmp["estimators"].unique().astype(int)
    threads = tmp["threads"].unique().astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True, dpi=200)
    axes[0].set_title(f"Training Weak Scaling")
    axes[1].set_title(f"Prediction Weak Scaling")

    blues = mpl.colormaps["Blues"](np.linspace(0.4, 0.8, 2))
    reds = mpl.colormaps["Reds"](np.linspace(0.4, 0.8, 2))

    omp_magic_train = weak_scaling_of(
        df, estimators, threads, "magic", "omp", "train_time"
    )
    ff_magic_train = weak_scaling_of(
        df, estimators, threads, "magic", "ff", "train_time"
    )
    omp_susy_train = weak_scaling_of(
        df, estimators, threads, "susy20000", "omp", "train_time"
    )
    ff_susy_train = weak_scaling_of(
        df, estimators, threads, "susy20000", "ff", "train_time"
    )

    omp_magic_predict = weak_scaling_of(
        df, estimators, threads, "magic", "omp", "predict_time"
    )
    ff_magic_predict = weak_scaling_of(
        df, estimators, threads, "magic", "ff", "predict_time"
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
        marker=".",
        label=f"OpenMP Magic",
        color=blues[0],
    )

    axes[0].plot(
        threads,
        ff_magic_train,
        marker=".",
        label=f"FastFlow Magic",
        color=reds[0],
    )

    axes[0].plot(
        threads,
        omp_susy_train,
        marker=".",
        label=f"OpenMP SUSY",
        color=blues[1],
    )

    axes[0].plot(
        threads,
        ff_susy_train,
        marker=".",
        label=f"FastFlow SUSY",
        color=reds[1],
    )

    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Threads-Estimators")
    axes[0].set_ylabel("Relative Speedup")
    threads = [1, 2, 4, 8, 16, 32]
    axes[0].set_xticks(threads, [f"{t}-{t * 8}" for t in threads])
    # axes[0].legend()
    axes[0].grid()

    axes[1].plot(
        threads,
        omp_magic_predict,
        marker=".",
        label=f"OpenMP Magic",
        color=blues[0],
    )

    axes[1].plot(
        threads,
        ff_magic_predict,
        marker=".",
        label=f"FastFlow Magic",
        color=reds[0],
    )

    axes[1].plot(
        threads,
        omp_susy_predict,
        marker=".",
        label=f"OpenMP SUSY",
        color=blues[1],
    )

    axes[1].plot(
        threads,
        ff_susy_predict,
        marker=".",
        label=f"FastFlow SUSY",
        color=reds[1],
    )

    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Threads-Estimators")
    axes[1].set_xticks(threads, [f"{t}-{t * 8}" for t in threads])
    # axes[1].legend()
    axes[1].grid()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    plt.savefig("../report/images/shared_weak_scaling.svg", dpi=300)
    plt.show()
