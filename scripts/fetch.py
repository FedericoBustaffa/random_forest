import argparse
import os

import kagglehub
import pandas as pd
from ucimlrepo import fetch_ucirepo


def kaggle_fetch(name):
    if f"{name}.csv" in os.listdir("datasets"):
        print(f"{name} dataset already fetched")
        return

    path_susy = kagglehub.dataset_download("janus137/supersymmetry-dataset")
    df_susy = pd.read_csv(
        path_susy + "/supersymmetry_dataset.csv", header=None, low_memory=False
    )
    df_susy = df_susy.iloc[:, 1:].assign(last_col=df_susy.iloc[:, 0])
    df_susy = df_susy.iloc[1:].reset_index(drop=True)

    df_susy.to_csv("datasets/susy.csv", header=False, index=False)


def uci_fetch(name: str, id: int):
    if f"{name}.csv" in os.listdir("datasets"):
        print(f"{name} dataset already fetched")
        return

    print(f"fetching {name} with ID {id}")
    ds = fetch_ucirepo(id=id)
    if ds.data is not None:
        X = ds.data.features
        y = ds.data.targets
        df = pd.concat([X, y], axis=1)

        df.to_csv(f"datasets/{name}.csv", index=False, header=False)
        print(f"{name} dataset fetched")


if __name__ == "__main__":
    datasets = {
        "iris": 53,
        "breast_cancer": 17,
        "magic": 159,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, choices=datasets.keys(), help="name of the dataset"
    )
    args = parser.parse_args()

    if "datasets" not in os.listdir("."):
        os.mkdir("datasets")

    if args.name is None:
        for name in datasets.keys():
            uci_fetch(name, datasets[name])
        kaggle_fetch("susy")
    else:
        uci_fetch(args.name, datasets[args.name])
