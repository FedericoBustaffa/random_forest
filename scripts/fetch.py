import argparse
import os

import pandas as pd
from ucimlrepo import fetch_ucirepo

datasets = {
    53: "iris",
    109: "wine",
    17: "breast_cancer",
    159: "magic",
    59: "letter",
    80: "digits",
    31: "covertype",
}


def fetch(id: int):
    if "datasets" not in os.listdir("."):
        os.mkdir("datasets")

    if id in datasets.keys():
        if f"{datasets[id]}.csv" in os.listdir("datasets"):
            return

    ds = fetch_ucirepo(id=id)
    if ds.data is not None:
        X = ds.data.features
        y = ds.data.targets
        df = pd.concat([X, y], axis=1)

        name = ""
        if id in datasets.keys():
            name = datasets[id]
        elif ds.metadata is not None:
            name = ds.metadata.name
            name = name.lower()
            name = name.replace(" ", "_")
            name = name.replace("(", "")
            name = name.replace(")", "")

        df.to_csv(f"datasets/{name}.csv", index=False, header=False)
        print(f"{name} fetched")


parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help="id of the UCI dataset")
args = parser.parse_args()


if args.id is None:
    for id in datasets.keys():
        fetch(id)
else:
    fetch(args.id)
