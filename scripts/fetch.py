import argparse
import os

import pandas as pd
from ucimlrepo import fetch_ucirepo


def fetch(id: int):
    if "datasets" not in os.listdir("."):
        os.mkdir("datasets")

    ds = fetch_ucirepo(id=id)
    if ds.data is not None:
        X = ds.data.features
        y = ds.data.targets
        df = pd.concat([X, y], axis=1)
        if ds.metadata is not None:
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

IDs = [53, 17, 159, 80, 31]

if args.id is None:
    for id in IDs:
        fetch(id)
else:
    fetch(args.id)
