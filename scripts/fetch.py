import argparse

import pandas as pd
from ucimlrepo import fetch_ucirepo

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int, help="id of the UCI dataset")
args = parser.parse_args()


dataset = fetch_ucirepo(id=args.id)
if dataset.data is not None:
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)
    if dataset.metadata is not None:
        name = dataset.metadata.name
        df.to_csv(f"datasets/{name.lower()}.csv", index=False, header=False)
        print(f"{name} fetched")
