import json
import os

import pandas as pd

batch = {
    "threading": [],
    "estimators": [],
    "max_depth": [],
    "train_time": [],
    "predict_time": [],
    "accuracy": [],
    "nthreads": [],
}

paths = [f"results/{fp}" for fp in os.listdir("results/")]

for fp in paths:
    if fp.endswith(".csv"):
        continue
    f = open(fp, "r")
    content = json.load(f)
    for k in batch.keys():
        batch[k].append(content[k])

    os.remove(fp)

df = pd.DataFrame(batch)
df.to_csv("results/results.csv", header=True, index=False)
