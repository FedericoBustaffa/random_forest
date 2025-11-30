import json
import os
import sys

import pandas as pd

batch = {
    "dataset": [],
    "backend": [],
    "estimators": [],
    "max_depth": [],
    "accuracy": [],
    "train_time": [],
    "predict_time": [],
    "threads": [],
    "nodes": [],
}

paths = [f"tmp/{fp}" for fp in os.listdir("tmp/")]

for fp in paths:
    f = open(fp, "r")
    content = json.load(f)
    for k in batch.keys():
        if k == "dataset":
            batch[k].append(content[k].split("/")[1].split(".")[0])
        else:
            batch[k].append(content[k])

    os.remove(fp)

if "results" not in os.listdir("."):
    os.mkdir("results")

nfiles = len(os.listdir("results"))
df = pd.DataFrame(batch)
df.to_csv(f"results/{sys.argv[1]}_{nfiles + 1}.csv", header=True, index=False)
