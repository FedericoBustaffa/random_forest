import json
import os
import sys

import pandas as pd

batch = {
    "estimators": [],
    "max_depth": [],
    "backend": [],
    "threads": [],
    "nodes": [],
    "dataset": [],
    "train_accuracy": [],
    "train_f1": [],
    "test_accuracy": [],
    "test_f1": [],
    "train_time": [],
    "train_predict_time": [],
    "test_predict_time": [],
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

param_cols = df.columns.to_list()[:6]
res_cols = df.columns.to_list()[6:]

df = df.groupby(by=param_cols, as_index=False)[res_cols].mean()

prefix = sys.argv[1]
dataset = sys.argv[2].split("/")[1].split(".")[0]
df.to_csv(f"results/{prefix}_{dataset}_{nfiles + 1}.csv", header=True, index=False)
