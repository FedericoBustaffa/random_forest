import pandas as pd

df = pd.read_csv("./results/cluster_iris_1.csv")

param_cols = df.columns.tolist()[:6]
result_cols = df.columns.tolist()[6:]

df = df.groupby(param_cols, as_index=False)[result_cols].mean()

df.to_csv("./results/cluster_iris.csv", header=True, index=False)
