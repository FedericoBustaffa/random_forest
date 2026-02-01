import pandas as pd

df = pd.read_csv("results/forest.csv")
INPUT_COLS = ["dataset", "estimators", "max_depth", "backend", "nodes", "threads"]
OUTPUT_COLS = ["accuracy", "f1", "train_time", "predict_time"]

df = df.groupby(by=INPUT_COLS, as_index=False)[OUTPUT_COLS].mean()
df = df.drop(columns=["max_depth", "train_time", "predict_time"])
df = df[df["estimators"] == 128]
df = df[(df["dataset"] != "iris") & (df["dataset"] != "susy100000")]
mask = df["backend"] == "seq"
mask = mask | (df["backend"] == "omp") & (df["threads"] == 16)
mask = mask | (df["backend"] == "ff") & (df["threads"] == 16)
mask = mask | (df["backend"] == "joblib") & (df["threads"] == 16)
mask = mask | (df["backend"] == "mpi") & (df["nodes"] == 8) & (df["threads"] == 16)

df = df.drop(columns=["estimators"])

df = df[mask]
print(df)

print(df.to_latex(index=False, float_format="%.2f"))
