import pandas as pd

df = pd.read_csv("results/forest.csv")
INPUT_COLS = ["dataset", "estimators", "max_depth", "backend"]
print(df.columns)

# df = df[df["estimators"] == 128]
# df = df.drop(columns=["estimators", "max_depth", "train_time", "predict_time"])
#
# df = df[~((df["backend"] == "mpi") & ~((df["nodes"] == 8) & (df["threads"] == 16)))]
# df = df[~((df["backend"] == "ff") & ~((df["nodes"] == 1) & (df["threads"] == 16)))]
# df = df[~((df["backend"] == "omp") & ~((df["nodes"] == 1) & (df["threads"] == 16)))]
# df = df[~((df["backend"] == "joblib") & ~((df["nodes"] == 1) & (df["threads"] == 1)))]
# df = df[df["dataset"] != "iris"]
#
# print(df)
#
# print(df.to_latex(index=False, float_format="%.2f"))
