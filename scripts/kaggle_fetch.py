import os

import kagglehub
import pandas as pd

if __name__ == "__main__":
    if "datasets" not in os.listdir("."):
        os.mkdir("datasets")

    # MiniBooNE
    path_mb = kagglehub.dataset_download("alexanderliapatis/miniboone")
    df_mb = pd.read_csv(path_mb + "/MiniBooNE_PID.csv", header=None)
    df_mb = df_mb.iloc[1:].reset_index(drop=True)

    df_mb.to_csv("datasets/miniboone.csv", header=False, index=False)
    print(df_mb.head())

    # SUSY
    path_susy = kagglehub.dataset_download("janus137/supersymmetry-dataset")
    df_susy = pd.read_csv(
        path_susy + "/supersymmetry_dataset.csv", header=None, low_memory=False
    )
    df_susy = df_susy.iloc[:, 1:].assign(last_col=df_susy.iloc[:, 0])
    df_susy = df_susy.iloc[1:].reset_index(drop=True)

    df_susy.to_csv("datasets/susy.csv", header=False, index=False)
    print(df_susy.head())
