import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="dataset filepath")
    parser.add_argument("samples", type=int, help="max number of samples")

    args = parser.parse_args()

    df = pd.read_csv(args.path, header=None)
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()

    _, X, _, y = [
        np.asarray(i)
        for i in train_test_split(
            X, y, test_size=args.samples, random_state=42, stratify=y
        )
    ]

    y = y.reshape(-1, 1)
    data = np.hstack((X, y))
    df = pd.DataFrame(data, columns=None)
    print(df)

    base, ext = os.path.splitext(args.path)
    new_path = f"{base}_{args.samples}{ext}"
    df.to_csv(new_path, index=False, header=False)
