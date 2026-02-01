import argparse
import json
import os
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("estimators", type=int, help="number of estimators")
    parser.add_argument("max_depth", type=int, help="max depth of trees")
    parser.add_argument("njobs", type=int, help="number of parallel processes")
    parser.add_argument("dataset", type=str, help="dataset filepath")
    parser.add_argument(
        "--log", action="store_true", help="flag to log results in a file"
    )
    args = parser.parse_args()
    max_depth = args.max_depth if args.max_depth > 0 else None

    df = pd.read_csv(args.dataset, header=None)
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=args.estimators,
        criterion="entropy",
        max_depth=max_depth,
        max_features=None,
        n_jobs=args.njobs,
    )

    start = time.perf_counter()
    rf.fit(X_train, y_train)
    end = time.perf_counter()
    train_time = end - start

    start = time.perf_counter()
    test_pred = rf.predict(X_test)
    end = time.perf_counter()
    predict_time = end - start

    accuracy = accuracy_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred, average="macro")

    data = {
        "dataset": args.dataset,
        "estimators": args.estimators,
        "max_depth": args.max_depth,
        "accuracy": accuracy,
        "f1": f1,
        "backend": "joblib",
        "nodes": 1,
        "threads": args.njobs,
        "train_time": train_time * 1000,
        "predict_time": predict_time * 1000,
    }
    print(json.dumps(data, indent=2))

    if args.log:
        if "tmp" not in os.listdir("."):
            os.mkdir("tmp")

        nfiles = len(os.listdir("tmp"))
        filepath = f"tmp/result_{nfiles}.json"
        with open(filepath, "w") as fp:
            json.dump(data, fp, indent=4)
