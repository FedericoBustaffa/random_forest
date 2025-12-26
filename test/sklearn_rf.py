import argparse
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("estimators", type=int, help="number of estimators")
    parser.add_argument("max_depth", type=int, help="max depth of trees")
    parser.add_argument("dataset", type=str, help="dataset filepath")
    parser.add_argument("njobs", type=int, help="number of parallel processes")
    args = parser.parse_args()
    max_depth = args.max_depth if args.max_depth > 0 else None

    df = pd.read_csv(args.dataset)
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
    print(f"train time: {end - start:.4f} s")

    start = time.perf_counter_ns() * 1000
    train_pred = rf.predict(X_train)
    end = time.perf_counter_ns() * 1000
    print(f"train predict time: {end - start:.4f} ms")

    train_accuracy = accuracy_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred, average="macro")
    print(f"train accuracy: {train_accuracy:.2f}")
    print(f"train f1: {train_f1:.2f}")

    start = time.perf_counter()
    test_pred = rf.predict(X_test)
    end = time.perf_counter()
    print(f"test predict time: {end - start:.4f} s")

    test_accuracy = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="macro")
    print(f"test accuracy: {test_accuracy:.2f}")
    print(f"test f1: {test_f1:.2f}")
