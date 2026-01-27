import argparse
import json
import os

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=str, help="prefix for the resulting file")
    args = parser.parse_args()

    # read the first file to infer the parameters
    filepath = os.listdir("tmp")[0]
    with open(f"tmp/{filepath}", "r") as fp:
        keys = json.load(fp).keys()

    df = {k: [] for k in keys}

    # read all the files and merge them
    paths = [f"tmp/{fp}" for fp in os.listdir("tmp/")]
    for fp in paths:
        f = open(fp, "r")
        content = json.load(f)
        for k in df.keys():
            if k == "dataset":
                df[k].append(content[k].split("/")[1].split(".")[0])
            else:
                df[k].append(content[k])

        # delete consumed files
        os.remove(fp)

    # build a DataFrame
    df = pd.DataFrame(df)

    # mean multiple runs with same parameters
    param_cols = df.columns.to_list()[:6]
    res_cols = df.columns.to_list()[6:]
    df = df.groupby(by=param_cols, as_index=False)[res_cols].mean()
    assert isinstance(df, pd.DataFrame)

    # if not present create a "results" directory
    if "results" not in os.listdir("."):
        os.mkdir("results")

    if f"{args.prefix}.csv" not in os.listdir("results"):
        # write results in CSV file
        filename = f"results/{args.prefix}.csv"
        df.to_csv(filename, header=True, index=False)
    else:
        # merge current data with existing results
        old_df = pd.read_csv(f"results/{args.prefix}.csv")
        new_df = pd.concat([old_df, df], ignore_index=True)
        new_df.to_csv(f"results/{args.prefix}.csv", header=True, index=False)
