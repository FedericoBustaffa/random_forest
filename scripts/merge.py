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

    # get file id
    prefixed_files = filter(
        lambda x: x.startswith(f"{args.prefix}"), os.listdir("results")
    )
    nfiles = len(list(prefixed_files))

    # write results in CSV file
    dataset = df["dataset"].iloc[0]
    filename = f"results/{args.prefix}_{dataset}_{nfiles + 1}.csv"
    df = df.drop(columns="dataset")
    df.to_csv(filename, header=True, index=False)
