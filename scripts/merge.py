import argparse
import json
import os

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="filepath to aggregate results")
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

    # if not present create a "results" directory
    if "results" not in os.listdir("."):
        os.mkdir("results")

    if args.filepath.split("/")[1] not in os.listdir("results"):
        # write results in CSV file
        df.to_csv(args.filepath, header=True, index=False)
    else:
        # merge current data with existing results
        old_df = pd.read_csv(args.filepath)
        new_df = pd.concat([old_df, df], ignore_index=True)
        new_df.to_csv(args.filepath, header=True, index=False)
