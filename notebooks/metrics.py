import argparse

import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="path of the file")
    args = parser.parse_args()

    df = pd.read_csv(args.filepath)
