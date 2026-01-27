import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def runtime_thread(df):
    df = df[df["backend"] == "mpi"]
    return df


if __name__ == "__main__":
    pass
