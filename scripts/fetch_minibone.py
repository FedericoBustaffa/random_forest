import io

import pandas as pd
import requests

url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt"
)
print("Scaricamento MiniBooNE...")

response = requests.get(url)
lines = response.text.splitlines()

counts = list(map(int, lines[0].split()))
signal_count = counts[0]
background_count = counts[1]

df = pd.read_csv(io.StringIO("\n".join(lines[1:])), sep=r"\s+", header=None)

y = [1] * signal_count + [0] * background_count
df["target"] = y

output_path = "datasets/miniboone.csv"
df.to_csv(output_path, index=False, header=False)
print(f"MiniBooNE pronto: {output_path} (50 feature continue, 1 target)")
