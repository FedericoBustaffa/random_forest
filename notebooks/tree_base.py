import pandas as pd

df = pd.read_csv("results/tree.csv")

print(df)
print(df.to_latex(index=False, float_format="%.3f"))
