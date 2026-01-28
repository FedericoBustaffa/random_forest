import pandas as pd

df = pd.read_csv("results/tree.csv")

print(df)

with open("tree_table.tex", "w") as fp:
    print(df.to_latex(index=False, float_format="%.2f"), file=fp)
