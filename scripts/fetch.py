import pandas as pd
from ucimlrepo import fetch_ucirepo

iris = fetch_ucirepo(id=53)
breast_cancer = fetch_ucirepo(id=17)
magic = fetch_ucirepo(id=159)
covertype = fetch_ucirepo(id=31)


datasets = [
    ("iris", iris),
    ("breast_cancer", breast_cancer),
    ("magic", magic),
    ("covertype", covertype),
]


for name, ds in datasets:
    if ds.data is not None:
        X = ds.data.features
        y = ds.data.targets
        df = pd.concat([X, y], axis=1)
        df.to_csv(f"datasets/{name}.csv", index=False, header=False)
        print(df)
