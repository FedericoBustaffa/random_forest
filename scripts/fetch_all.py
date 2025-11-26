import pandas as pd
from ucimlrepo import fetch_ucirepo

IDs = [53, 17, 159, 31]


for id in IDs:
    ds = fetch_ucirepo(id=id)
    if ds.data is not None:
        X = ds.data.features
        y = ds.data.targets
        df = pd.concat([X, y], axis=1)
        if ds.metadata is not None:
            name = ds.metadata.name
            name = name.lower()
            name = name.replace(" ", "_")
            name = name.replace("(", "")
            name = name.replace(")", "")
            df.to_csv(f"datasets/{name}.csv", index=False, header=False)
            print(f"{name} fetched")
