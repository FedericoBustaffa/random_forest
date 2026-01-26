import os

import pandas as pd


def balance_and_reduce_csv(file_path, max_samples):
    # 1. Caricamento del file
    print(f"Caricamento di {file_path}...")
    # Leggiamo senza header come da tua necessità per il C++
    df = pd.read_csv(file_path, header=None)

    # Assumiamo che l'ultima colonna sia il target
    target_col = df.columns[-1]

    # 2. Identificazione delle classi e bilanciamento
    classes = df[target_col].unique()
    n_classes = len(classes)

    # Calcoliamo quanti campioni prendere per ogni classe
    samples_per_class = max_samples // n_classes

    reduced_parts = []

    print(f"Bilanciamento classi per {n_classes} etichette...")
    for c in classes:
        class_subset = df[df[target_col] == c]

        # Se la classe ha meno campioni di quelli richiesti, li prendiamo tutti
        # Altrimenti facciamo il sampling
        n_to_take = min(len(class_subset), samples_per_class)
        reduced_parts.append(class_subset.sample(n=n_to_take, random_state=42))

    # Uniamo le parti bilanciate
    df_balanced = pd.concat(reduced_parts)

    # 3. Shuffle finale (per mischiare le classi che ora sono ordinate)
    df_final = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Se a causa di classi piccole non abbiamo raggiunto esattamente max_samples,
    # ma vogliamo essere sicuri di non superarlo:
    if len(df_final) > max_samples:
        df_final = df_final.head(max_samples)

    # 4. Creazione del nuovo nome file
    base, ext = os.path.splitext(file_path)
    new_path = f"{base}_reduced{ext}"

    # 5. Salvataggio
    df_final.to_csv(new_path, index=False, header=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="dataset filepath")
    parser.add_argument("samples", type=int, help="max number of samples")

    args = parser.parse_args()

    if os.path.exists(args.path):
        balance_and_reduce_csv(args.path, args.samples)
    else:
        print("file not found")
