"""augment_data.py
Genera un CSV aumentado a partir de data/bank-full.csv creando copias perturbadas
de filas positivas ('yes') para mejorar balance de clases. No sobrescribe el archivo
original; crea data/bank-full-augmented.csv.

Uso:
    python scripts/augment_data.py --target-ratio 0.25 --out data/bank-full-augmented.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

NUMERIC_COLS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]


def perturb_row(row, numeric_cols):
    r = row.copy()
    for col in numeric_cols:
        try:
            val = r[col]
            if pd.isna(val):
                continue
            if isinstance(val, (int, np.integer)):
                # small integer perturbation
                if col == "age":
                    delta = int(np.random.normal(0, 2))
                    r[col] = max(18, int(val + delta))
                else:
                    delta = int(np.random.normal(0, max(1, abs(val) * 0.02)))
                    r[col] = int(max(-999999, val + delta))
            else:
                # floats
                delta = np.random.normal(0, max(1.0, abs(float(val) * 0.02)))
                r[col] = float(val + delta)
        except Exception:
            # ignore perturbation if any issue
            r[col] = r[col]
    return r


def augment(csv_path: Path, out_path: Path, target_positive_ratio: float = 0.25):
    df = pd.read_csv(csv_path, sep=";", quotechar='"')
    # ensure y exists
    if "y" not in df.columns:
        raise SystemExit("Columna 'y' no encontrada en CSV")
    counts = df['y'].value_counts()
    n_pos = int(counts.get('yes', 0))
    n_total = len(df)
    n_neg = n_total - n_pos
    # compute how many positive to add to reach target ratio
    # (n_pos + x) / (n_total + x) = target => x = (target*n_total - n_pos) / (1 - target)
    x = int(max(0, np.round((target_positive_ratio * n_total - n_pos) / (1 - target_positive_ratio))))
    print(f"Original: total={n_total}, pos={n_pos}, neg={n_neg}. Añadiendo x={x} positivos para alcanzar ratio {target_positive_ratio}")
    if x == 0:
        print("No se requiere augmentación; copia directa")
        df.to_csv(out_path, index=False, sep=';')
        return out_path
    df_pos = df[df['y'] == 'yes']
    if df_pos.empty:
        raise SystemExit("No hay ejemplos positivos para muestrear")
    new_rows = []
    for i in range(x):
        sample = df_pos.sample(n=1, replace=True).iloc[0]
        new_row = perturb_row(sample, NUMERIC_COLS)
        new_rows.append(new_row)
    df_new = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    # shuffle
    df_new = df_new.sample(frac=1, random_state=42).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(out_path, index=False, sep=';')
    print(f"Escrito CSV aumentado: {out_path} (n={len(df_new)})")
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generar CSV aumentado con positivos sintéticos')
    parser.add_argument('--data-path', type=str, default='data/bank-full.csv')
    parser.add_argument('--out', type=str, default='data/bank-full-augmented.csv')
    parser.add_argument('--target-ratio', type=float, default=0.25, help='Proporción de positivos deseada en el dataset final')
    args = parser.parse_args()
    augment(Path(args.data_path), Path(args.out), args.target_ratio)
