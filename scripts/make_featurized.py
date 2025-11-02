"""Simple script to generate a featurized CSV from the original bank-full.csv
Usage: python scripts/make_featurized.py --in data/bank-full.csv --out data/bank-full-feat.csv
"""
from pathlib import Path
import argparse
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()


NUMERIC_COLS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]


def perturb_row(row, numeric_cols, rng: np.random.Generator):
    r = row.copy()
    for col in numeric_cols:
        try:
            val = r[col]
            if pd.isna(val):
                continue
            # integer-like
            if isinstance(val, (int, np.integer)):
                if col == "age":
                    delta = int(rng.normal(0, 2))
                    r[col] = max(18, int(val + delta))
                else:
                    delta = int(rng.normal(0, max(1, abs(val) * 0.02)))
                    r[col] = int(max(-999999, val + delta))
            else:
                delta = float(rng.normal(0, max(1.0, abs(float(val) * 0.02))))
                r[col] = float(val + delta)
        except Exception:
            r[col] = r[col]
    return r


def augment_df(df: pd.DataFrame, target_positive_ratio: float, rng: np.random.Generator) -> pd.DataFrame:
    if "y" not in df.columns:
        raise SystemExit("Columna 'y' no encontrada en CSV")
    counts = df['y'].value_counts()
    # accept 'yes'/'no' or 1/0
    if 'yes' in counts.index:
        n_pos = int(counts.get('yes', 0))
    else:
        n_pos = int(counts.get(1, 0))
    n_total = len(df)
    # compute x: (n_pos + x) / (n_total + x) = target
    x = int(max(0, np.round((target_positive_ratio * n_total - n_pos) / (1 - target_positive_ratio))))
    print(f"Original: total={n_total}, pos={n_pos}. Añadiendo x={x} positivos para alcanzar ratio {target_positive_ratio}")
    if x == 0:
        return df.copy()
    # get positive rows (support both 'yes' and 1)
    if 'yes' in df['y'].unique():
        df_pos = df[df['y'] == 'yes']
    else:
        df_pos = df[df['y'] == 1]
    if df_pos.empty:
        raise SystemExit("No hay ejemplos positivos para muestrear")
    new_rows = []
    for i in range(x):
        sample = df_pos.sample(n=1, replace=True, random_state=int(rng.integers(0, 2**31-1))).iloc[0]
        new_row = perturb_row(sample, NUMERIC_COLS, rng)
        new_rows.append(new_row)
    df_new = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    # shuffle
    df_new = df_new.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_new


def featurize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["age", "balance", "pdays", "previous", "duration", "campaign"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "age" in df.columns:
        df["age_bin"] = pd.cut(df["age"], bins=[0, 25, 35, 45, 55, 65, 120], labels=["<25","25-34","35-44","45-54","55-64",">=65"]) 
    if "balance" in df.columns:
        df["balance_log"] = np.log1p(df["balance"].clip(lower=0))
    if "pdays" in df.columns:
        df["pdays_flag"] = (df["pdays"] == -1).astype(int)
    if "previous" in df.columns:
        df["previous_flag"] = (df["previous"] > 0).astype(int)
    if "duration" in df.columns:
        df["duration_bin"] = pd.cut(df["duration"], bins=[-1, 60, 180, 600, 10000], labels=["short","mid","long","very_long"]).astype(object)
    if "campaign" in df.columns:
        df["campaign_bin"] = pd.cut(df["campaign"], bins=[-1,0,1,2,3,5,100], labels=["0","1","2","3","4-5","6+"]).astype(object)
    if "campaign" in df.columns and "previous" in df.columns:
        df["camp_prev"] = df["campaign"] * df["previous"]
    return df



def main():
    p = argparse.ArgumentParser()
    # make input/output optional; read defaults from .env if available
    default_in = os.getenv("DATA_PATH", "data/bank-full.csv")
    default_out = os.getenv("FEAT_OUTPUT", "data/bank-full-feat.csv")
    default_sep = os.getenv("CSV_SEP", ";")
    if default_sep == "":
        default_sep = None

    p.add_argument("--in", dest="in_path", required=False, default=default_in, help=f"Input CSV (default: {default_in})")
    p.add_argument("--out", dest="out_path", required=False, default=default_out, help=f"Output featurized CSV path (default: {default_out})")
    p.add_argument("--sep", dest="sep", default=default_sep, help="CSV separator (default from .env or ';')")
    # by default we WILL augment and featurize when the user runs the script without flags
    p.add_argument("--no-augment", dest="augment", action="store_false", help="Disable augmentation; by default augmentation is applied")
    p.add_argument("--target-ratio", dest="target_ratio", type=float, default=float(os.getenv("AUGMENT_TARGET", 0.35)), help="Target positive ratio if augment is used")
    p.add_argument("--random-state", dest="random_state", type=int, default=int(os.getenv("RANDOM_STATE", 42)), help="Random seed for augmentation sampling")
    args = p.parse_args()

    inp = Path(args.in_path)
    out = Path(args.out_path)
    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    sep = args.sep
    if sep in ("None", "", None):
        sep = None

    df = pd.read_csv(inp, sep=sep, engine="python" if sep is None else None)
    print(f"Loaded {inp} -> rows={len(df)}")
    before_pos = df['y'].value_counts().to_dict() if 'y' in df.columns else None
    if before_pos is not None:
        print("Original target counts:", before_pos)

    rng = np.random.default_rng(args.random_state)
    if args.augment:
        df = augment_df(df, args.target_ratio, rng)
        print(f"After augmentation rows={len(df)}; target counts={df['y'].value_counts().to_dict()}")

    df_feat = featurize_df(df)
    # normalize target to 0/1 if present
    if 'y' in df_feat.columns and df_feat['y'].dtype == object:
        df_feat['y'] = df_feat['y'].map({'yes': 1, 'no': 0}).fillna(df_feat['y'])

    out.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(out, index=False, sep=sep if sep is not None else ';')
    print(f"Featurized CSV written to: {out} -> rows={len(df_feat)}")
    if 'y' in df_feat.columns:
        print("Featurized target counts:", df_feat['y'].value_counts().to_dict())


if __name__ == '__main__':
    main()
