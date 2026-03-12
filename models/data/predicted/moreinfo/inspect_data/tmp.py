import pandas as pd
import ast
import sys
from contextlib import redirect_stdout

COLS_TO_INSPECT = [
    "lcc_main_name_pred",
    "lcc_name_pred", 
    "journal",
    "publisher",
]

def try_parse(val):
    if isinstance(val, (list, tuple)):
        return list(val)
    try:
        if pd.isna(val):
            return None
    except (ValueError, TypeError):
        pass
    try:
        parsed = ast.literal_eval(str(val))
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        return [str(val)]

def inspect_col(df, col, top_n=50):
    print(f"\n{'═'*70}")
    print(f"  📌 {col}")
    print(f"{'═'*70}")

    series = df[col]
    n_null = series.isna().sum()
    print(f"  rows: {len(series):,}  |  filled: {len(series)-n_null:,}  |  null: {n_null:,}\n")

    sample = series.dropna().head(100)
    is_list_col = sample.apply(lambda v: isinstance(v, list) or str(v).strip().startswith("[")).mean() > 0.5

    if is_list_col:
        parsed   = series.apply(try_parse)
        exploded = parsed.explode().dropna()
        exploded = exploded[exploded.astype(str).str.strip() != ""]
        vc       = exploded.astype(str).value_counts()
        print(f"  ⚑ list column  |  unique items: {len(vc):,}  |  total items: {vc.sum():,}")
    else:
        vc = series.dropna().astype(str).value_counts()
        print(f"  unique values: {len(vc):,}")

    print(f"\n  {'#':<6} {'value':<55} {'count':>8}  {'%':>6}")
    print(f"  {'-'*6} {'-'*55} {'-'*8}  {'-'*6}")
    for i, (val, cnt) in enumerate(vc.head(top_n).items(), 1):
        label = str(val)[:55].ljust(55)
        pct   = cnt / len(series) * 100
        print(f"  {i:<6} {label} {cnt:>8,}  {pct:>5.2f}%")

    if len(vc) > top_n:
        print(f"\n  … {len(vc) - top_n:,} more unique values not shown (increase top_n to see more)")

    print(f"\n  ── all unique values ({'items' if is_list_col else 'values'}) ──")
    for i, v in enumerate(vc.index.tolist(), 1):
        print(f"    {i:>4}. {v}")

def inspect_all(df, cols=COLS_TO_INSPECT, top_n=50, output_file="inspect_output.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            missing = [c for c in cols if c not in df.columns]
            if missing:
                print(f"⚠  not found in df: {missing}")
            for col in [c for c in cols if c in df.columns]:
                inspect_col(df, col, top_n=top_n)
            print(f"\n{'═'*70}\n  ✅ DONE\n{'═'*70}\n")

    print(f"✅ Results saved to: {output_file}")  # this still prints to terminal


df = pd.read_parquet("models/data/predicted/moreinfo/predict_moreinfo_emb_final.parquet")
inspect_all(df, output_file="models/data/predicted/moreinfo/inspect_output.txt")