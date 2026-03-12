"""
Exploratory Data Analysis Script
=================================
Run with: python eda_explore.py --file your_data.csv
(or adjust load_df() to match your actual data source)
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
import warnings
import sys
import os

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ── helpers ──────────────────────────────────────────────────────────────────

def sep(title=""):
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * pad)
    else:
        print("\n" + "─" * width)


def load_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".xls", ".xlsx"):
        return pd.read_excel(path)
    elif ext == ".parquet":
        return pd.read_parquet(path)
    elif ext == ".json":
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── 1. OVERVIEW ──────────────────────────────────────────────────────────────

def overview(df: pd.DataFrame):
    sep("BASIC OVERVIEW")
    print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory usage   : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    # Some columns may contain lists/arrays (unhashable) — drop them before dedup check
    hashable_cols = [c for c in df.columns if df[c].map(lambda x: isinstance(x, (list, np.ndarray))).sum() == 0]
    dup_count = df[hashable_cols].duplicated().sum() if hashable_cols else "N/A (all cols unhashable)"
    print(f"  Duplicate rows : {dup_count:,}" if isinstance(dup_count, int) else f"  Duplicate rows : {dup_count}")

    sep("DTYPES")
    dtype_counts = df.dtypes.value_counts()
    for dt, cnt in dtype_counts.items():
        print(f"  {str(dt):<15} → {cnt} column(s)")

    sep("COLUMN LIST")
    for col in df.columns:
        print(f"  {col}")


# ── 2. MISSING VALUES ────────────────────────────────────────────────────────

def missing_values(df: pd.DataFrame):
    sep("MISSING VALUES")
    miss = df.isnull().sum()
    miss_pct = miss / len(df) * 100
    miss_df = pd.DataFrame({"missing": miss, "pct": miss_pct})
    miss_df = miss_df[miss_df["missing"] > 0].sort_values("pct", ascending=False)

    if miss_df.empty:
        print("  ✓ No missing values found.")
        return

    print(miss_df.to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(10, max(4, len(miss_df) * 0.4)))
    bars = ax.barh(miss_df.index, miss_df["pct"], color=sns.color_palette("Reds_r", len(miss_df)))
    ax.set_xlabel("Missing %")
    ax.set_title("Missing Values by Column")
    for bar, pct in zip(bars, miss_df["pct"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("plot_missing_values.png", dpi=130)
    plt.show()
    print("  → saved: plot_missing_values.png")


# ── 3. CLASSIFICATION COLUMNS ────────────────────────────────────────────────

def classification_columns(df: pd.DataFrame):
    pred_cols = ["lcc_main_name_pred", "lcc_name_pred", "sub_model_used", "type"]
    existing = [c for c in pred_cols if c in df.columns]
    if not existing:
        return

    sep("CLASSIFICATION / CATEGORY COLUMNS")
    fig, axes = plt.subplots(1, len(existing), figsize=(6 * len(existing), 5))
    if len(existing) == 1:
        axes = [axes]

    for ax, col in zip(axes, existing):
        vc = df[col].value_counts().head(20)
        vc.plot(kind="bar", ax=ax, color=sns.color_palette("tab20", len(vc)))
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        print(f"\n  [{col}]  unique={df[col].nunique()}")
        print(vc.to_string())

    plt.suptitle("Category Distributions", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("plot_categories.png", dpi=130)
    plt.show()
    print("\n  → saved: plot_categories.png")


# ── 4. KNN CONFIDENCE SCORES ─────────────────────────────────────────────────

def confidence_scores(df: pd.DataFrame):
    conf_cols = [c for c in ["main_knn_confidence", "sub_knn_confidence"] if c in df.columns]
    if not conf_cols:
        return

    sep("KNN CONFIDENCE SCORES")
    for col in conf_cols:
        print(f"\n  [{col}]")
        print(df[col].describe().round(4).to_string())

    fig, axes = plt.subplots(1, len(conf_cols), figsize=(7 * len(conf_cols), 4))
    if len(conf_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, conf_cols):
        df[col].dropna().plot(kind="hist", bins=40, ax=ax,
                               color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.axvline(df[col].median(), color="red", linestyle="--", label=f"median={df[col].median():.3f}")
        ax.set_title(col)
        ax.set_xlabel("Confidence")
        ax.legend(fontsize=8)

    plt.suptitle("KNN Confidence Distributions", fontsize=13)
    plt.tight_layout()
    plt.savefig("plot_confidence.png", dpi=130)
    plt.show()
    print("  → saved: plot_confidence.png")

    # Confidence vs category
    if "lcc_main_name_pred" in df.columns and "main_knn_confidence" in df.columns:
        sep("Confidence per Main LCC Category (top 15)")
        top_cats = df["lcc_main_name_pred"].value_counts().head(15).index
        sub = df[df["lcc_main_name_pred"].isin(top_cats)]
        fig, ax = plt.subplots(figsize=(12, 5))
        sub.boxplot(column="main_knn_confidence", by="lcc_main_name_pred", ax=ax,
                    vert=True, rot=45)
        ax.set_title("main_knn_confidence by lcc_main_name_pred (top 15)")
        plt.suptitle("")
        plt.tight_layout()
        plt.savefig("plot_conf_by_category.png", dpi=130)
        plt.show()
        print("  → saved: plot_conf_by_category.png")


# ── 5. TEXT / LANGUAGE COLUMNS ───────────────────────────────────────────────

def text_columns(df: pd.DataFrame):
    sep("TEXT & LANGUAGE COLUMNS")

    # Language detection results
    lang_cols = [c for c in ["title_lang", "abstract_lang"] if c in df.columns]
    if lang_cols:
        fig, axes = plt.subplots(1, len(lang_cols), figsize=(6 * len(lang_cols), 4))
        if len(lang_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, lang_cols):
            vc = df[col].value_counts().head(15)
            vc.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", len(vc)))
            ax.set_title(col)
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            print(f"\n  [{col}] top values:\n{vc.to_string()}")
        plt.tight_layout()
        plt.savefig("plot_languages.png", dpi=130)
        plt.show()
        print("  → saved: plot_languages.png")

    # Abstract / title length distributions
    for col, clean_col in [("abstract", "clean_abstract"), ("title", "title")]:
        src = clean_col if clean_col in df.columns else col
        if src not in df.columns:
            continue
        lengths = df[src].dropna().astype(str).str.len()
        print(f"\n  [{src}] length stats:")
        print(lengths.describe().round(1).to_string())

    if "abstract" in df.columns or "clean_abstract" in df.columns:
        src = "clean_abstract" if "clean_abstract" in df.columns else "abstract"
        lengths = df[src].dropna().astype(str).str.len()
        fig, ax = plt.subplots(figsize=(8, 4))
        lengths.clip(upper=lengths.quantile(0.99)).plot(kind="hist", bins=50, ax=ax,
            color="#55A868", edgecolor="white", alpha=0.85)
        ax.set_title(f"Abstract Length Distribution ({src})")
        ax.set_xlabel("Characters")
        plt.tight_layout()
        plt.savefig("plot_abstract_length.png", dpi=130)
        plt.show()
        print("  → saved: plot_abstract_length.png")


# ── 6. PUBLICATION METADATA ──────────────────────────────────────────────────

def publication_metadata(df: pd.DataFrame):
    sep("PUBLICATION METADATA")

    # Dates
    if "publication_date" in df.columns:
        dates = pd.to_datetime(df["publication_date"], errors="coerce")
        print(f"\n  publication_date range : {dates.min()} → {dates.max()}")
        print(f"  unparseable dates      : {dates.isna().sum():,}")
        years = dates.dt.year.dropna()
        if not years.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            years.value_counts().sort_index().plot(kind="bar", ax=ax,
                color="#DD8452", edgecolor="white", alpha=0.85, width=0.8)
            ax.set_title("Publications per Year")
            ax.set_xlabel("Year")
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            plt.tight_layout()
            plt.savefig("plot_pub_years.png", dpi=130)
            plt.show()
            print("  → saved: plot_pub_years.png")

    # Top journals / publishers
    for col in ["journal", "publisher"]:
        if col in df.columns:
            vc = df[col].value_counts().head(15)
            print(f"\n  Top 15 [{col}]:\n{vc.to_string()}")

    # DOI coverage
    if "doi" in df.columns:
        has_doi = df["doi"].notna().sum()
        print(f"\n  DOI coverage: {has_doi:,} / {len(df):,} ({has_doi/len(df)*100:.1f}%)")

    # Errors
    if "error" in df.columns:
        errs = df["error"].notna().sum()
        print(f"\n  Rows with error field populated: {errs:,}")
        if errs > 0:
            print(df["error"].value_counts().head(10).to_string())


# ── 7. TOPICS & RELATED TITLES ───────────────────────────────────────────────

def topics_column(df: pd.DataFrame):
    for col in ["topics", "related_titles"]:
        if col not in df.columns:
            continue
        sep(f"{col.upper()} COLUMN")
        sample = df[col].dropna().head(3)
        print(f"  Sample values:")
        for v in sample:
            print(f"    {str(v)[:120]}")
        # Try to parse as list and get top tokens
        try:
            import ast
            flat = []
            for v in df[col].dropna():
                parsed = ast.literal_eval(v) if isinstance(v, str) else v
                if isinstance(parsed, list):
                    flat.extend([str(x).strip() for x in parsed])
            if flat:
                top = Counter(flat).most_common(20)
                print(f"\n  Top 20 tokens in [{col}]:")
                for tok, cnt in top:
                    print(f"    {tok:<50} {cnt:>6}")
        except Exception:
            pass  # column isn't a list-like string, skip


# ── 8. CORRELATIONS (numeric) ────────────────────────────────────────────────

def numeric_correlations(df: pd.DataFrame):
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return
    sep("NUMERIC CORRELATIONS")
    corr = num.corr()
    print(corr.round(3).to_string())
    fig, ax = plt.subplots(figsize=(max(6, num.shape[1]), max(5, num.shape[1] - 1)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax)
    ax.set_title("Numeric Correlation Matrix")
    plt.tight_layout()
    plt.savefig("plot_correlations.png", dpi=130)
    plt.show()
    print("  → saved: plot_correlations.png")


# ── 9. TIMESTAMPS ────────────────────────────────────────────────────────────

def timestamps(df: pd.DataFrame):
    ts_cols = [c for c in ["created_at", "modified_at"] if c in df.columns]
    if not ts_cols:
        return
    sep("TIMESTAMPS")
    for col in ts_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        print(f"\n  [{col}] range: {parsed.min()} → {parsed.max()}")
        print(f"  nulls: {parsed.isna().sum():,}")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame):
    overview(df)
    missing_values(df)
    classification_columns(df)
    confidence_scores(df)
    text_columns(df)
    publication_metadata(df)
    topics_column(df)
    numeric_correlations(df)
    timestamps(df)
    sep("EDA COMPLETE")
    print("  All plots saved to current directory.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA script for document classification dataframe")
    parser.add_argument("--file", required=True, help="Path to data file (csv, xlsx, parquet, json)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Random sample N rows for plot-heavy sections (e.g. --sample 200000). "
                             "Overview and missing-value checks always use the full df.")
    args = parser.parse_args()

    print(f"\nLoading: {args.file}")
    df = load_df(args.file)

    # Always run overview & missing on full df (column-level, fast)
    overview(df)
    missing_values(df)

    # Optionally downsample for heavy / plot-heavy sections
    if args.sample and args.sample < len(df):
        print(f"\n  ⚡ Sampling {args.sample:,} rows for remaining sections "
              f"(full df has {len(df):,} rows)")
        dfs = df.sample(args.sample, random_state=42)
    else:
        dfs = df

    classification_columns(dfs)
    confidence_scores(dfs)
    text_columns(dfs)
    publication_metadata(dfs)
    topics_column(dfs)
    numeric_correlations(dfs)
    timestamps(dfs)
    sep("EDA COMPLETE")
    print("  All plots saved to current directory.\n")