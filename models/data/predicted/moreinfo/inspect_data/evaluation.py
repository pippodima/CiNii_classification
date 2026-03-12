import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_parquet("models/data/predicted/moreinfo/predict_moreinfo_emb_final.parquet")

# ── 2. Journal ground truth dictionary ───────────────────────────────────────
journal_ground_truth = {
    # SCIENCE / Chemistry
    "The Journal of Chemical Physics":               ("Science", "Chemistry"),
    "Bulletin of the Chemical Society of Japan":     ("Science", "Chemistry"),
    "Chemistry Letters":                             ("Science", "Chemistry"),
    "Angewandte Chemie International Edition":       ("Science", "Chemistry"),
    "Chemistry – A European Journal":                ("Science", "Chemistry"),

    # SCIENCE / Biology
    "Journal of Virology":                           ("Science", "Natural history - Biology"),
    "Applied and Environmental Microbiology":        ("Science", "Natural history - Biology"),
    "Journal of Bacteriology":                       ("Science", "Natural history - Biology"),
    "Development":                                   ("Science", "Natural history - Biology"),
    "Journal of Cell Science":                       ("Science", "Natural history - Biology"),
    "Bioscience, Biotechnology, and Biochemistry":   ("Science", "Natural history - Biology"),
    "Agricultural and Biological Chemistry":         ("Science", "Natural history - Biology"),

    # SCIENCE / Physics
    "Journal of the Physical Society of Japan":      ("Science", "Physics"),

    # SCIENCE / Geology
    "Geophysical Research Letters":                  ("Science", "Geology"),
    "Journal of Geophysical Research: Atmospheres":  ("Science", "Geology"),
    "Journal of Geophysical Research: Space Physics":("Science", "Geology"),
    "Journal of Geophysical Research: Solid Earth":  ("Science", "Geology"),

    # SCIENCE / Astronomy
    "The Astrophysical Journal":                     ("Science", "Astronomy"),

    # MEDICINE / Internal medicine
    "Blood":                                         ("Medicine", "Internal medicine"),
    "Circulation":                                   ("Medicine", "Internal medicine"),
    "Cancer Research":                               ("Medicine", "Internal medicine"),
    "Journal of Clinical Oncology":                  ("Medicine", "Internal medicine"),
    "Stroke":                                        ("Medicine", "Internal medicine"),
    "The Journal of Neuroscience":                   ("Medicine", "Internal medicine"),
    "Journal of Neurochemistry":                     ("Medicine", "Internal medicine"),
    "Journal of Neurophysiology":                    ("Medicine", "Internal medicine"),
    "The Journal of Immunology":                     ("Medicine", "Internal medicine"),
    "Infection and Immunity":                        ("Medicine", "Internal medicine"),

    # MEDICINE / General
    "Internal Medicine":                             ("Medicine", "Medicine (General)"),
    "The Tohoku Journal of Experimental Medicine":   ("Medicine", "Medicine (General)"),

    # MEDICINE / Physiology
    "The Journal of Physiology":                     ("Medicine", "Physiology"),
    "Journal of Applied Physiology":                 ("Medicine", "Physiology"),

    # TECHNOLOGY / Electrical engineering
    "Japanese Journal of Applied Physics":           ("Technology", "Electrical engineering. Electronics. Nuclear engineering"),
    "Applied Physics Letters":                       ("Technology", "Electrical engineering. Electronics. Nuclear engineering"),
    "Journal of Applied Physics":                    ("Technology", "Electrical engineering. Electronics. Nuclear engineering"),
    "IEICE Proceeding Series":                       ("Technology", "Electrical engineering. Electronics. Nuclear engineering"),

    # TECHNOLOGY / Engineering general
    "Review of Scientific Instruments":              ("Technology", "Engineering (General). Civil engineering"),
    "ISIJ International":                            ("Technology", "Engineering (General). Civil engineering"),
    "MATERIALS TRANSACTIONS":                        ("Technology", "Engineering (General). Civil engineering"),
}

# ── 3. Publisher ground truth dictionary ─────────────────────────────────────
publisher_ground_truth = {
    # SCIENCE / Zoology
    "Zoological Society of London":                  ("Science", "Zoology"),

    # SCIENCE / Mathematics
    "American Mathematical Society":                 ("Science", "Mathematics"),
    "London Mathematical Society":                   ("Science", "Mathematics"),

    # SCIENCE / Physics
    "THE PHYSICAL SOCIETY OF JAPAN":                 ("Science", "Physics"),
    "American Physical Society":                     ("Science", "Physics"),

    # SCIENCE / Astronomy
    "American Astronomical Society":                 ("Science", "Astronomy"),

    # TECHNOLOGY / Mechanical engineering
    "The Japan Society of Mechanical Engineers":     ("Technology", "Mechanical engineering and machinery"),
    "ASME International":                            ("Technology", "Mechanical engineering and machinery"),

    # TECHNOLOGY / Chemical technology
    "American Institute of Chemical Engineers":      ("Technology", "Chemical technology"),

    # TECHNOLOGY / Environmental engineering
    "Water Environment Federation":                  ("Technology", "Environmental technology. Sanitary engineering"),

    # NAVAL SCIENCE
    "Naval Institute Press":                         ("Naval Science", "Naval architecture. Shipbuilding. Marine engineering"),

    # SOCIAL SCIENCES
    "SAGE Publications":                             ("Social Sciences", "Social sciences"),
    "American Sociological Association":             ("Social Sciences", "Social sciences"),

    # MEDICINE / Dentistry
    "American Dental Association":                   ("Medicine", "Dentistry"),
    "International & American Associations for Dental Research": ("Medicine", "Dentistry"),

    # MEDICINE / Pharmacology
    "American Society for Pharmacology and Experimental Therapeutics": ("Medicine", "Therapeutics. Pharmacology"),

    # BIBLIOGRAPHY
    "American Library Association":                  ("Bibliography & Information Science", "Bibliography & Information Science"),

    # EDUCATION
    "American Educational Research Association":     ("Education", "Theory and practice of education"),
}

# ── 4. Build evaluation subsets ───────────────────────────────────────────────
# Journal-based
df_eval = df[df["journal"].isin(journal_ground_truth)].copy()
df_eval["gt_main"] = df_eval["journal"].map(lambda j: journal_ground_truth[j][0])
df_eval["gt_sub"]  = df_eval["journal"].map(lambda j: journal_ground_truth[j][1])

# Publisher-based (no overlap with journal-based)
df_pub_eval = df[df["publisher"].isin(publisher_ground_truth)].copy()
df_pub_eval["gt_main"] = df_pub_eval["publisher"].map(lambda p: publisher_ground_truth[p][0])
df_pub_eval["gt_sub"]  = df_pub_eval["publisher"].map(lambda p: publisher_ground_truth[p][1])

# Combined (journal takes priority, no duplicate rows)
df_combined = pd.concat([
    df_eval,
    df_pub_eval[~df_pub_eval.index.isin(df_eval.index)]
]).copy()

print(f"Total rows             : {len(df):>10,}")
print(f"Journal-based GT rows  : {len(df_eval):>10,}  ({len(df_eval)/len(df)*100:.1f}%)")
print(f"Publisher-based GT rows: {len(df_pub_eval):>10,}  ({len(df_pub_eval)/len(df)*100:.1f}%)")
print(f"Combined (no overlap)  : {len(df_combined):>10,}  ({len(df_combined)/len(df)*100:.1f}%)\n")

# ── 5. Helper: fix Categorical dtype ─────────────────────────────────────────
def add_other_category(series):
    if hasattr(series, 'cat'):
        if "Other" not in series.cat.categories:
            series = series.cat.add_categories("Other")
    return series

# ── 6. Classification reports ─────────────────────────────────────────────────
df_comb_main = df_combined.dropna(subset=["lcc_main_name_pred"])
df_comb_sub  = df_combined.dropna(subset=["lcc_name_pred"])

print("=" * 70)
print("MAIN CATEGORY — combined journal + publisher GT")
print("=" * 70)
print(classification_report(
    df_comb_main["gt_main"],
    df_comb_main["lcc_main_name_pred"],
    zero_division=0
))

print("=" * 70)
print("SUB CATEGORY — combined journal + publisher GT")
print("=" * 70)
print(classification_report(
    df_comb_sub["gt_sub"],
    df_comb_sub["lcc_name_pred"],
    zero_division=0
))

# ── 7. Confusion matrices ─────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, title, figsize=(10, 8)):
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
    for ax, data, fmt, subtitle in zip(
        axes, [cm, cm_norm], ["d", ".2f"],
        ["Raw counts", "Normalized by true label (recall)"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, linewidths=0.5)
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    fname = f"confusion_matrix_{'main' if 'MAIN' in title else 'sub'}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}\n")


active_main = df_comb_main["gt_main"].unique()
active_sub  = df_comb_sub["gt_sub"].unique()

pred_main = add_other_category(df_comb_main["lcc_main_name_pred"].copy())
pred_sub  = add_other_category(df_comb_sub["lcc_name_pred"].copy())

plot_confusion_matrix(
    df_comb_main["gt_main"],
    pred_main.where(pred_main.isin(active_main), other="Other"),
    "MAIN CATEGORY", figsize=(8, 6)
)

plot_confusion_matrix(
    df_comb_sub["gt_sub"],
    pred_sub.where(pred_sub.isin(active_sub), other="Other"),
    "SUB CATEGORY", figsize=(14, 10)
)

# ── 8. Per-journal accuracy table ────────────────────────────────────────────
journal_acc = []
for journal, (gt_main, gt_sub) in journal_ground_truth.items():
    group = df_eval[df_eval["journal"] == journal]
    if len(group) == 0:
        continue
    n        = len(group)
    acc_main = group["lcc_main_name_pred"].eq(gt_main).mean()
    acc_sub  = group["lcc_name_pred"].eq(gt_sub).mean()
    journal_acc.append({
        "journal":   journal,
        "n_papers":  n,
        "gt_main":   gt_main,
        "gt_sub":    gt_sub,
        "acc_main":  round(acc_main, 3),
        "acc_sub":   round(acc_sub,  3),
    })

journal_acc_df = pd.DataFrame(journal_acc).sort_values("acc_main")

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_colwidth", 60)
print("=" * 70)
print("PER-JOURNAL ACCURACY")
print("=" * 70)
print(journal_acc_df.to_string(index=False))

# ── 9. Low-performing journals deep dive ─────────────────────────────────────
LOW_PERF_THRESHOLD = 0.70

low_perf_journals = journal_acc_df[
    journal_acc_df["acc_main"] < LOW_PERF_THRESHOLD
]["journal"].tolist()

print(f"\nLow-performing journals (acc_main < {LOW_PERF_THRESHOLD}): {len(low_perf_journals)}\n")

print("=" * 80)
print("WHERE MISCLASSIFIED PAPERS GO — MAIN CATEGORY")
print("=" * 80)

summary = []
for journal in low_perf_journals:
    gt_main, gt_sub = journal_ground_truth[journal]
    group = df_eval[df_eval["journal"] == journal]
    n = len(group)

    pred_dist = (
        group["lcc_main_name_pred"]
        .value_counts(normalize=True)
        .mul(100).round(1)
        .reset_index()
    )
    pred_dist.columns = ["predicted", "pct"]

    print(f"\n  {journal}  (n={n:,} | gt={gt_main})")
    print(f"  {'Predicted':<45} {'%':>6}")
    print(f"  {'-'*52}")
    for _, row in pred_dist.iterrows():
        marker = " ✓" if row["predicted"] == gt_main else " ✗"
        print(f"  {row['predicted']:<45} {row['pct']:>5}%{marker}")

    wrong         = group[group["lcc_main_name_pred"] != gt_main]["lcc_main_name_pred"]
    top_wrong     = wrong.value_counts().idxmax() if len(wrong) > 0 else "—"
    top_wrong_pct = wrong.value_counts(normalize=True).max() * 100 if len(wrong) > 0 else 0
    acc_main      = group["lcc_main_name_pred"].eq(gt_main).mean()
    acc_sub       = group["lcc_name_pred"].eq(gt_sub).mean()

    summary.append({
        "journal":        journal,
        "n_papers":       n,
        "gt_main":        gt_main,
        "acc_main":       round(acc_main, 3),
        "acc_sub":        round(acc_sub,  3),
        "top_wrong_pred": top_wrong,
        "top_wrong_pct":  round(top_wrong_pct, 1),
    })

print("\n")
print("=" * 80)
print("WHERE MISCLASSIFIED PAPERS GO — SUB CATEGORY")
print("=" * 80)

for journal in low_perf_journals:
    gt_main, gt_sub = journal_ground_truth[journal]
    group = df_eval[df_eval["journal"] == journal].dropna(subset=["lcc_name_pred"])
    n = len(group)

    pred_dist = (
        group["lcc_name_pred"]
        .value_counts(normalize=True)
        .mul(100).round(1)
        .reset_index()
    )
    pred_dist.columns = ["predicted", "pct"]

    print(f"\n  {journal}  (n={n:,} | gt={gt_sub})")
    print(f"  {'Predicted':<55} {'%':>6}")
    print(f"  {'-'*62}")
    for _, row in pred_dist.iterrows():
        marker = " ✓" if row["predicted"] == gt_sub else " ✗"
        print(f"  {row['predicted']:<55} {row['pct']:>5}%{marker}")

print("\n")
print("=" * 80)
print("LOW-PERFORMING JOURNALS — SUMMARY TABLE")
print("=" * 80)
summary_df = pd.DataFrame(summary).sort_values("acc_main")
print(summary_df.to_string(index=False))