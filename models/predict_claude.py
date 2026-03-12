"""
predict_simple.py

Single-call approach: load all embeddings into RAM once via sequential
mmap read (fast), then one kneighbors() call per model.

Bad emb_ids are read from BAD_EMB_IDS_PATH (one per line, # for comments).
Bad physical mmap rows are defined in BAD_MMAP_ROWS and skipped during read.
"""

import os
import json

os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"
os.environ["LOKY_MAX_CPU_COUNT"]   = "1"

import numpy as np
import pandas as pd
from joblib import load, parallel_backend
from tqdm.auto import tqdm

# ----------------------------
# Paths
# ----------------------------
MAIN_MODEL_PATH  = "models/main/new_lcc_main_knn.joblib"
META_PATH        = "data/embedded/embedded_data.parquet"
STATE_PATH       = "data/embedded/checkpoints/embeddings_ckpt.state.json"
IDMAP_PATH       = "data/embedded/checkpoints/embeddings_ckpt.emb_id.parquet"
OUTPUT_DIR       = "models/data/predicted"
OUTPUT_PATH      = os.path.join(OUTPUT_DIR, "predict_moreinfo_emb.parquet")
SUB_ROOT         = "models/sub/data"
SUB_MODEL_NAME   = "knn_k3.joblib"
BAD_EMB_IDS_PATH = "data/embedded/bad_emb_ids.txt"  # one emb_id per line

# Physical mmap row positions that hang on sequential read.
# These are disk-level bad spots — add new ones here as discovered.
BAD_MMAP_ROWS = {
    1_775_675,  # emb_id 387c53a7c5f98068...  hangs on sequential read
}

BATCH_SIZE = 4096
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Label maps
# ----------------------------
def norm(x):
    if not isinstance(x, str):
        return ""
    return " ".join(x.strip().upper().replace("&", "AND").split())

MAIN_TO_SUBFOLDER = {
    norm("SCIENCE"):                             "SCIENCE",
    norm("TECHNOLOGY"):                          "TECHNOLOGY",
    norm("MEDICINE"):                            "MEDICINE",
    norm("GEOGRAPHY. ANTHROPOLOGY. RECREATION"): "GEOGRAPHY_ANTHROPOLOGY_RECREATION",
}

SINGLE_SUBCLASS = {
    norm("EDUCATION"):                          "Theory and practice of education",
    norm("AGRICULTURE"):                        "Forestry",
    norm("SOCIAL SCIENCES"):                    "Social sciences",
    norm("NAVAL SCIENCE"):                      "Naval architecture. Shipbuilding. Marine engineering",
    norm("Bibliography & Information Science"):  "Bibliography & Information Science",
    norm("LANGUAGE & LITERATURE"):               "Language & Literature",

}

# ----------------------------
# Helpers
# ----------------------------
def load_bad_emb_ids(path):
    """Load bad emb_ids from txt file (one per line, # for comments)."""
    if not os.path.exists(path):
        print(f"  no bad_emb_ids file found at {path} — skipping")
        return set()
    with open(path) as f:
        ids = {
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        }
    print(f"  loaded {len(ids)} bad emb_id(s) from {path}")
    return ids


def patch_model(model):
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            patch_model(step)
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    return model


def get_knn(model):
    if hasattr(model, "named_steps") and "knn" in model.named_steps:
        return model[:-1].transform, model.named_steps["knn"]
    return (lambda X: X), model


def majority_vote(knn, neigh_idx):
    y = getattr(knn, "_y", None)
    if y is None:
        raise ValueError("knn._y not found")
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]

    neighbor_labels = y[neigh_idx]
    pred = np.empty(len(neigh_idx), dtype=object)
    for i in range(len(neigh_idx)):
        vals, counts = np.unique(neighbor_labels[i], return_counts=True)
        pred[i] = vals[counts.argmax()]

    classes_ = getattr(knn, "classes_", None)
    if classes_ is not None and neighbor_labels.dtype.kind in ("i", "u"):
        pred = np.asarray(classes_, dtype=object)[pred.astype(int)]
    return pred


def run_predict(model, X):
    transform, knn = get_knn(model)
    Xt = transform(X)
    with parallel_backend("threading", n_jobs=1):
        dists, neigh_idx = knn.kneighbors(Xt, return_distance=True)
    conf = (1.0 / (1.0 + dists[:, 0])).astype(np.float32)
    pred = majority_vote(knn, neigh_idx)
    return pred, conf


def read_mmap_safe(embs, r0, r1, bad_rows, chunk=200_000):
    """
    Read embs[r0:r1] into RAM sequentially, skipping known bad physical rows.
    Bad rows are filled with zeros (they will be filtered from df anyway).
    """
    span  = r1 - r0
    X_seq = np.empty((span, embs.shape[1]), dtype=embs.dtype)

    with tqdm(total=span, unit="rows", unit_scale=True, desc="  reading mmap") as pbar:
        for c_start in range(0, span, chunk):
            c_end = min(c_start + chunk, span)

            # find bad physical rows inside this chunk
            bad_in_chunk = sorted(
                b for b in bad_rows
                if r0 + c_start <= b < r0 + c_end
            )

            if not bad_in_chunk:
                # fast path — no bad rows, read whole chunk at once
                X_seq[c_start:c_end] = embs[r0 + c_start : r0 + c_end]
            else:
                # split chunk around each bad row
                sub_start = r0 + c_start
                for bad in bad_in_chunk:
                    if sub_start < bad:
                        s = sub_start - r0
                        e = bad - r0
                        X_seq[s:e] = embs[sub_start:bad]
                    # fill bad row with zeros (document already dropped from df)
                    X_seq[bad - r0] = 0.0
                    print(f"\n  skipped bad mmap row {bad:,}")
                    sub_start = bad + 1
                # remainder after last bad row
                if sub_start < r0 + c_end:
                    s = sub_start - r0
                    e = r0 + c_end - r0
                    X_seq[s:e] = embs[sub_start : r0 + c_end]

            pbar.update(c_end - c_start)

    return X_seq


# ----------------------------
# Load data
# ----------------------------
print("Loading metadata...")
df = pd.read_parquet(META_PATH, columns=["file", "emb_id"]).copy()
print(f"  rows: {len(df):,}")

print("Loading mmap state...")
with open(STATE_PATH) as f:
    state = json.load(f)

embs = np.memmap(
    state["mmap_path"], mode="r",
    dtype=np.dtype(state["dtype"]),
    shape=(state["n_docs"], state["dim"]),
)

print("Building emb_id → mmap row index map...")
id_map    = pd.read_parquet(IDMAP_PATH, columns=["emb_id"])
id_to_row = dict(zip(id_map["emb_id"].values, range(len(id_map))))

print("Mapping emb_id → row index...")
row_idx = df["emb_id"].map(id_to_row)
missing = row_idx.isna().sum()
if missing:
    print(f"  WARNING: dropping {missing:,} rows with missing emb_id")
    df      = df[row_idx.notna()].copy()
    row_idx = row_idx[row_idx.notna()]

df["_row"] = row_idx.astype(np.int32)

# ----------------------------
# Drop bad emb_ids
# ----------------------------
print("Checking for bad emb_ids...")
bad_emb_ids = load_bad_emb_ids(BAD_EMB_IDS_PATH)
if bad_emb_ids:
    bad_mask = df["emb_id"].isin(bad_emb_ids)
    print(f"  dropping {bad_mask.sum():,} rows with bad emb_id(s)")
    if bad_mask.sum() > 0:
        print(df[bad_mask][["file", "emb_id"]].to_string())
    df = df[~bad_mask].copy().reset_index(drop=True)

n       = len(df)
row_arr = df["_row"].values
print(f"  rows after filtering: {n:,}")

# ----------------------------
# Load all embeddings into RAM (sequential read, skipping bad rows)
# ----------------------------
gb = n * embs.shape[1] * 4 / 1e9
print(f"\nLoading all {n:,} embeddings into RAM (~{gb:.1f} GB)...")
print(f"  bad physical mmap rows to skip: {sorted(BAD_MMAP_ROWS)}")

sort_idx       = np.argsort(row_arr)
row_arr_sorted = row_arr[sort_idx]

r0   = int(row_arr_sorted[0])
r1   = int(row_arr_sorted[-1]) + 1

X_seq = read_mmap_safe(embs, r0, r1, BAD_MMAP_ROWS)

print("  selecting and reordering rows...")
X_sorted = X_seq[row_arr_sorted - r0]
del X_seq

restore = np.argsort(sort_idx)
X_all   = X_sorted[restore]
del X_sorted

print(f"  done. shape={X_all.shape} dtype={X_all.dtype}")

# ----------------------------
# Load + patch main model
# ----------------------------
print("\nLoading main model...")
main_model = load(MAIN_MODEL_PATH)
patch_model(main_model)
_, main_knn = get_knn(main_model)
print(f"  n_jobs={main_knn.n_jobs}  algorithm={main_knn.algorithm}  metric={main_knn.metric}  fit_shape={main_knn._fit_X.shape}")

# ----------------------------
# MAIN prediction — single call
# ----------------------------
print("\nMAIN prediction (single kneighbors call)...")
with tqdm(total=1, desc="  kneighbors") as pbar:
    main_pred, main_conf = run_predict(main_model, X_all)
    pbar.update(1)

main_norm = np.vectorize(norm)(main_pred)
print(f"  done. unique classes: {len(np.unique(main_pred))}")

# ----------------------------
# Load sub-models
# ----------------------------
print("\nLoading sub-models...")
SUB_MODELS = {}
ID_TO_NAME = {}

for key, folder in MAIN_TO_SUBFOLDER.items():
    path = os.path.join(SUB_ROOT, folder, "models", SUB_MODEL_NAME)
    if not os.path.exists(path):
        print(f"  missing: {path}")
        continue
    m = patch_model(load(path))
    SUB_MODELS[key] = m
    print(f"  loaded: {folder}  n_jobs={get_knn(m)[1].n_jobs}")

    name_map = os.path.join(SUB_ROOT, folder, "id_to_name.json")
    if os.path.exists(name_map):
        with open(name_map) as f:
            ID_TO_NAME[key] = {int(k): v for k, v in json.load(f).items()}

# ----------------------------
# SUB prediction — per class
# ----------------------------
print("\nSUB prediction...")
sub_pred = np.empty(n, dtype=object)
sub_conf = np.full(n, np.nan, dtype=np.float32)
sub_used = np.empty(n, dtype=object)

unique_mains = np.unique(main_norm)
for main_name in tqdm(unique_mains, desc="  sub-models"):
    idx = np.where(main_norm == main_name)[0]

    if main_name in SINGLE_SUBCLASS:
        sub_pred[idx] = SINGLE_SUBCLASS[main_name]
        sub_conf[idx] = 1.0
        sub_used[idx] = "direct_map"
        continue

    if main_name not in SUB_MODELS:
        sub_used[idx] = f"no_submodel:{main_name}"
        continue

    sp, sc = run_predict(SUB_MODELS[main_name], X_all[idx])

    if main_name in ID_TO_NAME:
        try:
            mp = ID_TO_NAME[main_name]
            sp = np.array([mp.get(int(v), v) for v in sp], dtype=object)
        except Exception:
            pass

    sub_pred[idx] = sp
    sub_conf[idx] = sc
    sub_used[idx] = main_name

# ----------------------------
# Save
# ----------------------------
print("\nSaving...")
out = df[["file", "emb_id"]].copy()
out["lcc_main_name_pred"] = pd.Categorical(main_pred)
out["main_knn_confidence"] = main_conf
out["lcc_name_pred"]       = pd.Categorical(sub_pred)
out["sub_knn_confidence"]  = sub_conf
out["sub_model_used"]      = pd.Categorical(sub_used)

out.to_parquet(OUTPUT_PATH, index=False)
print("DONE →", OUTPUT_PATH)