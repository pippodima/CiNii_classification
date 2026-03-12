"""
predict_moreinfo_emb_fast.py

Key changes vs your script (to prevent “stuck at 50%” swap-thrashing):
- Load ONLY needed metadata columns (default: emb_id) instead of 23 object/text cols.
- Force single-thread BLAS/OpenMP + single-thread KNN (avoids oversubscription).
- Avoid doing KNN twice per batch (no predict()+kneighbors()).
  We call kneighbors() once and compute BOTH:
    - prediction (majority vote, k=3)
    - confidence = 1/(1 + nearest_dist)
- Avoid df.loc writes inside tight loops; fill preallocated numpy arrays by position.

If you want extra metadata columns in output, add them to KEEP_COLS.
"""

import os
import json
import numpy as np
import pandas as pd
from joblib import load
from tqdm.auto import tqdm

# ----------------------------
# Thread limits (set BEFORE heavy compute)
# ----------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

MAIN_MODEL_PATH = "models/main/new_lcc_main_knn.joblib"
META_PATH = "data/embedded/embedded_data.parquet"

STATE_PATH = "data/embedded/checkpoints/embeddings_ckpt.state.json"
IDMAP_PATH = "data/embedded/checkpoints/embeddings_ckpt.emb_id.parquet"

OUTPUT_DIR = "models/data/predicted"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "predict_moreinfo_emb.parquet")

SUB_ROOT = "models/sub/data"
SUB_MODEL_NAME = "knn_k3.joblib"

BATCH_SIZE = 4096
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Only load the columns you actually need (add more if you must)
KEEP_COLS = ["file", "emb_id"]

# ----------------------------
def norm_str(x: str) -> str:
    if not isinstance(x, str):
        return ""
    return " ".join(x.strip().upper().replace("&", "AND").split())

MAIN_TO_SUBFOLDER = {
    norm_str("SCIENCE"): "SCIENCE",
    norm_str("TECHNOLOGY"): "TECHNOLOGY",
    norm_str("MEDICINE"): "MEDICINE",
    norm_str("GEOGRAPHY. ANTHROPOLOGY. RECREATION"): "GEOGRAPHY_ANTHROPOLOGY_RECREATION",
}

SINGLE_SUBCLASS = {
    norm_str("EDUCATION"): "Theory and practice of education",
    norm_str("AGRICULTURE"): "Forestry",
    norm_str("SOCIAL SCIENCES"): "Social sciences",
    norm_str("NAVAL SCIENCE"): "Naval architecture. Shipbuilding. Marine engineering",
    norm_str("Bibliography & Information Science"): "Bibliography & Information Science",
}

def read_mmap_batch(memmap: np.memmap, rows: np.ndarray) -> np.ndarray:
    """Read memmap rows with sequential access inside the batch."""
    order = np.argsort(rows)
    sorted_rows = rows[order]
    X = memmap[sorted_rows]
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return X[inv]

def get_knn_parts(model):
    """
    Returns (transformer, knn_estimator).
    transformer: callable that maps X -> X_transformed (identity if none)
    knn: sklearn KNN estimator with kneighbors()
    """
    if hasattr(model, "named_steps") and "knn" in model.named_steps:
        knn = model.named_steps["knn"]
        # pipeline[:-1] exists in sklearn Pipeline
        transformer = model[:-1].transform
        return transformer, knn
    # plain knn
    return (lambda X: X), model

def knn_predict_and_confidence(model, X: np.ndarray):
    """
    Compute prediction + confidence using ONE kneighbors() call.
    Assumes classifier with uniform weights. Works great for k=3.
    """
    transformer, knn = get_knn_parts(model)

    Xt = transformer(X)

    # Make sure we don't spawn tons of workers
    if hasattr(knn, "n_jobs"):
        knn.n_jobs = 1

    dists, neigh = knn.kneighbors(Xt, return_distance=True)
    conf = (1.0 / (1.0 + dists[:, 0])).astype(np.float32)

    # Majority vote from neighbor labels (no second KNN pass)
    # sklearn stores encoded labels in knn._y (usually ints 0..n_classes-1)
    y = getattr(knn, "_y", None)
    if y is None:
        # Fallback: if internals differ, use predict() (this does extra work)
        pred = knn.predict(Xt)
        return pred, conf

    y = np.asarray(y)
    # y can be (n_samples,) or (n_samples, 1) for single output
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]

    # labels per row: shape (batch, k)
    labels = y[neigh]

    # k is small (3). Do a fast vote per row.
    # labels might be floats/objects if trained that way; try to handle ints best.
    if labels.dtype.kind in ("i", "u"):
        # fast path for ints
        pred_enc = np.empty(labels.shape[0], dtype=labels.dtype)
        for i in range(labels.shape[0]):
            vals, counts = np.unique(labels[i], return_counts=True)
            pred_enc[i] = vals[np.argmax(counts)]
        # decode to original class labels
        classes_ = getattr(knn, "classes_", None)
        if classes_ is not None:
            pred = np.asarray(classes_, dtype=object)[pred_enc]
        else:
            pred = pred_enc
    else:
        # general path
        pred = np.empty(labels.shape[0], dtype=object)
        for i in range(labels.shape[0]):
            vals, counts = np.unique(labels[i].astype(object), return_counts=True)
            pred[i] = vals[np.argmax(counts)]

    return pred, conf

# ----------------------------
print("Loading metadata (minimal columns)...")
df = pd.read_parquet(META_PATH, columns=KEEP_COLS).copy()
n = len(df)
print("rows:", n)

print("Loading mmap state...")
with open(STATE_PATH) as f:
    state = json.load(f)

embs = np.memmap(
    state["mmap_path"],
    mode="r",
    dtype=np.dtype(state["dtype"]),
    shape=(state["n_docs"], state["dim"]),
)

print("Loading emb_id map...")
id_map = pd.read_parquet(IDMAP_PATH, columns=["emb_id"])
id_to_row = dict(zip(id_map.emb_id.values, range(len(id_map))))

print("Mapping emb_id -> row_idx...")
row_idx = df["emb_id"].map(id_to_row)
missing = int(row_idx.isna().sum())
if missing:
    # drop missing to avoid memmap index errors
    print(f"WARNING: missing emb_id mappings: {missing} (dropping them)")
    df = df.loc[row_idx.notna()].copy()
    row_idx = row_idx.loc[row_idx.notna()]
df["row_idx"] = row_idx.astype(np.int32)
# Sort once for true sequential mmap access
df["_orig_pos"] = np.arange(len(df))
df = df.sort_values("row_idx").reset_index(drop=True)
n = len(df)

print("Loading main model...")
main_model = load(MAIN_MODEL_PATH)
# Force single-thread KNN
if hasattr(main_model, "named_steps") and "knn" in main_model.named_steps:
    if hasattr(main_model.named_steps["knn"], "n_jobs"):
        main_model.named_steps["knn"].n_jobs = 1

# Preallocate outputs (avoid df.loc in hot loop)
main_pred = np.empty(n, dtype=object)
main_conf = np.empty(n, dtype=np.float32)

sub_pred = np.empty(n, dtype=object)
sub_conf = np.full(n, np.nan, dtype=np.float32)
sub_used = np.empty(n, dtype=object)

# ----------------------------
# 1) MAIN prediction with global sequential mmap scan
# ----------------------------
print("\nMAIN prediction (true sequential mmap scan)")

for start in tqdm(range(0, n, BATCH_SIZE)):
    end = min(start + BATCH_SIZE, n)

    rows = df["row_idx"].values[start:end].astype(np.int64)

    # contiguous slice (critical fix)
    r0 = rows[0]
    r1 = rows[-1] + 1
    X = embs[r0:r1]

    # select only needed rows inside the slice
    X = X[rows - r0]

    pred, conf = knn_predict_and_confidence(main_model, X)

    main_pred[start:end] = pred
    main_conf[start:end] = conf

# Normalize main names once (vectorized-ish)
main_norm = pd.Series(main_pred, dtype="object").map(norm_str).values

# restore original row order
restore = np.argsort(df["_orig_pos"].values)
main_pred = main_pred[restore]
main_conf = main_conf[restore]
df = df.iloc[restore].reset_index(drop=True)

# ----------------------------
# Load submodels once
# ----------------------------
print("\nLoading submodels...")
SUB_MODELS = {}
ID_TO_NAME = {}

for k, v in MAIN_TO_SUBFOLDER.items():
    sub_dir = os.path.join(SUB_ROOT, v)
    p = os.path.join(sub_dir, "models", SUB_MODEL_NAME)
    if os.path.exists(p):
        m = load(p)
        # force single-thread if possible
        if hasattr(m, "named_steps") and "knn" in m.named_steps and hasattr(m.named_steps["knn"], "n_jobs"):
            m.named_steps["knn"].n_jobs = 1
        elif hasattr(m, "n_jobs"):
            m.n_jobs = 1
        SUB_MODELS[k] = m

        name_map_path = os.path.join(sub_dir, "id_to_name.json")
        if os.path.exists(name_map_path):
            with open(name_map_path) as f:
                ID_TO_NAME[k] = {int(a): b for a, b in json.load(f).items()}

# ----------------------------
# 2) SUB prediction (batched, no df.loc)
# ----------------------------
print("\nSUB prediction")
for start in tqdm(range(0, n, BATCH_SIZE)):
    end = min(start + BATCH_SIZE, n)
    rows = df["row_idx"].values[start:end].astype(np.int64)
    X = read_mmap_batch(embs, rows)

    mains = main_norm[start:end]

    # process per unique main in the batch
    for main_name in np.unique(mains):
        mask = mains == main_name
        idx = np.arange(start, end)[mask]

        if main_name in SINGLE_SUBCLASS:
            sub_pred[idx] = SINGLE_SUBCLASS[main_name]
            sub_conf[idx] = 1.0
            sub_used[idx] = "direct_map"
            continue

        if main_name not in SUB_MODELS:
            sub_used[idx] = f"no_submodel:{main_name}"
            continue

        model = SUB_MODELS[main_name]
        Xsub = X[mask]

        sp, sc = knn_predict_and_confidence(model, Xsub)

        # Optional id->name decoding if submodel returns integer ids
        if main_name in ID_TO_NAME:
            # only if the prediction looks integer-like
            try:
                sp2 = []
                mp = ID_TO_NAME[main_name]
                for v in sp:
                    sp2.append(mp.get(int(v), None))
                sp = np.array(sp2, dtype=object)
            except Exception:
                pass

        sub_pred[idx] = sp
        sub_conf[idx] = sc
        sub_used[idx] = main_name

# ----------------------------
# Build output (keep minimal cols + predictions)
# ----------------------------
out = df[KEEP_COLS].copy()
out["lcc_main_name_pred"] = pd.Categorical(main_pred)  # cheaper than raw object
out["main_knn_confidence"] = main_conf.astype(np.float32)
out["lcc_name_pred"] = pd.Categorical(sub_pred)
out["sub_knn_confidence"] = sub_conf.astype(np.float32)
out["sub_model_used"] = pd.Categorical(sub_used)

print("\nSaving...")
out.to_parquet(OUTPUT_PATH, index=False)
print("DONE →", OUTPUT_PATH)