# import numpy as np
# import json
# import time

# STATE_PATH = "data/embedded/checkpoints/embeddings_ckpt.state.json"

# with open(STATE_PATH) as f:
#     state = json.load(f)

# embs = np.memmap(
#     state["mmap_path"], mode="r",
#     dtype=np.dtype(state["dtype"]),
#     shape=(state["n_docs"], state["dim"]),
# )

# CHUNK = 100_000
# n = state["n_docs"]
# TIMEOUT = 10  # seconds before we flag it as suspicious

# print(f"Scanning {n:,} rows in chunks of {CHUNK:,}...")
# for start in range(0, n, CHUNK):
#     end = min(start + CHUNK, n)
#     t0 = time.time()
#     x = np.array(embs[start:end])  # force full read
#     elapsed = time.time() - t0
#     pct = start / n * 100

#     flag = " ← SLOW" if elapsed > TIMEOUT else ""
#     print(f"  [{pct:5.1f}%] rows {start:>9,}-{end:>9,}  {elapsed:6.2f}s  mean={x.mean():.4f}{flag}")


import numpy as np
import pandas as pd
import json
import time

# STATE_PATH = "data/embedded/checkpoints/embeddings_ckpt.state.json"
# IDMAP_PATH = "data/embedded/checkpoints/embeddings_ckpt.emb_id.parquet"

# with open(STATE_PATH) as f:
#     state = json.load(f)

# embs = np.memmap(
#     state["mmap_path"], mode="r",
#     dtype=np.dtype(state["dtype"]),
#     shape=(state["n_docs"], state["dim"]),
# )

# id_map = pd.read_parquet(IDMAP_PATH, columns=["emb_id"])


# # binary search within 1,700,000 - 1,800,000
# CHUNK = 10_000
# print("Narrowing down bad region...")
# for start in range(1_700_000, 1_800_000, CHUNK):
#     end = min(start + CHUNK, 1_800_000)
#     t0 = time.time()
#     x = np.array(embs[start:end])
#     elapsed = time.time() - t0
#     print(f"  rows {start:,}-{end:,}  {elapsed:.2f}s")


# for start in range(1_770_000, 1_780_000, 1_000):
#     end = start + 1_000
#     t0 = time.time()
#     x = np.array(embs[start:end])
#     elapsed = time.time() - t0
#     print(f"  rows {start:,}-{end:,}  {elapsed:.2f}s")


# for row in range(1_775_670, 1_775_690):
#     emb_id = str(id_map.iloc[row]["emb_id"])
#     print(f"  reading row {row:,}  emb_id={emb_id} ...", flush=True)
#     t0 = time.time()
#     x = np.array(embs[row:row+1])
#     elapsed = time.time() - t0
#     print(f"  done  {elapsed:.2f}s")

# import pandas as pd
# import json

# META_PATH  = "data/embedded/embedded_data.parquet"
# IDMAP_PATH = "data/embedded/checkpoints/embeddings_ckpt.emb_id.parquet"

# BAD_MMAP_ROW = 1_775_675

# # find which emb_id corresponds to mmap row 1,775,675
# id_map = pd.read_parquet(IDMAP_PATH, columns=["emb_id"])
# bad_emb_id = id_map.iloc[BAD_MMAP_ROW]["emb_id"]
# print(f"emb_id at mmap row {BAD_MMAP_ROW:,}: {bad_emb_id}")

# # find that emb_id in the full metadata
# df = pd.read_parquet(META_PATH)
# row = df[df["emb_id"] == bad_emb_id]
# print(f"\nMatching row(s) in metadata: {len(row)}")
# print(row.to_string())


df = pd.read_parquet("models/data/predicted/moreinfo/predict_moreinfo_emb_final.parquet")

def analyze_lcc_columns(df):
    for col in ['lcc_main_name_pred', 'lcc_name_pred']:
        print(f"\n{'='*60}")
        print(f"COLUMN: {col}")
        print(f"{'='*60}")
        
        total = len(df)
        null_count = df[col].isna().sum()
        non_null = total - null_count
        unique_vals = df[col].nunique(dropna=True)

        print(f"Total rows       : {total:,}")
        print(f"Non-null         : {non_null:,} ({non_null/total*100:.1f}%)")
        print(f"NaN / missing    : {null_count:,} ({null_count/total*100:.1f}%)")
        print(f"Unique values    : {unique_vals:,}")

        print(f"\nTop 20 value counts (excl. NaN):")
        print(df[col].value_counts(dropna=True).head(20).to_string())

        print(f"\nBottom 5 (least frequent):")
        print(df[col].value_counts(dropna=True).tail(5).to_string())

        null_by_main = (df[df['lcc_name_pred'].isna()]
                        .groupby('lcc_main_name_pred')
                        .size()
                        .sort_values(ascending=False)
                        .reset_index(name='null_count'))

        null_by_main['pct_of_total_nulls'] = (null_by_main['null_count'] / df['lcc_name_pred'].isna().sum() * 100).round(1)
        null_by_main['pct_of_main_rows'] = (null_by_main['null_count'] / df['lcc_main_name_pred'].value_counts()[null_by_main['lcc_main_name_pred']].values * 100).round(1)

        print(null_by_main.to_string(index=False))

analyze_lcc_columns(df)
