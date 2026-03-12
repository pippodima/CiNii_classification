import pandas as pd
import numpy as np
import json

# --- Load metadata ---
meta = pd.read_parquet("data/moreinfo_emb.parquet")

# --- Load embedding state ---
with open("data/embedded/checkpoints/embeddings_ckpt.state.json") as f:
    state = json.load(f)

embs = np.memmap(
    state["mmap_path"],
    mode="r",
    dtype=np.dtype(state["dtype"]),
    shape=(state["n_docs"], state["dim"])
)

# --- Load emb_id -> row mapping ---
id_map = pd.read_parquet(
    "data/embedded/checkpoints/embeddings_ckpt.emb_id.parquet"
)

id_to_row = dict(zip(id_map.emb_id, range(len(id_map))))

# --- Print title + embedding for first 10 papers ---
for i in range(10):
    emb_id = meta.loc[i, "emb_id"]
    title = meta.loc[i, "title"]

    row_idx = id_to_row[emb_id]
    vec = embs[row_idx]

    print(f"\nTitle: {title}")
    print(f"Embedding:\n{vec}")

print(meta.columns)
print(meta.shape)