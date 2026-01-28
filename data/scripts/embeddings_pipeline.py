import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datetime import datetime
import os, json


# -----------------------------------------------------
# Embedding helpers
# -----------------------------------------------------
def getModel(device="cpu", modelName="Qwen/Qwen3-Embedding-0.6B"):
    return SentenceTransformer(modelName, device=device)


def get_title_and_abstract(df: pd.DataFrame, title_col="title_en", abs_col="abstract_en"):
    df["full_text"] = df[title_col].fillna("") + ". " + df[abs_col].fillna("")
    return df["full_text"].tolist()


def getEmbeddings(model, query, documents, device="cpu", batch_size=32):
    _ = model.encode(query, prompt_name="query", device=device)

    embeddings = []
    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding batches"):
        batch = documents[i:i + batch_size]
        batch_embs = model.encode(batch)
        embeddings.extend(batch_embs)

    return np.array(embeddings)
    


def getEmbeddings_checkpointed(
    model,
    query,
    documents,
    device="cpu",
    batch_size=32,
    checkpoint_dir="output",
    checkpoint_name="embeddings_ckpt",
    save_every_n_batches=10,
):
    """
    Compute embeddings with periodic checkpoints and resume support.

    Saves:
      - {checkpoint_name}.npz containing:
          embs: float32 array [n_done, dim]
          next_i: int (next document index to compute)
          dim: int
          n_docs: int
      - {checkpoint_name}.json (human-readable progress)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    npz_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.npz")
    json_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.json")

    # "prime" the query embedding (you had this already)
    _ = model.encode(query, prompt_name="query", device=device)

    n_docs = len(documents)
    next_i = 0
    embs_list = []

    # ---- Resume if checkpoint exists ----
    if os.path.exists(npz_path):
        ckpt = np.load(npz_path, allow_pickle=False)
        saved_embs = ckpt["embs"]
        next_i = int(ckpt["next_i"])
        saved_n_docs = int(ckpt.get("n_docs", n_docs))

        if saved_n_docs != n_docs:
            raise ValueError(
                f"Checkpoint documents length mismatch: ckpt n_docs={saved_n_docs}, current n_docs={n_docs}. "
                "This usually means your input data/order changed. Regenerate checkpoint."
            )

        # Convert saved portion to list of rows for easy extend
        embs_list = [row for row in saved_embs]
        print(f"🔁 Resuming embeddings from checkpoint: next_i={next_i}/{n_docs}")

    # If already complete
    if next_i >= n_docs:
        out = np.array(embs_list, dtype=np.float32)
        print("✅ Embeddings already complete from checkpoint.")
        return out

    # ---- Main loop ----
    start_range = range(next_i, n_docs, batch_size)
    pbar = tqdm(start_range, desc="Embedding batches (checkpointed)")

    batches_since_save = 0
    dim = None

    def _atomic_save_npz(path, **arrays):
        tmp = path + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, path)

    for i in pbar:
        batch = documents[i : i + batch_size]
        batch_embs = model.encode(batch)  # returns np array
        batch_embs = np.asarray(batch_embs)

        if dim is None:
            dim = int(batch_embs.shape[1])

        # keep memory manageable; store float32
        batch_embs = batch_embs.astype(np.float32, copy=False)

        # append rows
        for row in batch_embs:
            embs_list.append(row)

        next_i = i + len(batch)

        batches_since_save += 1
        pbar.set_postfix(done=next_i, total=n_docs)

        if batches_since_save >= save_every_n_batches or next_i >= n_docs:
            # save checkpoint
            embs_arr = np.vstack(embs_list) if len(embs_list) else np.zeros((0, dim), dtype=np.float32)
            _atomic_save_npz(
                npz_path,
                embs=embs_arr,
                next_i=np.array(next_i, dtype=np.int64),
                dim=np.array(embs_arr.shape[1] if embs_arr.size else (dim or 0), dtype=np.int64),
                n_docs=np.array(n_docs, dtype=np.int64),
            )
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "next_i": next_i,
                        "n_docs": n_docs,
                        "batch_size": batch_size,
                        "save_every_n_batches": save_every_n_batches,
                        "checkpoint_npz": npz_path,
                    },
                    f,
                    indent=2,
                )
            batches_since_save = 0

    return np.vstack(embs_list).astype(np.float32, copy=False)
