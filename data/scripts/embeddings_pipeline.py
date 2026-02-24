# =========================
# FILE 2: embeddings_pipeline.py
# (save checkpoint every batch + write ordered emb_id mapping)
# =========================

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datetime import datetime
import os, json, hashlib, tempfile


# -----------------------------------------------------
# Embedding helpers
# -----------------------------------------------------
def getModel(device="cpu", modelName="Qwen/Qwen3-Embedding-0.6B"):
    return SentenceTransformer(modelName, device=device)


def get_title_and_abstract(df: pd.DataFrame, title_col="title_en", abs_col="abstract_en", max_chars=2200):
    df["full_text"] = df[title_col].fillna("") + ". " + df[abs_col].fillna("").astype(str)
    df["full_text"] = df["full_text"].str.slice(0, max_chars)
    return df["full_text"].tolist()


def getEmbeddings(model, query, documents, device="cpu", batch_size=32):
    _ = model.encode(query, prompt_name="query", device=device)

    embeddings = []
    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding batches"):
        batch = documents[i:i + batch_size]
        batch_embs = model.encode(batch)
        embeddings.extend(batch_embs)

    return np.array(embeddings)


# -----------------------------------------------------
# Atomic JSON write (unchanged behavior)
# -----------------------------------------------------
def _atomic_write_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path),
        suffix=".tmp",
        dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# -----------------------------------------------------
# Fingerprint (KEEP EXACTLY to resume your existing work)
# -----------------------------------------------------
def _fingerprint_documents_full(documents):
    """
    Stable fingerprint of inputs to prevent resuming on different data/order.
    NOTE: This is intentionally the SAME as your current approach so that
    resumes will match partial work already done.
    """
    h = hashlib.sha256()
    for d in documents:
        if d is None:
            d = ""
        if not isinstance(d, str):
            d = str(d)
        h.update(d.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def _normalize_dtype(dtype_like):
    """
    Robustly normalize dtype from:
      - np.float32 / np.dtype('float32') / 'float32'
      - "<class 'numpy.float32'>" (your old saved form)
    """
    if isinstance(dtype_like, np.dtype):
        return dtype_like
    if dtype_like in (np.float32, np.float16, np.float64):
        return np.dtype(dtype_like)

    s = str(dtype_like).strip()
    # Handle old saved strings like "<class 'numpy.float32'>"
    if "float16" in s:
        return np.dtype("float16")
    if "float32" in s:
        return np.dtype("float32")
    if "float64" in s:
        return np.dtype("float64")

    # Try direct parse
    try:
        return np.dtype(s)
    except Exception:
        # Safe default
        return np.dtype("float32")


# -----------------------------------------------------
# Optional lock (won't break if filelock isn't installed)
# -----------------------------------------------------
class _NullLock:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False

def _maybe_file_lock(lock_path: str):
    """
    Prevent two processes from writing the same checkpoint concurrently.
    Uses filelock if available; otherwise no-op.
    """
    try:
        from filelock import FileLock
        return FileLock(lock_path)
    except Exception:
        return _NullLock()


# -----------------------------------------------------
# Helper: save ordered emb_id mapping (small, disk-friendly)
# -----------------------------------------------------
def _save_emb_id_mapping(checkpoint_dir: str, checkpoint_name: str, emb_ids):
    """
    Saves the row-order mapping from embedding row index -> emb_id.
    This links your metadata parquet (which stores emb_id) to the memmap rows.
    """
    if emb_ids is None:
        return None

    path = os.path.join(checkpoint_dir, f"{checkpoint_name}.emb_id.parquet")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save as parquet for easy joins/merges later
    pd.DataFrame({"emb_id": list(emb_ids)}).to_parquet(path, index=False)
    return path


# -----------------------------------------------------
# Crash-proof embeddings with resume that works with your existing ckpts
# -----------------------------------------------------
def getEmbeddings_checkpointed(
    model,
    query,
    documents,
    device="cpu",
    batch_size=32,
    checkpoint_dir="output",
    checkpoint_name="embeddings_ckpt",
    save_every_n_batches=1,   # ✅ default to every batch
    emb_ids=None,             # ✅ ordered ids matching 'documents'
):
    """
    Drop-in replacement that:
      - DOES NOT load all embeddings into RAM at the end (returns memmap ndarray)
      - Resumes from your existing checkpoint files (same doc fingerprint method)
      - Handles old dtype serialization safely
      - Adds a best-effort lock to avoid concurrent corruption
      - Saves a row-order emb_id mapping on disk to link metadata <-> embeddings
      - Checkpoints every batch by default

    Return:
      - a numpy.memmap (ndarray-compatible) of shape (n_docs, dim)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    state_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.state.json")
    mmap_path  = os.path.join(checkpoint_dir, f"{checkpoint_name}.embs.mmap")
    lock_path  = os.path.join(checkpoint_dir, f"{checkpoint_name}.lock")

    # Save emb_id mapping (id list must match documents order)
    emb_id_map_path = _save_emb_id_mapping(checkpoint_dir, checkpoint_name, emb_ids)

    # Keep your behavior: "prime" query embedding
    _ = model.encode(query, prompt_name="query", device=device)

    n_docs = len(documents)
    doc_fp = _fingerprint_documents_full(documents)

    with _maybe_file_lock(lock_path):
        next_i = 0
        dim = None
        dtype = np.dtype("float32")

        embs = None
        state = None

        # -------------------------
        # Resume / initialize state
        # -------------------------
        if os.path.exists(state_path) and os.path.exists(mmap_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)

                if int(state.get("n_docs", -1)) != n_docs:
                    raise ValueError(f"n_docs mismatch (ckpt {state.get('n_docs')} vs current {n_docs})")
                if state.get("doc_fingerprint") != doc_fp:
                    raise ValueError("Input fingerprint mismatch (data/order changed). Refusing to resume.")

                dim = int(state["dim"])
                next_i = int(state["next_i"])
                if next_i < 0 or next_i > n_docs:
                    raise ValueError("Corrupt checkpoint: next_i out of range")

                dtype = _normalize_dtype(state.get("dtype", "float32"))

                embs = np.memmap(mmap_path, mode="r+", dtype=dtype, shape=(n_docs, dim))
                print(f"🔁 Resuming: next_i={next_i}/{n_docs}, dim={dim}, dtype={dtype.name}")

            except Exception as e:
                print(f"⚠️ Checkpoint exists but failed to load safely: {e}")
                print("   -> Starting fresh (keeping old files; not deleting automatically).")
                next_i = 0
                dim = None
                embs = None
                state = None

        # -------------------------
        # If starting fresh, infer dim with 1 batch
        # -------------------------
        if dim is None:
            if n_docs == 0:
                return np.zeros((0, 0), dtype=np.float32)

            first_batch = documents[0:min(batch_size, n_docs)]
            first_embs = np.asarray(model.encode(first_batch))
            if first_embs.ndim != 2:
                raise ValueError(f"Expected 2D embeddings, got shape {first_embs.shape}")

            dim = int(first_embs.shape[1])
            dtype = np.dtype("float32")

            embs = np.memmap(mmap_path, mode="w+", dtype=dtype, shape=(n_docs, dim))
            embs[0:first_embs.shape[0]] = first_embs.astype(dtype, copy=False)
            embs.flush()

            next_i = int(first_embs.shape[0])

            _atomic_write_json(state_path, {
                "timestamp": datetime.now().isoformat(),
                "next_i": next_i,
                "n_docs": n_docs,
                "dim": dim,
                "dtype": dtype.name,
                "batch_size": batch_size,
                "save_every_n_batches": save_every_n_batches,
                "doc_fingerprint": doc_fp,
                "mmap_path": mmap_path,
                "emb_id_map_path": emb_id_map_path,
            })

            print(f"🆕 Initialized: next_i={next_i}/{n_docs}, dim={dim}, dtype={dtype.name}")

        # Already complete?
        if next_i >= n_docs:
            print("✅ Embeddings already complete.")
            return np.memmap(mmap_path, mode="r", dtype=dtype, shape=(n_docs, dim))

        # -------------------------
        # Main loop with crash safety
        # -------------------------
        start_range = range(next_i, n_docs, batch_size)
        pbar = tqdm(start_range, desc="Embedding batches (resume-safe)")

        try:
            for i in pbar:
                batch = documents[i:i + batch_size]
                batch_embs = np.asarray(model.encode(batch))

                if batch_embs.ndim != 2 or batch_embs.shape[1] != dim:
                    raise ValueError(f"Embedding shape changed: got {batch_embs.shape}, expected (*, {dim})")

                batch_len = int(batch_embs.shape[0])
                embs[i:i + batch_len] = batch_embs.astype(dtype, copy=False)

                next_i = i + batch_len
                pbar.set_postfix(done=next_i, total=n_docs)

                # ✅ checkpoint every batch if save_every_n_batches=1 (or per the given interval)
                # Keep interval semantics, but default is now 1.
                # This avoids changing behavior if you pass >1.
                if save_every_n_batches <= 1 or ( ( (i - start_range.start) // batch_size + 1 ) % save_every_n_batches == 0 ) or next_i >= n_docs:
                    embs.flush()
                    _atomic_write_json(state_path, {
                        "timestamp": datetime.now().isoformat(),
                        "next_i": next_i,
                        "n_docs": n_docs,
                        "dim": dim,
                        "dtype": dtype.name,
                        "batch_size": batch_size,
                        "save_every_n_batches": save_every_n_batches,
                        "doc_fingerprint": doc_fp,
                        "mmap_path": mmap_path,
                        "emb_id_map_path": emb_id_map_path,
                    })

        except KeyboardInterrupt:
            print("\n🛑 Interrupted. Saving progress...")
            try:
                embs.flush()
            except Exception:
                pass
            _atomic_write_json(state_path, {
                "timestamp": datetime.now().isoformat(),
                "next_i": next_i,
                "n_docs": n_docs,
                "dim": dim,
                "dtype": dtype.name,
                "batch_size": batch_size,
                "save_every_n_batches": save_every_n_batches,
                "doc_fingerprint": doc_fp,
                "mmap_path": mmap_path,
                "emb_id_map_path": emb_id_map_path,
                "note": "Interrupted by user",
            })
            raise

        except Exception as e:
            print(f"\n💥 Error during embedding: {e}")
            print("   Saving progress before exiting...")
            try:
                embs.flush()
            except Exception:
                pass
            _atomic_write_json(state_path, {
                "timestamp": datetime.now().isoformat(),
                "next_i": next_i,
                "n_docs": n_docs,
                "dim": dim,
                "dtype": dtype.name,
                "batch_size": batch_size,
                "save_every_n_batches": save_every_n_batches,
                "doc_fingerprint": doc_fp,
                "mmap_path": mmap_path,
                "emb_id_map_path": emb_id_map_path,
                "note": f"Exception: {type(e).__name__}",
            })
            raise

        # Done
        print("✅ Done embedding. Final flush...")
        embs.flush()
        _atomic_write_json(state_path, {
            "timestamp": datetime.now().isoformat(),
            "next_i": n_docs,
            "n_docs": n_docs,
            "dim": dim,
            "dtype": dtype.name,
            "batch_size": batch_size,
            "save_every_n_batches": save_every_n_batches,
            "doc_fingerprint": doc_fp,
            "mmap_path": mmap_path,
            "emb_id_map_path": emb_id_map_path,
        })

        return np.memmap(mmap_path, mode="r", dtype=dtype, shape=(n_docs, dim))
