# =========================
# FILE 1: your main script
# (remove RAM-unsafe embeddings.tolist()
#  add stable emb_id + save mapping file)
# =========================

import os
import json
import hashlib
from utils import load_df, save, _get_device
from clustering import auto_optimize_clustering
from labeling import label_clusters
from plots import plot_static, plot_interactive
from datetime import datetime
from embeddings_pipeline import getEmbeddings, getModel, get_title_and_abstract, getEmbeddings_checkpointed
import numpy as np
import pandas as pd
from process_data import apply_clean_text_to_df

OUTPUT_DIR = "data/embedded"
query_prompt = "Given a scientific paper title and abstract, produce an embedding that captures the research topic."

def _make_emb_id_from_full_text(s: str) -> str:
    if s is None:
        s = ""
    if not isinstance(s, str):
        s = str(s)
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def main():
    print("📥 Loading data...")
    df = load_df("data/final/data_english.parquet")

    df = apply_clean_text_to_df(df)

    # ✅ Freeze row order for stable mapping (important)
    df = df.reset_index(drop=True)

    print("🧠 Loading embedding model...")
    device = _get_device()
    model = getModel(device=device)

    print("📝 Preparing documents...")
    documents = get_title_and_abstract(df, title_col="title", abs_col="abstract")

    # ✅ Build stable IDs that correspond 1:1 with 'documents' order
    # get_title_and_abstract writes df["full_text"], so we can hash it
    df["emb_id"] = df["full_text"].apply(_make_emb_id_from_full_text)

    print("🔢 Generating embeddings...")
    ckpt_dir = f"{OUTPUT_DIR}/checkpoints"
    ckpt_name = "embeddings_ckpt"

    embeddings = getEmbeddings_checkpointed(
        model,
        query_prompt,
        documents,
        device=device,
        batch_size=8,
        checkpoint_dir=ckpt_dir,
        checkpoint_name=ckpt_name,
        save_every_n_batches=1,  # ✅ checkpoint EVERY batch
        emb_ids=df["emb_id"].tolist(),  # ✅ save ordered mapping next to ckpt
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("💾 Saving outputs...")
    # ✅ Save only metadata (+ emb_id); embeddings stay on disk in the mmap
    df_out = df.drop(columns=["full_text"], errors="ignore")
    save(df_out, os.path.join(OUTPUT_DIR, "embedded_data.parquet"))

    print("✅ Done.")

if __name__ == "__main__":
    main()
