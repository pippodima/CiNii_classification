import os
import json
from utils import load_df, save, _get_device
from clustering import auto_optimize_clustering
from labeling import label_clusters
from plots import plot_static, plot_interactive
from datetime import datetime
from embeddings_pipeline import getEmbeddings, getModel, get_title_and_abstract
import numpy as np
import pandas as pd

OUTPUT_DIR = "../../output"
query_prompt = "Given a scientific paper title and abstract, produce an embedding that captures the research topic."



def main():
    print("📥 Loading data...")
    df = load_df("../final/data_sci_Cleaned.parquet")

    print("🧠 Loading embedding model...")
    model = getModel()

    print("📝 Preparing documents...")
    documents = get_title_and_abstract(df)

    print("🔢 Generating embeddings...")
    embeddings = getEmbeddings(model, query_prompt, documents, batch_size=64)
    df["embeddings"] = embeddings.tolist()

    print("🌐 Running auto-optimized clustering...")
    umap_embeddings, labels, best_cfg = auto_optimize_clustering(embeddings)

    print("🏷️ Labeling clusters...")
    df_labeled, cluster_info = label_clusters(df, labels)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("💾 Saving outputs...")
    save(df_labeled, os.path.join(OUTPUT_DIR, "clustered_data.parquet"))
    cluster_info.to_parquet(os.path.join(OUTPUT_DIR, "cluster_metadata.parquet"), index=False)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "best_config": best_cfg,
        "n_clusters": len(cluster_info),
        "n_outliers": int(list(labels).count(-1)),
    }

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("📊 Generating plots...")
    plot_static(umap_embeddings, labels, os.path.join(OUTPUT_DIR, "umap_clusters.png"))
    plot_interactive(df_labeled, umap_embeddings, os.path.join(OUTPUT_DIR, "umap_clusters.html"))

    print("✅ Done.")





if __name__ == "__main__":
    main()


