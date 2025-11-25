import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd


def plot_static(umap_embeddings, labels, path="umap_clusters.png"):
    plt.figure(figsize=(10, 8))
    plt.scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        c=labels,
        s=8,
        cmap="Spectral"
    )
    plt.title("UMAP Clusters (Static)")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"📊 Saved static plot → {path}")


def plot_interactive(df, umap_embeddings, path="umap_clusters.html"):
    df_plot = pd.DataFrame({
        "x": umap_embeddings[:, 0],
        "y": umap_embeddings[:, 1],
        "title": df["title_en"].fillna(""),
        "cluster": df["cluster_id"].astype(str),
        "name": df["name"].fillna("")
    })

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="cluster",
        hover_data=["title", "name"],
        title="UMAP Interactive Clusters"
    )

    fig.write_html(path)
    print(f"🌐 Saved interactive plot → {path}")
