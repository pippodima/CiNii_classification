import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# -----------------------------------------------------
# Embedding helpers
# -----------------------------------------------------
def getModel(modelName="Qwen/Qwen3-Embedding-0.6B"):
    return SentenceTransformer(modelName)


def get_title_and_abstract(df: pd.DataFrame, title_col="title_en", abs_col="abstract_en"):
    df["full_text"] = df[title_col].fillna("") + ". " + df[abs_col].fillna("")
    return df["full_text"].tolist()


def getEmbeddings(model, query, documents, batch_size=32):
    _ = model.encode(query, prompt_name="query", device="cpu")

    embeddings = []
    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding batches"):
        batch = documents[i:i + batch_size]
        batch_embs = model.encode(batch)
        embeddings.extend(batch_embs)

    return np.array(embeddings)
    