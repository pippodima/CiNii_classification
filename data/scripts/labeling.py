import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_top_tfidf_terms(texts, top_n=10):
    """
    Extract top-N TF-IDF terms using ONLY the default English stopwords.
    No custom stopword removal.
    """

    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return ["unknown"]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        token_pattern=r"(?u)\b\w+\b",
        lowercase=True
    )

    x = vectorizer.fit_transform(texts)
    tfidf_sum = np.asarray(x.sum(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())

    top_idx = np.argsort(tfidf_sum)[::-1][:top_n]
    return terms[top_idx].tolist()


def label_clusters(df, labels, text_col="full_text"):
    """
    Assign cluster labels using TF-IDF keywords only.
    """

    df["cluster_id"] = labels

    cluster_info = []

    for cid in sorted(set(labels)):
        if cid == -1:
            continue

        cluster_texts = df.loc[df["cluster_id"] == cid, text_col].dropna().tolist()
        if not cluster_texts:
            continue

        top_terms = extract_top_tfidf_terms(cluster_texts, 10)
        name = " / ".join(top_terms[:3])

        cluster_info.append({
            "cluster_id": cid,
            "keywords": top_terms,
            "name": name,
            "size": len(cluster_texts),
        })

    info_df = pd.DataFrame(cluster_info)
    df = df.merge(info_df[["cluster_id", "name"]], on="cluster_id", how="left")

    return df, info_df
