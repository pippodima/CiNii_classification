import ollama
import pandas as pd
from utils import load_df, save
import re
from tqdm import tqdm
import json

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_colwidth", None)

def clean_response(text):
    # Remove <think> ... </think> blocks (multi-line safe)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


# -----------------------------
# Function to generate subtopic
# -----------------------------
def generate_subtopic(keywords):
    prompt = (
        "Given this list of keywords: "
        f"{', '.join(keywords)}\n\n"
        "Create a short, meaningful subtopic title (2–5 words) "
        "that best summarizes the topic.\n"
        "Do NOT output <think> tags or chain-of-thought."
    )

    response = ollama.chat(
        model="qwen3:1.7b",  # Change model name to whichever you use in Ollama
        messages=[{"role": "user", "content": prompt}]
    )

    content = response["message"]["content"]
    cleaned = clean_response(content)
    return cleaned
 
def generate_main_topic(all_keywords):
    prompt = (
        "Given this list of keywords:\n"
        f"{', '.join(all_keywords)}\n\n"
        "Create a short, meaningful MAIN TOPIC title (2–4 words) that "
        "summarizes the entire topic. Avoid generic labels. "
    )

    response = ollama.chat(
        model="qwen3:1.7b",  # same as your subtopic script
        messages=[{"role": "user", "content": prompt}]
    )

    content = response["message"]["content"]
    return clean_response(content)




def main():
    df = load_df("../../output/cluster_metadata.parquet")
    tqdm.pandas(desc="Generating subtopics")
    df["subtopic"] = df["keywords"].progress_apply(generate_subtopic)

    save(df, "../../output/merge/metadata_clusetered_names.parquet")
    
    merged_path = "../../output/merge/cluster_output_merged_topics.json"
    with open(merged_path, "r") as f:
        merged_clusters = json.load(f)
    
    name_to_keywords = dict(zip(df["name"], df["keywords"]))

    rows = []
    for big_cluster_id, subcluster_names in merged_clusters.items():

        # Collect keywords from all subclusters
        collected_keywords = []

        for name in subcluster_names:
            if name in name_to_keywords:
                collected_keywords.extend(name_to_keywords[name])
            else:
                print(f"WARNING: subcluster '{name}' not found in metadata")

        # Remove duplicates while preserving order
        collected_keywords = list(dict.fromkeys(collected_keywords))

        rows.append({
            "merged_cluster_id": int(big_cluster_id),
            "subclusters": subcluster_names,
            "keywords": collected_keywords
        })

    merged_df = pd.DataFrame(rows)

    tqdm.pandas(desc="Generating main topics")
    merged_df["main_topic"] = merged_df["keywords"].progress_apply(generate_main_topic)

    output_path = "../../output/merge/merged_clusters_with_main_topics.parquet"
    save(merged_df, output_path)
    print(merged_df.head(10))



if __name__=="__main__":
    main()