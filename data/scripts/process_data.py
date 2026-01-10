import os
import re
from html import unescape
import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm
from utils import load_df, save, _get_device, M2M_LANG_MAP
from time import sleep
import torch
tqdm.pandas()


# ============================================================================
# TEXT CLEANING
# ============================================================================

def clean_abstract(text: str) -> str:
    """Clean and normalize abstract text by removing HTML, LaTeX, and formatting."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r'\\"([a-zA-Z])', lambda m: m.group(1), text)
    text = re.sub(r"\\'", "", text)
    text = re.sub(r"\$(.*?)\$", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\(begin|end)\{.*?\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("``", '"').replace("''", '"')
    text = re.sub(r"\s+", " ", text)


    return text.strip()


# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return None


# ============================================================================
# DOCUMENT CLASSIFICATION
# ============================================================================

DIAGNOSTIC_PATTERNS = re.compile(
    r'\blhd\s*(bolometer|flxloop|fpellet|interferometer|cxrs|ece|diagnostic|probe)\b|'
    r'^lhd\b|'
    r'\blhd\s*#?\d+',
    flags=re.IGNORECASE
)


def classify_document_type(title: str) -> str:
    if pd.isna(title):
        return "unknown"
    title_clean = str(title).strip().lower()
    return "diagnostic_report" if DIAGNOSTIC_PATTERNS.search(title_clean) else "scientific_paper"


# ============================================================================
# DATAFRAME OPERATIONS
# ============================================================================

def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    print("Dropping rows where title or abstract is empty or null")

    tqdm.pandas(desc="Checking for empty title/abstract")

    df_clean = df.dropna(subset=["title", "abstract"])

    df_clean = df_clean[
        df_clean["title"].progress_apply(lambda x: x.strip() != "") &
        df_clean["abstract"].progress_apply(lambda x: x.strip() != "")
    ]

    return df_clean


def drop_uninformative_abstracts(df: pd.DataFrame) -> pd.DataFrame:
    print("Dropping rows with non-informative abstracts")
    tqdm.pandas(desc="Filtering abstracts")

    META_KEYWORDS = [
        "article", "論文", "type:", "identifier:", "application/",
        "text/", "pdf", "html", "doi", "report"
    ]

    citation_pattern = re.compile(
    r".*(pp?\.?\s*[0-9一二三四五六七八九十]+[-‐~〜][0-9一二三四五六七八九十]+).*;?\s*(\d{4})$"
    )


    def is_informative(text: str) -> bool:
        t = text.strip().lower()

        # 1. Too short to be meaningful
        if len(t) < 100:
            return False

        # 2. Matches known metadata patterns
        for kw in META_KEYWORDS:
            if t.startswith(kw) or t == kw:
                return False

        # 3. Pure metadata (e.g. 論文(Article))
        if re.match(r".*\b(article|論文)\b.*$", t):
            if len(t) < 40:  # still metadata-level
                return False
                
        # 4. CITATION CHECK: The important new part
        if citation_pattern.match(t):
            return False

        # 5. No sentence markers -> very likely metadata
        if not any(p in t for p in [".", "。", "、", ","]):
            return False

        return True

    return df[df["abstract"].progress_apply(is_informative)]


def apply_clean_text_to_df(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning text")
    df["clean_abstract"] = df["abstract"].progress_apply(clean_abstract)
    return df


def add_abstract_language(df: pd.DataFrame) -> pd.DataFrame:
    print("Detecting languages")
    df['title_lang'] = df['title'].astype(str).progress_apply(detect_language)
    df['abstract_lang'] = df['clean_abstract'].astype(str).progress_apply(detect_language)
    return df


def remove_empty_lang(df: pd.DataFrame, title_col: str = 'title_lang',
                      abstract_col: str = 'abstract_lang') -> pd.DataFrame:
    return df.dropna(subset=[title_col, abstract_col])


def classify_documents(df: pd.DataFrame) -> pd.DataFrame:
    print("Classifying document types")
    df["type"] = df["title"].progress_apply(classify_document_type)
    df_diag = df[df["type"] == "diagnostic_report"].copy()
    df_sci = df[df["type"] == "scientific_paper"].copy()

    print(f"Diagnostic reports: {len(df_diag)} | Scientific papers: {len(df_sci)}")
    return df_diag, df_sci


# ============================================================================
# I/O OPERATIONS
# ============================================================================



def drop_boilerplate_only_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows whose titles and abstracts contain only boilerplate words like
    'type:text', 'application/pdf', 'Departmental Bulletin Paper', etc.
    """
    boilerplate_patterns = re.compile(
        r"^(type|application|departmental|bulletin|text|論文|記事|表紙|裏表紙|奥付|目次)\b",
        flags=re.IGNORECASE
    )

    mask = df["clean_abstract"].fillna("").str.match(boilerplate_patterns)
    dropped = df[mask]
    if len(dropped) > 0:
        print(f"🧹 Dropping {len(dropped)} boilerplate-only rows (no semantic content)")
    return df[~mask].copy()



# Load model once globally (fast)
DEVICE = _get_device()
print(f"Using device: {DEVICE}")

tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(DEVICE)


def normalize_lang(code):
    if code is None:
        return None
    return M2M_LANG_MAP.get(code.lower(), None)


def translate_batch_m2m(text_list, src_lang, batch_size=4):
    """Translate list of texts from src_lang → English using GPU M2M100."""

    def safe_generate(**kwargs):
        try:
            return model.generate(**kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            sleep(1)
            return model.generate(**kwargs)


    if src_lang is None:
        return text_list  # identity: skip

    tokenizer.src_lang = src_lang
    results = []

    for i in tqdm(range(0, len(text_list), batch_size), desc=f"{src_lang} → en"):
        batch = text_list[i:i + batch_size]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(DEVICE)

        generated = safe_generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id("en"),
            num_beams=1,
            max_length=256
        )

        outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
        results.extend(outputs)

    return results


def translate_to_eng(df):
    print("Translating to English using local M2M100 model")

    df["title_en"] = df["title"].astype(str)
    df["abstract_en"] = df["clean_abstract"].astype(str)

    # ---------- TITLES ----------
    mask_t = df["title_lang"] != "en"
    langs_t = df.loc[mask_t, "title_lang"].tolist()

    for lang in set(langs_t):
        norm = normalize_lang(lang)

        # skip unsupported languages
        if norm is None or norm == "en":
            continue

        idx = df.index[(df["title_lang"] == lang)]
        texts = df.loc[idx, "title"].tolist()

        translated = translate_batch_m2m(texts, norm)
        df.loc[idx, "title_en"] = translated

    # ---------- ABSTRACTS ----------
    mask_a = df["abstract_lang"] != "en"
    langs_a = df.loc[mask_a, "abstract_lang"].tolist()

    for lang in set(langs_a):
        norm = normalize_lang(lang)

        # skip unsupported languages
        if norm is None or norm == "en":
            continue

        idx = df.index[(df["abstract_lang"] == lang)]
        texts = df.loc[idx, "clean_abstract"].tolist()

        translated = translate_batch_m2m(texts, norm)
        df.loc[idx, "abstract_en"] = translated
        torch.cuda.empty_cache()

    return df


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    df = load_df("../processed/rdf_results_final.parquet").sample(n=80000, random_state=42)
    print(df.shape)
    df = drop_empty_rows(df)
    print(df.shape)
    df = drop_uninformative_abstracts(df)
    print(df.shape)
    df = apply_clean_text_to_df(df)
    print(df.shape)
    df = add_abstract_language(df)
    print(df.shape)
    df = translate_to_eng(df)
    print(df.shape)
    df_diag, df_sci = classify_documents(df)
    save(df_sci, "../final/data_sci_Cleaned.parquet")
    save(df_diag, "../final/data_diag_Cleaned.parquet")



if __name__ == "__main__":
    main()