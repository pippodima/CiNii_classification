import xml.etree.ElementTree as ET
import glob
import argparse
import os
import random
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Set, Optional
import pandas as pd

from configs import NAMESPACES
from utils import save


# ============================================================================
# RDF PARSING
# ============================================================================

def extract_title(root: ET.Element) -> Optional[str]:
    """Extract title from RDF, preferring non-English versions."""
    titles = root.findall(".//dc:title", NAMESPACES)
    if not titles:
        return None

    # Prefer non-English titles (typically Japanese)
    for title_elem in titles:
        lang = title_elem.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
        if lang != "en":
            return title_elem.text

    # Fallback to first title
    return titles[0].text


def extract_abstract(root: ET.Element) -> Optional[str]:
    """Extract abstract from RDF notation element."""
    abstract_elem = root.find(".//cinii:description/cinii:notation", NAMESPACES)
    return abstract_elem.text.strip() if abstract_elem is not None and abstract_elem.text else None


def parse_rdf(file_path: str) -> Dict[str, Optional[str]]:
    """
    Parse a single RDF file and extract metadata.
    Never crashes the whole pipeline: errors go into the 'error' field.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        return {
            "file": file_path,
            "title": extract_title(root),
            "abstract": extract_abstract(root),
            "error": None,
        }
    except Exception as e:
        return {
            "file": file_path,
            "title": None,
            "abstract": None,
            "error": f"{type(e).__name__}: {e}",
        }


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def find_rdf_files(root_folder: str, pattern: str = "*.rdf") -> List[str]:
    """Recursively find all RDF files in a folder and its subfolders."""
    search_pattern = os.path.join(root_folder, "**", pattern)
    return glob.glob(search_pattern, recursive=True)


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def load_checkpoint(checkpoint_file: str) -> Set[int]:
    """Load completed batch numbers from checkpoint file."""
    if not os.path.exists(checkpoint_file):
        return set()

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        return set(int(line.strip()) for line in f if line.strip().isdigit())


def append_to_checkpoint(checkpoint_file: str, batch_number: int) -> None:
    """Record that a batch was completed."""
    with open(checkpoint_file, "a", encoding="utf-8") as f:
        f.write(f"{batch_number}\n")


# ============================================================================
# PARQUET REPAIR + MERGE (SCHEMA-SAFE)
# ============================================================================

TARGET_SCHEMA = pa.schema([
    ("file", pa.string()),
    ("title", pa.string()),
    ("abstract", pa.string()),
    ("error", pa.string()),
])


def cast_table_to_schema(t: pa.Table, schema: pa.Schema) -> pa.Table:
    """
    Ensure all required columns exist and have the exact target types.
    Missing columns are created as all-null arrays with the correct type.
    """
    for name in schema.names:
        if name not in t.column_names:
            t = t.append_column(name, pa.array([None] * t.num_rows, type=schema.field(name).type))
    # Reorder to match schema order, then cast
    t = t.select(schema.names)
    return t.cast(schema, safe=False)


def repair_batch_parquet_files(output_folder: str) -> None:
    """
    One-time (or every run) repair:
    casts every batch parquet file to TARGET_SCHEMA so merging never fails.
    """
    batch_files = sorted(glob.glob(os.path.join(output_folder, "rdf_results_batch_*.parquet")))
    if not batch_files:
        return

    repaired = 0
    for f in batch_files:
        try:
            t = pq.read_table(f)
            t2 = cast_table_to_schema(t, TARGET_SCHEMA)
            pq.write_table(t2, f)  # overwrite
            repaired += 1
        except Exception as e:
            print(f"Warning: could not repair {f}: {e}")
    print(f"Repaired {repaired} batch parquet files to a consistent schema.")


def merge_batch_parquet_files(batch_files: List[str], final_output: str) -> None:
    """
    Merge parquet batches safely by casting each batch to TARGET_SCHEMA, then concatenating.
    (This avoids the pyarrow.dataset schema unification edge cases.)
    """
    tables = []
    for f in batch_files:
        t = pq.read_table(f)
        t = cast_table_to_schema(t, TARGET_SCHEMA)
        tables.append(t)

    merged = pa.concat_tables(tables, promote=True)
    pq.write_table(merged, final_output)


def cleanup_temporary_files(batch_files: List[str], checkpoint_file: str) -> None:
    """Remove temporary batch files and checkpoint file."""
    for batch_file in batch_files:
        try:
            os.remove(batch_file)
        except Exception as e:
            print(f"Warning: could not delete {batch_file}: {e}")

    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print("Removed checkpoint file.")
        except Exception as e:
            print(f"Warning: could not delete checkpoint file: {e}")


def merge_batches(output_folder: str, final_output: str, checkpoint_file: str) -> None:
    """Merge all batch Parquet files into one Parquet file, then clean up temporary files."""
    batch_files = sorted(glob.glob(os.path.join(output_folder, "rdf_results_batch_*.parquet")))
    if not batch_files:
        print("No batch parquet files found to merge.")
        return

    # Ensure any previously-written batches are schema-consistent
    repair_batch_parquet_files(output_folder)

    # Merge
    merge_batch_parquet_files(batch_files, final_output)
    print(f"Merged {len(batch_files)} batch parquet files into {final_output}")

    # Cleanup
    cleanup_temporary_files(batch_files, checkpoint_file)
    print("Temporary batch files deleted. Cleanup complete.")


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def select_sample_files(rdf_files: List[str], max_docs: Optional[int], random_state: int = 42) -> List[str]:
    """
    Select a random sample of files if max_docs is specified.

    NOTE: We sort rdf_files before sampling to keep sampling stable across runs.
    """
    if max_docs is None or max_docs >= len(rdf_files):
        return rdf_files

    random.seed(random_state)
    return random.sample(rdf_files, max_docs)


def process_batch(batch_files: List[str], output_file: str, checkpoint_file: str, batch_number: int) -> None:
    """Process a single batch of RDF files."""
    results = [parse_rdf(f) for f in batch_files]
    df = pd.DataFrame(results)

    # Force stable schema across batches (prevents 'null' Arrow columns)
    for col in ["file", "title", "abstract", "error"]:
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df), dtype="string")
        else:
            df[col] = df[col].astype("string")

    save(df, output_file)
    append_to_checkpoint(checkpoint_file, batch_number)
    print(f"Saved batch {batch_number} ({len(results)} files)")


def process_rdf_folder_batches(
    root_folder: str,
    batch_size: int = 1000,
    max_docs: Optional[int] = None,
    output_folder: str = "processed",
) -> None:
    """Process RDF files recursively in batches with crash recovery."""
    os.makedirs(output_folder, exist_ok=True)
    checkpoint_file = os.path.join(output_folder, "checkpoint.txt")
    completed_batches = load_checkpoint(checkpoint_file)

    # Deterministic order so batch numbers always map to the same files
    rdf_files = sorted(find_rdf_files(root_folder))
    rdf_files = select_sample_files(rdf_files, max_docs)

    total_files = len(rdf_files)
    print(f"Found {total_files} files. Processing in batches of {batch_size}...")

    batch_number = 1
    for i in range(0, total_files, batch_size):
        if batch_number in completed_batches:
            print(f"Skipping batch {batch_number} (already processed)")
            batch_number += 1
            continue

        batch_files = rdf_files[i:i + batch_size]
        output_file = os.path.join(output_folder, f"rdf_results_batch_{batch_number}.parquet")
        process_batch(batch_files, output_file, checkpoint_file, batch_number)
        batch_number += 1

    final_output = os.path.join(output_folder, "rdf_results_final.parquet")
    merge_batches(output_folder, final_output, checkpoint_file)
    print(f"All results saved to {final_output}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RDF parsing pipeline")
    parser.add_argument(
        "-max-document",
        "--max-document",
        type=int,
        default=None,
        help="Maximum number of documents to process (default: all)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for processing (default: 10000)",
    )
    parser.add_argument(
        "-r",
        "--root-folder",
        type=str,
        default="data/raw",
        help="Root folder containing RDF files (default: data/raw)",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        type=str,
        default="data/processed",
        help="Output folder for processed files (default: data/processed)",
    )
    args = parser.parse_args()

    process_rdf_folder_batches(
        root_folder=args.root_folder,
        batch_size=args.batch_size,
        max_docs=args.max_document,
        output_folder=args.output_folder,
    )


if __name__ == "__main__":
    main()
