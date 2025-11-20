import xml.etree.ElementTree as ET
import glob
import argparse
import os
import random
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from typing import Dict, List, Set, Optional
from configs import NAMESPACES, CSV_FIELDS
from utils import save
import pandas as pd


# ============================================================================
# RDF PARSING
# ============================================================================

def extract_title(root: ET.Element) -> Optional[str]:
    """
    Extract title from RDF, preferring non-English versions.

    Args:
        root: Root element of the XML tree

    Returns:
        Title text or None if not found
    """
    titles = root.findall('.//dc:title', NAMESPACES)
    if not titles:
        return None

    # Prefer non-English titles (typically Japanese)
    for title_elem in titles:
        lang = title_elem.attrib.get('{http://www.w3.org/XML/1998/namespace}lang')
        if lang != 'en':
            return title_elem.text

    # Fallback to first title
    return titles[0].text


def extract_abstract(root: ET.Element) -> Optional[str]:
    """
    Extract abstract from RDF notation element.

    Args:
        root: Root element of the XML tree

    Returns:
        Abstract text or None if not found
    """
    abstract_elem = root.find('.//cinii:description/cinii:notation', NAMESPACES)
    return abstract_elem.text.strip() if abstract_elem is not None else None


def parse_rdf(file_path: str) -> Dict[str, Optional[str]]:
    """
    Parse a single RDF file and extract metadata.

    Args:
        file_path: Path to the RDF file

    Returns:
        Dictionary containing file path, title, and abstract
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    return {
        'file': file_path,
        'title': extract_title(root),
        'abstract': extract_abstract(root)
    }


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def find_rdf_files(root_folder: str, pattern: str = "*.rdf") -> List[str]:
    """
    Recursively find all RDF files in a folder and its subfolders.

    Args:
        root_folder: Root directory to search
        pattern: File pattern to match (default: "*.rdf")

    Returns:
        List of file paths
    """
    search_pattern = os.path.join(root_folder, '**', pattern)
    return glob.glob(search_pattern, recursive=True)


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def load_checkpoint(checkpoint_file: str) -> Set[int]:
    """
    Load completed batch numbers from checkpoint file.

    Args:
        checkpoint_file: Path to checkpoint file

    Returns:
        Set of completed batch numbers
    """
    if not os.path.exists(checkpoint_file):
        return set()

    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        return set(int(line.strip()) for line in f if line.strip().isdigit())


def append_to_checkpoint(checkpoint_file: str, batch_number: int) -> None:
    """
    Record that a batch was completed.

    Args:
        checkpoint_file: Path to checkpoint file
        batch_number: Batch number to record
    """
    with open(checkpoint_file, 'a', encoding='utf-8') as f:
        f.write(f"{batch_number}\n")


# ============================================================================
# BATCH MERGING
# ============================================================================

def merge_batch_parquet_files(batch_files: List[str], final_output: str) -> None:
    """
    Merge multiple batch Parquet files into one Parquet file using PyArrow Dataset
    for maximum performance and minimal memory usage.
    """
    # Create a dataset from the list of parquet files
    dataset = ds.dataset(batch_files, format="parquet")

    # Read the dataset into a single Arrow table (streamed, parallel)
    table = dataset.to_table()  # uses optimized C++ engine

    # Write merged table to a Parquet file
    pq.write_table(table, final_output)


def cleanup_temporary_files(batch_files: List[str], checkpoint_file: str) -> None:
    """
    Remove temporary batch files and checkpoint file.

    Args:
        batch_files: List of batch file paths to delete
        checkpoint_file: Path to checkpoint file to delete
    """
    # Remove batch files
    for batch_file in batch_files:
        try:
            os.remove(batch_file)
        except Exception as e:
            print(f"Warning: could not delete {batch_file}: {e}")

    # Remove checkpoint file
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print("Removed checkpoint file.")
        except Exception as e:
            print(f"Warning: could not delete checkpoint file: {e}")


def merge_batches(output_folder: str, final_output: str, checkpoint_file: str) -> None:
    """
    Merge all batch Parquet files into one Parquet file, then clean up temporary files.

    Args:
        output_folder: Directory containing batch parquet files
        final_output: Path to final merged output file
        checkpoint_file: Path to checkpoint file
    """
    batch_files = sorted(glob.glob(os.path.join(output_folder, 'rdf_results_batch_*.parquet')))

    if not batch_files:
        print("No batch parquet files found to merge.")
        return

    # Merge all batches
    merge_batch_parquet_files(batch_files, final_output)
    print(f"Merged {len(batch_files)} batch parquet files into {final_output}")

    # Cleanup temporary files
    cleanup_temporary_files(batch_files, checkpoint_file)
    print("Temporary batch files deleted. Cleanup complete.")


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def select_sample_files(rdf_files: List[str], max_docs: Optional[int],
                        random_state: int = 42) -> List[str]:
    """
    Select a random sample of files if max_docs is specified.

    Args:
        rdf_files: List of all RDF file paths
        max_docs: Maximum number of documents to process (None for all)
        random_state: Random seed for reproducibility

    Returns:
        List of selected file paths
    """
    if max_docs is None or max_docs >= len(rdf_files):
        return rdf_files

    random.seed(random_state)
    return random.sample(rdf_files, max_docs)


def process_batch(batch_files: List[str], output_file: str,
                  checkpoint_file: str, batch_number: int) -> None:
    """
    Process a single batch of RDF files.

    Args:
        batch_files: List of file paths to process in this batch
        output_file: Path to save batch results
        checkpoint_file: Path to checkpoint file
        batch_number: Current batch number
    """
    results = [parse_rdf(f) for f in batch_files]
    df = pd.DataFrame(results)
    save(df, output_file)
    append_to_checkpoint(checkpoint_file, batch_number)
    print(f"Saved batch {batch_number} ({len(results)} files)")


def process_rdf_folder_batches(root_folder: str, batch_size: int = 1000,
                               max_docs: Optional[int] = None,
                               output_folder: str = "processed") -> None:
    """
    Process RDF files recursively in batches with crash recovery.

    Args:
        root_folder: Root directory containing RDF files
        batch_size: Number of files to process per batch
        max_docs: Maximum number of documents to process (None for all)
        output_folder: Directory to save output files
    """
    # Setup
    os.makedirs(output_folder, exist_ok=True)
    checkpoint_file = os.path.join(output_folder, 'checkpoint.txt')
    completed_batches = load_checkpoint(checkpoint_file)

    # Find and optionally sample files
    rdf_files = find_rdf_files(root_folder)
    rdf_files = select_sample_files(rdf_files, max_docs)

    total_files = len(rdf_files)
    print(f"Found {total_files} files. Processing in batches of {batch_size}...")

    # Process batches
    batch_number = 1
    for i in range(0, total_files, batch_size):
        # Skip already completed batches
        if batch_number in completed_batches:
            print(f"Skipping batch {batch_number} (already processed)")
            batch_number += 1
            continue

        # Process current batch
        batch_files = rdf_files[i:i + batch_size]
        output_file = os.path.join(output_folder, f"rdf_results_batch_{batch_number}.parquet")
        process_batch(batch_files, output_file, checkpoint_file, batch_number)
        batch_number += 1

    # Merge all batches into final output
    final_output = os.path.join(output_folder, "rdf_results_final.parquet")
    merge_batches(output_folder, final_output, checkpoint_file)
    print(f"All results saved to {final_output}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    """Execute the RDF parsing pipeline."""
    # -------------------------------
    # Command-line arguments
    # -------------------------------
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
        default="../raw",
        help="Root folder containing RDF files (default: raw)",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        type=str,
        default="../processed",
        help="Output folder for processed files (default: processed)",
    )
    args = parser.parse_args()

    # -------------------------------
    # Configuration
    # -------------------------------
    ROOT_FOLDER = args.root_folder
    BATCH_SIZE = args.batch_size
    MAX_DOCUMENTS = args.max_document
    OUTPUT_FOLDER = args.output_folder

    # -------------------------------
    # Run the processing
    # -------------------------------
    process_rdf_folder_batches(
        root_folder=ROOT_FOLDER,
        batch_size=BATCH_SIZE,
        max_docs=MAX_DOCUMENTS,
        output_folder=OUTPUT_FOLDER,
    )


if __name__ == "__main__":
    main()
