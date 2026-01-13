#!/bin/bash
# ==========================================================
# sample_resourcedumps.sh
# Randomly copy N unzipped resource dump folders from source to destination
# ==========================================================

# --- Default values ---
N=20
SRC="/mnt/nas-lae/cinii-kg/resourcedump_rdf_product_20240404"
DEST="/home/fdmartino/work/cinii/CiNii_classification/data/raw"
LIST_FILE="selected_folders.txt"

# --- Parse arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --n) N="$2"; shift ;;
        --src) SRC="$2"; shift ;;
        --dest) DEST="$2"; shift ;;
        --list) LIST_FILE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# --- Check paths ---
if [[ ! -d "$SRC" ]]; then
    echo "❌ Source directory not found: $SRC"
    exit 1
fi

mkdir -p "$DEST"
cd "$DEST" || exit 1

# --- Generate list of random unzipped folders ---
echo "📋 Selecting $N random unzipped folders from:"
echo "   $SRC"
ls -d "$SRC"/resourcedump_* | grep -v '\.zip$' | shuf -n "$N" > "$LIST_FILE"

echo "✅ List saved to: $DEST/$LIST_FILE"
echo "------------------------------"
cat "$LIST_FILE"
echo "------------------------------"

# --- Confirm before copying ---
read -p "⚠️  Proceed to copy these folders to $DEST ? (y/N): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "❌ Aborted."
    exit 0
fi

# --- Copy using rsync (safe, read-only on source) ---
echo "🚀 Copying files..."
while read -r folder; do
    rsync -av --progress "$folder" "$DEST/"
done < "$LIST_FILE"

echo "✅ Done! Copied $N folders to:"
echo "   $DEST"
