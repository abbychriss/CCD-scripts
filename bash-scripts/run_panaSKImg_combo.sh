#!/usr/bin/env bash
set -e

# -------- USER INPUT --------
if [ -z "$1" ]; then
    echo "Usage: $0 <min_index>"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Usage: $0 <max_index>"
    exit 1
fi

if [ -z "$3" ]; then
    echo "Usage: $0 <avg or itp>"
    exit 1
fi

MIN_INDEX=$1
MAX_INDEX=$2
PATH_EXT=$3
SEARCH_EXT=$4

BASE_DIR="/Users/abbychriss/Desktop/Privitera_335"
INPUT_DIR="${BASE_DIR}/data/Am241-Spectra-data/1x1-bin/"
CONFIG_FILE="${BASE_DIR}/json/panaSKImg_config_LBC_ACM_4DQM_PA08_103.json"
OUTPUT_BASE="${BASE_DIR}/panaSKImg_output"

SEARCH_DIR="${INPUT_DIR}/${SEARCH_EXT}"
FILES=()

# Collect files whose index is 0–MAX_INDEX
echo "Searching in directory ${SEARCH_DIR}"
for f in "${SEARCH_DIR}"/*${SEARCH_EXT}*/${PATH_EXT}_img_*.fz; do
    fname=$(basename "$f")
    idx="${fname##*_}"
    idx="${idx%.fz}"

    if (( idx >= MIN_INDEX && idx <= MAX_INDEX )); then
        FILES+=("$f")
    fi
done

TOTAL=${#FILES[@]}
COUNT=0

echo "Found $TOTAL files to process ($MIN_INDEX - $MAX_INDEX)."
echo "------------------------------------"

for INPUT_FILE in "${FILES[@]}"; do
    COUNT=$((COUNT + 1))
    
    IMAGE_NAME=$(basename "$INPUT_FILE")
    echo "${IMAGE_NAME}"
    OUTPUT_DIR="${OUTPUT_BASE}/${PATH_EXT}-img/${IMAGE_NAME%.fz}"

    printf "[%4d/%4d] Processing %s\n" "$COUNT" "$TOTAL" "$IMAGE_NAME"

    if [ -d "$OUTPUT_DIR" ]; then
        echo "          ↳ Skipping (already exists)"
        continue
    fi

    mkdir -p "$OUTPUT_DIR"

    panaSKImg -j "$CONFIG_FILE" \
              -o "$OUTPUT_DIR" \
              "$INPUT_FILE" #\
              #--save-plots
done

echo "------------------------------------"
echo "Done."
