#!/usr/bin/env bash
set -e

# -------- INPUTS --------
SEARCH_EXT="${1:-*}"     # directory pattern
PATH_EXT="${2:-avg}"     # avg or itp

SAVE_PLOTS=false

for arg in "$@"; do
    if [[ "$arg" == "--save-plots" ]]; then
        SAVE_PLOTS=true
    fi
done

# Validate image type
if [[ "$PATH_EXT" != "avg" && "$PATH_EXT" != "itp" ]]; then
    echo "Error: second argument must be 'avg' or 'itp'"
    exit 1
fi

BASE_DIR="/Users/abbychriss/Desktop/Privitera_335"
INPUT_DIR="${BASE_DIR}/data/test_chamber/Am241-Spectra-data/1x1-bin"
OUTPUT_BASE="${BASE_DIR}/panaSKImg_output"
CONFIG_FILE="${BASE_DIR}/json/panaSKImg_config_LBC_ACM_4DQM_PA08_103.json"

echo "Searching in directory: ${INPUT_DIR}/${SEARCH_EXT}*"
echo "Image type: ${PATH_EXT}"
echo "Save plots: ${SAVE_PLOTS}"

shopt -s nullglob globstar

found=0

echo "DEBUG: Expanded glob:"
printf '%s\n' "${INPUT_DIR}/${SEARCH_EXT}"*/**/"${PATH_EXT}"_img_*.fz

for f in "${INPUT_DIR}/${SEARCH_EXT}"*/**/"${PATH_EXT}"_img_*.fz; do
    found=$((found + 1))

    # -------- BUILD RELATIVE PATH --------
    REL_PATH="$f"

    if [[ "$f" == *"/data/"* ]]; then
        REL_PATH="${f#*/data/}"
    fi

    # remove leading slash if present
    REL_PATH="${REL_PATH#/}"

    # optional: strip file extension
    REL_PATH="${REL_PATH%.fits}"
    REL_PATH="${REL_PATH%.fz}"

    # -------- FINAL OUTPUT DIR --------
    OUTPUT_DIR="${OUTPUT_BASE}/${REL_PATH}"

    echo "------------------------------------"
    echo "Processing: $f"
    echo "Output dir: $OUTPUT_DIR"

    mkdir -p "$OUTPUT_DIR"

    # -------- OPTIONAL FLAG --------
    SAVE_FLAG=""
    if [[ "$SAVE_PLOTS" == "true" ]]; then
        SAVE_FLAG="--save-plots"
    fi

    # -------- RUN --------
    panaSKImg -j "$CONFIG_FILE" \
              -o "$OUTPUT_DIR" \
              "$f" \
              $SAVE_FLAG

done

if (( found == 0 )); then
    echo "No files found."
    exit 1
fi

echo "------------------------------------"
echo "Done. Processed $found files."
