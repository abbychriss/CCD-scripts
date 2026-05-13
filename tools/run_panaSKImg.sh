#!/usr/bin/env bash

set -e

# -------- INPUTS --------
if [ -z "$1" ]; then
  echo "Usage: $0 <image_name> <output_name> [save-plots]"
  exit 1
fi

if [ -z "$2" ]; then
  echo "Usage: $0 <image_name> <output_name> [save-plots]"
  exit 1
fi

IMAGE_NAME="$1"
OUTPUT_NAME="$2"
SAVE_PLOTS="${3:-false}"   # default = false

# -------- PATH HANDLING --------
REL_PATH="$IMAGE_NAME"

# If path contains "/data/", strip everything up to it
if [[ "$IMAGE_NAME" == *"/data/"* ]]; then
    REL_PATH="${IMAGE_NAME#*/data}"
fi

# -------- PATHS --------
BASE_DIR=/Users/abbychriss/Privitera_335
OUTPUT_DIR="${BASE_DIR}/panaSKImg_output/${REL_PATH}/${OUTPUT_NAME}"
CONFIG_FILE="${BASE_DIR}/json/panaSKImg_config_LBC_ACM_4DQM_PA08_103.json"
INPUT_FILE="$IMAGE_NAME"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# -------- OPTIONAL FLAG --------
SAVE_FLAG=""
if [[ "$SAVE_PLOTS" == "--save-plots" || "$SAVE_PLOTS" == "true" ]]; then
    SAVE_FLAG="--save-plots"
fi

# -------- RUN --------
panaSKImg -j "$CONFIG_FILE" \
          -o "$OUTPUT_DIR" \
          "$INPUT_FILE" \
          $SAVE_FLAG
