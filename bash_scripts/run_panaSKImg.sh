#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if image name was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <image_name>"
  exit 1
fi

#Check if output name was provided
if [ -z "$2" ]; then
  echo "Usage: $0 <output_name>"
  exit 1
fi

IMAGE_NAME="$1"

OUTPUT_DIR="/Users/abbychriss/Desktop/Privitera_335/panaSKImg_output/${IMAGE_NAME}/$2"
CONFIG_FILE="/Users/abbychriss/Desktop/Privitera_335/json/panaSKImg_config_LBC_ACM_4DQM_PA08_103.json"
INPUT_FILE="${IMAGE_NAME}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run panaSKImg
panaSKImg -j "$CONFIG_FILE" \
          -o "$OUTPUT_DIR" \
          "$INPUT_FILE" \
          --save-plots
