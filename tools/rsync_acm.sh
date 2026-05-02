#!/bin/bash

# Check required argument
if [[ -z "$1" ]]; then
    echo "Usage: $0 YYYY-MM-DD-directory [include_pattern]"
    exit 1
fi

src="$1"
pattern="${2:-*.fz}"   # default include pattern

# Split YYYY-MM-DD-directory
date_part="${src:0:10}"   # first 10 characters
dir_part="${src:11}"      # everything after YYYY-MM-DD-

# Split date components
IFS="-" read -r yyyy mm dd <<< "$date_part"

dest_date="${mm}-${dd}-${yyyy}"
dest="$HOME/Privitera_335/data/test_chamber/$dest_date/$dir_part"

mkdir -p "$dest"

rsync -av --include="$pattern" --exclude='*' \
    damicm@acmdev:/home/damicm/Soft/cdaq/run/$src/ \
    "$dest/"
