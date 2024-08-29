#!/bin/sh

# Sanitize trailing slash if it is added
# FIXME: Not pretty
PKG=$(realpath --relative-to $1/.. $1)

# Ensure the necessary folders exist at least
mkdir -p results/$PKG
mkdir -p weights/$PKG
mkdir -p .tmp_home

# Pass everything except the PKG (folder)
shift

apptainer run --nvccli --no-home \
    --bind $PKG/workspace:/workspace \
    --bind .tmp_home:$HOME \
    --bind weights/$PKG:/weights \
    --bind results/$PKG:/results \
    --bind data-config:/data-config \
    --bind ${DATA_DIR:-data}:/data \
    $PKG/$PKG.sif $@
