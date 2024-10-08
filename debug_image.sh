#!/bin/sh

# Sanitize trailing slash if it is added
# FIXME: Not pretty
PKG=$(realpath --relative-to $1/.. $1)

# Ensure the necessary folders exist at least
mkdir -p results/$PKG
mkdir -p weights/$PKG
mkdir -p .tmp_home

apptainer exec --nvccli --no-home \
    --bind $PKG/workspace:/workspace \
    --bind .tmp_home:$HOME \
    --bind weights/$PKG:/weights \
    --bind results/$PKG:/results \
    --bind ${DATA_DIR:-data}:/data \
    $PKG/$PKG.sif /bin/bash
