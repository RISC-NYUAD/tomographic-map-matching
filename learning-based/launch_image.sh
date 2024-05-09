#!/bin/sh

PKG=$1

# Ensure the necessary folders exist at least
mkdir -p results/$PKG
mkdir -p weights/$PKG

singularity run --nv --no-home \
    --bind $PKG/workspace:/workspace \
    --bind tmp_home:$HOME \
    --bind weights/$PKG:/weights \
    --bind results/$PKG:/results \
    --bind ../data-config:/data-config \
    --bind $DATA_DIR:/data \
    $PKG/$PKG.sif
