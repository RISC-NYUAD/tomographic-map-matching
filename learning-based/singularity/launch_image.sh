#!/bin/sh

# Pass the package folder name
PKG=$1
singularity run --nv --no-home \
    --bind $PKG/scripts:/workspace \
    --bind tmp_home:$HOME \
    --bind ../weights/$PKG:/weights \
    --bind ../results/$PKG:/results \
    --bind ../../data-config:/data-config \
    --bind $DATA_DIR:/data \
    $PKG/$PKG.sif
