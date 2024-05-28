#!/bin/sh

PKG=Consensus

# Ensure the necessary folders exist at least
mkdir -p results/$PKG

# Variations
NOISES="0-00 0-02 0-05"
# GRID_SIZES="0-02 0-05"
GRID_SIZES="0-02"


for PARAM_FULL in "${PKG}/workspace/config/"*.json; do
    for NOISE in $NOISES; do
        for GRID_SIZE in $GRID_SIZES; do
            PARAM=$(realpath --relative-to $PKG/workspace $PARAM_FULL)
            ARGS="--parameter_config $PARAM --data_config /data-config/interiornet/gsize$GRID_SIZE-noise$NOISE.json"

            echo $ARGS

            singularity run --nv --no-home \
                --bind $PKG/workspace:/workspace \
                --bind results/$PKG:/results \
                --bind data-config:/data-config \
                --bind $DATA_DIR:/data \
                $PKG/$PKG.sif $ARGS
        done
    done
done
