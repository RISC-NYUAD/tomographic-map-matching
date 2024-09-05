#!/bin/sh

set -e

# Some formatting utils
show_info ()
{
    # Green and bold
    FMT='\033[1;32m'
    OFF='\033[0m'
    echo "$FMT$1$OFF"
}

show_info ">> Downloading weights..."

# Function to automate downloads
download_weights ()
{
    mkdir -p $FOLDER

    for FILE in $FILES; do
        if [ -f "$FOLDER/$FILE" ]
        then
            echo "- File $FOLDER/$FILE exists. Skipping..."
        else
            show_info ">> Downloading $FOLDER/$FILE"
            wget --show-progress --quiet $URL_BASE/$FILE -P $FOLDER
        fi
    done
}

# Deep Global Registration
show_info ">> DeepGlobalRegistration"
FOLDER=DeepGlobalRegistration
URL_BASE=http://node2.chrischoy.org/data/projects/DGR
FILES="ResUNetBN2C-feat32-3dmatch-v0.05.pth ResUNetBN2C-feat32-kitti-v0.3.pth"
download_weights

# Rotation-Invariant Transformer
show_info ">> RoITr"
FOLDER=RoITr
URL_BASE=https://github.com/haoyu94/RoITr/releases/download/v1.0.0
FILES="model_3dmatch.pth"
download_weights

# GeoTransformer
show_info ">> GeoTransformer"
FOLDER=GeoTransformer
URL_BASE=https://github.com/qinzheng93/GeoTransformer/releases/download/1.0.0
FILES="geotransformer-3dmatch.pth.tar geotransformer-kitti.pth.tar"
download_weights

# BUFFER
show_info ">> BUFFER"
# Structure is somewhat different than the rest. Indoors (ThreeDMatch) first
PKG=BUFFER
URL_BASE_TDM=https://raw.githubusercontent.com/h-utkuunlu/BUFFER/main/ThreeDMatch/snapshot/06132318
SUBFOLDERS="Desc Inlier Keypt Ref"
FILES="best.pth"

for SUBFOLDER in $SUBFOLDERS; do
    FOLDER=$PKG/ThreeDMatch/$SUBFOLDER
    URL_BASE=$URL_BASE_TDM/$SUBFOLDER
    download_weights
done

URL_BASE_KITTI=https://raw.githubusercontent.com/h-utkuunlu/BUFFER/main/KITTI/snapshot/06050001
for SUBFOLDER in $SUBFOLDERS; do
    FOLDER=$PKG/KITTI/$SUBFOLDER
    URL_BASE=$URL_BASE_KITTI/$SUBFOLDER
    download_weights
done

show_info ">> Finished downloading weights!"
