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

show_info ">> Downloading data..."

FILE=tomographic-map-matching-data.zip

if [ -f "$FILE" ]
then
    echo "- File $FILE exists. Skipping download..."
else
    URL=https://ultraviolet.library.nyu.edu/records/m859g-t4p13/files/tomographic-map-matching-data.zip
    wget --show-progress --quiet $URL
fi

show_info ">> Extracting data..."
unzip -u $FILE

show_info ">> Finished retrieving data!"
