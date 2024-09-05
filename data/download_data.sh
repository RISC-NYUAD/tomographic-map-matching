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

# TODO

show_info ">> Finished downloading data!"
