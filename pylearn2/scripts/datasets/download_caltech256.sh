#!/bin/bash
# This script downloads CALTECH256 dataset to $PYLEARN2_DATA_PATH/caltech256
#set -e
[ -z "$PYLEARN2_DATA_PATH" ] && echo "PYLEARN2_DATA_PATH is not set" && exit 1
CALTECH256_DIR=$PYLEARN2_DATA_PATH/caltech256
CALTECH256_URL="http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"

which wget > /dev/null
WGET=$?
which curl > /dev/null
CURL=$?

if [ "$WGET" -eq 0 ]; then
    DL_CMD="wget --no-verbose -O -"
elif [ "$CURL" -eq 0 ]; then
    DL_CMD="curl --silent -o -"
else
    echo "You need wget or curl installed to download"
    exit 1
fi

[ -d $CALTECH256_DIR ] && echo "$CALTECH256_DIR already exists." && exit 1
mkdir -p $CALTECH256_DIR

echo "Downloading and unzipping CALTECH256 dataset into $CALTECH256_DIR..."
pushd $CALTECH256_DIR > /dev/null
$DL_CMD $CALTECH256_URL | tar xvf -
popd > /dev/null
