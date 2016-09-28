#!/usr/bin/env sh
# This scripts downloads the ptb data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

cd data
wget http://russellsstewart.com/s/tensorbox/googlenet.pb
wget https://www.dropbox.com/s/qyy7xw5zrk9o009/classification_model.ckpt?dl=1
wget https://www.dropbox.com/s/w7dvylkgpzri8ca/overfeat_checkpint.ckpt?dl=1

echo "Done."
