#! /bin/bash

set -e

pushd /tmp

  echo "Downloading speech dataset"
  rm -fr LJSpeech*
  wget -q \
    https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

  popd
