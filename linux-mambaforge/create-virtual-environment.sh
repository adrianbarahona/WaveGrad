#! /bin/bash

set -e

echo "Installing Mambaforge if needed"
./mambaforge.sh

echo "Enabling 'conda' and 'mamba' commands"
source $HOME/mambaforge/etc/profile.d/conda.sh
source $HOME/mambaforge/etc/profile.d/mamba.sh

echo "Creating fresh 'audio' conda environment"
mamba env create --force --quiet --file=audio.yml

echo "Checking GPU"
python check_gpu.py
