#! /bin/bash

set -e

echo "Enabling 'conda' and 'mamba' commands"
source $HOME/mambaforge/etc/profile.d/conda.sh
source $HOME/mambaforge/etc/profile.d/mamba.sh

echo "Activating audio environment"
mamba activate audio
python linux-mambaforge/check_gpu.py

export DIRECTORY=/data/znmeb/taiko

echo "Removing any existing OUTPUT_* files"
rm $DIRECTORY/OUTPUT_*

for infile in `ls -1 $DIRECTORY/*.wav`
do
  echo "Processing $infile"
  /usr/bin/time python ./single_inference.py \
    --input_checkpoint "/data/znmeb/NASH/Adrian checkpoints/checkpoint_32004.pt" \
    --input_config "/data/znmeb/NASH/Adrian checkpoints/config.json" \
    --input_sound "$infile"
  basefile=`basename $infile`
  echo "Saving output to $DIRECTORY/OUTPUT_$basefile"
  mv output.wav $DIRECTORY/OUTPUT_$basefile
done
