#! /bin/bash

set -e
export VENV_NAME=WaveGrad
rm -f /tmp/create-$VENV_NAME.log

echo "Installing Mambaforge if needed"
./mambaforge.sh \
  >> /tmp/create-$VENV_NAME.log 2>&1

echo "Enabling 'conda' and 'mamba' commands"
source $HOME/mambaforge/etc/profile.d/conda.sh
source $HOME/mambaforge/etc/profile.d/mamba.sh

echo "Creating fresh '$VENV_NAME' conda environment"
mamba env create --force --quiet --file=venv-spec.yml \
  >> /tmp/create-$VENV_NAME.log 2>&1

echo "Activating $VENV_NAME"
mamba activate $VENV_NAME

./install-jupyterlab.sh

echo "Installing pip requirements"
cd ..

# do we need an over-ride?
if [ -e "linux-mambaforge/pip-requirements.txt" ]
then
  echo "Over-riding requirements.txt with linux-mambaforge/pip-requirements.txt"
  pip install -r linux-mambaforge/pip-requirements.txt \
    >> /tmp/create-$VENV_NAME.log 2>&1
else
  pip install -r requirements.txt \
    >> /tmp/create-$VENV_NAME.log 2>&1
fi

echo "Checking GPU"
python linux-mambaforge/check_gpu.py

export NODE_NAME=`uname -n`
conda list \
  > linux-mambaforge/$VENV_NAME.$NODE_NAME.list
echo "$VENV_NAME.$NODE_NAME.list has package list"

echo "Check /tmp/create-$VENV_NAME.log for errors / warnings / version mismatches"
