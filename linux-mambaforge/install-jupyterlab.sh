#! /bin/bash

set -e

echo "Activating $VENV_NAME virtual environment"
source $HOME/mambaforge/etc/profile.d/conda.sh
source $HOME/mambaforge/etc/profile.d/mamba.sh
mamba activate $VENV_NAME

echo "Installing 'jupyterlab' and 'r-irkernel'"
mamba install --yes --quiet jupyterlab r-irkernel \
  >> /tmp/create-$VENV_NAME.log 2>&1

echo "Activating R kernel"
R -e "IRkernel::installspec()" \
  >> /tmp/create-$VENV_NAME.log 2>&1
