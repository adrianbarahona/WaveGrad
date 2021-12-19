# Convenience scripts for Mambaforge virtual environment and NASH testing

## Creating the virtual environment
To create the virtual environment, type

```
./create-virtual-environment.sh
```

This will:

1. Install `mambaforge` if it's not already there. If it installed `mambaforge`, you will
need to restart your shell. Both `bash` and `zsh` are supported. I use `zsh` with the
`powerlevel10k` theme.
2. Create a `mamba` virtual environment `WaveGrad` with everything needed to run the testing.
This includes PyTorch and all the packages defined in `requirements.txt`. It also
includes JupyterLab and the R language kernel, which I am pretty much useless without.
3. List the installed packages and tell you where the logfile is.

## Starting a JupyterLab server
To start a JupyterLab server:

```
mamba activate WaveGrad
cd ..
./linux-mambaforge/start-jupyter-lab.sh
```

## Running an inference test
To run an inference test, edit `test-command-line.sh` to point to the required files. I've listed all the files I uploaded to Drive but not the
others. Then:

```
mamba activate WaveGrad
cd ..
./linux-mambaforge/test-command-line.sh
```
