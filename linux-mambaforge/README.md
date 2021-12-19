# Installing on Windows 11 WSL Ubuntu 20.04 LTS with an NVIDIA GPU - Mambaforge version

This is tested on an Alienware Aurora R11 with an NVIDIA RTX 3090 GPU only.
The operating system is Windows 11 with WSL Ubuntu 20.04 LTS.

It probably cannot be made to work on Windows 10; Microsoft apparently adjusted
their Windows 10 priorities when they released Windows 11 and some features that were
initially supposed to be in Windows 10 21H2 didn't make the cut.

If you're adventurous and don't mind doing free troubleshooting for Microsoft and
NVIDIA and want to pursue this in Windows 10, start here:
<https://docs.nvidia.com/cuda/wsl-user-guide/index.html#installing-insider-preview-builds>.

I went down that path a few weeks ago on my laptop, which has a processor too
old for Windows 11, and got hopelessly entangled in Microsoft's feedback forum.
I gave up and reformatted the laptop back to Windows 10 stable and there it
will stay.

## Installing the NVIDIA tools
The directions are here: <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>.
I'm assuming you followed them and they worked.

Everything else is done by the script
`WaveGrad_NASH/linux-mambaforge/create-virtual-environment.sh`. So to install,

```
cd neural-waveshaping-synthesis/linux-mambaforge
./create-virtual-environment.sh
```

## Mambaforge
The first thing the script does is install the Mambaforge virtual environment /
package manager if needed. See <https://github.com/conda-forge/miniforge> for
documentation.

If your system already has a `mambaforge` install at `$HOME/mambaforge` the
script will not touch it. Otherwise, it will install Mambaforge into
`$HOME/mambaforge` and initialize the `bash` and `zsh` shells for interactive
`conda` and `mamba` use. In this case, you will need to restart your shell.

Why Mambaforge? `mamba` is a mostly drop-in replacement for `conda`, with a
completely rewritten dependency solver. It's in C++ and it's much faster
than the one that comes with `conda`.

## The `audio` virtual environment
Next, the script will create the `audio` conda enviroment. This is done
using a spec file `audio.yml` provided by <https://github.com/rodrigodzf>.
Then it activates `audio` and runs `check_gpu.py` to verify that PyTorch
can access the GPU.
