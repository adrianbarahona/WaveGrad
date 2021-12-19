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
2. Create a `mamba` virtual environment `audio` with everything needed to run the testing.
