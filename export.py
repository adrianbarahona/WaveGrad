import argparse
import json
import torch
import librosa as li
import torchaudio
import torchaudio.functional as F

from model import WaveGrad
from data import AudioDataset, MelSpectrogramFixed
from argparse import ArgumentParser
from pathlib import Path
from utils import ConfigWrapper

if __name__ == '__main__':
    parser = ArgumentParser(description='Wavegrad inference with a single file')
    parser.add_argument('--input_config', type=Path, default=".")
    parser.add_argument('--input_checkpoint', type=Path, default=".")
    parser.add_argument('--input_sound', type=Path, default=".")
    parser.add_argument('--device', type=int, default=0, choices=([-1] + list(range(torch.cuda.device_count()))), help="GPU to use; -1 is CPU (default 0)")

    args = parser.parse_args()

    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")

    with open(args.input_config) as f:
        config = ConfigWrapper(**json.load(f))
    model = WaveGrad().to(device)

    model.load_state_dict(torch.load(args.input_checkpoint)['model'], strict=False)
    # model.set_new_noise_schedule()
    model.eval()

    mel = torch.rand((1, 80, 2))
    print('mel shape', mel.shape)
    output = model.forward(
        mel, store_intermediate_states=False
    )
    torchaudio.save('output.wav', output.cpu(), config.data_config.sample_rate)

    # # export model
    # traced_script_module = torch.jit.trace(model, mel)
    # traced_script_module.save("traced_model.pt")
    # export model
    traced_script_module = torch.jit.script(model)
    traced_script_module.save("traced_model.pt")