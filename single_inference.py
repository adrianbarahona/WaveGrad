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
    model = WaveGrad(config).to(device)
    print(f'Number of parameters: {model.nparams}')

    model.load_state_dict(torch.load(args.input_checkpoint)['model'], strict=False)
    model.set_new_noise_schedule()

    # open and resample to fit model
    w, sr = torchaudio.load(args.input_sound)
    resampled_waveform = F.resample(w, sr, config.data_config.sample_rate, lowpass_filter_width=6)

    # get only the first channel
    resampled_waveform = resampled_waveform[0,:]

    # get mel spec
    mel_fn = MelSpectrogramFixed(
        sample_rate=config.data_config.sample_rate,
        n_fft=config.data_config.n_fft,
        win_length=config.data_config.win_length,
        hop_length=config.data_config.hop_length,
        f_min=config.data_config.f_min,
        f_max=config.data_config.f_max,
        n_mels=config.data_config.n_mels,
        window_fn=torch.hann_window
    ).to(device)

    mel = mel_fn(resampled_waveform[None].to(device))
    output = model.forward(
        mel, store_intermediate_states=False
    )
    torchaudio.save('output.wav', output.cpu(), config.data_config.sample_rate)