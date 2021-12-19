import json
import numpy as np

import torch

from model.base import BaseModule
from model.layers import Conv1dWithInitialization
from model.upsampling import UpsamplingBlock as UBlock
from model.downsampling import DownsamplingBlock as DBlock
from model.linear_modulation import FeatureWiseLinearModulation as FiLM
from torch import Tensor
from typing import List, Tuple
from utils import ConfigWrapper

class WaveGradNN(BaseModule):
    """
    WaveGrad is a fully-convolutional mel-spectrogram conditional
    vocoder model for waveform generation introduced in
    "WaveGrad: Estimating Gradients for Waveform Generation" paper (link: https://arxiv.org/pdf/2009.00713.pdf).
    The concept is built on the prior work on score matching and diffusion probabilistic models.
    Current implementation follows described architecture in the paper.
    """
    def __init__(self):
        super(WaveGradNN, self).__init__()
        # Building upsampling branch (mels -> signal)

        config = ConfigWrapper(**json.loads('{"model_config": {"factors": [5, 5, 3, 2, 2], "upsampling_preconv_out_channels": 768, "upsampling_out_channels": [512, 512, 256, 128, 128], "upsampling_dilations": [[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]], "downsampling_preconv_out_channels": 32, "downsampling_out_channels": [128, 128, 256, 512], "downsampling_dilations": [[1, 2, 4], [1, 2, 4], [1, 2, 4], [1, 2, 4]]}, "data_config": {"sample_rate": 16000, "n_fft": 1024, "win_length": 1024, "hop_length": 300, "f_min": 80.0, "f_max": 8000, "n_mels": 80}, "training_config": {"logdir": "logs/sfx", "continue_training": false, "train_filelist_path": "filelists/train.txt", "test_filelist_path": "filelists/test.txt", "batch_size": 96, "segment_length": 7200, "lr": 0.001, "grad_clip_threshold": 1, "scheduler_step_size": 1, "scheduler_gamma": 0.9, "n_epoch": 100000000, "n_samples_to_test": 4, "test_interval": 10, "use_fp16": true, "training_noise_schedule": {"n_iter": 1000, "betas_range": [1e-06, 0.01]}, "test_noise_schedule": {"n_iter": 50, "betas_range": [1e-06, 0.01]}}, "dist_config": {"MASTER_ADDR": "localhost", "MASTER_PORT": "600010"}}'))
        self.ublock_preconv = Conv1dWithInitialization(
            in_channels=config.data_config.n_mels,
            out_channels=config.model_config.upsampling_preconv_out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        upsampling_in_sizes = [config.model_config.upsampling_preconv_out_channels] \
            + config.model_config.upsampling_out_channels[:-1]
        self.ublocks = torch.nn.ModuleList([
            UBlock(
                in_channels=in_size,
                out_channels=out_size,
                factor=factor,
                dilations=dilations
            ) for in_size, out_size, factor, dilations in zip(
                upsampling_in_sizes,
                config.model_config.upsampling_out_channels,
                config.model_config.factors,
                config.model_config.upsampling_dilations
            )
        ])
        self.ublock_postconv = Conv1dWithInitialization(
            in_channels=config.model_config.upsampling_out_channels[-1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Building downsampling branch (starting from signal)
        self.dblock_preconv = Conv1dWithInitialization(
            in_channels=1,
            out_channels=config.model_config.downsampling_preconv_out_channels,
            kernel_size=5,
            stride=1,
            padding=2
        )
        downsampling_in_sizes = [config.model_config.downsampling_preconv_out_channels] \
            + config.model_config.downsampling_out_channels[:-1]
        self.dblocks = torch.nn.ModuleList([
            DBlock(
                in_channels=in_size,
                out_channels=out_size,
                factor=factor,
                dilations=dilations
            ) for in_size, out_size, factor, dilations in zip(
                downsampling_in_sizes,
                config.model_config.downsampling_out_channels,
                config.model_config.factors[1:][::-1],
                config.model_config.downsampling_dilations
            )
        ])

        # Building FiLM connections (in order of downscaling stream)
        film_in_sizes = [32] + config.model_config.downsampling_out_channels
        film_out_sizes = config.model_config.upsampling_out_channels[::-1]
        film_factors = [1] + config.model_config.factors[1:][::-1]
        self.films = torch.nn.ModuleList([
            FiLM(
                in_channels=in_size,
                out_channels=out_size,
                input_dscaled_by=np.product(film_factors[:i+1])  # for proper positional encodings initialization
            ) for i, (in_size, out_size) in enumerate(
                zip(film_in_sizes, film_out_sizes)
            )
        ])

    # @torch.jit.script_method
    def forward(self, mels, yn, noise_level):
        """
        Computes forward pass of neural network.
        :param mels (torch.Tensor): mel-spectrogram acoustic features of shape [B, n_mels, T//hop_length]
        :param yn (torch.Tensor): noised signal `y_n` of shape [B, T]
        :param noise_level (float): level of noise added by diffusion
        :return (torch.Tensor): epsilon noise
        """
        # Prepare inputs
        assert len(mels.shape) == 3  # B, n_mels, T
        yn = yn.unsqueeze(1)
        assert len(yn.shape) == 3  # B, 1, T

        # Downsampling stream + Linear Modulation statistics calculation
        scales = []
        shifts = []
        #  = torch.jit.annotate(List[Tuple[Tensor, Tensor]], [])
        dblock_outputs = self.dblock_preconv(yn)
        scale, shift = self.films[0](x=dblock_outputs, noise_level=noise_level)
        # statistics.append([scale, shift])
        scales.append(scale)
        shifts.append(shift)
        for dblock, film in zip(self.dblocks, self.films[1:]):
            dblock_outputs = dblock(dblock_outputs)
            scale, shift = film.forward(x=dblock_outputs, noise_level=noise_level)
            # statistics.append([scale, shift])
            scales.append(scale)
            shifts.append(shift)
        # statistics = statistics[::-1]
        scales = scales[::-1]
        shifts = shifts[::-1]

        # Upsampling stream
        ublock_outputs = self.ublock_preconv(mels)
        for i, ublock in enumerate(self.ublocks):
            scale = scales[i]
            shift = shifts[i]
            ublock_outputs = ublock(x=ublock_outputs, scale=scale, shift=shift)
        outputs = self.ublock_postconv(ublock_outputs)
        return outputs.squeeze(1)
