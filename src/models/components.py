from operator import mul
from functools import reduce
from typing import Tuple, List

import torch
from torch import nn, tensor


class ConvEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 conv_encoder_layers: Tuple[int],
                 use_batch_norm: bool = False
                 ) -> None:
        super().__init__()

        layers = []

        for (out_channels, kernel, stride) in conv_encoder_layers:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel, stride=stride))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.LeakyReLU(0.05))
            in_channels = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x: tensor) -> tensor:
        return self.layers(x)


class ConvDecoder(nn.Module):
    def __init__(self,
                 in_features: int,
                 encoder_output_dim: Tuple[int],
                 conv_decoder_layers: List[Tuple],
                 use_batch_norm: bool = False
                 ) -> None:
        super().__init__()

        layers = [
            nn.Linear(in_features, reduce(mul, encoder_output_dim)),
            nn.Unflatten(1, encoder_output_dim)
        ]

        in_channels = encoder_output_dim[0]
        for i, (out_channels, kernel, stride, output_padding) in enumerate(conv_decoder_layers):
            layers.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride=stride,
                    output_padding=output_padding))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))

            # Don't add activation for last layer
            if i != len(conv_decoder_layers) - 1:
                layers.append(nn.LeakyReLU(0.05))

            in_channels = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x: tensor) -> tensor:
        return self.layers(x)


class AE(nn.Module):
    def __init__(self,
                 encoder: ConvEncoder,
                 decoder: ConvDecoder,
                 encoder_output_dim: Tuple[int],
                 encoding_dim: int
                 ) -> None:
        super(AE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        flattened_dim = reduce(mul, encoder_output_dim)
        self.encoding = nn.Linear(flattened_dim, encoding_dim)

    def encode(self, x: tensor) -> tensor:
        output = torch.flatten(self.encoder(x), start_dim=1)
        return self.encoding(output)

    def decode(self, encoding: tensor) -> tensor:
        return self.decoder(encoding)

    def forward(self, x: tensor) -> tensor:
        return self.decode(self.encode(x))


class VAE(nn.Module):
    def __init__(self,
                 encoder: ConvEncoder,
                 decoder: ConvDecoder,
                 encoder_output_dim: Tuple[int],
                 latent_dim: int
                 ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        flattened_dim = reduce(mul, encoder_output_dim)
        self.latent_mean = nn.Linear(flattened_dim, latent_dim)
        self.latent_log_var = nn.Linear(flattened_dim, latent_dim)

    def encode(self, x: tensor) -> Tuple[tensor, tensor]:
        output = torch.flatten(self.encoder(x), start_dim=1)
        return self.latent_mean(output), self.latent_log_var(output)

    @staticmethod
    def sample(mean: tensor, log_var: tensor) -> tensor:
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return eps * std + mean

    def decode(self, z: tensor) -> tensor:
        return self.decoder(z)

    def forward(self, x: tensor) -> Tuple[tensor, tensor, tensor]:
        mean, log_var = self.encode(x)
        z = self.sample(mean, log_var)
        return mean, log_var, self.decode(z)
