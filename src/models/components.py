from operator import mul
from functools import reduce
from typing import Tuple, List

import torch
from torch import nn, tensor


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, conv_encoder_layers: List[Tuple[int]], use_batch_norm: bool = False) -> None:
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
    def __init__(self, in_features: int,
                 encoder_output_dim: Tuple[int],
                 conv_decoder_layers: List[Tuple[int]],
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


class LinearEmbedder(nn.Module):
    def __init__(self, in_features, linear_layers: List[int], use_batch_norm: bool = False) -> None:
        super(LinearEmbedder, self).__init__()
        layers = []
        for out_features in linear_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.LeakyReLU(0.05))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(1))
            in_features = out_features
        layers.append(nn.Linear(in_features, 2))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: tensor) -> tensor:
        return self.layers(x)


class BlockCurveEmbedder(nn.Module):
    def __init__(self, curve_in_features, block_in_features, linear_layers: List[int],
                 use_batch_norm: bool = False) -> None:
        super(BlockCurveEmbedder, self).__init__()
        self.curve_in_features = curve_in_features
        self.block_in_features = block_in_features
        self.curve_embedder = LinearEmbedder(curve_in_features, linear_layers, use_batch_norm=use_batch_norm)
        self.block_embedder = LinearEmbedder(block_in_features, linear_layers, use_batch_norm=use_batch_norm)
        self.output_layer = nn.Linear(4, 2)

    def forward(self, x: tensor) -> tensor:
        last_dim = len(x.shape)

        curve_feat_batch = torch.unsqueeze(x[..., :self.curve_in_features], dim=1)
        block_feat_batch = torch.unsqueeze(x[..., self.curve_in_features:], dim=1)
        curve_output = self.curve_embedder(curve_feat_batch)
        block_output = self.block_embedder(block_feat_batch)

        combined_output = torch.cat((curve_output, block_output), dim=last_dim)

        return self.output_layer(combined_output)


class FeatureTransformer(nn.Module):
    def __init__(self, linear_layers: List[int], use_batch_norm: bool = False) -> None:
        super(FeatureTransformer, self).__init__()
        layers = []
        in_features = 1
        for out_features in linear_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.LeakyReLU(0.05))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(1))
            in_features = out_features
        layers.append(nn.Linear(in_features, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: tensor) -> tensor:
        return self.layers(x)


class MultiFeatureTransformer(nn.Module):
    def __init__(self, in_features, linear_layers: List[int], use_batch_norm: bool = False) -> None:
        super(MultiFeatureTransformer, self).__init__()
        self.model = nn.ModuleList([FeatureTransformer(linear_layers, use_batch_norm=use_batch_norm)
                                    for _ in range(in_features)])

    def forward(self, x: tensor) -> tensor:
        y_hat = torch.zeros_like(x)
        for i in range(x.shape[-1]):
            feat_batch = torch.unsqueeze(x[..., i], dim=1)
            y_hat[..., i] = torch.squeeze(self.model[i](feat_batch), dim=2)
        return y_hat
