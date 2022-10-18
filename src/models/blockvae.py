from typing import Tuple

import numpy as np
from pytorch_lightning import LightningModule
import torch
from torch import tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset

from src.models.components import ConvEncoder, ConvDecoder, VAE
from src.models.losses import gaussian_nll, kl_divergence


class BlockVAE(LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

        model_config = config["model"]
        encoder = ConvEncoder(model_config["in_channels"],
                              model_config["conv_encoder_layers"],
                              use_batch_norm=model_config.get("use_batch_norm", True))
        decoder = ConvDecoder(model_config["latent_dim"],
                              model_config["encoder_output_dim"],
                              model_config["conv_decoder_layers"],
                              use_batch_norm=model_config.get("use_batch_norm", True))
        self.model = self.init_model(encoder, decoder, model_config)
        self.train_dataset, self.val_dataset = self.prepare_data()

    @staticmethod
    def init_model(encoder: ConvEncoder, decoder: ConvDecoder, model_config: dict) -> VAE:
        return VAE(
            encoder,
            decoder,
            model_config["encoder_output_dim"],
            model_config["latent_dim"]
        )

    def prepare_data(self) -> Tuple[TensorDataset, TensorDataset]:
        data_config = self.config["data"]
        train_templates = np.load(data_config["train_data_path"])
        val_templates = np.load(data_config["val_data_path"])

        x_train = torch.from_numpy(train_templates).float()
        x_val = torch.from_numpy(val_templates).float()
        train_dataset = TensorDataset(x_train, x_train)
        val_dataset = TensorDataset(x_val, x_val)
        return train_dataset, val_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.config["data"]["train_batch_size"])

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.config["data"]["val_batch_size"])

    def forward(self, x: tensor) -> tensor:
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(
            self.parameters(), lr=(self.config["learning_rate"] or 1e-4))

    @staticmethod
    def loss(batch: tensor, outputs: Tuple[tensor, tensor, tensor]) -> Tuple[float, float, float]:
        x, _ = batch
        mean, log_var, x_hat = outputs

        reconstruction_loss = gaussian_nll(x, x_hat)
        kld = kl_divergence(mean, log_var)
        elbo = (reconstruction_loss + kld).mean()

        return (
            elbo,
            reconstruction_loss.mean(),
            kld.mean()
        )

    def training_step(self, batch: tensor, batch_idx: int) -> float:
        output = self.model(batch[0])
        elbo, reconstruction_loss, kld = self.loss(batch, output)

        self.log_dict({
            'train_loss': elbo,
            'train_recon_loss': reconstruction_loss,
            'train_kld': kld
        }, on_step=False, on_epoch=True, prog_bar=True)

        return elbo

    def validation_step(self, batch: tensor, batch_idx: int) -> float:
        output = self.model(batch[0])
        elbo, reconstruction_loss, kld = self.loss(batch, output)

        self.log_dict({
            'val_loss': elbo,
            'val_recon_loss': reconstruction_loss,
            'val_kld': kld
        }, prog_bar=True)

        return elbo
