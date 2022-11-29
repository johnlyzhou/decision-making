from typing import Tuple

import numpy as np
from pytorch_lightning import LightningModule
from sklearn.model_selection import train_test_split
import torch
from torch import tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset

from src.models.components import LinearEmbedder
from src.models.losses import SupConLoss


class SigmoidNet(LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.loss = SupConLoss()

        model_config = config["model"]
        self.num_feats = model_config["in_features"]
        self.model = LinearEmbedder(self.num_feats, model_config["linear_layers"])
        print(self.model)

        self.train_dataset, self.val_dataset = self.prepare_data()

    def prepare_data(self) -> Tuple[TensorDataset, TensorDataset]:
        data_config = self.config["data"]
        X = np.load(data_config["feature_path"])
        y = np.load(data_config["label_path"])
        train_prop = data_config["train_proportion"]

        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_prop, random_state=99,
                                                          shuffle=True, stratify=y)
        X_train = torch.unsqueeze(torch.from_numpy(X_train).float(), 1)
        X_val = torch.unsqueeze(torch.from_numpy(X_val).float(), 1)
        y_train = torch.unsqueeze(torch.from_numpy(y_train).float(), 1)
        y_val = torch.unsqueeze(torch.from_numpy(y_val).float(), 1)
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
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

    def training_step(self, batch: tensor, batch_idx: int) -> float:
        x, y = batch
        y = torch.squeeze(y).type(torch.LongTensor)
        x_hat = self.model(x)
        loss = self.loss(x_hat, labels=y)

        self.log_dict({
            'train_loss': loss
        }, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tensor, batch_idx: int) -> float:
        x, y = batch
        y = torch.squeeze(y).type(torch.LongTensor)
        x_hat = self.model(x)
        loss = self.loss(x_hat, labels=y)

        self.log_dict({
            'val_loss': loss
        }, on_step=False, on_epoch=True, prog_bar=True)

        return loss
