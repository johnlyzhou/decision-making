from typing import Tuple, List, Union
import multiprocessing

import numpy as np
from pytorch_lightning import LightningModule
import torch
from sklearn.model_selection import train_test_split
from torch import tensor, nn
from torchmetrics import Accuracy
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset


class LinearClassifier(nn.Module):
    def __init__(self, in_features: int, linear_layers: List[int], use_batch_norm: bool = False,
                 num_workers: Union[int, str] = 0) -> None:
        if num_workers == 'max':
            self.num_workers = multiprocessing.cpu_count()
        elif type(int):
            if num_workers > multiprocessing.cpu_count() or num_workers < 0:
                raise ValueError
            self.num_workers = num_workers
        else:
            raise ValueError

        super(LinearClassifier, self).__init__()
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


class AgentNet(LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(num_classes=2, average=None)
        self.val_acc = Accuracy(num_classes=2, average=None)

        model_config = config["model"]
        print(model_config)

        self.model = LinearClassifier(model_config["in_features"],
                                      model_config["linear_layers"],
                                      use_batch_norm=model_config["use_batch_norm"])
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
        y_hat = self.model(x)
        y = torch.squeeze(y).type(torch.LongTensor)
        y_hat = torch.squeeze(y_hat)
        loss = self.loss(y_hat, y)
        acc = self.train_acc(y_hat, y)
        mf = acc[0].item()
        mb = acc[1].item()

        self.log_dict({
            'train_loss': loss,
            'train_acc_mf': mf,
            'train_acc_mb': mb
        }, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tensor, batch_idx: int) -> float:
        x, y = batch
        y_hat = self.model(x)
        y = torch.squeeze(y).type(torch.LongTensor)
        y_hat = torch.squeeze(y_hat)
        loss = self.loss(y_hat, y)
        acc = self.val_acc(y_hat, y)
        mf = acc[0].item()
        mb = acc[1].item()

        self.log_dict({
            'val_loss': loss,
            'val_acc_mf': mf,
            'val_acc_mb': mb
        }, on_step=False, on_epoch=True, prog_bar=True)

        return loss
