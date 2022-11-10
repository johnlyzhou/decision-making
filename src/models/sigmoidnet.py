from typing import Tuple, List

import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
import torch
from sklearn.model_selection import train_test_split
from torch import tensor, nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset

from src.models.losses import SupConLoss


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


if __name__ == "__main__":
    data_path = "/Users/johnzhou/research/decision-making/data/processed"
    config = OmegaConf.create({
        "name": "test_expt",
        "random_seed": 4995,
        "model": {
            "in_features": 3,
            "linear_layers": [2],
            "use_batch_norm": False
        },
        "learning_rate": 1e-4,
        "data": {
            "feature_path": f"{data_path}/ql_mse_sig.npy",
            "label_path": f"{data_path}/ql_labels.npy",
            "train_proportion": 0.8,
            "train_batch_size": 100,
            "val_batch_size": 100
        },
        "trainer": {
            "gpus": 0,
            "max_epochs": 100
        },

    })

    batch_size = 100
    num_feats = 3
    test_data = torch.ones((batch_size, num_feats))
    net = SigmoidNet(OmegaConf.to_container(config))
    out = net(test_data)
    print(test_data.shape)
    print(out.shape)
