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


class SigmoidNet(LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.loss = nn.CrossEntropyLoss()
