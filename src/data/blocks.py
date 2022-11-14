from typing import List, Tuple, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.real_data import DynamicForagingData, generate_real_block_params, convert_real_actions
from src.utils import blockify, normalize_choice_block_side


class RealDataset(Dataset):
    def __init__(self, data_file: str, min_len: int = 15) -> None:
        self.data_file = data_file
        self.min_len = min_len
        self.data = self.__ingest_real_data()

    def __getitem__(self, idx: int) -> Tensor:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

    def __ingest_real_data(self) -> List[Tensor]:
        real_expt = DynamicForagingData(self.data_file)
        blocks = generate_real_block_params(real_expt.block, real_expt.correct_side)
        actions = convert_real_actions(real_expt.response_side)
        blocked_actions = blockify(blocks, actions)
        normalized_actions = [torch.tensor(normalize_choice_block_side(blocked_actions[block_idx], side=blocks[block_idx][0]
                                                                       )[: self.min_len])
                              for block_idx in range(len(blocks)) if blocks[block_idx][2] >= self.min_len]
        return normalized_actions


class SynthDataset(Dataset):
    def __init__(self, data_file: str, label_file: str, min_len: int = 15) -> None:
        self.min_len = min_len
        self.data = [torch.tensor(sample) for sample in list(np.load(data_file).T)]
        self.labels = list(np.load(label_file).T)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Any]:
        return self.data[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.data)
