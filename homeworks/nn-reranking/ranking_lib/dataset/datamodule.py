import lightning as L
import torch

from functools import partial
from typing import List

from hydra.utils import instantiate
from torch.utils.data import DataLoader, WeightedRandomSampler

from omegaconf import DictConfig


class BaseDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset: DictConfig,
        valid_dataset: DictConfig,
        train_dataloader: DictConfig,
        valid_dataloader: DictConfig,
        preprocessor: DictConfig,
        train_sampler_type: str = 'uniform'
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.train_dataloader_instance = train_dataloader
        self.valid_dataloader_instance = valid_dataloader

        self.train_sampler_type = train_sampler_type

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = instantiate(self.train_dataset)
            self.valid_dataset = instantiate(self.valid_dataset)

    def train_dataloader(self) -> DataLoader:
        if self.train_sampler_type != 'uniform':
            weights = self.train_dataset.get_weights()
            sampler = WeightedRandomSampler(weights, len(self.train_dataset), True)
        return instantiate(
            self.train_dataloader_instance,
            dataset=self.train_dataset,
            drop_last=True,
            shuffle=True
        )

    def val_dataloader(self) -> List[DataLoader]:
        return instantiate(self.valid_dataloader_instance,
            dataset=self.valid_dataset,
            shuffle=False,
            drop_last=False
        )