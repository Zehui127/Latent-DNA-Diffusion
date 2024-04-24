import os
import numpy as np
import  pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

class OHEDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        self.labels = torch.load(self.data_path)
        print(f"data loaded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!{self.labels.shape}")

    def __getitem__(self, index):
        # label is the same as the original input
        return self.labels[index]

    def __len__(self):
        return len(self.labels)

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: object
    ):
        super().__init__()
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.train_prop = config.train_prop
        self.valid_prop = config.valid_prop
        self.generator = config.generator
        self.shuffle = config.shuffle
        self.train_dataset = (
            self.val_dataset
        ) = self.test_dataset = None

    def prepare_data(self) -> None:
        """
        prepare_data is called from the main process.
        It is not recommended to assign state here (e.g. self.x = y) since it is called on a single process.
        """
        pass

    def setup(self, stage: str = None) -> None:
        dataset = OHEDataset(data_path = self.data_path)
        train_size = int(self.train_prop * len(dataset))
        val_size = int(self.valid_prop * len(dataset))
        test_size = len(dataset)-train_size-val_size
        (
            self.train_dataset,
            self.test_dataset,
            self.val_dataset,
        ) =  random_split(dataset = dataset, lengths = [train_size, test_size, val_size], generator = self.generator)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
