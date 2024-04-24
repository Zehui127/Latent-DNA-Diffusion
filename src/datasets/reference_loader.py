import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class LoaderConfig():
    def __init__(self, loader_cfg=None):
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), loader_cfg.data_path)

        if loader_cfg is None:
            self.num_workers: int = 2 # cpus
            self.num_processes: int = 8 # gpus
            self.batch_size: int = 64 #// self.num_processes
            self.train_prop: float = 0.5
            self.val_prop: float = 0.1
            self.shuffle: bool = True
            self.generator = torch.Generator().manual_seed(50)
        else:
            self.num_workers: int = loader_cfg.num_workers
            self.num_processes: int = loader_cfg.num_processes
            self.batch_size: int = loader_cfg.batch_size #// self.num_processes
            self.train_prop: float = loader_cfg.train_prop
            self.shuffle: bool = loader_cfg.shuffle
            self.generator = torch.Generator().manual_seed(loader_cfg.seed)


class DataSet(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        self.labels = torch.load(self.data_path)


    def __getitem__(self, index):
        # label is the same as the original input
        return self.labels[index]

    def __len__(self):
        return len(self.labels)


def prepare_loader(loader_config=None):
    config = loader_config if loader_config else LoaderConfig()

    dataset = DataSet(data_path = config.data_path)

    train_size = int(config.train_prop * len(dataset))
    val_size = int(config.valid_prop * len(dataset))
    test_size = len(dataset)-train_size-val_size

    train_dataset, test_dataset, val_dataset = random_split(dataset = dataset, lengths = [train_size, test_size, val_size], generator = config.generator)
    print(len(train_dataset), len(test_dataset), len(val_dataset))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers
    )

    return train_loader, val_loader, test_loader
