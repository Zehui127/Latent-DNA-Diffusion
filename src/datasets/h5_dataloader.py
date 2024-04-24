import os
import sys
import math
import signal
import functools
import numpy as np

import torch
import pathlib
torch.multiprocessing.set_sharing_strategy("file_system")


class DataLoader(torch.utils.data.DataLoader):
    """
    usage:
    from src.datasets.h5_dataloader import DataLoader
    from src.datasets.h5_ref_dataset import ReferenceGene
    dataset = ReferenceGene(split='test')
    batch_size = 32
    num_workers =10 # cpu number
    dataloader = DataLoader(dataset,batch_size,num_workers)
    # iterate through dataloader and get the first item
    item = next(iter(dataloader))
    item.shape # 2048*32*4
    """
    pin_memory = True

    def __init__(
        self,
        dataset,
        batch_size,
        workers,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
    ):
        super().__init__(
            dataset,
            batch_size,
            pin_memory=self.pin_memory,
            num_workers=workers,
            worker_init_fn=self.worker_init,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
        )
        self.single_graph = single_graph
        self.workers = workers

    @staticmethod
    def worker_init(x):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def __iter__(self):
        for i, item in enumerate(super().__iter__()):
            yield item
