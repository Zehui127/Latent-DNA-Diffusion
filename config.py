import os
import torch

class LoaderConfig():
    def __init__(self, loader_cfg=None):
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), loader_cfg.data_path)

        if loader_cfg is None:
            self.num_workers: int = 2 # cpus
            self.num_processes: int = 8 # gpus
            self.batch_size: int = 12
            self.train_prop: float = 0.5
            self.valid_prop: float = 0.1
            self.shuffle: bool = True
            self.generator = torch.Generator().manual_seed(50)
        else:
            self.num_workers: int = loader_cfg.num_workers
            self.num_processes: int = loader_cfg.num_processes
            self.batch_size: int = loader_cfg.batch_size
            self.train_prop: float = loader_cfg.train_prop
            self.valid_prop: float = loader_cfg.valid_prop
            self.shuffle: bool = loader_cfg.shuffle
            self.generator = torch.Generator().manual_seed(loader_cfg.seed)

class TrainerConfig():
    def __init__(self, save_path, trainer_cfg=None, loader_cfg=None, model_cfg=None):
        self.loader_config = LoaderConfig(loader_cfg=loader_cfg)
        self.save_path = save_path

        # training parameters
        for attr in trainer_cfg:
            self.__setattr__(attr, trainer_cfg[attr])
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        # model parameters
        for attr in model_cfg:
            self.__setattr__(attr, model_cfg[attr])
