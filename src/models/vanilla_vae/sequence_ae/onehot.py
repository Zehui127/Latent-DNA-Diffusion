import torch
import numpy as np


def onehot_encoder(sequences):
    max_len = max([len(s) for s in sequences])
    dictionary = {
        "A": [1, 0, 0, 0],
        "T": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "C": [0, 0, 0, 1],
    }
    if not isinstance(sequences, (list, tuple, np.ndarray)):
        sequences = list(sequences)
    for i in range(len(sequences)):
        sequences[i] = sequences[i].upper()[:max_len]
    shape = [len(sequences), max_len, len(dictionary)]
    onehot = np.zeros(shape, dtype=np.float32)
    for i, s in enumerate(sequences):
        for j, el in enumerate(s):
            onehot[i, j] = dictionary[el]
    return torch.tensor(onehot)
