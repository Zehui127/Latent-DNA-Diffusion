import torch
import numpy as np


def onehot_encoder(sequences):
    max_len = max([len(s) for s in sequences])
    dictionary = {
        "A": [1, 0, 0, 0],
        "T": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "C": [0, 0, 0, 1],
        # 'N': [0, 0, 0, 0],
        'N': [0.25, 0.25, 0.25, 0.25],
    }
    if not isinstance(sequences, (list, tuple, np.ndarray)):
        sequences = list(sequences)
    for i in range(len(sequences)):
        sequences[i] = sequences[i].upper()[:max_len]
    shape = [len(sequences), max_len, 4]
    onehot = np.zeros(shape, dtype=np.float32)
    for i, s in enumerate(sequences):
        for j, el in enumerate(s):
            onehot[i, j] = dictionary[el]
    return torch.tensor(onehot)

def pad_on_both_side(sequence,target_len=2048):
    n = len(sequence)
    if n>=target_len:
        return sequence
    left_padding = (target_len - n)//2
    right_padding = target_len - n - left_padding
    return 'N'*left_padding + sequence + 'N'*right_padding

def padding_df(df,target_len=2048):
    df['merged_padding'] = [pad_on_both_side(row['merged']) for _, row in df.iterrows()]
    df['merged_padding'] = df.apply(lambda row: pad_on_both_side(row['merged']), axis=1)
    return df
