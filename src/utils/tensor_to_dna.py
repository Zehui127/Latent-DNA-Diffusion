import torch

def tensor_to_dna(tensor, inference=False):
    """Convert a 4xn tensor into a DNA sequence.
    If inference is True and the largest logit is 'N', output the index of the second largest value instead."""
    # Define the mapping from one-hot encoding to nucleotide
    mapping = {
        0: 'A',
        1: 'T',
        2: 'G',
        3: 'C',
        4: 'N'
    }

    if inference:
        # Use topk to get indices of the two largest values
        top2_vals, top2_indices = torch.topk(tensor, 2, dim=1)
        # Select the largest values' indices
        discrete_tensor = top2_indices[:, 0]
        # Check if the largest value is 'N'
        is_N = discrete_tensor == 4
        # For elements where largest value is 'N', use the second largest
        discrete_tensor[is_N] = top2_indices[is_N, 1]
    else:
        # Use argmax for normal mode
        discrete_tensor = torch.argmax(tensor, dim=1)

    # Ensure tensor is on CPU and convert to list
    tensor_list = discrete_tensor.cpu().tolist()

    # Map each index to its corresponding nucleotide
    dna_sequence = ''.join(mapping[base] for base in tensor_list)
    return dna_sequence
