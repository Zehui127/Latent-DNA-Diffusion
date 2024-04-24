# from selene_sdk.utils import NonStrandSpecific
from .sei import *
import os
import pandas as pd

current_path = os.path.dirname(os.path.abspath(__file__))

def load_model_sei(is_cuda_available,model_path='sei.pth', names_path='target.names'):
    """
    Loads the Sei model with the given state dictionary and returns the model and feature names.

    Args:
        model_path (str): Path to the saved model's state dictionary.
        names_path (str): Path to the CSV containing feature names.

    Returns:
        sei (torch.nn.Module): The loaded Sei model.
        seifeatures (pandas.DataFrame): DataFrame containing the feature names.
    """
    model_path = os.path.join(current_path, model_path)
    names_path = os.path.join(current_path, names_path)

    # Initialize the model
    # sei = NonStrandSpecific(Sei(4096, 21907))
    sei = Sei(4096, 21907)
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location='cpu')
    # Remove 'module.' prefix from state_dict keys
    new_state_dict = {k.replace('module.model.', ''): v for k, v in state_dict.items()}
    # Load the modified state dictionary into the model
    sei.load_state_dict(new_state_dict)
    # Load feature names from CSV
    seifeatures = pd.read_csv(names_path, sep='|', header=None)
    if is_cuda_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    sei = sei.to(device)
    return sei, seifeatures
