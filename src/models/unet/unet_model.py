#define unet
from diffusers import UNet2DModel
from typing import List
#only has 4 up and down blocks to make it work for 16 x 16. for higher resolutions add more blocks
class UNetModel():
    """UNetModel predicts noise of latent encoded sequences.
    input: [Bx16x16xC] tensor
    output: predicted noise [Bx16x16xC] tensor
    details:
    """
    def __init__(self,
                 sample_size: int,
                 in_channels: int,
                 out_channels: int,
                 layers_per_block: int,
                 block_out_channels: List,
                 down_block_types: List,
                 up_block_types: List,
                 **kwargs) -> None:

        UNet2DModel(sample_size = sample_size,
        in_channels = in_channels,
        out_channels = out_channels,
        layers_per_block = layers_per_block,
        block_out_channels = block_out_channels,
        down_block_types = down_block_types,
        up_block_types = up_block_types,
        **kwargs,
        )
