import torch
from torch import nn
from torch.nn import functional as F
from .sequence_ae.multikernel_conv_block import *
from .sequence_ae.dna_encoder import *
from .distribution import *
from ...utils.types_ import *
from .two_d_res_block import ResnetBlock

class VanillaVAE(nn.Module):
    """VanillaVAE encodes sequences into latent space.
    input: [Bx2048x4] tensor
    output: reconstructed [Bx2048x4] tensor
    details: for submodule layers check DNAEncoder and DNADecoder block
    """
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 seq2img_num_layers: int = 5,
                 seq2img_img_width: int = 64,
                 layer_per_block: int = 0,
                 **kwargs) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [4, 8]
        self.hidden_dims = hidden_dims.copy()
        self.seq2img_num_layers = seq2img_num_layers
        self.seq2img_img_width = seq2img_img_width

        # Define Sequence to Image Encoder-Decoder
        # length reduces by 2^(num_layers), width increases to target_width
        self.sequence_encoder = DNAEncoder(num_layers=self.seq2img_num_layers, target_width=self.seq2img_img_width)
        self.sequence_decoder = DNADecoder(num_layers=self.seq2img_num_layers, target_width=self.seq2img_img_width)

        # Define Encoder Layers
        modules = []
        for h_dim in hidden_dims:
            #### Add Resnet Blocks ####
            for j in range(layer_per_block):
                modules.append(ResnetBlock(in_channels, dropout=0.0))
            ###########################
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            ) # this shrinks the length and width by 2 each time
            in_channels = h_dim

        #### Add Resnet Blocks ####
        for j in range(layer_per_block):
            modules.append(ResnetBlock(in_channels, dropout=0.0))
        ###########################
        self.encoder = nn.Sequential(*modules)

        self.activation_map_size = seq2img_img_width // pow(2, len(hidden_dims))

        # Define 2D Conv. Layers for learning multivariate Gaussian distribution
        self.latent_encoder = nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size=1)

        # self.fc_mu = nn.Linear(hidden_dims[-1]*self.activation_map_size**2, latent_dim)
        # self.fc_var = nn.Linear(hidden_dims[-1]*self.activation_map_size**2, latent_dim)

        # Define Decoder Layers
        modules = []
        self.latent_decoder = nn.Conv2d(hidden_dims[-1]//2, hidden_dims[-1], kernel_size=1)

        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            #### Add Resnet Blocks ####
            for j in range(layer_per_block):
                modules.append(ResnetBlock(hidden_dims[i], dropout=0.0))
            ###########################
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        #### Add Resnet Blocks ####
        for j in range(layer_per_block):
            modules.append(ResnetBlock(hidden_dims[-1], dropout=0.0))
        ###########################
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=1,
                                      kernel_size=3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # encode sequence into image-like representation
        input = self.sequence_encoder(input)
        # print(f"encoder input.shape: {input.shape}")
        # encode image-like representation into latent representation
        input = input.view(-1, 1, input.shape[1], input.shape[2])
        # print(f"encoder after viewing input.shape: {input.shape}")
        input = self.encoder(input)
        # print(f"encoder after encoder input.shape: {input.shape}")
        # print(f"encoder after latent encoder input.shape: {self.latent_encoder(input).shape}")
        dist = DiagonalGaussianDistribution(self.latent_encoder(input))
        return dist

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        # decode latent representation into image-like representation
        result = self.latent_decoder(z)
        # print(f"decoder latent_decoder result.shape: {result.shape}")
        result = result.view(-1, self.hidden_dims[-1], self.activation_map_size, self.activation_map_size)
        # print(f"decoder after viewing result.shape: {result.shape}")
        result = self.decoder(result)
        result = self.final_layer(result)
        # decode image-like representation into sequence
        result = self.sequence_decoder(result)
        return result

    def get_emb(self, input: Tensor) -> Tensor:
        # encode sequence into image-like representation
        input = self.encode(input).mean
        input = input.view(input.size(0), -1)
        return input
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        dist = self.encode(input)
        z = dist.sample()
        res = self.decode(z)

        return  res, dist

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        dist = args[2]
        kld_weight = kwargs['kld_weight'] # Account for the minibatch samples from the dataset

        kld_loss = dist.kl().sum()
        # recons_loss = F.mse_loss(recons, input,reduction='sum')
        # recons_loss = self.mean_sum_square_error(recons,input)
        recons_loss = self.cross_entropy_prob(recons,input)

        return recons_loss + kld_weight * kld_loss, recons_loss, kld_weight * kld_loss

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    def mean_sum_square_error(self,recons: Tensor,input: Tensor) -> Tensor:
        square_diff = (recons-input)**2
        mse = square_diff.mean(dim=[1,2])
        sum_mse = mse.sum()
        return sum_mse

    def cross_entropy_sum(self, predicted_tensor, target_tensor):
        return F.cross_entropy(predicted_tensor.reshape(-1, 4), torch.argmax(target_tensor, dim=2).reshape(-1), reduction='sum')

    def cross_entropy_prob(self, predicted_tensor, target_tensor):
        # TODO make all the N token to be 0.25, 0.25, 0.25, 0.25
        return F.cross_entropy(predicted_tensor.reshape(-1, 4), target_tensor.reshape(-1,4), reduction='sum')
