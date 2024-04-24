import torch
import unittest
import numpy as np
from vae import VanillaVAE
from torchsummary import summary

class TestVAE(unittest.TestCase):

    def setUp(self) -> None:
        self.model =  VanillaVAE(in_channels=1, latent_dim=256, hidden_dims=[8, 16, 32], seq2img_num_layers=4, seq2img_img_width=128)
        self.data_size = [50, 2048, 4] # shape of input (sequence encoded 'image')
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("No of params:",params)

    # def test_summary(self):
    #     print(summary(self.model, self.data_size, device='cpu'))

    def test_encode(self):
        print("Testing Encoder")
        x = torch.randn(*self.data_size)
        d = self.model.encode(x)
        print("Encoder Output size:", d.sample().shape)

    def test_forward(self):
        print("Testing Forward")
        x = torch.randn(*self.data_size)
        y, d = self.model(x)
        print("Model Output size:", y.size(), "initial: ", x.size())

    def test_loss(self):
        print("Testing Loss")
        x = torch.randn(*self.data_size)
        res, d = self.model(x)
        loss = self.model.loss_function(res, x, d, M_N=50).shape
        print(loss)


if __name__ == '__main__':
    unittest.main()