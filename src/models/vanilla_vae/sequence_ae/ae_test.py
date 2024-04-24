import torch
import unittest
import numpy as np
from dna_encoder import SequenceAE
from torchsummary import summary

class TestAE(unittest.TestCase):
    def setUp(self) -> None:
        self.num_layers = 3
        self.funnel_width = 256
        self.model =  SequenceAE(num_layers=self.num_layers, funnel_width=self.funnel_width)
        self.data_size = [5, 2048, 4]
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("No of params:",params)

    # def test_summary(self):
    #     print("Testing Summary")
    #     print(summary(self.model, tuple([*self.data_size]), device='cpu'))

    def test_encode(self):
        print("Testing Encoder")
        x = torch.randn(*self.data_size)
        y = self.model.encoder(x)
        print("Encoder Output size:", y.size())

    def test_forward(self):
        print("Testing Forward")
        x = torch.randn(*self.data_size)
        y = self.model(x)
        print("AE Input size:", x.size())
        print("AE Output size:", y.size())

    def test_loss(self):
        print("Testing Loss")
        x = torch.randn(*self.data_size)
        result = self.model(x)

        loss = self.model.loss_function(result, x)
        print(loss)


if __name__ == '__main__':
    unittest.main()