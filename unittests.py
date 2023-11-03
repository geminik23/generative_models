import unittest
import torch
import torch.nn as nn
from arm.arm_model import CasualConv1d, AutoregressiveModel
from arm.pixelcnn import MaskedConv2D, ResidualBlock


class PixelCNNCase(unittest.TestCase):
    def test_shape_masked_conv_and_residual(self):
        batch_size = 4
        in_ch = 4
        out_ch = 8
        kernel_size = 3
        padding = 1



        # input
        x = torch.randn(batch_size, in_ch, 32, 32)

        ## MaskedConv2D
        maskedconv2d = MaskedConv2D(True, in_ch, out_ch, kernel_size, padding)
        # output
        y = maskedconv2d(x)
        target_size = torch.Size((batch_size, out_ch, 32, 32))

        self.assertEqual(y.shape, target_size)

        ## ResidualBlock
        residual = ResidualBlock(in_ch)
        y = residual(x)

        target_size = torch.Size((batch_size, in_ch, 32, 32))
        self.assertEqual(y.shape, target_size)
        

    

class ARMMTestCase(unittest.TestCase):
    def setUp(self):
        NUM_CLASS = 17
        HIDDEN_DIM = 512

        self.net = nn.Sequential(
            CasualConv1d(1, HIDDEN_DIM, dilation=1, kernel_size = 2, exclude_last=True, bias=True),
            nn.LeakyReLU(),
            CasualConv1d(HIDDEN_DIM, HIDDEN_DIM, dilation=1, kernel_size = 2, exclude_last=False, bias=True),
            nn.LeakyReLU(),
            CasualConv1d(HIDDEN_DIM, HIDDEN_DIM, dilation=1, kernel_size = 2, exclude_last=False, bias=True),
            nn.LeakyReLU(),    
            CasualConv1d(HIDDEN_DIM, NUM_CLASS, dilation=1, kernel_size = 2, exclude_last=False, bias=True)
        )

    def test_causalconv1d(self):
        DIM = 64
        NUM_CLASS = 17
        BATCH = 4
        ##
        ## FORWARD PASS
        # construct the network
        x = torch.randint(0, 15, (BATCH, DIM)).float()
        x = x.unsqueeze(1) # -> [B, C, D]
        y = self.net(x)
        self.assertEqual(y.shape, torch.Size([BATCH, NUM_CLASS, DIM]))

        ##
        ## SOFTMAX
        # apply the softmax
        y = y.permute(0,2,1) # -> [B, D, C]
        softmax = nn.Softmax(2)
        y = softmax(y)
        self.assertEqual(y.shape, torch.Size([BATCH, DIM, NUM_CLASS]))

        y = torch.argmax(y, dim=2)
        self.assertEqual(y.shape, torch.Size([BATCH, DIM]))

    def test_autoregressive_model(self):
        DIM = 64
        NUM_CLASS = 17
        BATCH = 4
        ##
        ## GET LOSS
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AutoregressiveModel(self.net, DIM, NUM_CLASS).to(device)
        
        x = torch.randint(0, 15, (BATCH, DIM), device=device).float()
        y = model(x)
        # y is scalar
        assert y.shape == torch.Size([])

        ##
        ## SAMPLING
        out = model.inference(1, device=device)
        assert out.shape == torch.Size([1, 64])


if __name__ == '__main__':
    unittest.main(verbosity=2)
