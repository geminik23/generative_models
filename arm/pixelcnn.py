import torch
import torch.nn as nn
import torch.nn.functional as F


# implmenetation from the paper 'https://arxiv.org/pdf/1601.06759v3.pdf'
class MaskedConv2D(nn.Conv2d):
    def __init__(self, mask_include_center, in_channels, out_channels, kernel_size, padding, *args, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, *args, **kwargs)

        self.include_center = mask_include_center
        # register the mask as buffer
        self.register_buffer('mask', self.weight.data.clone())

        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + self.include_center:] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        # # mask the weight
        self.weight.data *= self.mask
        return super().forward(x)



class ResidualBlock(nn.Module):
    """ ResidualBlock include the MaskedConv2D"""
    def __init__(self, filters, include_center=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # keep same dimension (W, H)
        self.conv1 = nn.Conv2d(filters, filters // 2, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.pixel_conv = MaskedConv2D(include_center, filters // 2, filters // 2, kernel_size=3, padding=1) 
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(filters // 2, filters, kernel_size=1)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pixel_conv(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.relu3(out)
        out = out + x
        return out

        
class PixelCNN(nn.Module):
    def __init__(self, image_size, n_filters, residual_blocks, kernel_size, pixel_levels):
        super().__init__()

        self.image_size = image_size
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.residual_blocks = residual_blocks
        self.pixel_levels = pixel_levels

        # like autoregressive model exclude the central in first layer.
        self.masked_conv_ex = MaskedConv2D( False, in_channels=1, out_channels=n_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu1 = nn.ReLU()
        self.res_blocks = nn.Sequential(*[ResidualBlock(n_filters) for _ in range(residual_blocks)])
        self.masked_conv_in1 = MaskedConv2D( True, in_channels=n_filters, out_channels=n_filters, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU()
        self.masked_conv_in2 = MaskedConv2D( True, in_channels=n_filters, out_channels=n_filters, kernel_size=1, padding=0)
        self.relu3 = nn.ReLU()
        self.conv_out = nn.Conv2d( in_channels=n_filters, out_channels=pixel_levels, kernel_size=1, stride=1, padding=0)

    def _f(self, x):
        x = self.masked_conv_ex(x)
        x = self.relu1(x)
        x = self.res_blocks(x)
        x = self.masked_conv_in1(x)
        x = self.relu2(x)
        x = self.masked_conv_in2(x)
        x = self.relu3(x)
        x = self.conv_out(x)
        return x

    def forward(self, x):
        logits = self._f(x)
        target = x.long().squeeze(1)  # remove the channel dimension
        return F.cross_entropy(logits, target, reduction='none').mean()

    def inference(self, num_img, device, temperature=1.0):
        with torch.no_grad():
            self.eval()
            generated_images = torch.zeros((num_img, 1, self.image_size, self.image_size), device=device)
            batch, channels, rows, cols = generated_images.shape

            for row in range(rows):
                for col in range(cols):
                    logits = self._f(generated_images)[:, :, row, col]
                    probs = F.softmax(logits / temperature, dim=1)
                    new_pixel = torch.multinomial(probs, num_samples=1)
                    generated_images[:, 0, row, col] = new_pixel.squeeze()
        return generated_images