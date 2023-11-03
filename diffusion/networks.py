import torch
import torch.nn as nn


class MLPNet(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_steps):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_steps = num_steps
        self.prenet = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.nets =  nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim*2)) for _ in range(self.num_steps)])

    def forward(self, x, t):
        """Return mu, log_var"""
        x = self.nets[t](self.prenet(x))
        mu, log_var = torch.chunk(x, 2, dim=1)
        return mu, log_var


class MLPModel(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_steps):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_steps = num_steps

        self.emb_1 = nn.Embedding(num_steps, latent_dim)
        self.emb_2 = nn.Embedding(num_steps, hidden_dim)
        self.prenet = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.LeakyReLU()
        )
        self.nets_1 = self.postnet =  nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.nets_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim*2))

    def forward(self, x, t):
        if not torch.is_tensor(t):
            t = torch.tensor([t]).to(x.device).long()
        emb1 = self.emb_1(t)
        emb2 = self.emb_2(t)
        x = self.nets_1(self.prenet(x)*emb1)
        x = self.nets_2(x*emb2)
        mu, log_var = torch.chunk(x, 2, dim=1)
        return mu, log_var

class DecoderNet(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.Tanh(),
        )

    def forward(self, x):
        x = self.model(x) # [batch_size, latent_dim] -> [batch_size, latent_dim]
        # -1.0 ~ 1.0 -> 0.0 ~ 1.0
        return (x+1.0) / 2.0


class SimpleUNet(nn.Module):
    def __init__(self, T, in_channels, out_channels, emb_dim=32, hidden_channels=64):
        super().__init__()

        self.T = T
        self.emb_dim = emb_dim
        self.emb = nn.Embedding(T, emb_dim)
        self.linear1 = nn.Linear(emb_dim, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, hidden_channels*2)



        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def up_conv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.encoder1 = conv_block(in_channels, hidden_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = conv_block(hidden_channels, hidden_channels*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.middle1 = conv_block(hidden_channels*2, hidden_channels*2)
        self.middle2 = conv_block(hidden_channels*2, hidden_channels*2)
                                    

        self.up_conv1 = up_conv(hidden_channels*2, hidden_channels*2)
        self.decoder1 = conv_block(hidden_channels*4, hidden_channels*2)
        self.up_conv2 = up_conv(hidden_channels*2, hidden_channels)
        self.decoder2 = conv_block(hidden_channels*2, hidden_channels)

        self.out_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)


    def forward(self, x, t):
        if not torch.is_tensor(t):
            t = torch.tensor([t]).to(x.device).long()
        batch_size = x.shape[0]
        emb = self.emb(t)
        emb = self.linear1(emb)
        emb2 = self.linear2(emb)
        emb2 = emb2.reshape(batch_size, -1, 1, 1)
        emb = emb.reshape(batch_size, -1, 1, 1)

        # down
        enc1 = self.encoder1(x)
        pool1 = self.pool1(enc1*emb)
        enc2 = self.encoder2(pool1)
        pool2 = self.pool2(enc2*emb2)

        middle = self.middle1(pool2*emb2)
        middle = self.middle1(middle*emb2)

        # up
        up1 = self.up_conv1(middle)
        concat1 = torch.cat([up1, enc2], dim=1)
        dec1 = self.decoder1(concat1)
        up2 = self.up_conv2(dec1*emb2)
        concat2 = torch.cat([up2, enc1], dim=1)
        dec2 = self.decoder2(concat2)

        out = self.out_conv(dec2)
        return out