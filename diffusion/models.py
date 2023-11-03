import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


def reparameterize(mu, log_var):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    return mu + std*eps

# http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
def log_likelihood_of_gaussian(x, mu, log_var):
    log_p = -(0.5 * torch.tensor(2. * torch.pi).to(x.device).log_() + log_var) - 0.5 * torch.exp(-log_var) * (x - mu)**2.
    return log_p

class NaiveDiffusionModel(nn.Module):
    def __init__(self, latent_dim:int, hidden_dim:int, ModelNet, DecoderNet, betas:torch.Tensor, device='cpu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.T = len(betas)
        self.betas = betas.to(device)

        # network model
        self.model = ModelNet(latent_dim, hidden_dim, self.T)
        self.decoder = DecoderNet(latent_dim, hidden_dim)

    def q_process(self, xt_1, t):
        assert t>0, "t should be greater than 0"
        t -= 1
        # 
        mu = xt_1*torch.sqrt(1 - self.betas[t])
        std = torch.sqrt(self.betas[t])
        return mu + std*torch.randn_like(xt_1).to(xt_1.device)

    
    def p_process(self, xs):
        mus = []
        log_vars = []
        for t in range(self.T-1, -1, -1):
            mu, log_var = self.model(xs[t+1], t)
            mus.append(mu)
            log_vars.append(log_var)
        return mus, log_vars
    
    def forward(self, x0):
        """compute the loss"""

        device = x0.device

        # forward process
        xs = [x0] # x0~xT
        for t in range(1, self.T+1):
            xs.append(self.q_process(xs[-1],t))

        # backward process
        mus, log_vars = self.p_process(xs)
        
        # append zeros for convenience in the computation
        mus.append(torch.full_like(mus[0], 0.0).to(device))
        log_vars.append(torch.full_like(log_vars[0], 0.0).to(device))  # exp(log_var=0) = 1
        mus = list(reversed(mus))
        log_vars = list(reversed(log_vars))

        z1 = self.decoder(reparameterize(mus[1], log_vars[1]))
    
        # compute the ELBO
        # : calculate the log likelyhood of gaussian distribution directly.
        # :term1: ln{p(x|z_1)}
        L0 = log_likelihood_of_gaussian(xs[0] - z1, mus[0], log_vars[0]).sum(-1)
        
        # :term2: KL[q(z_T|z_{T-1}||p(z_T)]
        KL =  (log_likelihood_of_gaussian(xs[-1], torch.sqrt(1. - self.betas[-1]) * xs[-1], torch.log(self.betas[-1])) 
               - log_likelihood_of_gaussian(xs[-1], mus[0], log_vars[0])).sum(-1)

        # :rest_of_terms: KL[q(z_t|z_{t-1}||p(z_t)|z_{t+1})]
        for i in range(1, self.T):
            KL += (log_likelihood_of_gaussian(xs[i], torch.sqrt(1. - self.betas[i-1]) * xs[i], torch.log(self.betas[i-1])) - log_likelihood_of_gaussian(xs[i], mus[i], log_vars[i])).sum(-1)

        # the negative log-likelihood loss
        loss = -(L0 - KL).mean()
        return loss

    def inference(self, batch_size:int, device='cpu'):
        z = torch.randn([batch_size, self.latent_dim]).to(device)
        for t in range(self.T-1, -1, -1):
            mu, log_var = self.model(z, t)
            z = reparameterize(mu, log_var)
        return self.decoder(z)


class DiffusionModel(nn.Module):
    """
    Algorithms in Denoising Diffusion Probabilistic Model Paper
    """
    def __init__(self, T:int, model, betas:torch.Tensor, img_size:Tuple[int, int], device='cpu'):
        super().__init__()
        self.T = T
        self.img_size = img_size
        # used linear
        self.betas = betas.to(device).to(device)
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, 0)
        
        self.model = model.to(device)

    def inference(self, sample_size:int, n_channels:int=1, device='cpu'):
        """From algorithm 2 from paper"""
        xt = torch.randn([sample_size, n_channels, self.img_size[0], self.img_size[1]]).to(device)
        
        for t in range(self.T-1, -1, -1):
            z = torch.randn_like(xt).to(device) if t > 0 else torch.zeros_like(xt).to(device)
            t = torch.full((sample_size,), t, dtype=torch.long, device=device)
            # needs alpha, alphabar and beta(sigma) of t
            beta_t = self.betas[t].reshape(-1, 1, 1, 1)
            alpha_t = self.alphas[t].reshape(-1, 1, 1, 1)
            alpha_bar_t = self.alphas_bar[t].reshape(-1, 1, 1, 1)
            xt = (1 / torch.sqrt(alpha_t) * (xt - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * self.model(xt, t))) + torch.sqrt(beta_t) * z
        return xt

        
    def forward(self, x0):
        """compute the loss : one step of gradient descent. algorithm 1 from paper"""
        batch_size = x0.shape[0]
        
        # randomly select the `t`
        t = torch.randint(0, self.T, (batch_size,)).to(x0.device)

        eps = torch.randn_like(x0)
        
        # x0 [B, C, W, H]
        # alpha_bar_t also need to be 4 dim.
        alpha_bar_t = self.alphas_bar[t].reshape(-1, 1, 1, 1)
        eps_theta = self.model(torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps, t)
        loss = F.mse_loss(eps, eps_theta)
        return loss

