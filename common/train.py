
import torch
import pandas as pd
import numpy as np
import time
# from tqdm.autonotebook import tqdm
from tqdm import tqdm
from common.utils import get_logger



log = get_logger('train')

## ----------------------------------------------------
## train generative network 
# assume that the return value of forward method is loss value
def train_gen_network(save_filepath, model, max_patience, train_loader, val_loader, epochs=50, device=torch.device("cpu"), optimizer=None, plot_func=None, plot_img_interval=1, save_whenever_improved=True):
    to_track = ["epoch", "total time", "train loss"]

    if val_loader is not None:
        to_track.append("val loss")

    total_train_time = 0 

    results = {}
    for item in to_track:
        results[item] = []

    prev_loss = torch.finfo(torch.float32).max
    p = 0
    loss_val = None

    model.to(device)
    for epoch in tqdm(range(1, epochs+1), desc="Epoch"):
        model = model.train()

        total_train_time += run_epoch(model, optimizer, train_loader, device, results, prefix="train", desc="Training")
        
        results["epoch"].append( epoch )
        results["total time"].append( total_train_time )

        if val_loader is not None:
            model = model.eval() 
            with torch.no_grad():
                run_epoch(model, optimizer, val_loader, device, results, prefix="val", desc="Validating")
                loss_val = results["val loss"][-1]
        

        if loss_val is not None:
            log.info(f"Epoch {str(epoch)} - val_loss={loss_val}")

            # when improved
            if loss_val < prev_loss:
                if save_whenever_improved:
                    torch.save(model, save_filepath.format(epoch))
                prev_loss = loss_val
                p = 0
            else:
                p += 1
            
            if epoch % plot_img_interval==0 and plot_func is not None:
                digits = model.inference(16, device)
                plot_func(digits.detach().cpu())
                    
        if p > max_patience:
            break
    return results

def run_epoch(model, optimizer, data_loader, device, results, prefix="", desc=None):
    running_loss = []
    start = time.time()
    for inputs in tqdm(data_loader, desc=desc, leave=False):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            inputs = inputs[0]

        inputs = inputs.float().to(device)
        loss = model(inputs)

        if model.training:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        running_loss.append(loss.item())
    end = time.time() 
    results[prefix + " loss"].append(np.mean(running_loss))
    return end-start




## ----------------------------------------------------
## train code for GAN
def train_gan(fullpath_template, G, D, latent_size, loss_func, optimizer_D, optimizer_G, train_loader, epochs, device):
    G.to(device)
    D.to(device)

    real_label = 1
    fake_label = 0

    g_losses = []
    d_losses = []

    for epoch in tqdm(range(epochs), desc="Epoch"):
        for data in tqdm(train_loader, leave=False):
            if type(data) == tuple or type(data) == list:
                data = data[0]

            real_data = data.to(device)
            bsize = real_data.size(0)
            y_real = torch.full((bsize, 1), real_label, dtype=torch.float32, device=device)
            y_fake = torch.full((bsize, 1), fake_label, dtype=torch.float32, device=device)

            D.zero_grad()

            # loss
            errd_real = loss_func(D(real_data), y_real)
            errd_real.backward()

            z = torch.randn(bsize, latent_size, device=device)
            fake = G(z) 
            errd_fake = loss_func(D(fake.detach()), y_fake)
            errd_fake.backward()

            ##
            # update D
            # sum real err and fake err
            errd = errd_real + errd_fake
            optimizer_D.step()


            ##
            # update G
            G.zero_grad()

            # calculate g loss
            errg = loss_func(D(fake), y_real)
            errg.backward()

            optimizer_G.step()
            
            g_losses.append(errg.item())
            d_losses.append(errd.item())

        torch.save(G, fullpath_template.format('g'))
        torch.save(D, fullpath_template.format('d'))
        pass

    return d_losses, g_losses

def sample_gan(G, D, batch_size, latent_size, device):
    D.eval()
    G.eval()
    with torch.no_grad():
        z = torch.randn(batch_size, latent_size, device=device)
        fake_digits = G(z)
        scores = torch.sigmoid(D(fake_digits))
        fake_digits = fake_digits.cpu()
        scores = scores.cpu().numpy().flatten()
    return fake_digits, scores


def train_wgan_gp(fullpath_template, G, D, latent_size, optimizer_D, optimizer_G, train_loader, epochs, device):
    G.to(device)
    D.to(device)

    g_losses = []
    d_losses = []
 
    for epoch in tqdm(range(1, epochs+1)):
        _glosses = []
        _dlosses = []
        for data in tqdm(train_loader, leave=False):
            if type(data) == tuple or type(data) == list:
                data = data[0]

            batch_size = data.size(0)

            ###
            # update D
            D.zero_grad()
            G.zero_grad()

            # real
            real = data.to(device)
            d_real = D(real)

            # fake
            z = torch.randn(batch_size, latent_size, device=device)
            fake = G(z) 
            d_fake = D(fake)

            ###
            # gradient penalty
            eps_shape = [batch_size]+[1]*(len(data.shape)-1)
            eps = torch.rand(eps_shape, device=device)
            fake = eps*real + (1-eps)*fake
            output = D(fake) 

            grad = torch.autograd.grad(outputs=output, inputs=fake,
                                  grad_outputs=torch.ones(output.size(), device=device),
                                  create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]
                                  
            d_grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
            ####

            errd = (d_fake-d_real).mean() + d_grad_penalty.mean()*10
            errd.backward()
            optimizer_D.step()
            
            _dlosses.append(errd.item())

            ## 
            # update G
            D.zero_grad()
            G.zero_grad()

            noise = torch.randn(batch_size, latent_size, device=device)
            output = -D(G(noise))
            errg = output.mean()
            errg.backward()
            optimizer_G.step()
            
            _glosses.append(errg.item())

        d_losses.append(np.mean(_dlosses))
        g_losses.append(np.mean(_glosses))

        torch.save(G, fullpath_template.format('g'))
        torch.save(D, fullpath_template.format('d'))
        pass


    return d_losses, g_losses
