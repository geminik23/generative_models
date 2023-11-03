import torch
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

# 
from diffusion.networks import *
from diffusion.models import NaiveDiffusionModel, DiffusionModel
from diffusion.utils import create_betas
from common.train import train_gen_network
from common.utils import plot_imgs, setup_logger, get_logger
from common.dataset import get_digit_dataset

import os

result_dir = 'result'
if not(os.path.exists(result_dir)):
    os.mkdir(result_dir)




import datetime
def get_filename_with_timestamp(prefix:str):
    now = datetime.datetime.now()
    timestamp = now.strftime('%y%m%d_%H%M%S')
    filename = f'{prefix}_data_{timestamp}.pt'
    return filename
    

def main():
    parser = argparse.ArgumentParser(description='Diffusion Generative Model')
    parser.add_argument('mode', choices=['train', 'inference'], help='Choose mode: train or inference')
    parser.add_argument('--model', type=str, default='naive-lin', choices=['naive-lin', 'ddm-unet'], help='Select the model to use (default: naive-lin)')
    parser.add_argument('--save_file', type=str, default='model', help='File name to save the model. {} is for epoch count(default: model)')
    parser.add_argument('--load_file', type=str, default='model', help='File name to load the model (default: model)')
    parser.add_argument('--plot_interval', type=int, default=-1, help='During training, plot the generated image every {int} epochs. If value is negative, it will be ignored (default: -1)')
    args = parser.parse_args()

    setup_logger(None)
    log = get_logger('main')
    log.info(f'MODEL: {args.model}')
    log.info(f'MODE: {args.mode}')
    log.info(f'Save model to {args.save_file}' if args.mode == 'train' else f'Load model from {args.load_file}')


    # Hyperparameters and other settings
    IMG_SIZE = 12
    DIM = IMG_SIZE * IMG_SIZE
    LEARNING_RATE = 1e-4
    EPOCH = 2500
    BATCH_SIZE = 64
    MAX_PATIENCE = 20
    BETAS = (1e-5, 2e-2)


    plot_func = lambda x: plot_imgs(x, IMG_SIZE, (4, 4))
    if args.plot_interval < 0:
        plot_func = None
        args.plot_interval = 1
        
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"PyTorch Device is {device}")


    # Model
    if args.model == 'naive-lin':
        LEARNING_RATE = 1e-3
        latent_dim = DIM
        hidden_dim = 256
        steps = 12

        dataset = get_digit_dataset(IMG_SIZE, transform=None)

        betas = create_betas(steps, BETAS[0], BETAS[1], 'linear')
        model = NaiveDiffusionModel(latent_dim, hidden_dim, MLPModel, DecoderNet, betas, device=device)
    else:
        MAX_PATIENCE = 1000
        steps = 40

        dataset = get_digit_dataset(IMG_SIZE, flatten=False, transform=None)

        net = SimpleUNet(steps, 1, 1)
        betas = create_betas(steps, BETAS[0], BETAS[1], 'linear')
        model = DiffusionModel(steps, net, betas, (IMG_SIZE, IMG_SIZE), device=device)
        pass

    val_len = int(len(dataset)*0.05)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE) 

    if args.mode == 'train':
        args.save_file = args.save_file + '_{}.pt'

        optimizer = torch.optim.Adamax(model.parameters(), LEARNING_RATE)

        # Training
        log.info('Starting training...')
        results = train_gen_network(os.path.join(result_dir, args.save_file), model, MAX_PATIENCE, train_loader, val_loader, EPOCH, device, optimizer=optimizer, plot_func=plot_func, plot_img_interval=args.plot_interval)
        log.info('Save the results...')
        torch.save(results, os.path.join(result_dir, get_filename_with_timestamp(args.model)))
        log.info('Training completed...')
        
        # Plot losses
        results = pd.DataFrame(results)
        sns.lineplot(x="epoch", y="val loss", data=results)
        plt.show()

    elif args.mode == 'inference':
        # Load saved model
        model = torch.load(os.path.join(result_dir, args.load_file))

        # Generate images
        model.eval()
        imgs = model.inference(16, device=device).detach().cpu()
        plot_imgs(imgs, IMG_SIZE, (4,4), save_file= f"./images/diffusion_{args.model}_digit.png")


