import torch
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
from torch.utils.data import DataLoader, random_split
import numpy as np

# 
from gan.gan_models import get_gan_model
from common.train import train_wgan_gp, sample_gan
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
    parser = argparse.ArgumentParser(description='GAN-based Generative Model')
    parser.add_argument('mode', choices=['train', 'inference'], help='Choose mode: train or inference')
    parser.add_argument('--model', type=str, default='gan', choices=['gan'], help='Select the model to use (default: gan)')
    parser.add_argument('--save_file', type=str, default='model', help='File name to save the model. _g is generator and _d is discriminator(default: model)')
    parser.add_argument('--load_file', type=str, default='model', help='File name to load the model (default: model)')

    args = parser.parse_args()

    setup_logger(None)
    log = get_logger('main')
    log.info(f'MODEL: {args.model}')
    log.info(f'MODE: {args.mode}')
    log.info(f'Save model to {args.save_file}' if args.mode == 'train' else f'Load model from {args.load_file}')

    # Hyperparameters and other settings
    IMG_SIZE = 12
    PIXEL_LEVELS = 16
    LEARNING_RATE = 1e-3
    EPOCH = 500
    LATENT_SIZE = 32
    HIDDEN_DIM = 128
    BATCH_SIZE = 128
    OUT_SHAPE = (-1, 1, IMG_SIZE, IMG_SIZE)

    plot_func = lambda x: plot_imgs(x, IMG_SIZE, (4, 4))

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"PyTorch Device is {device}")

    # Dataset
    def collate_fn(batch):
        return torch.stack(batch).view(-1, 1, IMG_SIZE, IMG_SIZE).float() / float(PIXEL_LEVELS - 1)

    train_dataset = get_digit_dataset(IMG_SIZE, PIXEL_LEVELS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


    # Model
    G, D = get_gan_model(LATENT_SIZE, HIDDEN_DIM, OUT_SHAPE)
    optimizer_D = torch.optim.AdamW(D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))
    optimizer_G = torch.optim.AdamW(G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))

    if args.mode == 'train':
        args.save_file = args.save_file + '_{}.pt'
        # Model

        # Training
        log.info('Starting training...')
        d_losses, g_losses = train_wgan_gp(os.path.join(result_dir, args.save_file), G, D, LATENT_SIZE, optimizer_D, optimizer_G, train_loader, EPOCH, device)
        log.info('Training completed...')
        
        # Plot losses
        plt.figure(figsize=(10,4))
        plt.title('Generator and Discriminator losses ')
        plt.plot(np.convolve(g_losses, np.ones((100,))/100, mode='valid'), label='G')
        plt.plot(np.convolve(d_losses, np.ones((100,))/100, mode='valid'), label='D')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()


    elif args.mode == 'inference':
        args.load_file = args.load_file + '_{}.pt'

        # Load saved models
        G = torch.load(os.path.join(result_dir, args.load_file.format('g')))
        D = torch.load(os.path.join(result_dir, args.load_file.format('d')))

        # Generate samples
        imgs, _ = sample_gan(G, D, 16, LATENT_SIZE, device)
        plot_imgs(imgs, IMG_SIZE, (4,4), save_file= f"./images/gan_{args.model}_digit.png")


