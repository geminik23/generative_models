import torch
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
from torch.utils.data import DataLoader, random_split

# 
from flow_based.realnvp import RealNVP
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
    parser = argparse.ArgumentParser(description='Flow-based Generative Model')
    parser.add_argument('mode', choices=['train', 'inference'], help='Choose mode: train or inference')
    parser.add_argument('--model', type=str, default='realnvp', choices=['realnvp', 'glow'], help='Select the model to use (default: realnvp)')
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
    LATENT_DIM = IMG_SIZE * IMG_SIZE
    PIXEL_LEVELS = 16
    HIDDEN_DIM = 256
    LEARNING_RATE = 1e-3
    EPOCH = 500
    BATCH_SIZE = 64
    N_BLOCKS = 12
    MAX_PATIENCE = 20


    plot_func = lambda x: plot_imgs(x, IMG_SIZE, (4, 4))
    if args.plot_interval < 0:
        plot_func = None
        args.plot_interval = 1
        
    is_2d = args.model == 'glow'

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"PyTorch Device is {device}")

    # Dataset
    dataset = get_digit_dataset(IMG_SIZE, PIXEL_LEVELS, flatten=not(is_2d))

    val_len = int(len(dataset)*0.05)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE) 

    # Model
    if args.model == 'realnvp':
        LEARNING_RATE = 1e-3
        model = RealNVP(LATENT_DIM, HIDDEN_DIM, N_BLOCKS)
    else: # for pixelcnn
        # N_FILTERS = 128
        # LEARNING_RATE = 5e-4
        # model = PixelCNN(IMG_SIZE, n_filters=N_FILTERS, residual_blocks=5, kernel_size=KERNEL_SIZE, pixel_levels=PIXEL_LEVELS).to(device)
        model = RealNVP(LATENT_DIM, HIDDEN_DIM, N_BLOCKS)

    if args.mode == 'train':
        args.save_file = args.save_file + '_{}.pt'
        # Model
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
        args.load_file = args.load_file + '.pt'

        # Load saved model
        model = torch.load(os.path.join(result_dir, args.load_file))

        # Generate images
        imgs = model.inference(16, device).detach().cpu()
        plot_imgs(imgs, IMG_SIZE, (4,4), save_file= f"./images/flow_{args.model}_digit.png")


