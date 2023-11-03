import os
from pathlib import Path

from matplotlib import font_manager
from typing import Optional
# import dotenv

# dotenv.load_dotenv()

DATASET_PATH = os.environ.get('DATASET_PATH')
if DATASET_PATH is None:
    DATASET_PATH = './data'
    os.makedirs(DATASET_PATH, exist_ok=True)


## ===== logger
import logging

def setup_logger(name:Optional[str]):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter( logging.Formatter('%(asctime)s [%(name)s] - %(levelname)s - %(message)s'))

    logger.addHandler(sh)
    return logger

def get_logger(name:str):
    return logging.getLogger(name)


# setup_logger("test")
# get_logger("test").info("Hello")


## ===== visualize
import matplotlib.pyplot as plt

## plot image 
def plot_imgs(imgs, img_size, size, scores=None, save_file=None):
    imgs = imgs.view(-1, img_size, img_size)
    f, axarr = plt.subplots(size[0], size[1], figsize=(10, 10))
    for i in range(size[1]):
        for j in range(size[0]):
            idx = i*size[0]+j
            axarr[i, j].imshow(imgs[idx,:].numpy(), cmap='gray')
            axarr[i, j].set_axis_off()

            # only for discriminator
            if scores is not None:
                axarr[i,j].text(0.0, 0.5, str(round(scores[idx], 2)), dict(size=20, color='red'))

    if save_file is not None:
        plt.savefig(save_file)
    plt.show()




