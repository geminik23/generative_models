from common.dataset import get_digit_dataset, get_fashion_dataset
from common.utils import plot_imgs

import torch
import numpy as np

from typing import Optional


def plot_image_from(dataset, img_size, save_file:Optional[str]=None):
    # randomly select 16 image
    imgs = np.stack([dataset[i] for i in range(16)])
    plot_imgs(torch.tensor(imgs),img_size, (4,4), save_file=save_file)


if __name__ == '__main__':
    img_size = 16
    num_classes = 16

    ## digit dataset
    dataset = get_digit_dataset(img_size, num_classes)
    plot_image_from(dataset, img_size, "./images/original_digit.png")

    ## fashion dataset
    dataset = get_fashion_dataset(img_size, num_classes)
    plot_image_from(dataset, img_size, "./images/original_fashion.png")



