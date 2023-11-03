import torch
from torch.utils.data import DataLoader, random_split
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from common import train_gen_network, get_digit_dataset, get_fashion_dataset
from arm_model import AutoregressiveModel, create_network
from common import plot_digit_imgs


import os
result_dir = 'results'
if not(os.path.exists(result_dir)):
    os.mkdir(result_dir)



##


## 
# Model
model = AutoregressiveModel(create_network(HIDDEN_DIM, NUM_CLASS, KERNEL_SIZE, 3), DIM, NUM_CLASS).to(device)

optimizer = torch.optim.Adamax(model.parameters(), LEARNING_RATE)


## TRAIN DATA
results = train_gen_network(os.path.join(result_dir,"arm_e_{}.pt"), model, MAX_PATIENCE, train_loader, val_loader, EPOCH, device, optimizer=optimizer, plot_func=plot_func, plot_img_interval=10)
results = pd.DataFrame(results)

## plot the losses
sns.lineplot(x="epoch", y="val loss", data=results)
plt.show()

## LOAD DATA
model = torch.load(os.path.join(result_dir, 'arm_e_38.pt'))

## GENERATE THE IMGS
imgs = model.inference(16, device).detach().cpu()
plot_digit_imgs(imgs, IMG_SIZE, (4,4))


