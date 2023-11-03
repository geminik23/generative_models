import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from typing import Optional

## ----------------------------------------------------
## dataset
class ResizedMNISTDataset(Dataset):
    def __init__(self, path, mnist, img_size:int, num_classes:Optional[int]=None, flatten:bool=True, transform=None):
        super().__init__()

        trans_list = [T.ToTensor(), T.Resize((img_size, img_size), antialias=True)]
        if flatten: trans_list.append(T.Lambda(lambda x: torch.flatten(x)))
        if num_classes is not None: trans_list.append(T.Lambda(lambda x: (x*(num_classes-1)).long()))
        if transform is not None: trans_list.append(transform)
        transforms = T.Compose(trans_list)
        datasets = mnist(path, train=True, download=True, transform=transforms)
        self.data = [d[0] for d in datasets]
    
    def __len__(self):
        # return len(self.data)
        return 2000 # limit the size of dataset

    def __getitem__(self, i):
        return self.data[i]

def get_fashion_dataset(img_size, num_classes:Optional[int]=None, flatten=True, transform=None, dataset_root_dir:str='./data'):
    return ResizedMNISTDataset(dataset_root_dir, torchvision.datasets.FashionMNIST, img_size, num_classes, flatten, transform)

def get_digit_dataset(img_size, num_classes:Optional[int]=None, flatten=True, transform=None, dataset_root_dir:str='./data'):
    return ResizedMNISTDataset(dataset_root_dir, torchvision.datasets.MNIST, img_size, num_classes, flatten, transform)

