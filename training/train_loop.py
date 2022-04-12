'''
Main training loop

'''
import torch
import os
import logging
from torch.utils.data import DataLoader

from imagesearch import LoggingHandler
from imagesearch.datasets import download_cifar10, load_cifar10, TripletDataset, RandomSubsetSampler
from imagesearch.models import ImageEncoder

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info("Device used: {}".format(device))

#### Download CIFAR10 dataset ####
dataset_dir = "./datasets/cifar-10-batches-py"
if not os.path.isdir(dataset_dir):
    download_cifar10()

#### Load CIFAR10 dataset ####
train_dic, test_dic = load_cifar10()

train_ds = TripletDataset(train_dic, device=device)
test_ds = TripletDataset(test_dic, device=device)

n_samples = 50000
n_epochs = 20  # total number of epochs

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    sampler=RandomSubsetSampler(len(train_ds), n_samples)
)
train_loader

net = ImageEncoder()
net.to(device)

optimizer = torch.optim.Adam(net.parameters())

loss_fn = torch.nn.TripletMarginLoss()


def train_one_epoch():
    for epoch in range(n_epochs):
        for i, data in enumerate(train_loader):
            a, p, n = data

            optimizer.zero_grad()

            z_a = net(a)
            z_p = net(p)
            z_n = net(n)

            loss = loss_fn(z_a, z_p, z_n)
            loss.backward()

            optimizer.step()

            # if i % 100 == 99:
            print(f'epoch {i}: {loss}')


train_one_epoch()