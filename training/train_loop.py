'''
Main training loop

'''
import argparse
import torch
import os
import logging
from torch.utils.data import DataLoader
import sys

from imagesearch import LoggingHandler
# from imagesearch.dataset import download_cifar10, load_cifar10, TripletDataset, RandomSubsetSampler
from imagesearch.dataset import load_cifar10, TripletDataset, RandomSubsetSampler
from imagesearch.models import ImageEncoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', dest='samples', type=int, default=50000)
    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    args = vars(parser.parse_args(sys.argv))

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Device used: {}".format(device))

    #### Download CIFAR10 dataset ####
    dataset_dir = "./datasets/cifar-10-batches-py"
    # if not os.path.isdir(dataset_dir):
    #     download_cifar10()

    #### Load CIFAR10 dataset ####
    train_dic, test_dic = load_cifar10()

    train_ds = TripletDataset(train_dic, device=device)
    test_ds = TripletDataset(test_dic, device=device)

    n_samples = args['samples']
    n_epochs = args['epochs']  # total number of epochs

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

    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0

        for i, data in enumerate(train_loader):
            a, p, n = data

            optimizer.zero_grad()

            z_a = net(a)
            z_p = net(p)
            z_n = net(n)

            loss = loss_fn(z_a, z_p, z_n)
            loss.backward()

            optimizer.step()
            total_loss += loss
            n_batches += 1
        
        logging.info("epoch: {} loss: {:.2f}".format(epoch, total_loss/n_batches))


    # train_one_epoch()