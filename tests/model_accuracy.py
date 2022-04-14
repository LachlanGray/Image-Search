import argparse
import logging
import sys
import torch

from imagesearch.models import ImageEncoder
from imagesearch.dataset import CIFAR_LABELS, load_cifar10
from training.train_loop import train

def train_model(train_ds, test_ds, n_samples, n_epochs, device):
    train_ds = TripletDataset(train_dic, device=device)
    test_ds = TripletDataset(test_dic, device=device)

    logging.info("training. epochs={} samples={}".format(n_epochs, n_samples))
    train(train_ds, test_ds, n_samples, n_epochs, model_path, device)
    logging.info("done training")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-device", dest="train_device", type=str, default='cpu')
    parser.add_argument("--test-device", dest="test_device", type=str, default='cpu')
    parser.add_argument("--train-samples", dest="train_samples", type=int, default=50000)
    parser.add_argument("--epochs", dest="epochs", type=int, default=20)
    parser.add_argument("--test-samples", dest="test_samples", type=int, default=500)
    parser.add_argument("--min-k", dest="min_k", type=int, default=5)
    parser.add_argument("--max-k", dest="max_k", type=int, default=10)
    parser.add_argument("--k-step", dest="k_step", type=int, default=1)
    parser.add_argument("--output", dest="output", type=str, default='acc.png')
    args = vars(parser.parse_args(sys.argv[1:]))

    train_device = torch.device(args['train_device'])
    test_device = torch.device(args['test_device'])
    n_train_samples = args['train_samples']
    n_test_samples = args['test_samples']
    n_epochs = args['epochs']
    min_k = args['min_k']
    max_k = args['max_k']
    k_step = args['k_step']

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    train_ds, test_ds = load_cifar10()

    for k in range(min_k, max_k+1, k_step):
        net = train_model(train_ds, test_ds)

        # TODO: Sample the dataset.

        # TODO: Evaluate the model.

        # TODO: Add a data point.
    
    # TODO: Plot results.