import argparse
import logging
import matplotlib.pyplot as plt
import sys
import torch

from imagesearch import LoggingHandler
from imagesearch.db import ImageDatabase
from imagesearch.models import ImageEncoder
from imagesearch.dataset import CIFAR_LABELS, load_cifar10, TripletDataset
from training.train_loop import train

def train_model(train_ds, test_ds, n_samples, n_epochs, output_vector_size, device):
    train_ds = TripletDataset(train_ds, device=device)
    test_ds = TripletDataset(test_ds, device=device)

    logging.info("training. epochs={} samples={} output-vector-size={}".format(n_epochs, n_samples, output_vector_size))
    net = train(train_ds, test_ds, n_samples, n_epochs, output_vector_size=output_vector_size, device=device)
    logging.info("done training")

    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-device", dest="train_device", type=str, default='cpu')
    parser.add_argument("--test-device", dest="test_device", type=str, default='cpu')
    parser.add_argument("--train-samples", dest="train_samples", type=int, default=50000)
    parser.add_argument("--epochs", dest="epochs", type=int, default=20)
    parser.add_argument("--test-samples", dest="test_samples", type=int, default=500)
    parser.add_argument("--k", dest="k", type=int, default=5)
    parser.add_argument("--min-vector-size", dest="min_vector_size", type=int, default=10)
    parser.add_argument("--max-vector-size", dest="max_vector_size", type=int, default=100)
    parser.add_argument("--vector-size-step", dest="vector_size_step", type=int, default=10)
    parser.add_argument("--output", dest="output", type=str, default='acc.png')
    args = vars(parser.parse_args(sys.argv[1:]))

    train_device = torch.device(args['train_device'])
    test_device = torch.device(args['test_device'])
    n_train_samples = args['train_samples']
    n_test_samples = args['test_samples']
    n_epochs = args['epochs']
    min_vector_size = args['min_vector_size']
    max_vector_size = args['max_vector_size']
    vector_size_step = args['vector_size_step']
    vector_size_range = range(min_vector_size, max_vector_size+1, vector_size_step)
    k = args['k']

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    logging.info("using training device: {}".format(train_device))
    logging.info("using test device: {}".format(test_device))

    logging.info("loading dataset")
    train_ds, test_ds = load_cifar10()
    logging.info("loaded dataset")

    score_accs = []
    dist_accs = []

    for output_vector_size in vector_size_range:
        net = train_model(train_ds, test_ds, n_train_samples, n_epochs, output_vector_size, train_device)
        net = net.to(test_device)

        logging.info("loading database")
        db = ImageDatabase(train_ds, net, test_device)
        logging.info("loaded database. size={}".format(len(db)))

        logging.info("evaluating model on search by score")
        score_acc = db.evaluate(test_ds, n_test_samples, k)
        logging.info("search by score accuracy: {:.2f}".format(100*score_acc))

        logging.info("evaluating model on search by distance")
        dist_acc = db.evaluate(test_ds, n_test_samples, k, by_score=False)
        logging.info("search by score distance: {:.2f}".format(100*dist_acc))

        score_accs.append(score_acc)
        dist_accs.append(dist_acc)
    
    plt.figure(figsize=(11, 8), dpi=300)
    plt.plot(vector_size_range, score_accs, label='score')
    plt.plot(vector_size_range, dist_accs, label='distance')
    plt.xlabel('Latent Vector Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(args['output'])