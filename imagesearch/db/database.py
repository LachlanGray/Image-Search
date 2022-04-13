import argparse
import logging
from imagesearch import LoggingHandler
from imagesearch.models import ImageEncoder, load_model
import os
import sys
import torch
import torch.nn.functional as F 

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def default_similarity(x1,x2):
    return F.cosine_similarity(x1, x2, dim=0)

class ImageDatabase (object):
    '''
       Database containing images indexed by latent vector.
    '''

    def __init__(self, dataset, encoder, device=None):
        '''
            dataset - tensor or list of examples ?
            encoder - ?
        '''
        self.dataset = dataset
        self.encoder = encoder
        if device is not None:
            self.device = device
        else:
            self.device = get_device()
        self.db = self.encode_images()

    def encode_image(self, img):
        n = len(img.shape)
        img = torch.FloatTensor(img / 255).to(self.device)

        if n == 3:
            img = img.unsqueeze(0)
        elif n == 4:
            raise Exception('Invalid image')

        enc = self.encoder.forward(img)
        return enc.squeeze()

    def encode_images(self):
        db = []

        for label in self.dataset:
            for img in self.dataset[label]:
                db.append((self.encode_image(img), img, label))

        return db

    def __len__(self):
        return len(self.db)

    def search(self, img, k=0, similarity=default_similarity, min_sim=-1.0, max_sim=1.0):
        '''
            Search for k similar images according to user-provided similarity
            measure.

            img - input image
        '''
        results = []
        n = 0
        enc = self.encode_image(img)

        for db_enc, db_img, label in self.db:
            if k > 0 and n == k:
                return results
            sim = similarity(enc, db_enc)
            if sim >= min_sim and sim <= max_sim:
                results.append((db_img, label, sim))
                n += 1

        return results

if __name__ == '__main__':
    from imagesearch.models import ImageEncoder
    from imagesearch.dataset import CIFAR_LABELS, load_cifar10

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    
    device = get_device()
    logging.info("Device used: {}".format(device))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model_path", type=str)
    parser.add_argument("--label", dest="label", type=int, default=0)
    parser.add_argument("--index", dest="index", type=int, default=0)
    parser.add_argument("--k", dest="k", type=int, default=5)
    parser.add_argument("--min-sim", dest="min_sim", type=float, default=-1.0)
    parser.add_argument("--max-sim", dest="max_sim", type=float, default=1.0)
    args = vars(parser.parse_args(sys.argv[1:]))
    model_path = args['model_path']
    k = args['k']
    min_sim = args['min_sim']
    max_sim = args['max_sim']

    train, test = load_cifar10()
    if model_path:
        logging.info("loading model from {}".format(os.path.abspath(model_path)))
        net = load_model(model_path, device)
        logging.info("loaded model")
    else:
        net = ImageEncoder()

    logging.info("loading database")
    db = ImageDatabase(train, net, device)
    logging.info("loaded database. size={}".format(len(db)))

    logging.info("searching for k={} similar images to image (label={}, index={}) in test".format(k, CIFAR_LABELS[args['label']], args['index']))
    search_img = test[args['label']][args['index']]
    search_results = db.search(search_img, k, min_sim=min_sim, max_sim=max_sim)
    logging.info("search returned {} results".format(len(search_results)))

    k = len(search_results)
    if k > 0:
        pass