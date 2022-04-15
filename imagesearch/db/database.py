import argparse
import logging
import random
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
        self.index, self.imgs, self.labels = self.encode_images()

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
        index = []
        imgs = []
        labels = []

        for label in self.dataset:
            for img in self.dataset[label]:
                index.append(self.encode_image(img))
                imgs.append(torch.tensor(img))
                labels.append(label)

        index = torch.stack(index).to(self.device)
        imgs = torch.stack(imgs).to(self.device)
        labels = torch.tensor(labels).long().to(self.device)

        return index, imgs, labels

    def __len__(self):
        return self.index.shape[0]

    def search_by_score(self, img, k=0):
        '''
            Search for k similar images by score.
            Images are scored by their cosine similarity to the search image.
            For images with score 1.0 (vector in the same direction as the search image),
            we add the reciprocal of the Euclidean distance.

            img - input image
        '''
        results = []
        enc = self.encode_image(img).to(self.device)
        db_size = self.__len__()

        for i in range(db_size):
            db_enc = self.index[i]
            db_img = self.imgs[i]
            label = self.labels[i].item()
            sim = F.cosine_similarity(enc, db_enc, dim=0).item()
            if round(sim) == 1:
                sim += 1/max(1e-8, torch.norm(db_enc-enc).item())
            if k > 0:
                if len(results) == k:
                    j = 0
                    for i in range(1,k):
                        if results[i][2] < results[j][2]:
                            j = i
                    if sim > results[j][2]:
                        results[j] = (db_img, label, sim)
                else:
                    results.append((db_img, label, sim))
            else:
                results.append((db_img, label, sim))

        results.sort(reverse=True, key=lambda x: x[2])

        return results

    def search_by_distance(self, img, k=0):
        results = []
        enc = self.encode_image(img).to(self.device)
        db_size = self.__len__()

        for i in range(db_size):
            db_enc = self.index[i]
            db_img = self.imgs[i]
            label = self.labels[i].item()
            dist = torch.norm(db_enc-enc).item()
            if k > 0:
                if len(results) == k:
                    j = 0
                    for i in range(1,k):
                        if results[i][2] > results[j][2]:
                            j = i
                    if dist < results[j][2]:
                        results[j] = (db_img, label, dist)
                else:
                    results.append((db_img, label, dist))
            else:
                results.append((db_img, label, dist))

        results.sort(key=lambda x: x[2])

        return results

    def evaluate(self, test_ds, n_samples, k, by_score=True):
        '''
        Evaluate the encoder by sample n_samples images from each class in test_ds.
        Return the mean accuracy where accuracy is the proportion of images returned
        by the search in the same class.
        '''
        test_imgs = []
        acc = 0

        for label in test_ds:
            sample = random.sample(test_ds[label], n_samples)
            for test_img in sample:
                test_imgs.append((test_img, label))
        
        for test_img, label in test_imgs:
            if by_score:
                results = self.search_by_score(test_img, k)
            else:
                results = self.search_by_distance(test_img, k)
            
            acc += len(list(filter(lambda x: x[1] == label, results)))/k
        
        return acc/len(test_imgs)


if __name__ == '__main__':
    from imagesearch.models import ImageEncoder
    from imagesearch.dataset import CIFAR_LABELS, load_cifar10

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", dest="device", type=str, default='cpu')
    parser.add_argument("--model", dest="model_path", type=str)
    parser.add_argument("--distance", dest="search_by_distance", action='store_true')
    parser.add_argument("--score", dest="search_by_score", action='store_true')
    parser.add_argument("--label", dest="label", type=int, default=0)
    parser.add_argument("--index", dest="index", type=int, default=0)
    parser.add_argument("--k", dest="k", type=int, default=5)
    parser.add_argument("--evaluate-samples", dest="evaluate_samples", type=int, default=0)
    parser.add_argument("--output", dest="output", type=str, default="search-results.png")
    args = vars(parser.parse_args(sys.argv[1:]))
    model_path = args['model_path']
    k = args['k']

    device = torch.device(args['device'])
    logging.info("Device used: {}".format(device))

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

    evaluate_samples = args['evaluate_samples']
    if evaluate_samples > 0:
        if args['search_by_score']:
            logging.info("search by score accuracy = {:.2f}".format(100*db.evaluate(test, evaluate_samples, k)))
        else:
            logging.info("search by distance accuracy = {:.2f}".format(100*db.evaluate(test, evaluate_samples, k, by_score=False)))
    else:
        logging.info("searching for k={} similar images to image (label={}, index={}) in test".format(k, CIFAR_LABELS[args['label']], args['index']))
        search_img = test[args['label']][args['index']]
        if args['search_by_score']:
            search_results = db.search_by_score(search_img, k)
        else:
            search_results = db.search_by_distance(search_img, k)
        logging.info("search returned {} results".format(len(search_results)))

        import matplotlib.pyplot as plt

        k = len(search_results)
        if k > 0:
            plt.subplots(1, k+1, figsize=(11,3), dpi=300)
            plt.subplot(1, k+1, 1)
            plt.imshow(search_img.reshape(32, 32, 3))
            plt.title("{}".format(CIFAR_LABELS[args['label']]))
            for i in range(k):
                plt.subplot(1, k+1, i+2)
                result_img, label, d = search_results[i]
                plt.imshow(result_img.cpu().reshape(32, 32, 3))
                if args['search_by_score']:
                    plt.title("{}\nscore={:.2f}".format(CIFAR_LABELS[label], d))
                else:
                    plt.title("{}\ndist={:.2f}".format(CIFAR_LABELS[label], d))
            plt.tight_layout()
            plt.savefig(args['output'])