'''
Functions relevant to loading/processing the dataset

'''

import torch
from torch.utils.data import Dataset, Sampler

import pickle


def unpickle(file):
    with open(file, 'rb') as f:
        dic = pickle.load(f, encoding='latin1')

    return dic


def load_cifar10(dir='./datasets/cifar-10-batches-py'):
    '''
    load cifar 10 into a train and test dict consisting of {label:[examples]}.

    download cifar 10 from https://www.cs.toronto.edu/~kriz/cifar.html
    and unzip it into datasets folder.
    '''
    train = {i: [] for i in range(10)}
    file_names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    for file in file_names:
        data = unpickle(dir + '/' + file)

        for label, img in zip(data['labels'], data['data']):
            train[label].append(img.reshape(3, 32, 32))

    test = {i: [] for i in range(10)}
    data = unpickle(dir + '/test_batch')
    for label, img in zip(data['labels'], data['data']):
        test[label].append(img.reshape(3, 32, 32))

    return train, test


class TripletDataset(Dataset):
    '''
    Custom dataset for producing (anchor, positive, negative) triplets.

    Stores all of the examples by class, allows the retrieval of all possible
    triplets using one index.

    Initialize with a dicitonary {label:[examples]}.

    '''

    def __init__(self, class_dic, device=None):
        '''
        class_dic is a dictionary containing {class:[examples]}, where each
        class has an equal number of exmaples.
        '''
        class_size = len(next(iter(class_dic.values())))
        n_classes = len(class_dic)
        if not all(class_size == len(x) for x in class_dic.values()):
            raise Exception("All classes must have the same number of examples")

        self.class_size = class_size
        self.n_classes = n_classes
        self.num_triplets = class_size**2 * (n_classes - 1) * class_size * n_classes

        # examples = [x for x in class_dic.values()]
        self.examples = [[] for _ in range(n_classes)]
        for i, X in enumerate(class_dic.values()):
            for x in X:
                self.examples[i].append((torch.from_numpy(x) / 255).to(device))

        # self.examples = [torch.from_numpy(x).to(device) for x in class_dic.values()]

    def __getitem__(self, idx):
        anchor_class = idx % self.n_classes
        idx = int(idx / self.n_classes)

        a = idx % self.class_size
        idx = int(idx / self.class_size)

        p = idx % self.class_size
        idx = int(idx / self.class_size)

        negative_class = idx % (self.n_classes - 1)
        # negative class can't be same as anchor class
        if negative_class >= anchor_class:
            negative_class += 1

        idx = int(idx / (self.n_classes - 1))

        n = idx % self.class_size

        return self.examples[anchor_class][a], self.examples[anchor_class][p], self.examples[negative_class][n]

    def __len__(self):
        return self.n_classes * (self.n_classes - 1) * self.class_size**3


class RandomSubsetSampler(Sampler):
    """
    Yields sample_size random indices between 0 and dataset_size.

    Useful when there are way too many indeces to iterate over in one epoch.
    """

    def __init__(self, dataset_size, sample_size):
        self.sample_size = sample_size
        self.dataset_size = dataset_size

    def __iter__(self):
        indices = torch.randint(0, self.dataset_size, (self.sample_size, ))
        for i in torch.randperm(self.sample_size):
            yield indices[i]

    def __len__(self):
        return self.sample_size
