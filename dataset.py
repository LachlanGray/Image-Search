'''
Functions relevant to loading/processing the dataset

'''

import pickle


def unpickle(file):
    with open(file, 'rb') as f:
        dic = pickle.load(f, encoding='latin1')

    return dic


def load_cifar10(dir='./datasets/cifar-10-batches-py'):
    '''
    load cifar 10 into a train and test dict consisting of {label:[examples]}.

    download cifar 10 from https://www.cs.toronto.edu/~kriz/cifar.html and unzip it into datasets folder.
    '''
    train = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    for file in train_files:
        data = unpickle(dir + '/' + file)

        for label, img in zip(data['labels'], data['data']):
            train[label].append(img.reshape(3, 32, 32))

    test = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    data = unpickle(dir + '/test_batch')
    for label, img in zip(data['labels'], data['data']):
        test[label].append(img.reshape(3, 32, 32))

    return train, test
