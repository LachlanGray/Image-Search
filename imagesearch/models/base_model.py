'''
A base network class that implements whatever will be shared across all of
our experiment models.

'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(torch.nn.Module):

    def __init__(self, output_vector_size=10):
        super(ImageEncoder, self).__init__()
        self.output_vector_size = output_vector_size
        self.conv1 = nn.Conv2d(3, 8, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, padding='same')
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, output_vector_size)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.bn1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.bn2(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = self.bn3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x
    
    def get_output_vector_size(self):
        return self.output_vector_size

class ImageDecoder(torch.nn.module):

    def __init__(self, input_vector_size=10, output_shape=(3, 32, 32)):
        self.input_vector_size = input_vector_size
        self.output_vector_size = output_vector_size = np.prod(output_shape)
        self.fc1 = nn.Linear(input_vector_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, output_vector_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x

class AutoEncoder(torch.nn.module):

    def __init__(self, output_vector_size=10, output_shape=(3, 32, 32)):
        self.enc = ImageEncoder(output_vector_size=output_vector_size)
        self.dec = ImageDecoder(input_vector_size=output_vector_size, output_shape=output_shape)
    
    def forward(self, x):
        y = self.enc(x)
        y = self.dec(y)

        return y

def load_model(model_path, device=None):
    checkpoint = torch.load(model_path)
    net = ImageEncoder()
    if device:
        net = net.to(device)
    net.load_state_dict(checkpoint['model'])
    return net