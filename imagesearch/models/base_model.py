'''
A base network class that implements whatever will be shared across all of
our experiment models.

'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(torch.nn.Module):

    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def load_model(model_path, device=None):
    checkpoint = torch.load(model_path)
    net = ImageEncoder()
    if device:
        net.to(device)
    net.load_state_dict(checkpoint['model'])
    return net