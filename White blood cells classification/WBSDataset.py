import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from torch.optim import lr_scheduler
import glob


class WBSDataset(Dataset):
    def __init__(self, root, train, device, transform=None):
        self.root = root
        self.train = train
        self.device = device
        self.transform = transform
        self.data, self.labels = self.load_data()

    def load_data(self):
        image_classes = []
        image_classes.append(glob.glob(self.root + '/Basophil/*.jpg'))
        image_classes.append(glob.glob(self.root + '/Eosinophil/*.jpg'))
        image_classes.append(glob.glob(self.root + '/Lymphocyte/*.jpg'))
        image_classes.append(glob.glob(self.root + '/Monocyte/*.jpg'))
        image_classes.append(glob.glob(self.root + '/Neutrophil/*.jpg'))

        labels = None
        for i in range(len(image_classes)):
            images = image_classes[i]
            temp = torch.ones((len(images, ))) * i

            if labels is None:
                labels = temp
            else:
                labels = torch.cat((labels, temp), dim=0)
        data = None
        for images in image_classes:
            if data is None:
                data = images
            else:
                data += images
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = Image.open(self.data[item])
        label = self.labels[item]
        if self.transform:
            image = self.transform(image)
        return image, label

