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
from tqdm.auto import tqdm
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, trainloader, testloader, loss_fn, optimizer, scheduler=None):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, epochs, device, print_loss=False, print_interval=1, save=False):
        self.model = self.model.to(device)
        itr_loss_train = []
        itr_loss_test = []
        ratio = len(self.trainloader) / len(self.testloader)
        for epoch in tqdm(range(epochs)):
            self.model.train()
            train_loss = 0
            for batch_number, (X, y) in enumerate(self.trainloader):
                X, y = X.to(device), y.to(device)
                X, y = X.type(torch.cuda.FloatTensor), y.type(torch.int64)

                out_raw = self.model(X)
                out_preds = torch.softmax(out_raw, dim=1).argmax(dim=1)

                loss = self.loss_fn(out_raw, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss
                if batch_number % round(10*ratio) == 0:
                    itr_loss_train.append(loss)

            train_loss /= len(self.trainloader)
            if self.scheduler:
                self.scheduler.step()

            self.model.eval()
            with torch.inference_mode():
                test_loss = 0
                test_acc = 0

                for batch_number, (X, y) in enumerate(self.testloader):
                    X, y = X.to(device), y.to(device)
                    X, y = X.type(torch.cuda.FloatTensor), y.type(torch.int64)

                    out_raw = self.model(X)
                    out_preds = torch.softmax(out_raw, dim=1).argmax(dim=1)

                    loss = self.loss_fn(out_raw, y)
                    test_loss += loss
                    test_acc += torch.sum(out_preds == y) / len(y)
                    if batch_number % 10 == 0:
                        itr_loss_test.append(loss)

                test_loss /= len(self.testloader)
                test_acc /= len(self.testloader)

            if epoch % print_interval == 0 and print_loss:
                print(f'Train loss = {train_loss} | Test loss = {test_loss} | Test accuracy = {test_acc}')
        return itr_loss_train, itr_loss_test
