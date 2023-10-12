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
from sklearn.metrics import confusion_matrix


class Tester:
    def __init__(self, testloader, model, class_names):
        self.loader = testloader
        self.model = model
        self.class_names = class_names

    def initialize_eval(self):
        evaluation = []

        for i in range(len(self.class_names)):
            evaluation.append({
                'TP': 0,
                'TF': 0,
                'FP': 0,
                'FM': 0,
            })
        return evaluation

    def test(self, device):
        self.model.eval()
        with torch.inference_mode():
            Y = None
            Y_preds = None
            for batch_number, (X, y) in enumerate(self.loader):
                X, y = X.to(device), y.to(device)
                X, y = X.type(torch.cuda.FloatTensor), y.type(torch.int64)

                out_raw = self.model(X)
                out_preds = torch.softmax(out_raw, dim=1).argmax(dim=1)

                y = y.to('cpu').numpy()
                out_preds = out_preds.to('cpu').numpy()

                if Y is None:
                    Y = y
                else:
                    Y = np.concatenate((Y, y))

                if Y_preds is None:
                    Y_preds = out_preds
                else:
                    Y_preds = np.concatenate((Y_preds, out_preds))

            Y = np.concatenate((Y, np.array([0, 1, 2, 3, 4])))
            Y_preds = np.concatenate((Y_preds, np.array([0, 1, 2, 3, 4])))

            return confusion_matrix(Y, Y_preds)

    def evaluate_model(self, matrix):
        print(matrix)
        total = np.sum(matrix)
        eval = self.initialize_eval()
        for i in range(len(self.class_names)):
            tp = matrix[i, i]
            fn = np.sum(matrix[i, :]) - tp
            fp = np.sum(matrix[:, i]) - tp
            tn = total - tp - fn - fp
            eval[i]['TP'] = tp
            eval[i]['TN'] = tn
            eval[i]['FP'] = fp
            eval[i]['FN'] = fn

        return eval

    def parametric_evaluation(self, eval):
        for i, class_name in enumerate(self.class_names):
            tp = eval[i]['TP']
            tn = eval[i]['TN']
            fp = eval[i]['FP']
            fn = eval[i]['FN']
            precs = tp/(tp+fp)
            sens = tp/(tp+fn)
            print(class_name)
            print(f'Precision = {round(precs * 100, 2)}% | Sensitivity = {round(sens * 100, 2)}% | '
                  f'F1 = {round(200 * ((precs * sens) / (precs + sens)), 2)}% | '
                  f'Accuracy = {round(100 * ((tn+tp) / (tp+tn+fp+fn)), 2)}%')





