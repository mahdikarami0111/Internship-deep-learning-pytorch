import torch
from WBSDataset import WBSDataset
from torch import nn
from Trainer import Trainer
import torchvision
from Tester import Tester
import torchvision.transforms as transforms
from Normalizer import Normalizer
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import pickle
from PIL import Image
from torch.optim import lr_scheduler
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainset = datasets.ImageFolder('data/Train', transform=transforms.PILToTensor())
batch_size = 32
loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
normalizer = Normalizer(loader)
mean, std = normalizer.get_mean_std()


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomAffine(degrees=(-45, 45), translate=(0.08, 0.08), scale=(0.7, 1.3)),
    transforms.ColorJitter(brightness=0.3, hue=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=mean, std=std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=mean, std=std),
    ])
#
EPOCHS = 20
BATCH_SIZE = 32
#
trainset = WBSDataset(root='data/Train', train=True, device=device, transform=transform_train)
testset = WBSDataset(root='data/Test-B', train=True, device=device, transform=transform_test)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

model = torchvision.models.resnet18().to(device)
fc = model.fc
in_features = fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5, inplace=True),
    nn.Linear(in_features=in_features, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=5)
)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

trainer = Trainer(model, trainloader, testloader, loss_fn, optimizer, scheduler)
train_loss, test_loss = trainer.train(EPOCHS, device, print_loss=True, print_interval=1)

train_loss = torch.Tensor(train_loss).to('cpu')
test_loss = torch.Tensor(test_loss).to('cpu')

plt.plot(range(len(train_loss)), train_loss, label='Train loss', color='blue')
plt.plot(range(len(test_loss)), test_loss, label='Test_loss', color='red')
plt.show()

torch.save(model.state_dict(), 'ResNet18_ELR_Aug4.pt')

model = torchvision.models.resnet18().to(device)
fc = model.fc
in_features = fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5, inplace=True),
    nn.Linear(in_features=in_features, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=5)
).to(device)
model.load_state_dict(torch.load('ResNet18_ELR_Aug4.pt'))
model.to(device)

tester = Tester(testloader, model, ['B', 'E', 'L', 'M', 'N'])
cf_matrix = tester.test(device)
eval = tester.evaluate_model(cf_matrix)
tester.parametric_evaluation(eval)


