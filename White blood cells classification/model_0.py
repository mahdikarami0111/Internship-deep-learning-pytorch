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

# trainset = datasets.ImageFolder('data/Train_u', transform=transforms.PILToTensor())
# batch_size = 32
# loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
# normalizer = Normalizer(loader)
# mean, std = normalizer.get_mean_std()


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=(-45, 45), translate=(0.08, 0.08), scale=(0.7, 1.3)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((575, 575))
    ])
#
EPOCHS = 5
BATCH_SIZE = 16
#
trainset = WBSDataset(root='data/Train_u', train=True, device=device, transform=transform_train)
testset = WBSDataset(root='data/TestA_u', train=True, device=device, transform=transform_test)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

model = torchvision.models.resnet18().to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1)

trainer = Trainer(model, trainloader, testloader, loss_fn, optimizer, scheduler)
train_loss, test_loss = trainer.train(EPOCHS, device, print_loss=True, print_interval=1)

train_loss = torch.Tensor(train_loss).to('cpu')
test_loss = torch.Tensor(test_loss).to('cpu')

plt.plot(range(len(train_loss)), train_loss, label='Train loss', color='blue')
plt.plot(range(len(test_loss)), test_loss, label='Test_loss', color='red')
plt.show()

torch.save(model.state_dict(), 'ResNet18_ELR0.pt')

model = torchvision.models.resnet18().to(device)
model.load_state_dict(torch.load('ResNet18_ELR0.pt'))
model.to(device)

tester = Tester(testloader, model, ['B', 'E', 'L', 'M', 'N'])
cf_matrix = tester.test(device)
eval = tester.evaluate_model(cf_matrix)
tester.parametric_evaluation(eval)


