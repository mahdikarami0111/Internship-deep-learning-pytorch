import numpy as np
from layers.fullyconnected import FC
from activations import*
from model import Model
from optimizers.gradientdescent import GD
from PIL import Image
from losses.binarycrossentropy import BinaryCrossEntropy
import glob

images_2 = glob.glob('datasets/MNIST/2/*.jpg')
images_5 = glob.glob('datasets/MNIST/5/*.jpg')

list_2 = np.array([np.array(Image.open(image)) for image in images_2])
list_5 = np.array([np.array(Image.open(image)) for image in images_5])

list_2 = list_2.reshape(list_2.shape[0], -1).T/255
list_5 = list_5.reshape(list_5.shape[0], -1).T/255
# print(list_2.shape)
# print(list_5.shape)
X_train = np.concatenate((list_2[:,0:700], list_5[:,0:700]), axis=1)
X_test = np.concatenate((list_2[:,700:1000], list_5[:,700:1000]), axis=1)
# print(X_train.shape)
# print(X_test.shape)
Y_train = np.concatenate((np.zeros((1, 700)), np.ones((1, 700))), axis=1)
Y_test = np.concatenate((np.zeros((1, 300)), np.ones((1, 300))), axis=1)
arch = {
    'fc1': FC(784, 16, 'fc1'),
    'ac1': ReLU(),
    'fc2': FC(16, 16, 'fc2'),
    'ac2': ReLU(),
    'fc3': FC(16, 1, 'fc3'),
    'ac3': Sigmoid()
}
opt = GD(arch, 0.3)
bce = BinaryCrossEntropy()
model = Model(arch, bce, opt)

model.train(X_train, Y_train, X_test, Y_test, 100, batch_size=64, shuffling=False, verbose=10)

prediction = np.floor(2 * model.predict(X_test))



