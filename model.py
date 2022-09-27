import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam

import torchvision
import torch.nn as nn
from torch.autograd import Variable

from CONSTS import *
from helpers import *
from Network import Network


class Classifier():
    def __init__(self):
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.network = Network().to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.network.parameters(),
                              lr=0.001, weight_decay=0.0001)

        self.batch_size = 10
        self.number_of_labels = 10

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')

        self.train_loader = None
        self.test_loader = None

    def set_train_test_loaders(self):
        train_loader, test_laoder = load_ball_images(self.batch_size)
        self.train_loader = train_loader
        self.test_loader = test_laoder

    def train(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            for i, (images, labels) in enumerate(self.train_loader, 0):

                images = Variable(images.to(self.device))
                labels = Variable(labels.to(self.device))

                self.optimizer.zero_grad()
                outputs = self.network(images)

                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

            accuracy = self.test_accuracy()
            print('For epoch', epoch,
                  'the test accuracy over the whole test set is %d %%' % (accuracy))

    def test_accuracy(self):
        self.network.eval()
        accuracy = 0.0
        total = 0.0

        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data

                outputs = self.network(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()

        accuracy = (100 * accuracy / total)
        return (accuracy)


if __name__ == "__main__":

    classifier = Classifier()
    classifier.set_train_test_loaders()

    classifier.train(50)
    print('Finished Training')

    save_model(classifier.network)
