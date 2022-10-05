import torch
from torch.optim import Adam
import torch.nn as nn
from torch.autograd import Variable

from CONSTS import *
from helpers import *
from Network import Network


class Classifier():
    def __init__(self, lr, weight_decay, batch_size, num_of_classes):
        """
        Init classifier that uses CrossEntropyLoss, Adam and the Network in Network.py
        """
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # TODO
        self.device = torch.device("cpu")

        self.network = Network(num_of_classes).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.network.parameters(),
                              lr=lr, weight_decay=weight_decay)

        self.batch_size = batch_size

        self.train_data = None
        self.test_data = None

    def load_train_test_data(self, classes):
        """
        Load train and test data from the ball images from the provided classes.
        """
        train_data, test_data = load_ball_images(self.batch_size, classes)
        self.train_data = train_data
        self.test_data = test_data

    def train(self, num_epochs):
        """
        Train the model with the train_data for num_epochs epochs.
        """
        for epoch in range(1, num_epochs + 1):
            for i, (images, labels) in enumerate(self.train_data):
                # The length of images and labels are according to the batch size for the classifier.

                images = Variable(images.to(self.device))
                labels = Variable(labels.to(self.device))

                self.optimizer.zero_grad()
                outputs = self.network(images)

                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

            accuracy = self.test_accuracy()
            print('Epoch: ' + str(epoch) + ', accuracy: %d %%' % (accuracy))

    def test_accuracy(self):
        """
        Test with the test images and return the percentage of correct classifications.
        """
        self.network.eval()
        accuracy = 0.0
        total = 0.0

        with torch.no_grad():
            for data in self.test_data:
                images, labels = data

                outputs = self.network(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()

        accuracy = (100 * accuracy / total)
        return accuracy


if __name__ == "__main__":

    classes = ['basketball', 'bowling ball', 'brass',
               'soccer ball', 'volley ball', 'water polo ball',
               #'bowling ball', 'golf ball'
               ]

    classifier = Classifier(lr=0.00002, weight_decay=0.003,
                            batch_size=10, num_of_classes=len(classes))
    classifier.load_train_test_data(classes)

    classifier.train(num_epochs=100)
    print('Finished Training')

    save_model(classifier.network)
