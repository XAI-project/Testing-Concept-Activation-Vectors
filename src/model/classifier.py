import torch
from torch.optim import Adam
import torch.nn as nn
from torch.autograd import Variable

from src.CONSTS import *
from src.model.Network import Network
from src.model.helpers import *
from src.utils import *


class Classifier:
    def __init__(self, batch_size, num_of_classes):
        """
        Init classifier that uses CrossEntropyLoss, Adam and the Network in Network.py
        """
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # TODO
        self.device = torch.device("cpu")
        self.num_of_classes = num_of_classes

        self.network = Network(num_of_classes).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = None

        self.batch_size = batch_size

        self.train_data = None
        self.test_data = None

    def set_optim(self, lr, weight_decay):
        self.optimizer = Adam(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )

    def load_train_test_data(self, classes, data_path):
        """
        Load train and test data from the ball images from the provided classes.
        """
        train_data, test_data = load_train_test_images(
            self.batch_size, classes, data_path
        )
        self.train_data = train_data
        self.test_data = test_data

    def train(self, num_epochs, print_progress=True):
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

            if print_progress:
                print("Epoch: " + str(epoch) + ", accuracy: %d %%" % (accuracy))

        return accuracy

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

        accuracy = 100 * accuracy / total
        return accuracy


def create_classification_model(IMG_DATA_PATH=BALLS_PATH, classes=BALLS_CLASSES):

    classifier = Classifier(batch_size=10, num_of_classes=len(classes))
    classifier.set_optim(lr=0.00002, weight_decay=0.003)
    classifier.load_train_test_data(classes, IMG_DATA_PATH)

    classifier.train(num_epochs=100)
    print("Finished Training")

    save_model(classifier.network)
