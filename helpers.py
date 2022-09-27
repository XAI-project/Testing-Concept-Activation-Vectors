from inspect import stack
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from CONSTS import *
import torch
import torchvision
from PIL import Image
import os
import random


def save_model(model, path=MODEL_PATH):
    """Save the model to a specified file"""
    torch.save(model.state_dict(), path)


def load_images(batch_size):

    # TODO: Change this to our dataset s.t. it returns a data loaders with (image, label).
    # Also change self.classes and self.number_of_labels in Classifier, and the network structure accordingly.

    # Loading and normalizing the data.
    # Define transformations for the training and test sets
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR10 dataset consists of 50K training images. We define the batch size of 10 to load 5,000 batches of images.
    # Create an instance for training.
    # When we run this code for the first time, the CIFAR10 train dataset will be downloaded locally.
    train_set = CIFAR10(root="./data", train=True,
                        transform=transformations, download=True)

    # Create a loader for the training set which will read the data within batch size and put into memory.
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    # Create an instance for testing, note that train is set to False.
    # When we run this code for the first time, the CIFAR10 test dataset will be downloaded locally.
    test_set = CIFAR10(root="./data", train=False,
                       transform=transformations, download=True)

    # Create a loader for the test set which will read the data within batch size and put into memory.
    # Note that each shuffle is set to false for the test loader.
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    print("The number of batches per epoch is: ", len(train_loader))

    return train_loader, test_loader


def load_ball_images(batch_size):
    classes = ['basketball', 'baseball', 'bowling ball', 'football', 'eyeballs',
               'marble', 'tennis ball', 'golf ball', 'volley ball', 'beachballs']
    image_and_label_training = []
    image_and_label_testing = []
    for i, imagetype in enumerate(classes):
        directory_train = os.fsencode('archive/train/' + imagetype)
        for file in os.listdir(directory_train):
            filename = os.fsdecode(file)
            img = Image.open('archive/train/' + imagetype + '/' + filename)
            resize = torchvision.transforms.Resize([32, 32])
            img = resize(img)
            to_tensor = torchvision.transforms.ToTensor()
            tensor = to_tensor(img)
            image_and_label_training.append((tensor, torch.tensor(i)))
        directory_test = os.fsencode('archive/test/' + imagetype)
        for file in os.listdir(directory_test):
            filename = os.fsdecode(file)
            img = Image.open('archive/test/' + imagetype + '/' + filename)
            resize = torchvision.transforms.Resize([32, 32])
            img = resize(img)
            to_tensor = torchvision.transforms.ToTensor()
            tensor = to_tensor(img)
            image_and_label_testing.append((tensor, torch.tensor(i)))
    random.shuffle(image_and_label_training)
    random.shuffle(image_and_label_testing)
    train_data = generate_batches_from_list(
        batch_size, image_and_label_training)
    test_data = generate_batches_from_list(batch_size, image_and_label_testing)
    return train_data, test_data


def generate_batches_from_list(batch_size, tensorlist):
    data = []
    batches_input = [pair[0] for pair in tensorlist]
    batches_label = [pair[1] for pair in tensorlist]
    for i in range(1, len(batches_input) // 10):
        try:
            inputs = batches_input[batch_size * (i - 1):batch_size * i]
            labels = batches_label[batch_size * (i - 1):batch_size * i]
            stacked_inputs = torch.stack(inputs, dim=0)  # (10, 3, 32, 32)
            stacked_labels = torch.stack(labels, dim=0)  # (10)
            data.append((stacked_inputs, stacked_labels))
        except:
            continue
    return data


load_ball_images(10)
