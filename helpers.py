from CONSTS import *
import torch
import torchvision
from PIL import Image
import os
import random


def save_model(model, path=MODEL_PATH):
    """Save the model to a specified file"""
    torch.save(model.state_dict(), path)


def create_image_label_data(path, image_type, label):
    image_type_path = path + "/" + image_type
    directory_test = os.fsencode(image_type_path)
    image_label_data = []
    for file in os.listdir(directory_test):
        filename = os.fsdecode(file)
        img = Image.open(image_type_path + '/' + filename)
        resize = torchvision.transforms.Resize([32, 32])
        img = resize(img)
        to_tensor = torchvision.transforms.ToTensor()
        tensor = to_tensor(img)
        image_label_data.append((tensor, torch.tensor(label)))
    return image_label_data


def load_ball_images(batch_size):
    classes = ['basketball', 'baseball', 'bowling ball', 'football', 'eyeballs',
               'marble', 'tennis ball', 'golf ball', 'screwballs', 'meat ball']
    image_and_label_training = []
    image_and_label_testing = []
    for i, image_type in enumerate(classes):
        image_and_label_training += create_image_label_data(
            BALLS_TRAIN_PATH, image_type, i)
        image_and_label_testing += create_image_label_data(
            BALLS_TEST_PATH, image_type, i)
        image_and_label_testing += create_image_label_data(
            BALLS_VALID_PATH, image_type, i)  # We don't validate atm, so adding for more accurate accuracy

    random.shuffle(image_and_label_training)
    random.shuffle(image_and_label_testing)
    train_data = generate_batches_from_list(
        batch_size, image_and_label_training)
    test_data = generate_batches_from_list(batch_size, image_and_label_testing)
    return train_data, test_data


def generate_batches_from_list(batch_size, tensor_list):
    data = []
    batches_input = [pair[0] for pair in tensor_list]
    batches_label = [pair[1] for pair in tensor_list]
    for i in range(1, len(batches_input) // batch_size):
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
