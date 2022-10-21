import torch
import torchvision
from PIL import Image
import random
import splitfolders
import os

from src.CONSTS import *


def split_folder(input_folder, output_folder, ratio=(0.8, 0.1, 0.1)):
    """Split with a ratio"""
    splitfolders.ratio(
        input=input_folder,
        output=output_folder,
        seed=1337,
        ratio=ratio,
        group_prefix=None,
        move=False,
    )  # default values


def get_and_rescale_img(file, path):
    """
    Extracts an image, rescales to 32x32 and returns a tensor of the image.
    """
    filename = os.fsdecode(file)
    img = Image.open(path + "/" + filename)
    resize = torchvision.transforms.Resize(
        [32, 32]
    )  # Rescales any image size to 32 x 32
    img = resize(img)
    to_tensor = torchvision.transforms.ToTensor()
    tensor = to_tensor(img)
    return tensor


def create_image_label_data(path, label=0):
    """
    Extracts and rescales images from a path, and returns a list of (image_tensor, image_label) pairs.
    """
    directory_path = os.fsencode(path)
    image_label_data = []
    for file in os.listdir(directory_path):
        tensor = get_and_rescale_img(file, path)
        image_label_data.append((tensor, torch.tensor(label)))
    return image_label_data


def load_train_test_images(batch_size, classes, data_path):

    image_and_label_training = []
    image_and_label_testing = []
    image_and_label_validation = []
    for i, image_type in enumerate(classes):
        # print(i, image_type)
        image_and_label_training += create_image_label_data(
            data_path + "/train/" + image_type, i
        )
        image_and_label_testing += create_image_label_data(
            data_path + "/test/" + image_type, i
        )
        image_and_label_validation += create_image_label_data(
            data_path + "/valid/" + image_type, i
        )  # We don't validate atm, so adding for more accurate accuracy

    random.shuffle(image_and_label_training)
    random.shuffle(image_and_label_testing)
    random.shuffle(image_and_label_validation)
    train_data = generate_batches_from_list(batch_size, image_and_label_training)
    test_data = generate_batches_from_list(batch_size, image_and_label_testing)
    valid_data = generate_batches_from_list(batch_size, image_and_label_validation)
    return train_data, test_data


def load_images(path):
    images_and_labels = create_image_label_data(path)
    random.shuffle(images_and_labels)
    data = generate_batches_from_list(1, images_and_labels)
    return data


def generate_batches_from_list(batch_size, tensor_list):
    data = []
    batches_input = [pair[0] for pair in tensor_list]
    batches_label = [pair[1] for pair in tensor_list]
    for i in range(1, (len(batches_input) // batch_size) + 1):
        try:
            inputs = batches_input[batch_size * (i - 1) : batch_size * i]
            labels = batches_label[batch_size * (i - 1) : batch_size * i]
            stacked_inputs = torch.stack(inputs, dim=0)  # (10, 3, 32, 32)
            stacked_labels = torch.stack(labels, dim=0)  # (10)
            data.append((stacked_inputs, stacked_labels))
        except:
            continue
    return data
