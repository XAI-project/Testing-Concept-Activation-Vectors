import enum
from matplotlib.pyplot import draw
from CONSTS import *
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import os
import random

from Network import Network


def save_model(model, path=MODEL_PATH):
    """
    Save pytorch model to a specified file
    """
    torch.save(model, path)


def load_model(path=MODEL_PATH):
    """
    Load a pytorch model.
    """
    model = torch.load(path)
    model.eval()
    return model


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
    for i, image_type in enumerate(classes):
        # print(i, image_type)
        image_and_label_training += create_image_label_data(
            data_path + "/train/" + image_type, i
        )
        image_and_label_testing += create_image_label_data(
            data_path + "/test/" + image_type, i
        )
        image_and_label_testing += create_image_label_data(
            data_path + "/valid/" + image_type, i
        )  # We don't validate atm, so adding for more accurate accuracy

    random.shuffle(image_and_label_training)
    random.shuffle(image_and_label_testing)
    train_data = generate_batches_from_list(batch_size, image_and_label_training)
    test_data = generate_batches_from_list(batch_size, image_and_label_testing)
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


def generate_image_with_texts(percent_random=0):
    os.mkdir("." + DATA_PATH + "/textimages")
    for typefolder in os.listdir("." + BALLS_PATH):
        os.mkdir("." + DATA_PATH + "/textimages/" + typefolder)
        for sport in os.listdir("." + BALLS_PATH + "/" + typefolder):
            os.mkdir("." + DATA_PATH + "/textimages/" + typefolder + "/" + sport)
            for i, img in enumerate(
                os.listdir("." + BALLS_PATH + "/" + typefolder + "/" + sport)
            ):
                draw_and_save_copy(
                    imgpath="."
                    + BALLS_PATH
                    + "/"
                    + typefolder
                    + "/"
                    + sport
                    + "/"
                    + img,
                    savepath="."
                    + DATA_PATH
                    + "/textimages/"
                    + typefolder
                    + "/"
                    + sport,
                    text=sport,
                    name=str(i),
                )


def draw_and_save_copy(imgpath, savepath, text, name):
    image = Image.open(imgpath)
    copy = image.copy()
    draw_image = ImageDraw.Draw(copy)
    draw_image.text((50, 50), text, fill="white")
    copy.save(os.path.abspath(savepath) + "/" + name + ".jpg")


if __name__ == "__main__":
    generate_image_with_texts()
