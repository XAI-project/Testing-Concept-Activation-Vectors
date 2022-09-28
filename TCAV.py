
from CONSTS import *
from helpers import *


def prepare_concept_images():
    model = load_model()

    orange_images = load_images(ORANGE_PATH)
    orange_activations = []
    for (images, _) in orange_images:
        orange_activations.append(model.forward(images))

    random_images = load_images(RANDOM_IMAGES_PATH)
    random_activations = []
    for (images, _) in random_images:
        random_activations.append(model.forward(images))

    return orange_activations, random_activations


def linear_classifier():
    return


if __name__ == "__main__":

    orange_activations, random_activations = prepare_concept_images()

    linear_classifier()
