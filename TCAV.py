
from CONSTS import *
from helpers import *
from random import sample


def get_layer_one_activations(model, images):
    activations = []
    for (images, _) in images:
        model.forward(images)
        activations.append(torch.flatten(model.output1))
    return activations


def prepare_concept_activations():
    model = load_model()

    orange_images = load_images(ORANGE_PATH)
    orange_activations = get_layer_one_activations(model, orange_images)

    random_images = load_images(RANDOM_IMAGES_PATH)
    random_activations = get_layer_one_activations(model, random_images)

    return orange_activations, random_activations


def linear_classifier(orange_activations, random_activations):
    return


if __name__ == "__main__":

    orange_activations, random_activations = prepare_concept_activations()

    list_length = len(orange_activations)

    for i in range(0, 500):
        linear_classifier(orange_activations, sample(
            random_activations, list_length))
