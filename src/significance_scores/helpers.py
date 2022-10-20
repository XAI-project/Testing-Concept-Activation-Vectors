from random import sample
import numpy as np
from sklearn import svm

from src.utils import *
from src.CONSTS import *


def get_all_layer_activations(model, images, num_of_layers):
    """
    Get activations in each layer for all images
    """
    activations = np.empty((num_of_layers, 0))

    for (image, _) in images:

        model.forward_and_save_layers(image)

        img_activations = [
            torch.flatten(model.get_layer_activations(layer)).tolist()
            for layer in range(1, num_of_layers + 1)
        ]

        activations = np.c_[activations, img_activations]

    return activations.tolist()  # Shape: (# of layers, # of images)


def generate_concept_vectors(
    num_of_vectors, num_of_images, layer_concept_imgs_act, layer_random_act
):
    concept_vector_list = []
    for _ in range(0, num_of_vectors):
        clf = create_svm_classifier(
            layer_concept_imgs_act, sample(layer_random_act, num_of_images)
        )
        # Extract and append the concept vector to the list
        concept_vector_list.append(clf.coef_[0])
    return concept_vector_list


def prepare_concept_activations(
    model, num_of_layers, concept_images_path, include_random=True
):
    """
    Get activations for images belonging to a concept and random images.
    """

    concept_images = load_images(concept_images_path)
    concept_images_activations = get_all_layer_activations(
        model, concept_images, num_of_layers
    )

    if include_random:

        random_images = load_images(RANDOM_IMAGES_PATH)
        random_activations = get_all_layer_activations(
            model, random_images, num_of_layers
        )

        return concept_images_activations, random_activations

    return concept_images_activations


def create_svm_classifier(concept_images_activations, random_activations):
    """
    Get SVM classifier that classifies a set of concept images and a set of random images.
    """
    X = concept_images_activations + random_activations

    y = list(
        list(np.ones(len(concept_images_activations)))
        + list(np.zeros(len(random_activations)))
    )
    clf = svm.LinearSVC()

    clf.fit(X, y)
    return clf
