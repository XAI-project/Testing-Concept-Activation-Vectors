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
        try:

            model.forward_and_save_layers(image)

            img_activations = [
                torch.flatten(model.get_layer_activations(layer)).tolist()
                for layer in range(1, num_of_layers + 1)
            ]

            activations = np.c_[activations, img_activations]

        except:
            continue

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


def get_images_activations(model, num_of_layers, images_path):
    """
    Get activations for images in a provided path.
    """

    images = load_images(images_path)
    images_activations = get_all_layer_activations(model, images, num_of_layers)

    return images_activations


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
