from random import sample
from sklearn import svm
import numpy as np
import torch
from tqdm import tqdm

from CONSTS import *
from helpers import *

import warnings

warnings.filterwarnings("ignore")  # TODO: perhaps not the best solution


def get_all_layer_activations(model, images, num_of_layers):
    """
    Get activations in each layer for all images
    """
    activations = np.empty((6, 0))

    for (image, _) in images:

        model.forward_and_save_layers(image)

        img_activations = [
            torch.flatten(model.get_layer_activations(layer)).tolist()
            for layer in range(1, num_of_layers + 1)
        ]

        activations = np.c_[activations, img_activations]

    return activations.tolist()  # Shape: (# of layers, # of images)


def prepare_concept_activations(model, num_of_layers, concept_images_path):
    """
    Get activations for images belonging to a concept and random images.
    """

    concept_images = load_images(concept_images_path)
    concept_images_activations = get_all_layer_activations(
        model, concept_images, num_of_layers
    )

    random_images = load_images(RANDOM_IMAGES_PATH)
    random_activations = get_all_layer_activations(model, random_images, num_of_layers)

    return concept_images_activations, random_activations


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


def get_layer_TCAV_scores(
    layer,
    num_of_concept_vectors,
    num_of_images,
    layer_concept_images_activations,
    layer_random_activations,
    data_path,
    main_class_name,
    class_index,
):

    concept_vector_list = generate_concept_vectors(
        num_of_concept_vectors,
        num_of_images,
        layer_concept_images_activations,
        layer_random_activations,
    )

    TCAV_scores_layer = []

    for concept_vector in concept_vector_list:

        # All concept vectors are created the same way but differ due to inherent randomness by having many dimensions.

        data_train_path = data_path + "/train/" + main_class_name

        directory_path = os.fsencode(data_train_path)

        pos_s_count = 0
        neg_s_count = 0

        for file in os.listdir(directory_path):
            if (pos_s_count + neg_s_count) > 100:
                break
            try:  # TODO: fix errors
                tensor = get_and_rescale_img(file, data_train_path)
                image_label_data = [((tensor, torch.tensor(0)))]
                batch = generate_batches_from_list(
                    1, image_label_data
                )  # To have the correct dimensions

                image = batch[0][0]

                output = model.forward_and_save_layers(image)

                label = torch.zeros((1, output.shape[1]))
                label[0][class_index] = 1  # Class of interest

                loss = torch.nn.CrossEntropyLoss()(output, label)

                # Computing activation gradients for a layer
                layer_activation_grads = torch.autograd.grad(
                    loss,
                    model.get_layer_activations(layer),
                    create_graph=False,
                    only_inputs=True,
                )[0]

                sensitivity = -np.dot(
                    layer_activation_grads.flatten().detach().numpy(),
                    concept_vector,
                )

                if sensitivity > 0:
                    pos_s_count += 1
                else:
                    neg_s_count += 1
            except:
                continue

        # TCAV score for one of the concept vectors
        TCAV_score = pos_s_count / (pos_s_count + neg_s_count)

        # Add the TCAV score for this concept vector to a list for the current layer
        TCAV_scores_layer.append(TCAV_score)

    return TCAV_scores_layer


def get_layer_cluster_scores(
    layer,
    num_of_concept_vectors,
    num_of_images,
    layer_concept_images_activations,
    data_path,
    main_class_name,
    class_index,
):
    centroid = np.array(layer_concept_images_activations).mean(axis=0)

    data_train_path = data_path + "/train/" + main_class_name

    directory_path = os.fsencode(data_train_path)

    pos_s_count = 0
    neg_s_count = 0

    for file in os.listdir(directory_path):
        tensor = get_and_rescale_img(file, data_train_path)
        image_label_data = [((tensor, torch.tensor(0)))]
        batch = generate_batches_from_list(
            1, image_label_data
        )  # To have the correct dimensions

        image = batch[0][0]

        model.forward_and_save_layers(image)  # TODO: fix.

        orig_layer_activation = model.get_layer_activations(layer)
        activations_shape = orig_layer_activation.shape
        layer_activation = orig_layer_activation.flatten().detach().numpy()

        vector_img_to_centroid = centroid - layer_activation

        normalized_vector_img_to_centroid = vector_img_to_centroid / np.linalg.norm(
            vector_img_to_centroid
        )

        altered_layer_activations = layer_activation + normalized_vector_img_to_centroid

        sensitivity = (
            model.forward(
                torch.tensor(altered_layer_activations, dtype=torch.float).reshape(
                    activations_shape
                ),
                layer=layer,
            )[0][class_index]
            - model.forward(
                torch.tensor(layer_activation, dtype=torch.float).reshape(
                    activations_shape
                ),
                layer=layer,
            )[0][class_index]
        )

        if sensitivity > 0:
            pos_s_count += 1
        else:
            neg_s_count += 1

    # cluster sensitivity score for the concept centroid
    cluster_sensitivity_score = pos_s_count / (pos_s_count + neg_s_count)

    return cluster_sensitivity_score


def get_concept_significance_scores(
    model,
    num_of_layers,
    concept_images_path,
    data_path,
    main_class_name,
    class_index=0,
    num_of_concept_vectors=20,
    significance_type="tcav",
):
    """
    For each layer, create num_of_concept_vectors concept vectors and test all images on each concept vector.
    Average the ratio / significance score for all concept vectors and return a list of mean significance scores for each layer.
    """

    significance_scores = []

    concept_images_activations, random_activations = prepare_concept_activations(
        model, num_of_layers, concept_images_path
    )  # dim: num of layers x num of images ...

    num_of_images = len(concept_images_activations[0])

    for layer in tqdm(range(1, num_of_layers + 1)):

        layer_concept_images_activations = concept_images_activations[layer - 1]
        layer_random_activations = random_activations[layer - 1]

        if significance_type == "tcav":
            significance_scores_layer = get_layer_TCAV_scores(
                layer,
                num_of_concept_vectors,
                num_of_images,
                layer_concept_images_activations,
                layer_random_activations,
                data_path,
                main_class_name,
                class_index,
            )
            significance_scores.append(
                sum(significance_scores_layer) / len(significance_scores_layer)
            )
        elif significance_type == "cluster":
            significance_scores.append(
                get_layer_cluster_scores(
                    layer,
                    num_of_concept_vectors,
                    num_of_images,
                    layer_concept_images_activations,
                    data_path,
                    main_class_name,
                    class_index,
                )
            )
        else:
            raise Exception("Significance type not supported")

        # Get the average TCAV score for this layer and append it to a list of TCAV scores for all layers

    return significance_scores


if __name__ == "__main__":

    model = load_model()

    image_data_path = BALLS_PATH
    concept_data_path = COLORS_PATH + "/brown"

    # The main class must be the first in this list:
    classes = BALLS_CLASSES

    significance_scores = get_concept_significance_scores(
        model,
        num_of_layers=6,
        concept_images_path=concept_data_path,
        data_path=image_data_path,
        main_class_name=classes[0],
        num_of_concept_vectors=50,
        significance_type="cluster",
    )

    print(concept_data_path)
    print(significance_scores)
