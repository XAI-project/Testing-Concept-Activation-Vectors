import numpy as np
import torch
from tqdm import tqdm
import os

from src.CONSTS import *
from src.utils import *
from src.significance_scores.helpers import *

import warnings

warnings.filterwarnings("ignore")  # TODO: perhaps not the best solution


def get_cluster_significance_scores(
    model,
    num_of_layers,
    concept_images_path,
    data_path,
    main_class_name,
):
    """
    For each layer, create num_of_concept_vectors concept vectors and test all images on each concept vector.
    Average the ratio / significance score for all concept vectors and return a list of mean significance scores for each layer.
    """

    significance_scores = []

    concept_images_activations = get_images_activations(
        model, num_of_layers, concept_images_path, include_random=False
    )  # dim: num of layers x num of images ...

    for layer in tqdm(range(1, num_of_layers + 1)):

        layer_concept_images_activations = concept_images_activations[layer - 1]

        centroid = np.array(layer_concept_images_activations).mean(axis=0)

        data_main_class_train_path = data_path + "/train/" + main_class_name

        directory_path = os.fsencode(data_main_class_train_path)

        pos_s_count = 0
        neg_s_count = 0

        for file in os.listdir(directory_path):
            try:
                tensor = get_and_rescale_img(file, data_main_class_train_path)
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

                normalized_vector_img_to_centroid = (
                    vector_img_to_centroid / np.linalg.norm(vector_img_to_centroid)
                )

                altered_layer_activations = (
                    layer_activation + normalized_vector_img_to_centroid
                )

                sensitivity = (
                    model.forward(
                        torch.tensor(
                            altered_layer_activations, dtype=torch.float
                        ).reshape(activations_shape),
                        layer=layer,
                    )[0][
                        0
                    ]  # 0 is the index of the "main" class
                    - model.forward(
                        torch.tensor(layer_activation, dtype=torch.float).reshape(
                            activations_shape
                        ),
                        layer=layer,
                    )[0][
                        0
                    ]  # 0 is the index of the "main" class
                )

                if sensitivity > 0:
                    pos_s_count += 1
                else:
                    neg_s_count += 1
            except:
                continue

        if pos_s_count + neg_s_count < 50:
            print("very few test images when calculating sensitivity")

        # cluster sensitivity score for the concept centroid
        cluster_sensitivity_score = pos_s_count / (pos_s_count + neg_s_count)
        significance_scores.append(cluster_sensitivity_score)

        # Get the average cluster significance score for this layer and append it to a list of TCAV scores for all layers

    return significance_scores


def calculate_class_cluster_proximity(
    model,
    classes,
    num_of_layers,
    concept_images_path,
    data_path,
):
    """
    Calculate the cluster centroid
    """
    concept_images_activations = get_images_activations(
        model, num_of_layers, concept_images_path, include_random=False
    )

    differences = []

    for layer in range(1, num_of_layers + 1):

        layer_differences = []

        layer_concept_images_activations = concept_images_activations[layer - 1]
        concept_images_activations_centroid = np.array(
            layer_concept_images_activations
        ).mean(axis=0)

        for class_ in classes:
            class_images_path = data_path + "/train/" + class_

            class_images_activations = get_images_activations(
                model, num_of_layers, class_images_path, include_random=False
            )[layer - 1]
            class_images_activations_centroid = np.array(class_images_activations).mean(
                axis=0
            )

            activation_difference = abs(
                class_images_activations_centroid - concept_images_activations_centroid
            )
            difference_avg = sum(activation_difference) / len(activation_difference)

            layer_differences.append(difference_avg)

        differences.append(layer_differences)

    differences_by_class = np.array(
        differences
    ).T.tolist()  # Double list with num of classes x num og layers dimensions

    return differences_by_class
