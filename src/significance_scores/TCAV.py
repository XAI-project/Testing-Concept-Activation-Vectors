import numpy as np
import torch
import os
from tqdm import tqdm

from src.CONSTS import *
from src.utils import *
from src.significance_scores.helpers import *

import warnings

warnings.filterwarnings("ignore")  # TODO: perhaps not the best solution


def get_TCAV_significance_scores(
    model,
    num_of_layers,
    concept_images_path,
    data_path,
    main_class_name,
    class_index=0,
    num_of_concept_vectors=20,
):
    """
    For each layer, create num_of_concept_vectors concept vectors and test all images on each concept vector.
    Average the ratio / significance score for all concept vectors and return a list of mean significance scores for each layer.
    """

    significance_scores = []

    concept_images_activations, random_activations = get_images_activations(
        model, num_of_layers, concept_images_path
    )  # dim: num of layers x num of images ...

    num_of_images = len(concept_images_activations[0])

    for layer in tqdm(range(1, num_of_layers + 1)):

        layer_concept_images_activations = concept_images_activations[layer - 1]
        layer_random_activations = random_activations[layer - 1]

        concept_vector_list = generate_concept_vectors(
            num_of_concept_vectors,
            num_of_images,
            layer_concept_images_activations,
            layer_random_activations,
        )

        TCAV_scores_layer = []

        for concept_vector in concept_vector_list:

            # All concept vectors are created the same way but differ due to inherent randomness by having many dimensions.

            data_main_class_train_path = data_path + "/train/" + main_class_name

            directory_path = os.fsencode(data_main_class_train_path)

            pos_s_count = 0
            neg_s_count = 0

            for file in os.listdir(directory_path):
                if (pos_s_count + neg_s_count) > 100:
                    break
                try:  # TODO: fix errors
                    tensor = get_and_rescale_img(file, data_main_class_train_path)
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

        significance_scores_layer = TCAV_scores_layer
        significance_scores.append(
            sum(significance_scores_layer) / len(significance_scores_layer)
        )

        # Get the average TCAV score for this layer and append it to a list of TCAV scores for all layers

    return significance_scores
