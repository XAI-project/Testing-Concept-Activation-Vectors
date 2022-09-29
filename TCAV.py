
from CONSTS import *
from helpers import *
from random import sample
from sklearn import svm
import numpy as np
import torch
from tqdm import tqdm


import warnings
warnings.filterwarnings("ignore")


def get_layer_activations(model, images, layer):
    activations = []

    for layer in range(1, num_of_layers+1):
        img_activations = []
        for (image, _) in images:
            model.forward_and_save_layers(image)
            img_activations.append(torch.flatten(
                model.get_layer_activations(layer)).tolist())

        activations.append(img_activations)
    return activations


def prepare_concept_activations(model, layer):

    orange_images = load_images(ORANGE_PATH)
    orange_activations = get_layer_activations(model, orange_images, layer)

    random_images = load_images(RANDOM_IMAGES_PATH)
    random_activations = get_layer_activations(model, random_images, layer)

    return orange_activations, random_activations


def create_svm_classifier(orange_activations, random_activations):
    X = orange_activations + random_activations

    y = list(list(np.ones(len(orange_activations))) +
             list(np.zeros(len(random_activations))))
    clf = svm.LinearSVC()

    clf.fit(X, y)
    return clf


if __name__ == "__main__":

    TCAV_layer_averages = []
    num_of_layers = 6

    model = load_model()

    all_orange_activations = []
    all_random_activations = []

    orange_activations, random_activations = prepare_concept_activations(
        model, num_of_layers)  # dim: num of layers x num of images

    num_of_images = len(orange_activations[0])  # TODO

    num_of_dimensions = len(orange_activations[0][0])  # TODO

    for layer in tqdm(range(1, num_of_layers + 1)):

        v_l_C_list = []

        layer_orange_activations = orange_activations[layer-1]
        layer_random_activations = random_activations[layer-1]

        for i in range(0, 50):
            clf = create_svm_classifier(layer_orange_activations, sample(
                layer_random_activations, num_of_images))
            v_l_C_list.append(clf.coef_[0])

        TCAV_scores_layer = []

        for v_l_C in v_l_C_list:
            directory_path = os.fsencode(BASKETBALL_TRAIN_PATH)

            pos_s_count = 0
            neg_s_count = 0

            for file in os.listdir(directory_path):
                tensor = get_and_rescale_img(file, BASKETBALL_TRAIN_PATH)
                image_label_data = [((tensor, torch.tensor(0)))]
                batch = generate_batches_from_list(
                    1, image_label_data)

                image = batch[0][0]

                model.forward_and_save_layers(image)
                layer_activations = model.get_layer_activations(layer)

                v_unflattened = torch.tensor(
                    v_l_C.reshape(layer_activations.shape).astype(np.float32))

                v_delta = 0.001*v_unflattened

                changed_layer_activations = layer_activations + v_delta

                sensitivity = model(
                    changed_layer_activations, layer=layer)[0][0] - model(layer_activations, layer=layer)[0][0]

                if sensitivity > 0:
                    pos_s_count += 1
                else:
                    neg_s_count += 1

            TCAV_score = pos_s_count / (pos_s_count + neg_s_count)

            TCAV_scores_layer.append(TCAV_score)

        TCAV_layer_averages.append(
            sum(TCAV_scores_layer) / len(TCAV_scores_layer))

    print(TCAV_layer_averages)
