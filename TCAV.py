
from CONSTS import *
from helpers import *
from random import sample
from sklearn import svm
import numpy as np
import torch
from tqdm import tqdm


import warnings
warnings.filterwarnings("ignore")


def get_layer_one_activations(model, images):
    activations = []
    for (images, _) in images:
        model.forward_and_save_layers(images)
        activations.append(torch.flatten(model.output6).tolist())
    return activations


def prepare_concept_activations(model):

    orange_images = load_images(ORANGE_PATH)
    orange_activations = get_layer_one_activations(model, orange_images)

    random_images = load_images(RANDOM_IMAGES_PATH)
    random_activations = get_layer_one_activations(model, random_images)

    return orange_activations, random_activations


def create_svm_classifier(orange_activations, random_activations):
    X = orange_activations + random_activations

    y = list(list(np.ones(len(orange_activations))) +
             list(np.zeros(len(random_activations))))
    clf = svm.LinearSVC()

    clf.fit(X, y)
    return clf


if __name__ == "__main__":

    layer = 6

    model = load_model()
    orange_activations, random_activations = prepare_concept_activations(model)

    num_of_images = len(orange_activations)

    num_of_dimensions = len(orange_activations[0])

    v_l_C_list = []
    for i in tqdm(range(0, 100)):
        clf = create_svm_classifier(orange_activations, sample(
            random_activations, num_of_images))
        v_l_C_list.append(clf.coef_[0])

    for v_l_C in tqdm(v_l_C_list):
        directory_path = os.fsencode(BASKETBALL_TRAIN_PATH)

        pos_s_count = 0
        neg_s_count = 0

        for file in os.listdir(directory_path):
            filename = os.fsdecode(file)
            img = Image.open(BASKETBALL_TRAIN_PATH + '/' + filename)
            resize = torchvision.transforms.Resize([32, 32])
            img = resize(img)
            to_tensor = torchvision.transforms.ToTensor()
            tensor = to_tensor(img)
            image_label_data = [((tensor, torch.tensor(0)))]
            batch = generate_batches_from_list(
                1, image_label_data)

            image = batch[0][0]

            model.forward_and_save_layers(image)
            layer_activations = model.output6

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

        TCAV_Q = pos_s_count / (pos_s_count + neg_s_count)

        print(TCAV_Q)
