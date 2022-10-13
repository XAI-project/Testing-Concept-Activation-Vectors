
from TCAV import get_TCAV_layer_scores
from classifier import Classifier
from helpers import *


if __name__ == "__main__":

    DATA_PATH = WEATHER_PATH

    # The main class must be the first in this list:
    classes = ["snow", "sandstorm", "dew", "lightning", "rainbow"]

    model_scores = []

    # Iterate these paths 
    concept_data_paths = [RANDOM_2_IMAGES_PATH, ORANGE_PATH,
                          CIRCLES_IMAGES_PATH, CIRCLES_FILLED_IMAGES_PATH, PARQUET_IMAGES_PATH]

    for concept_data_path in concept_data_paths:

        print(concept_data_path)

        for i in range(3):

            classifier = Classifier(lr=0.00002, weight_decay=0.003,
                                    batch_size=10, num_of_classes=len(classes))
            classifier.load_train_test_data(classes, DATA_PATH)

            acc = classifier.train(num_epochs=25, print_progress=False)

            TCAV_scores = get_TCAV_layer_scores(
                classifier.network, num_of_layers=6, concept_images_path=concept_data_path, data_path=DATA_PATH, main_class_name=classes[0], num_of_concept_vectors=50)

            model_scores.append([acc, TCAV_scores])
            print(acc, TCAV_scores)
