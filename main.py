from src.TCAV import get_concept_significance_scores
from src.classifier import Classifier
from src.helpers import *


if __name__ == "__main__":

    IMAGE_DATA_PATH = WEATHER_PATH

    # The main class must be the first in this list:
    classes = ["snow", "sandstorm", "dew", "lightning", "rainbow"]

    model_scores = []

    # Iterate these paths
    concept_data_paths = [
        RANDOM_2_IMAGES_PATH,
        ORANGE_PATH,
        CIRCLES_IMAGES_PATH,
        CIRCLES_FILLED_IMAGES_PATH,
        PARQUET_IMAGES_PATH,
    ]

    for concept_data_path in concept_data_paths:

        print(concept_data_path)

        for i in range(3):

            classifier = Classifier(
                lr=0.00002,
                weight_decay=0.003,
                batch_size=10,
                num_of_classes=len(classes),
            )
            classifier.load_train_test_data(classes, IMAGE_DATA_PATH)

            acc = classifier.train(num_epochs=25, print_progress=False)

            significance_scores = get_concept_significance_scores(
                classifier.network,
                num_of_layers=6,
                concept_images_path=concept_data_path,
                data_path=IMAGE_DATA_PATH,
                main_class_name=classes[0],
                num_of_concept_vectors=50,
            )

            model_scores.append([acc, significance_scores])
            print(acc, significance_scores)
