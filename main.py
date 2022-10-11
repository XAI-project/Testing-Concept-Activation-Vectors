
from TCAV import get_TCAV_layer_scores
from classifier import Classifier
from helpers import *


if __name__ == "__main__":

    classes = ['basketball', 'bowling ball', 'brass',
               'soccer ball', 'volley ball', 'water polo ball',
               #'bowling ball', 'golf ball'
               ]

    model_scores = []
    for i in range(10):

        classifier = Classifier(lr=0.00002, weight_decay=0.003,
                                batch_size=10, num_of_classes=len(classes))
        classifier.load_train_test_data(classes)

        classifier.train(num_epochs=50)
        print('Finished Training')

        save_model(classifier.network)

        model = load_model()

        TCAV_scores = get_TCAV_layer_scores(
            model, num_of_layers=6, concept_images_path=ORANGE_PATH)

        model_scores.append(TCAV_scores)
    print(model_scores)
