from src.significance_scores import get_concept_significance_scores
from src.model.classifier import Classifier
from src.utils import *
from src.plotting import line_graph
from src.CONSTS import *


def main():
    image_data_path = BALLS_PATH

    # The main class must be the first in this list:
    classes = BALLS_CLASSES

    model_scores = []

    # Iterate these paths
    concept_data_paths = [
        # RANDOM_2_IMAGES_PATH,
        ORANGE_PATH,
        COLORS_PATH + "/violet",
        # COLORS_PATH + "/white",
        CIRCLES_IMAGES_PATH,
        # CIRCLES_FILLED_IMAGES_PATH,
        PARQUET_IMAGES_PATH,
    ]

    for significance_type in ["cluster", "tcav"]:

        for concept_data_path in concept_data_paths:

            print("Concept:", concept_data_path)

            accuracies = []

            for i in range(5):

                classifier = Classifier(batch_size=10, num_of_classes=len(classes))
                classifier.set_optim(lr=0.00002, weight_decay=0.003)
                classifier.load_train_test_data(classes, image_data_path)

                acc = classifier.train(num_epochs=20, print_progress=False)

                significance_scores = get_concept_significance_scores(
                    classifier.network,
                    num_of_layers=5,
                    concept_images_path=concept_data_path,
                    data_path=image_data_path,
                    main_class_name=classes[0],
                    significance_type=significance_type,
                )

                model_scores.append(significance_scores)
                print("acc:", acc, "; scores:", significance_scores)

                accuracies.append(acc)

            avg_acc = sum(accuracies) / len(accuracies)

            file_name = (
                image_data_path.split("/")[-1]
                + "_"
                + significance_type
                + "_"
                + concept_data_path.split("/")[-1].split(".")[0]
            )
            title = file_name + " (avg acc: " + str(int(avg_acc)) + ")"
            save_path = GRAPH_PATH + "/" + file_name
            line_graph(
                model_scores,
                range(1, 6),
                title,
                xlabel=significance_type + " score",
                save_path=save_path,
            )


if __name__ == "__main__":

    main()
