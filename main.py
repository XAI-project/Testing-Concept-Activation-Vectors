import seaborn as sns
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.significance_scores.TCAV import get_TCAV_significance_scores
from src.significance_scores.cluster import (
    calculate_class_cluster_proximity,
    get_cluster_significance_scores,
)
from src.model.classifier import Classifier
from src.utils import *
from src.plotting import line_graph
from src.CONSTS import *


def plot_significance_scores():
    image_data_path = WEATHER_PATH

    # The main class must be the first in this list:
    classes = WEATHER_CLASSES

    random_concept_path = WEATHER_PATH + "/class_random"
    random_concept_path_small = random_concept_path + "2"

    # Iterate these paths
    concept_data_paths = [
        # RANDOM_2_IMAGES_PATH,
        ORANGE_PATH,
        COLORS_PATH + "/violet",
        COLORS_PATH + "/white",
        # CIRCLES_IMAGES_PATH,
        # PARQUET_IMAGES_PATH,
        random_concept_path_small,
    ]

    for significance_type in ["tcav"]:

        for concept_data_path in concept_data_paths:

            print(
                "Significance type:", significance_type, "Concept:", concept_data_path
            )

            model_accuracies = []
            model_significance_scores = []

            for i in range(5):

                classifier = Classifier(batch_size=10, num_of_classes=len(classes))
                classifier.set_optim(lr=0.00002, weight_decay=0.003)
                classifier.load_train_test_data(classes, image_data_path)

                acc = classifier.train(num_epochs=20, print_progress=False)

                if significance_type == "tcav":

                    significance_scores = get_TCAV_significance_scores(
                        classifier.network,
                        num_of_layers=5,
                        concept_images_path=concept_data_path,
                        random_concept_path=random_concept_path,
                        data_path=image_data_path,
                        main_class_name=classes[0],
                    )

                elif significance_type == "cluster":
                    significance_scores = get_cluster_significance_scores(
                        classifier.network,
                        num_of_layers=5,
                        concept_images_path=concept_data_path,
                        data_path=image_data_path,
                        main_class_name=classes[0],
                    )

                model_significance_scores.append(significance_scores)
                print("acc:", acc, "; scores:", significance_scores)

                model_accuracies.append(acc)

            avg_acc = sum(model_accuracies) / len(model_accuracies)

            file_name = (
                image_data_path.split("/")[-1]
                + "_"
                + significance_type
                + "_"
                + concept_data_path.split("/")[-1].split(".")[0]
            )
            title = file_name + " (avg acc: " + str(int(avg_acc)) + ")"
            save_path = GRAPH_PATH + "/" + significance_type + "/" + file_name
            line_graph(
                model_significance_scores,
                range(1, 6),
                title,
                xlabel=significance_type + " score",
                save_path=save_path,
            )


def plot_cluster_concept_proximity():

    image_data_path = BALLS_PATH

    # The main class must be the first in this list:
    classes = BALLS_CLASSES

    random_concept_path_2 = BALLS_PATH + "/class_random2"

    # Iterate these paths
    concept_data_paths = [
        # RANDOM_2_IMAGES_PATH,
        # ORANGE_PATH,
        # COLORS_PATH + "/violet",
        # COLORS_PATH + "/white",
        # CIRCLES_IMAGES_PATH,
        # PARQUET_IMAGES_PATH,
        random_concept_path_2
    ]

    for concept_data_path in concept_data_paths:

        prox_df = pd.DataFrame([], columns=["score", "layer", "class", "model"])
        model_accuracies = []

        print("Proximity", concept_data_path)

        for i in tqdm(range(5)):

            classifier = Classifier(batch_size=10, num_of_classes=len(classes))
            classifier.set_optim(lr=0.00002, weight_decay=0.003)
            classifier.load_train_test_data(classes, image_data_path)

            acc = classifier.train(num_epochs=20, print_progress=False)

            proximities = calculate_class_cluster_proximity(
                classifier.network,
                classes,
                num_of_layers=5,
                concept_images_path=concept_data_path,
                data_path=image_data_path,
            )

            model_accuracies.append(acc)

            for class_index, class_scores in enumerate(proximities):
                for layer, layer_score in enumerate(class_scores):
                    prox_df.loc[len(prox_df.index)] = [
                        layer_score,
                        layer + 1,
                        classes[class_index],
                        i,
                    ]

        avg_acc = sum(model_accuracies) / len(model_accuracies)

        file_name = (
            image_data_path.split("/")[-1]
            + "_"
            + concept_data_path.split("/")[-1].split(".")[0]
        )
        title = (
            "cluster proximity " + file_name + " (avg acc: " + str(int(avg_acc)) + ")"
        )

        save_path = GRAPH_PATH + "/cluster_proximity/" + file_name

        sns.lineplot(
            data=prox_df,
            x="layer",
            y="score",
            hue="class",
            style="class",
            markers=True,
            dashes=False,
        )

        plt.ylabel("prox score")
        plt.xlabel("layer")
        plt.title(title)
        plt.savefig(save_path)
        plt.clf()
        plt.cla()


if __name__ == "__main__":

    # plot_cluster_concept_proximity()
    plot_significance_scores()
