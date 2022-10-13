import matplotlib.pyplot as plt
import seaborn as sns

from src.CONSTS import DATA_PATH


def line_graph(
    data, x_labels, title, xlabel=None, ylabel="Layer", save_path=None, ylim=[0, 1]
):
    """
    Plots a single line in the [0, 1] interval for the y-axis as default.
    """
    print(data)
    for line_points in data:
        plt.plot(x_labels, line_points, marker="o")
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.title(title)
    plt.ylim(ylim)
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    TCAV_orange_data = [
        [
            0.6358620689655172,
            0.8758620689655174,
            0.9732758620689655,
            0.720344827586207,
            0.6917241379310343,
            0.7132758620689655,
        ],
        [
            0.7358620689655172,
            0.8758620689655174,
            0.9732758620689655,
            0.320344827586207,
            0.6917241379310343,
            0.7132758620689655,
        ],
    ]
    save_path = DATA_PATH + "/" + "orange_TCAV_scores_by_layer"
    line_graph(
        TCAV_orange_data,
        range(1, 7),
        "TCAV scores in each layer for orange concept",
        xlabel="TCAV-score",
        save_path=save_path,
    )
