import matplotlib.pyplot as plt

from CONSTS import DATA_PATH


def line_graph(points, y, title, xlabel, ylabel, save_path, ylim):
    plt.plot(y, points, marker="o")
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.title(title)
    plt.ylim(ylim)
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    points = [0.6358620689655172, 0.8758620689655174, 0.9732758620689655,
              0.720344827586207, 0.6917241379310343, 0.7132758620689655]
    save_path = DATA_PATH + "/" + "TCAV_scores_by_layer"
    line_graph(points, range(1, 7), "Average TCAV scores in each layer",
               "TCAV-score", "Layer", save_path, [0, 1])
