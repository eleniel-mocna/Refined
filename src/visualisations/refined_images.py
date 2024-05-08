import copy
import json
import pickle

import numpy as np
from matplotlib import pyplot as plt

from config.config import Config
from config.constants import REFINED_ORDERS, IMAGES_FOLDER
from models.evaluation.ModelEvaluator import booleanize
from models.refined.Refined import Refined

interesting_refines = [0, 1, 2, 4, 7, 10, 15, 35]


def main():
    """
    Create plots for REFINED images.
    @return:
    """
    config = Config.get_instance()
    with open(config.train_surroundings, "rb") as file:
        data, labels = pickle.load(file)
    labels = np.vectorize(booleanize)(labels)
    true_samples = data[labels]

    orders = json.load(open(REFINED_ORDERS))
    refines = []
    scores = []
    refined = Refined(data, 38, config.surroundings_size, "temp", hca_starts=1)
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    axs = axs.flatten()
    for i, ax in zip(interesting_refines, axs):
        refined.from_pretrained(np.array(orders[i]["order"]))
        refines.append(copy.deepcopy(refined))
        score = orders[i]["score"]
        scores.append(score)
        transformed_data = refined.transform(true_samples).mean(axis=0).reshape((38, config.surroundings_size))
        im = ax.imshow(transformed_data, cmap='viridis', interpolation='nearest')
        ax.axis('off')
        score_string = "%.2E" % score
        ax.set_title(f"Epoch: {i}, score: {score_string}")
    plt.tight_layout()
    # noinspection PyUnboundLocalVariable
    plt.colorbar(im, ax=axs.ravel().tolist())
    plt.savefig(IMAGES_FOLDER / "combined_plot.png", dpi=1200, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    main()
