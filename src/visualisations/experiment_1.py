import json

import matplotlib.pyplot as plt
import numpy as np

from config.constants import EXPERIMENT1_STATS, IMAGES_FOLDER


def show_plots_for_metric(score):
    models = []
    x_values = []
    yerr = []
    for model in results:
        models.append(model)
        x_values.append(results[model][score])
        yerr.append(results[model][score] - results[model][f"{score} CI min"])
    # Python
    x_pos = list(range(len(models)))
    plt.bar(x_pos, x_values, yerr=yerr, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.xticks(x_pos, models, rotation=45)
    plt.ylabel(score)
    plt.tight_layout()
    plt.savefig(IMAGES_FOLDER / f"{score}.png", dpi=1200, bbox_inches='tight')
    plt.clf()



if __name__ == '__main__':
    results = json.load(open(EXPERIMENT1_STATS))
    show_plots_for_metric("F1_score")
    show_plots_for_metric("Accuracy")
    show_plots_for_metric("Recall")
    show_plots_for_metric("Precision")
