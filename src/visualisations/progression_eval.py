import json

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from config.constants import PROGRESSION_EVAL_RESULTS, DATA_FOLDER, PROGRESSION_MODEL_STATS, REFINED_ORDERS

if __name__ == '__main__':
    results = json.load(open(PROGRESSION_EVAL_RESULTS))
    stats = json.load(open(PROGRESSION_MODEL_STATS))
    orders = json.load(open(REFINED_ORDERS))
    x = []
    y = []
    y_min = []
    y_max = []
    mean = "F1_score"
    ci_min = "F1_score CI min"
    ci_max = "F1_score CI max"
    # mean = "Session F1_score Mean"
    # ci_min = "Session F1_score CI min"
    # ci_max = "Session F1_score CI max"
    for i in results.keys():
        # x.append(int(i))
        x.append(np.log(orders[int(i)]["score"]))
        min_loss = np.log(min([x["loss"] for x in stats[i]]))
        y.append(min_loss)
        y_min.append(min_loss)
        y_max.append(max([x["loss"] for x in stats[i]]))
        # y.append(stats[i][-1]["loss"])
        # y.append(results[i][mean])
        # y_min.append(results[i][ci_min])
        # y_max.append(results[i][ci_max])

    correlation, pearson_r = pearsonr(x, y)
    print('Pearsons correlation coefficient: %.3f' % correlation)
    fig, ax = plt.subplots()
    ax.scatter(x, y)

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--")
    # ax.fill_between(x, y_min, y_max, color='b', alpha=.1)
    file_to_save = DATA_FOLDER / "progression_train.png"
    plt.title("Validation loss per REFINED evolution")
    correlation_label = "Pearson's\ncorrelation\ncoefficient: %.3f" % correlation
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.5, 0.95, correlation_label, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    plt.ylabel("-ln(val_loss)")
    plt.xlabel("ln(REFINED_Score)")
    plt.savefig(file_to_save, dpi=1200, bbox_inches='tight')
    # plt.show()
