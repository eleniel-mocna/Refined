import glob
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from config.constants import DATA_FOLDER, IMAGES_FOLDER

refined_models = {
    'rfc_baseline': "P2Rank RFC",
    'rfc_surrounding': "RFC Surrounding",
    'surroundings_nn': "NN",
    'refined': "REFINED",
    'refined_random': "Random CNN",
    'refined_random_norm': "Normalized CNN",
}


def main():
    root = DATA_FOLDER / "for_transfer"
    metrics_files = glob.glob(str(root / '**/metrics.json'), recursive=True)
    metrics = [json.load(open(file)) for file in metrics_files]
    create_auc_plot(metrics)


alpha = 0.01


def create_auc_plot(metrics):
    aucs = dict()
    for metric in metrics:
        model_name = metric["model_name"].lower()
        auc = float(metric["AUC"])
        if model_name in aucs:
            aucs[model_name].append(auc)
        else:
            aucs[model_name] = [auc]
    aucs = {refined_models[k]: v for k, v in aucs.items()}
    mean_aucs = {model: np.mean(vals) for model, vals in aucs.items()}
    stds = {model: np.std(vals) for model, vals in aucs.items()}
    models_names = list(refined_models.values())
    means = [mean_aucs[model] for model in models_names]
    stds = [stds[model] for model in models_names]

    create_bar_plot(means, models_names, stds)

    p_values = calculate_p_values(aucs, models_names)
    show_ab_test_results(mean_aucs, models_names, p_values)


def show_ab_test_results(mean_aucs, models, p_values):
    p_val_show = np.vectorize(lambda x: x if x < alpha else alpha)(p_values)
    plt.imshow(-np.log10(p_val_show), cmap='hot', interpolation='nearest')
    for (j, i), label in np.ndenumerate(p_values):
        plt.text(i, j, -round(np.log10(p_values[j, i]), 2), ha="center", va="center")
    plt.colorbar(label='-log10(p-value)')
    plt.title('One way student t-test p-value on AUC')
    plt.xlabel('Model')
    plt.ylabel('Model')
    labels = [f"{model}\n\u03bc={round(mean_aucs[model], 2)}" for model in models]
    plt.xticks(np.arange(len(models)), labels, rotation=45)
    plt.yticks(np.arange(len(models)), labels)
    # Make space for x ticks
    plt.savefig(IMAGES_FOLDER / f"ab1.png", dpi=1200, bbox_inches='tight')
    plt.show()
    plt.clf()


def calculate_p_values(aucs, models):
    p_values = np.zeros((len(models), len(models)))
    # Comparing each model with every other model
    for i, model in enumerate(models):
        for j, other_model in enumerate(models):
            model_1 = model
            model_2 = other_model
            print(f"Comparing models: {model_1} and {model_2}")

            # Two sample t-test
            t_stat, p_val = stats.ttest_ind(aucs[model_1], aucs[model_2], equal_var=False, alternative="greater")

            p_values[i, j] = p_val
            # Using p-value to decide whether to reject or accept null hypothesis

            if p_val < alpha:
                print(f"t-statistic: {t_stat}, p-value: {p_val}")
                print(f"The difference in AUCs between {model_1} and {model_2} is statistically significant.\n")
    return p_values


def create_bar_plot(means, models, stds):
    plt.figure(figsize=(10, 5))
    plt.bar(models, means, yerr=stds, capsize=10)
    plt.xlabel("Model")
    plt.ylabel("Average AUC")
    plt.title("Model's ROC AUC comparison")
    for i in range(len(models)):
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(i - 0.0, means[i] - 0.1, f"\u03bc={round(means[i], 2)}\n \u03C3={round(stds[i], 3)}", ha='center',
                 fontsize=13,
                 verticalalignment='top', bbox=props)
    plt.savefig(IMAGES_FOLDER / f"auc.png", dpi=1200, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    main()
