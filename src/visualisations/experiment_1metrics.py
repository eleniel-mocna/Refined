import glob
import json

import matplotlib.pyplot as plt
import numpy as np

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
    """
    Create plots for F1_score, Precision, Accuracy, Recall
    """
    root = DATA_FOLDER / "for_transfer"
    metrics_files = glob.glob(str(root / '**/metrics.json'), recursive=True)
    metrics = [json.load(open(file)) for file in metrics_files]
    create_metrics_plot(metrics)


alpha = 0.01


def create_metrics_plot(metrics):
    metrics_list = ["F1_score", "Precision", "Accuracy", "Recall"]
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()

    for metric_index, metric_name in enumerate(metrics_list):
        aucs = dict()
        for metric in metrics:
            model_name = metric["model_name"].lower()
            auc = float(metric[metric_name])
            if model_name in aucs:
                aucs[model_name].append(auc)
            else:
                aucs[model_name] = [auc]
        models = list(refined_models.values())

        aucs = {refined_models[k]: v for k, v in aucs.items()}
        mean_aucs = {model: np.mean(vals) for model, vals in aucs.items()}
        means = [mean_aucs[model] for model in models]

        stds = {model: np.std(vals) for model, vals in aucs.items()}
        stds = [stds[model] for model in models]

        axs[metric_index].bar(models, means, yerr=stds, capsize=10)
        axs[metric_index].set_xticks(range(len(models)), models, rotation=45, ha='right')
        axs[metric_index].set_xlabel("Model")
        axs[metric_index].set_ylabel(metric_name)
        axs[metric_index].set_title(metric_name)

        for i in range(len(models)):
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axs[metric_index].text(i - 0.0, means[i] - 0.1, f"\u03bc={round(means[i], 2)}\n \u03C3={round(stds[i], 3)}",
                                   ha='center', fontsize=13, verticalalignment='top', bbox=props, rotation=90)

    plt.tight_layout()
    plt.savefig(IMAGES_FOLDER / f"metric_comparisons.png", dpi=1200, bbox_inches='tight')


if __name__ == '__main__':
    main()
