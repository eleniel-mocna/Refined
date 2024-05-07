import json

import matplotlib.pyplot as plt

from config.constants import DATA_FOLDER, IMAGES_FOLDER

refined_models = {
    'rfc_baseline': "P2Rank RFC",
    'rfc_surrounding': "RFC Surrounding",
    'NN': "NN (36k params, lr=5e-4)",
    'Refined': "REFINED (1.8M params, lr=5e-4)",
    'Random CNN': "Random CNN (1M params, lr=5e-5)",
    'Normalized CNN': "Normalized CNN (25M params, lr=5e-5)",
}

colors = ["b", "r", "g", "y", "c"]


def main():
    data = json.load(open(DATA_FOLDER / "training_logs.json"))
    for model, values in data.items():
        for i, run in enumerate(values):
            losses = [x["loss"] for x in run]
            val_losses = [x["val_loss"] for x in run]
            plt.plot(range(1, len(losses) + 1), losses, colors[i])
            plt.plot(range(1, len(losses) + 1), val_losses, colors[i] + "--")
        plt.title(refined_models[model])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["loss", "val_loss"])
        plt.savefig(IMAGES_FOLDER / f"{model}_loss.png", dpi=1200, bbox_inches='tight')
        plt.clf()


if __name__ == '__main__':
    main()
