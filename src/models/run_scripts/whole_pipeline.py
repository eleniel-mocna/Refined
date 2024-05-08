"""
Run the whole pipeline and train all the models based on the config file
It recognizes the following arguments:
--skip-preprocessing: If specified, only the models will be trained
--run-refined: If specified, a new REFINED image transformer will be trained, otherwise a pretrained one will be used
--tune-hyperparameters: If specified, hyperparameter tuning will be done,
    otherwise previously found hyperparameters will be used.
"""

import sys

from dataset.extract_data.extract_arffs import main as run_extract_arffs
from dataset.surroundings_calculation.calculate_lengths import main as run_calculate_lengths
from dataset.surroundings_calculation.extract_surroundings import main as run_extract_surroundings
from models.run_scripts.NN import main as run_nn
from models.run_scripts.RandomCNN import main as run_random_cnn
from models.run_scripts.RandomCNN_Normalized import main as run_random_cnn_normalized
from models.run_scripts.RefinedCNN import main as run_refined_cnn
from models.run_scripts.baseline_model import main as run_baseline_model
from models.run_scripts.big_rfc import main as run_big_rfc

if __name__ == '__main__':
    tune_hyperparams = "--tune-hyperparameters" in sys.argv
    run_refined = "--run-refined" in sys.argv

    if "--skip-preprocessing" not in sys.argv and False:
        run_extract_arffs()
        run_extract_surroundings()
        run_calculate_lengths()

    run_baseline_model()
    run_refined_cnn(run_refined, tune_hyperparams)
    run_random_cnn_normalized(tune_hyperparams)
    run_nn(tune_hyperparams)
    run_random_cnn(tune_hyperparams)
    run_big_rfc(tune_hyperparams)
