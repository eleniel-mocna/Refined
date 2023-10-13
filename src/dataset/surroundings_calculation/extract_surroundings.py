import os
import pickle

from config.constants import CLASS, SURROUNDINGS_FILE, DATASET_PICKLE_FILE
from dataset.surroundings_calculation.surroundings_extractor import SurroundingsExtractor


def main():
    folder = "refined"
    print("Loading the original dataset")
    with open(DATASET_PICKLE_FILE, "rb") as file:
        arffs = pickle.load(file)
    print("Original dataset loaded, modifying labels...")
    for i in range(arffs.__len__()):
        arffs[i][CLASS] = arffs[i][CLASS] == b'1'
    print("Labels modified, generating surrounding dataset...")
    surroundings_dataset = SurroundingsExtractor.extract_surroundings(arffs, 30)
    print("Dataset generated, pickling...")
    with open(os.path.join(folder, SURROUNDINGS_FILE), "wb") as file:
        pickle.dump(surroundings_dataset, file)
    print("Dataset saved succesfully.")


if __name__ == '__main__':
    main()
