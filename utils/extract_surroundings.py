import os
import pickle
from time import time
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd


def manhattan_distance(x, y):
    ret = 0
    for i in range(len(x)):
        ret += abs(x[i] - y[i])
    return ret


def get_surrounding(protein: pd.DataFrame, origin: int, surrounding_size: int) -> pd.DataFrame:
    origin_location = protein[["xyz.x", "xyz.y", "xyz.z"]].iloc[origin]
    closest = protein[["xyz.x", "xyz.y", "xyz.z"]].apply(lambda x: manhattan_distance(x, origin_location),
                                                         axis=1).sort_values().index[:surrounding_size]
    return protein.iloc[closest]


def get_complete_dataset(protein: pd.DataFrame, surrounding_size: int) -> List[pd.DataFrame]:
    n_samples = protein["@@class@@"].shape[0]
    return get_n_first_dataset(protein, surrounding_size, n_samples)


def get_balanced_dataset(protein: pd.DataFrame, surrounding_size: int) -> List[pd.DataFrame]:
    n_samples = (protein["@@class@@"] == b'1').sum() * 2
    return get_n_first_dataset(protein, surrounding_size, n_samples)


def get_n_first_dataset(protein: pd.DataFrame, surrounding_size: int, n_samples: int) -> List[pd.DataFrame]:
    executor: joblib.Parallel = joblib.Parallel(n_jobs=-1, verbose=0)
    return executor(
        joblib.delayed(lambda x: get_surrounding(protein, x, surrounding_size))(i) for i in range(n_samples))


def extract_surroundings(proteins: List[pd.DataFrame], surrounding_size: int, name="surroundings_dataset",
                         function=get_balanced_dataset) \
        -> Tuple[np.ndarray, np.ndarray]:
    pd_samples: List[pd.DataFrame] = []
    labels: List[pd.DataFrame] = []
    times = []
    for i in range(len(proteins)):
        print(f"Extracting from protein {i}/{len(proteins)}")
        start_time = time()
        new_samples = function(proteins[i], surrounding_size)
        pd_samples += [sample.drop("@@class@@", axis=1).to_numpy().flatten() for sample in new_samples]
        labels += [sample.iloc[0]["@@class@@"] for sample in new_samples]
        total_time = time() - start_time
        times.append(total_time)
        print(f"  took: {total_time}s, average: {np.mean(times)}s, ETA: {np.mean(times) * (len(proteins) - i - 1)}")
        if i % 100 == 0:
            file_name = f"{name}_{i}.pckl"
            print(f"Saving progress to {file_name}")
            with open(file_name, "wb") as file:
                pickle.dump((pd_samples, labels), file)
    return np.array(pd_samples), np.array(labels)


def main():
    folder = "refined"
    print("Loading the original dataset")
    with open("../allArffs.pckl", "rb") as file:
        arffs = pickle.load(file)

    print("Original dataset loaded, modifying labels...")
    for i in range(arffs.__len__()):
        arffs[i]["@@class@@"] = arffs[i]["@@class@@"] == b'1'
    print("Labels modified, generating surrounding dataset...")
    surroundings_dataset = extract_surroundings(arffs, 30)
    print("Dataset generated, pickling...")
    with open(os.path.join(folder, "surroundings_dataset.pckl"), "wb") as file:
        pickle.dump(surroundings_dataset, file)
    print("Dataset saved succesfully.")


if __name__ == '__main__':
    main()
