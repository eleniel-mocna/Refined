import pickle
from time import time
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

from config.constants import X_POSITION, Y_POSITION, Z_POSITION, CLASS


class SurroundingsExtractor:
    def __init__(self, protein:pd.DataFrame):
        self.protein = protein

    @staticmethod
    def manhattan_distance(x: List[int], y: List[int]) -> int:
        return np.abs(np.array(x) - np.array(y)).sum()

    @staticmethod
    def get_surrounding(protein: pd.DataFrame, origin: int, surrounding_size: int) -> pd.DataFrame:
        """
        @param protein: DataFrame containing protein data
        @param origin: Index of the origin position within the protein DataFrame
        @param surrounding_size: Number of closest positions to include in the surrounding
        @return: DataFrame containing the surrounding positions

        """
        origin_location = protein[[X_POSITION, Y_POSITION, Z_POSITION]].iloc[origin]
        closest = protein[[X_POSITION, Y_POSITION, Z_POSITION]].apply(
            lambda x: SurroundingsExtractor.manhattan_distance(x, origin_location), axis=1).sort_values().index[
                  :surrounding_size]
        return protein.iloc[closest]

    @staticmethod
    def get_complete_dataset(protein: pd.DataFrame, surrounding_size: int) -> List[pd.DataFrame]:
        n_samples = protein.shape[0]
        return SurroundingsExtractor.get_n_first_dataset(protein, surrounding_size, n_samples)

    @staticmethod
    def get_balanced_dataset(protein: pd.DataFrame, surrounding_size: int) -> List[pd.DataFrame]:
        n_samples = (protein[CLASS] == b'1').sum() * 2
        return SurroundingsExtractor.get_n_first_dataset(protein, surrounding_size, n_samples)

    @staticmethod
    def get_n_first_dataset(protein: pd.DataFrame, surrounding_size: int, n_samples: int) -> List[pd.DataFrame]:
        executor: joblib.Parallel = joblib.Parallel(n_jobs=-1, verbose=0)
        return executor(
            joblib.delayed(lambda x: SurroundingsExtractor.get_surrounding(protein, x, surrounding_size))(i) for i in
            range(n_samples))

    @staticmethod
    def extract_surroundings(proteins: List[pd.DataFrame], surrounding_size: int,
                             function=None) -> Tuple[np.ndarray, np.ndarray]:
        if function is None:
            function = SurroundingsExtractor.get_complete_dataset
        pd_samples: List[pd.DataFrame] = []
        labels: List[pd.DataFrame] = []
        times = []
        for i in range(len(proteins)):
            print(f"Extracting from protein {i}/{len(proteins)}")
            start_time = time()
            new_samples = function(proteins[i], surrounding_size)
            pd_samples += [sample.drop(CLASS, axis=1).to_numpy().flatten() for sample in new_samples]
            labels += [sample.iloc[0][CLASS] for sample in new_samples]
            total_time = time() - start_time
            times.append(total_time)
            print(f"  took: {total_time}s, average: {np.mean(times)}s, ETA: {np.mean(times) * (len(proteins) - i - 1)}")
        return np.array(pd_samples), np.array(labels)
