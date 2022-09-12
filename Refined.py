import datetime
import os
import sys

import numpy as np

import distances
from refined_functions import HOF, HCA, fitness, neighbours


class Refined:
    hof: HOF

    def __init__(self,
                 samples: np.ndarray,
                 rows: int,
                 cols: int,
                 folder_path: str,
                 distances_split: float = 1.,
                 distances_logging: int = 10,
                 distances_n_jobs: int = 0,
                 hca_logging: int = 10,
                 hca_starts: int = 10,
                 ):
        """
        Create an object for the REFINED algorithm
        :param samples: Array of shape (n_samples, n_dimensions)
        :param rows: Number of rows in resulting image from 1 sample
        :param cols: Number of columns in resulting image from 1 sample
        :param folder_path: Path where to save all needed data
        :param distances_split: What fraction of samples should be used to
            calculate the interdimensional distances (Use 1 by default,
            lower numbers may give wildly worse results, but will be
            definitely faster.
        :param distances_logging: How often should be progress from calculating
            distances logged (0 for none, 10 for all)
        :param distances_n_jobs: On how many cores should distances be
            calculated (0 for all)
        :param hca_logging: How often should HCA progress be logged (for `n`
            print log every `n` iterations)
        :param hca_starts: How many times should the HCA be restarted from
            a random position
        """
        self.dists: np.ndarray = None
        self.distances_n_jobs = distances_n_jobs
        self.distances_split = distances_split
        self.hca_logging = hca_logging
        self.distances_logging = distances_logging
        self.hca_starts = hca_starts
        self.path = folder_path
        self.cols = cols
        self.rows = rows
        self.samples = samples
        self._prepare_folder()
        self._check_inputs()
        self._log("Computing dists")
        self._compute_dists()

    def run(self):
        self._log("REFINED started.")
        self._compute_refined()

    @staticmethod
    def transform_from_vector(samples: np.ndarray,
                              refined_vector: np.ndarray,
                              rows: int,
                              cols: int) -> np.ndarray:
        """

        :param samples: Array of shape (n_samples, n_dimensions)
        :param refined_vector: Vector of shape (n_dimensions)
        :param rows: Number of rows in the resulting image
        :param cols: Number of cols in the resulting image
        :return: Array of shape (n_samples, rows, cols)
        """
        return np.array(
            [sample[refined_vector.reshape(1, rows * cols)]
             .reshape(rows, cols)
             for sample in samples])

    def transform(self, samples: np.ndarray) -> np.ndarray:
        """
        Transform given samples to REFINED images
        :param samples: Array of shape (n_samples, n_dimensions)
        :return: Array of shape (n_samples, rows, cols)
        """
        return Refined.transform_from_vector(samples, self.hof.best_individual, self.rows, self.cols)

    def _check_inputs(self):
        if len(self.samples.shape) != 2 or self.samples.shape[1] != self.rows * self.cols:
            error_msg = (
                f"Samples expected to have shape({self.samples.shape[0]}, {self.rows * self.cols}), "
                f"instead found {self.samples.shape}. "
                f"Are samples a np.array of shape (n_samples, n_dims) and have you "
                f"specified rows and cols correctly?")
            self._log(error_msg)
            raise AttributeError(error_msg)
        self._log("Inputs checked successfully.")

    def _prepare_folder(self):
        try:
            os.mkdir(self.path)
        except FileExistsError:
            pass
        self.logging_file = os.path.join(self.path, "log")

    def _log(self, msg: str):
        whole_msg: str = (
            f"[REFINED @ {datetime.datetime.now()}] - {msg}")
        print(whole_msg, file=sys.stderr)
        with open(self.logging_file, "w") as file:
            print(whole_msg, file=file)

    def _compute_dists(self):
        self.dists = distances.distances_parallel(self.samples,
                                                  verbosity=self.distances_logging,
                                                  used_split=self.distances_split,
                                                  n_jobs=self.distances_n_jobs)
        np.save(os.path.join(self.path, "dists.npy"), self.dists)

    def _compute_refined(self):
        if type(self.dists) is not np.ndarray:
            error_msg = "Dists have not been calculated when refined was called!"
            self._log(error_msg)
            raise AttributeError(error_msg)
        self.hof = HOF(logging_freq=self.hca_logging)
        for i in range(self.hca_starts):
            individual = np.arange(self.rows * self.cols)
            np.random.shuffle(individual)
            HCA(individual, lambda x: fitness(x, self.dists, self.rows, self.cols), neighbours, self.rows, self.cols, self.hof)
        np.save(os.path.join(self.path, "best_individual.npy"), self.hof.best_individual)
