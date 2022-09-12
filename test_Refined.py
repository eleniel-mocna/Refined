from unittest import TestCase

import numpy as np
from numpy import ndarray

from Refined import Refined


class TestRefined(TestCase):
    def test_refined(self):
        rows = 9
        cols = 10
        n_samples = 100
        samples = generate_samples(n_samples, rows * cols)[0]
        refined = Refined(samples, rows, cols, "test_folder")
        refined.run()
        self.assertTrue(refined.transform(samples).shape == (n_samples, rows, cols))


def generate_samples(n_samples, dimensions, base_params=None, noise_var=1) -> tuple[ndarray, ndarray]:
    if base_params is None:
        base_params = (dimensions // 10) + 1
    samples = []
    base_samples = []
    distances = []
    labels = []
    zeros = np.zeros(base_params)
    corrs = np.random.normal(0, 1, base_params)
    corrs_with = np.random.randint(0, base_params, size=dimensions)
    for i in range(n_samples):
        sample_base = np.random.normal(0, 1, base_params)
        base_samples.append(sample_base)
        sample = np.array([corrs[i] * sample_base[i] for i in corrs_with])
        distances.append(np.sum(np.square(zeros - sample_base)))
        samples.append(sample)
    quarter1 = np.quantile(distances, 0.25)
    quarter3 = np.quantile(distances, 0.75)
    for i in range(n_samples):
        labels.append(
            quarter1 < distances[i] + np.random.normal(0, noise_var) < quarter3)
    return np.array(samples), np.array(labels)
