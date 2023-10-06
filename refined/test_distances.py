from unittest import TestCase, skip

import numpy as np

from refined import distances


class Test(TestCase):
    @skip("TODO: Find a functional way to test this...")
    def test_distances_parallel(self):
        samples: np.ndarray = np.random.randint(0, 100, size=(10, 20))
        parallel_result: np.ndarray = distances.distances_parallel(samples)
        single_result: np.ndarray = distances.distances(samples)
        equals = (parallel_result == single_result).all()
        self.assertTrue(equals, "Parallel and single distances are different")

    def test_distances_split(self):
        samples: np.ndarray = np.random.randint(0, 100, size=(10, 20))
        result: np.ndarray = distances.distances_parallel(samples, verbosity=0)
        approximate_result: np.ndarray = distances.distances_parallel(samples, used_split=0.8, verbosity=0)
        difference = np.mean(result - approximate_result)
        relative_difference = difference / np.mean(result)
        self.assertTrue(relative_difference < 0.35)

    def test_distances_shape(self):
        samples: np.ndarray = np.random.randint(0, 100, size=(10, 20))
        self.assertEqual((20, 20), distances.distances_parallel(samples).shape)


if __name__ == '__main__':
    pass
