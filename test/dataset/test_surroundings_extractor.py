import unittest

import numpy as np
import pandas as pd

from config.constants import X_POSITION, Y_POSITION, Z_POSITION, CLASS
from src.dataset.surroundings_calculation import surroundings_extractor


class TestSurroundingsExtractor(unittest.TestCase):
    extractor = surroundings_extractor.SurroundingsExtractor

    def setUp(self):
        self.test_protein_data = pd.DataFrame({
            X_POSITION: [0, 1, 2, 0, 0],
            Y_POSITION: [0, 0, 0, 1, 2],
            Z_POSITION: [0, 0, 0, 0, 0],
            "id":       [0, 1, 2, 3, 4],
        })

    def test_manhattan_distance(self):
        x = [1, 2, 3]
        y = [6, 5, 4]
        expected_result = 9
        result = self.extractor.manhattan_distance(x, y)
        self.assertEqual(result, expected_result)

    def test_get_surrounding1(self):
        surrounding_size = 1
        origin = 0
        result = self.extractor.get_surrounding(self.test_protein_data, origin, surrounding_size)
        self.assertListEqual(result["id"].sort_values().to_list(), [0])

    def test_get_surrounding3(self):
        surrounding_size = 3
        origin = 0
        result = self.extractor.get_surrounding(self.test_protein_data, origin, surrounding_size)
        self.assertListEqual(result["id"].sort_values().to_list(), [0, 1, 3])

    def test_get_complete_dataset_simple(self):
        surrounding_size = 1
        result = self.extractor.get_complete_dataset(self.test_protein_data, surrounding_size)
        pd.testing.assert_frame_equal(self.test_protein_data, pd.concat(result))

    def test_get_balanced_dataset(self):
        self.test_protein_data[CLASS] = [b'1', b'1', b'0', b'0', b'0']
        surrounding_size = 3
        result = self.extractor.get_balanced_dataset(self.test_protein_data, surrounding_size)
        self.assertEqual(len(result), 4)

    def test_get_n_first_dataset(self):
        surrounding_size = 3
        n_samples = 2
        result = self.extractor.get_n_first_dataset(self.test_protein_data, surrounding_size, n_samples)
        self.assertEqual(len(result), n_samples)



if __name__ == '__main__':
    unittest.main()
