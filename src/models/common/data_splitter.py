import hashlib
import random
from typing import List, Tuple

import numpy as np

_hash_function = np.vectorize(lambda x: hash(hashlib.md5(str((((x + 5) * 31) ** 2) + 97).encode("ASCII")).hexdigest()))


class DataSplitter:
    def __init__(self, data: np.array, labels: np.array, protein_lengths: List[int] = None, splits: int = 5):
        if protein_lengths is None:
            protein_lengths = []
        self.labels = labels
        self.data = data
        self.protein_lengths = protein_lengths
        self.splits = splits
        self.data_splits = np.zeros(len(data))
        if self.splits == 1:
            return
        if self.protein_lengths:
            proteins_splits = _hash_function(np.arange(len(protein_lengths))) % splits
            current_index = 0
            for i, protein_length in enumerate(protein_lengths):
                current_index_end = current_index + protein_length
                self.data_splits[current_index:current_index_end] = proteins_splits[i]
                current_index = current_index_end
        else:
            self.data_splits = _hash_function(np.arange(len(data))) % splits

    def get_split(self, split: int) -> Tuple[np.array, np.array]:
        """
        Get (data, labels) for a specific split
        @param split:
        @return:
        """
        assert 0 <= split < self.splits
        if self.splits==1:
            return self.data, self.labels
        return self.data[self.data_splits != split], self.labels[self.data_splits != split]

if __name__ == '__main__':
    lengths = [random.randint(1, 10) for _ in range(100)]
    num_samples = sum(lengths)
    splitter = DataSplitter(np.zeros((num_samples, 10)), np.zeros(num_samples), lengths, 5)
    print(splitter.data_splits)