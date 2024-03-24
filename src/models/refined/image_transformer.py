from abc import ABC, abstractmethod

import numpy as np


class ImageTransformer(ABC):
    @abstractmethod
    def transform(self, samples: np.ndarray):
        """
        Transform given samples to REFINED images
        :param samples: Array of shape (n_samples, n_dimensions)
        :return: Array of shape (n_samples, rows, cols)
        """
        ...
