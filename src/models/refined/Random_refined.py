import numpy as np

from models.refined.image_transformer import ImageTransformer


class RandomRefined(ImageTransformer):
    def __init__(self,
                 rows: int,
                 cols: int):
        self.order = np.random.permutation(rows * cols)
        self.rows = rows
        self.cols = cols
    def transform(self,samples:np.ndarray):
        """
        Transform given samples to REFINED images
        :param samples: Array of shape (n_samples, n_dimensions)
        :return: Array of shape (n_samples, rows, cols)
        """
        return samples[:,self.order].reshape(samples.shape[0],self.rows,self.cols)

class RandomRefinedNormalized(ImageTransformer):
    def __init__(self,
                 samples: np.ndarray,
                 rows: int,
                 cols: int):
        self.stds = None
        self.means = None
        self.fit_normalize(samples)
        self.order = np.random.permutation(rows * cols)
        self.rows = rows
        self.cols = cols
    def transform(self,samples:np.ndarray):
        """
        Transform given samples to REFINED images
        :param samples: Array of shape (n_samples, n_dimensions)
        :return: Array of shape (n_samples, rows, cols)
        """
        return self.normalize(samples)[:,self.order].reshape(samples.shape[0],self.rows,self.cols)

    def fit_normalize(self, samples: np.ndarray):
        # Normalize each column to normal(0,1) and save the values for later
        self.means = np.mean(samples, axis=0)
        self.stds = np.std(samples, axis=0)

    def normalize(self, samples: np.ndarray) -> np.ndarray:
        return (samples - self.means) / self.stds
