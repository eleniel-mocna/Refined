from abc import ABC, abstractmethod
import numpy as np


class GenericProteinModel(ABC):
    @abstractmethod
    def predict(self, protein: np.ndarray) -> np.ndarray:
        ...
