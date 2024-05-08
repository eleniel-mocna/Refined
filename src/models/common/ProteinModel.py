import pickle
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.config import Config
from config.constants import CONFIG_JSON, MODEL_PKL, CLASS
from dataset.surroundings_calculation.surroundings_extractor import SurroundingsExtractor


class ProteinModel(ABC):
    result_folder: Optional[Path] = None

    @abstractmethod
    def predict(self, protein: pd.DataFrame) -> np.ndarray:
        ...

    @abstractmethod
    def predict_proba(self, protein: pd.DataFrame) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def get_result_folder(self) -> Path:
        if self.result_folder is None:
            self.result_folder = Config.get_model_folder(self.name)
            self.result_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Config.get_instance().config_file, self.result_folder / CONFIG_JSON)

        return self.result_folder

    def save_model(self):
        with open(self.get_result_folder() / MODEL_PKL, "wb") as file:
            pickle.dump(self, file)


class SurroundingsProteinModel(ProteinModel, ABC):
    @abstractmethod
    def predict_surroundings(self, protein: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def predict_surroundings_proba(self, protein: np.ndarray) -> np.ndarray:
        ...

    def predict(self, protein: pd.DataFrame) -> np.ndarray:
        input_dataset = SurroundingsExtractor.get_complete_dataset(protein, Config.default().surroundings_size)
        input_data = np.stack(
            [sample.drop(CLASS, axis=1, errors="ignore").to_numpy().flatten() for sample in input_dataset])
        return self.predict_surroundings(input_data)

    def predict_proba(self, protein: pd.DataFrame) -> np.ndarray:
        input_dataset = SurroundingsExtractor.get_complete_dataset(protein, Config.default().surroundings_size)
        input_data = np.stack(
            [sample.drop(CLASS, axis=1, errors="ignore").to_numpy().flatten() for sample in input_dataset])
        return self.predict_surroundings_proba(input_data)
