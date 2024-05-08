import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config.constants import DATASET_NAMES, RAW_DATA_FOLDER, EXTRACTED_DATA_FOLDER, DEFAULT_CONFIG_FILE, MODELS_FOLDER, \
    SURROUNDINGS_DATA_FOLDER, PROTEIN_LENGTHS_FOLDER


class Config:
    instance: Optional['Config'] = None

    def __init__(self, config_file: Path):
        """
        DO NOT CREATE THIS CLASS MANUALLY, USE `Config.create_instance()` or `Config.get_instance()` INSTEAD.
        """
        self.__config_file = config_file
        self._config_data = self.load_config(self.__config_file)

    @property
    def config_file(self) -> Path:
        return self.__config_file

    @staticmethod
    def load_config(config_file: Path) -> dict:
        with open(config_file, 'r') as file:
            data = json.load(file)
        return data

    @property
    def train_extracted(self) -> Path:
        return Config.get_extracted_path(self._config_data['train_dataset'])

    @property
    def test_extracted(self) -> Path:
        return Config.get_extracted_path(self._config_data['test_dataset'])

    @property
    def train_surroundings(self) -> Path:
        return Config.get_surroundings_path(self._config_data['train_dataset'])

    @property
    def test_surroundings(self) -> Path:
        return Config.get_surroundings_path(self._config_data['test_dataset'])

    @property
    def test_lengths(self) -> List[int]:
        try:
            with open(Config.get_lengths_path(self._config_data['test_dataset']), 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            with open(f"/storage/brno12-cerit/home/eleniel/refined/data/lengths/{self._config_data['test_dataset']}.json") as f:
                return json.load(f)

    @property
    def train_lengths(self) -> List[int]:
        try:
            with open(Config.get_lengths_path(self._config_data['train_dataset']), 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            with open(f"/storage/brno12-cerit/home/eleniel/refined/data/lengths/{self._config_data['train_dataset']}.json") as f:
                return json.load(f)

    @property
    def surroundings_size(self) -> int:
        return self._config_data['surroundings_size']

    @staticmethod
    def default() -> 'Config':
        if Config.instance is None:
            Config.create_instance(DEFAULT_CONFIG_FILE)
        return Config.instance

    @staticmethod
    def get_instance() -> 'Config':
        if Config.instance is None:
            return Config.default()
        return Config.instance

    @staticmethod
    def create_instance(config_file: Path) -> 'Config':
        if Config.instance is None:
            Config.instance = Config(config_file)
            return Config.instance
        raise ValueError("Config instance already exists.")

    @property
    def extract_dataset(self) -> List[Path]:
        """
        @return: A list of Path objects representing the dataset files to be extracted.
        """
        if self._config_data['extract_dataset'] == "*":
            return [RAW_DATA_FOLDER / name for name in DATASET_NAMES]
        if self._config_data['extract_dataset'] in DATASET_NAMES:
            return [RAW_DATA_FOLDER / self._config_data['extract_dataset']]
        raise ValueError(
            f"Unknown dataset name: {self._config_data['extract_dataset']},"
            f"if you want to add a new dataset, add it to `config/constants.py`#DATASET_NAMES.")

    @property
    def model_splits(self) -> int:
        return self._config_data['model_splits']

    @staticmethod
    def get_extracted_path(dataset_name: str) -> Path:
        return EXTRACTED_DATA_FOLDER / f"{dataset_name}.pckl"

    @staticmethod
    def get_surroundings_path(dataset_name: str) -> Path:
        return SURROUNDINGS_DATA_FOLDER / f"{dataset_name}.pckl"

    @staticmethod
    def get_lengths_path(dataset_name: str) -> Path:
        return PROTEIN_LENGTHS_FOLDER / f"{dataset_name}.json"

    @staticmethod
    def get_model_folder(model_name: str) -> Path:
        return MODELS_FOLDER / model_name / datetime.now().strftime('%y-%m-%d--%H-%M-%S')
