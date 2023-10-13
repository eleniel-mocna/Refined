import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config.constants import DATASET_NAMES, RAW_DATA_FOLDER, EXTRACTED_DATA_FOLDER, DEFAULT_CONFIG_FILE, MODELS_FOLDER


class Config:
    instance: Optional['Config'] = None

    def __init__(self, config_file: Path):
        """
        DO NOT CREATE THIS CLASS MANUALLY, USE `Config.create_instance()` or `Config.get_instance()` INSTEAD.
        """
        self.__config_file = config_file
        self.__config_data = self.load_config(self.__config_file)

    @property
    def config_file(self) -> Path:
        return self.__config_file

    @staticmethod
    def load_config(config_file: Path) -> dict:
        with open(config_file, 'r') as file:
            data = json.load(file)
        return data

    @property
    def train_dataset(self) -> Path:
        return Config.get_extracted_path(self.__config_data['train_dataset'])

    @property
    def test_dataset(self) -> Path:
        return Config.get_extracted_path(self.__config_data['test_dataset'])

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
        if self.__config_data['extract_dataset'] == "*":
            return [RAW_DATA_FOLDER / name for name in DATASET_NAMES]
        if self.__config_data['extract_dataset'] in DATASET_NAMES:
            return [RAW_DATA_FOLDER / self.__config_data['extract_dataset']]
        raise ValueError(
            f"Unknown dataset name: {self.__config_data['extract_dataset']},"
            f"if you want to add a new dataset, add it to `config/constants.py`#DATASET_NAMES.")

    @staticmethod
    def get_extracted_path(dataset_name: str) -> Path:
        return EXTRACTED_DATA_FOLDER / f"{dataset_name}.pckl"

    @staticmethod
    def get_model_folder(model_name: str) -> Path:
        return MODELS_FOLDER / model_name / datetime.now().strftime('%y-%m-%d--%H-%M-%S')
