import json
from pathlib import Path
from typing import List

from config.constants import DATASET_NAMES, RAW_DATA_FOLDER


class Config:
    def __init__(self, config_file: Path):
        self.__config_file = config_file
        self.__config_data = self.load_config(self.__config_file)

    @staticmethod
    def load_config(config_file: Path) -> dict:
        with open(config_file, 'r') as file:
            data = json.load(file)
        return data

    @staticmethod
    def default():
        return Config(Path('config.json'))

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
