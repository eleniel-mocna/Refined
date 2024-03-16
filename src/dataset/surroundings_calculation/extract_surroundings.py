import pickle
import sys
from typing import Optional

from config.config import Config
from config.constants import CLASS
from dataset.surroundings_calculation.surroundings_extractor import SurroundingsExtractor


def main(dataset: Optional[str] = None):
    config = Config.get_instance()
    config._config_data['test_dataset'] = dataset
    input_path = config.test_extracted
    output_path = config.test_surroundings

    # print("Loading the original dataset")
    with open(input_path, "rb") as file:
        arffs = pickle.load(file)

    for i in range(len(arffs)):
        arffs[i][CLASS] = (arffs[i][CLASS] == b'1')

    surroundings_dataset = SurroundingsExtractor.extract_surroundings(arffs, 30)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:
        pickle.dump(surroundings_dataset, file)


if __name__ == '__main__':
    dataset_name = sys.argv[1] if len(sys.argv)>1 else None
    print(f"Dataset name: {dataset_name}")
    main(dataset_name)
