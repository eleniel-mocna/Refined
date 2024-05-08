import pickle

from config.config import Config
from config.constants import CLASS
from dataset.surroundings_calculation.calculate_lengths import main as calculate_lengths
from dataset.surroundings_calculation.surroundings_extractor import SurroundingsExtractor


def main():
    config = Config.get_instance()
    for input_path in config.extracted_dataset:
        print(f"Extracting surroundings for dataset {input_path.name}...")
        output_path = config.get_surroundings_path(input_path.name.split(".")[0])
        with open(input_path, "rb") as file:
            arffs = pickle.load(file)

        num_elements = len(arffs)
        if config.surroundings_limit:
            num_elements = min(num_elements, config.surroundings_limit)

        arffs = arffs[:num_elements]
        for i in range(num_elements):
            arffs[i][CLASS] = (arffs[i][CLASS] == b'1')

        surroundings_dataset = SurroundingsExtractor.extract_surroundings(arffs, 30)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as file:
            pickle.dump(surroundings_dataset, file)
        print(f"Surroundings extracted succesfully from {input_path.name}.")


if __name__ == '__main__':
    main()
    # Just calculate all the lengths... It is not efficient, but that's life.
    calculate_lengths()
