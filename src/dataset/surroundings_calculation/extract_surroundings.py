import pickle

from config.config import Config
from config.constants import CLASS
from dataset.surroundings_calculation.surroundings_extractor import SurroundingsExtractor


def main():
    config = Config.get_instance()
    input_path = config.test_extracted
    output_path = config.test_surroundings

    # print("Loading the original dataset")
    with open(input_path, "rb") as file:
        arffs = pickle.load(file)

    for i in range(3):
        arffs[i][CLASS] = (arffs[i][CLASS] == b'1')

    surroundings_dataset = SurroundingsExtractor.extract_surroundings(arffs, 30)

    with open(output_path, "wb") as file:
        pickle.dump(surroundings_dataset, file)


if __name__ == '__main__':
    main()
