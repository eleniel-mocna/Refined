from config.constants import EXTRACTED_DATA_FOLDER, PROTEIN_LENGTHS_FOLDER
import os
import pickle
import pandas as pd
import json


def main():
    # List all pickle files in the EXTRACTED_DATA_FOLDER
    pickle_files = [f for f in os.listdir(EXTRACTED_DATA_FOLDER) if f.endswith('.pckl')]

    # Create lists of lengths from each pickle file
    for file in pickle_files:
        with open(os.path.join(EXTRACTED_DATA_FOLDER, file), 'rb') as f:
            data = pickle.load(f)
        lengths = [len(df) for df in data if isinstance(df, pd.DataFrame)]

        # Save the lengths to PROTEIN_LENGTHS_FOLDER as a json file
        with open(os.path.join(PROTEIN_LENGTHS_FOLDER, f'{os.path.splitext(file)[0]}.json'), 'w') as f:
            json.dump(lengths, f)


if __name__ == '__main__':
    main()