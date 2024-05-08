import json
import os
import pickle
from typing import Optional

import pandas as pd

from config.constants import EXTRACTED_DATA_FOLDER, PROTEIN_LENGTHS_FOLDER


def main(dataset: Optional[str] = None):
    if dataset is None:
        pickle_files = [f for f in os.listdir(EXTRACTED_DATA_FOLDER) if f.endswith('.pckl')]
        for file in pickle_files:
            with open(os.path.join(EXTRACTED_DATA_FOLDER, file), 'rb') as f:
                data = pickle.load(f)
            lengths = [len(df) for df in data if isinstance(df, pd.DataFrame)]
            protein_lengths_file = PROTEIN_LENGTHS_FOLDER / f'{os.path.splitext(file)[0]}.json'
            protein_lengths_file.parent.mkdir(parents=True, exist_ok=True)
            with open(protein_lengths_file, 'w') as f:
                json.dump(lengths, f)


if __name__ == '__main__':
    main()
