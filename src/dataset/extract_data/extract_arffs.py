import gzip
import os
import pickle
import shutil
from glob import glob
from pathlib import Path
from typing import List

import pandas as pd
from pandas import DataFrame
from scipy.io.arff import loadarff

from config.config import Config


def _unpack(file: str) -> str:
    unpacked_file = ".".join(file.split(".")[:-1])
    with gzip.open(file, 'rb') as f_in:
        with open(unpacked_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return unpacked_file


def _load_arffs(path) -> List[DataFrame]:
    all_files = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.arff.gz'))]
    limit = Config.default().extraction_size
    if limit:
        all_files = all_files[:limit]
    ret = [pd.DataFrame(loadarff(_unpack(file))[0]) for file in all_files]

    return ret


def _extract_arffs(original_folder: Path, pckl_file: Path):
    """
        Unpack all arffs from the `original_folder` into the `pckl_file`.
    @param original_folder: folder from which all arffs recursively are extracted
    @param pckl_file: pickle file of a list of pandas Dataframe
    """
    arffs = _load_arffs(original_folder)
    pckl_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pckl_file, "wb") as file:
        pickle.dump(arffs, file)


def main():
    config = Config.default()
    for dataset in config.extract_dataset:
        print(f"Extracting {dataset}...")
        output_file_path = Config.get_extracted_path(os.path.basename(dataset))
        _extract_arffs(dataset, output_file_path)
        print(f"Extracted {dataset} to {output_file_path}.")


if __name__ == '__main__':
    main()
