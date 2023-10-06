import gzip
import os
import pickle
import shutil
import sys
from glob import glob
from typing import List

import joblib
import pandas as pd
from pandas import DataFrame
from scipy.io.arff import loadarff


def unpack(file: str) -> str:
    unpacked_file = ".".join(file.split(".")[:-1])
    with gzip.open(file, 'rb') as f_in:
        with open(unpacked_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return unpacked_file


def load_arffs(path, n_jobs=8, verbosity=10) -> List[DataFrame]:
    all_files = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.arff.gz'))]
    executor: joblib.Parallel = joblib.Parallel(n_jobs=n_jobs, verbose=verbosity)
    return executor(
        joblib.delayed(lambda x: pd.DataFrame(loadarff(unpack(x))[0]))(file) for file in all_files)
    # return list(map(lambda x: pd.DataFrame(loadarff(unpack(x))[0]), all_files))


def extract_arffs(original_folder: str, pckl_file: str):
    """
        Unpack all arffs from the `original_folder` into the `pckl_file`.
    @param original_folder: folder from which all arffs recursively are extracted
    @param pckl_file: pickle file of a list of pandas Dataframe
    """
    print("Unpacking arffs...")
    arffs = load_arffs(original_folder)
    print("Arffs unpacked.\nWriting to a pickle file.")
    with open(pckl_file, "wb") as file:
        pickle.dump(arffs, file)
    print("Arffs extracted succesfully.")


if __name__ == '__main__':
    if sys.argv.__len__() == 3:
        extract_arffs(sys.argv[1], sys.argv[2])
    else:
        print("Usage: Usage: `python extract_arffs.py <folder with arffs> <pckl file to save to>`")
        print("  e.g.: `python extract_arffs.py \"David_arffs/holo4k\" \"allArffs.pckl\"`")
