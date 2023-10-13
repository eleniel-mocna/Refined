from pathlib import Path

DATA_FOLDER = Path("data")
RAW_DATA_FOLDER = DATA_FOLDER / "raw"
EXTRACTED_DATA_FOLDER = DATA_FOLDER / "extracted"


X_POSITION = "xyz.x"
Y_POSITION = "xyz.y"
Z_POSITION = "xyz.z"
CLASS = "@@class@@"
SURROUNDINGS_FILE = "surroundings_dataset.pckl"
DATASET_PICKLE_FILE = "allArffs.pckl"

DATASET_NAMES = ["chen11", "coach420", "holo4k", "joined"]
