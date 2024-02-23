from pathlib import Path

DEFAULT_CONFIG_FILE = Path("config.json")

DATA_FOLDER = Path("data")
RAW_DATA_FOLDER = DATA_FOLDER / "raw"
EXTRACTED_DATA_FOLDER = DATA_FOLDER / "extracted"
SURROUNDINGS_DATA_FOLDER = DATA_FOLDER / "surroundings"

MODELS_FOLDER = DATA_FOLDER / "models"


X_POSITION = "xyz.x"
Y_POSITION = "xyz.y"
Z_POSITION = "xyz.z"
CLASS = "@@class@@"

DATASET_PICKLE_FILE = "allArffs.pckl"

DATASET_NAMES = ["chen11", "coach420", "holo4k", "joined"]
CONFIG_JSON = "config.json"
MODEL_PKL = "model.pkl"
