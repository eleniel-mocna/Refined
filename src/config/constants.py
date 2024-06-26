from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_FILE = ROOT / "config.json"

DATA_FOLDER = ROOT / "data"
RAW_DATA_FOLDER = DATA_FOLDER / "raw"
EXTRACTED_DATA_FOLDER = DATA_FOLDER / "extracted"
SURROUNDINGS_DATA_FOLDER = DATA_FOLDER / "surroundings"
PROTEIN_LENGTHS_FOLDER = DATA_FOLDER / "lengths"
IMAGES_FOLDER = DATA_FOLDER / "images"

MODELS_FOLDER = DATA_FOLDER / "models"

REFINED_ORDERS = DATA_FOLDER / "REFINED_values.json"
PROGRESSION_EVAL_RESULTS = DATA_FOLDER / "progression_results.json"
PROGRESSION_MODEL_STATS = DATA_FOLDER / "progression_model_stats.json"
EXPERIMENT1_STATS = DATA_FOLDER / "experiment_1.json"


X_POSITION = "xyz.x"
Y_POSITION = "xyz.y"
Z_POSITION = "xyz.z"
CLASS = "@@class@@"

DATASET_PICKLE_FILE = "allArffs.pckl"

DATASET_NAMES = ["chen11", "coach420"]
CONFIG_JSON = "config.json"
MODEL_PKL = "model.pkl"
