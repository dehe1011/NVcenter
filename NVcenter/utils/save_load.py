import json
import yaml
import os
import logging

from .. import ROOT_DIR, DATA_DIR, PROJECT_NAME, VERBOSE

# -------------------------------------------------------------------

# ----------------------------- Logging -----------------------------

def setup_logging():
    log_folder = os.path.join(DATA_DIR, "logs")
    os.makedirs(log_folder, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(log_folder, "NVcenter.log"),
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# ----------------------------- YAML -----------------------------

def load_yaml(filepath):
    with open(filepath, 'r') as file:
            try:
                data = yaml.safe_load(file)
                return data
            except yaml.YAMLError as exc:
                message = f"Error in reading the file {filepath}."
                logger.warning(message)
                if VERBOSE:
                    print(message)
            
def get_defaults():
    filepath = os.path.join(ROOT_DIR, PROJECT_NAME, "defaults.yaml")
    return load_yaml(filepath)

# Load the default values
DEFAULTS = get_defaults()

# ----------------------------- JSON -----------------------------

def load_json(filepath):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        message = f"File {filepath} does not exist."
        logger.warning(message)
        if VERBOSE:
            print(message)