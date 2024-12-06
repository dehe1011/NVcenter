import json
import yaml
import os

from .. import ROOT_DIR, PROJECT_NAME

# -----------------------------------------------


def load_yaml(filepath):
    """Load a yaml file and return the data."""

    with open(filepath, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data


def get_defaults():
    """Load the default values from the defaults.yaml file."""

    filepath = os.path.join(ROOT_DIR, PROJECT_NAME, "defaults.yaml")
    return load_yaml(filepath)


# Load the default values
DEFAULTS = get_defaults()


def load_json(filepath):
    """Load a json file and return the data."""

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
