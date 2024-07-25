"""
This module contains some simple functions.
"""

import json
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).absolute().parent.parent)


def my_add(x, y):
    """
    Adds two numbers.
    """
    return x + y


def my_mul(x, y):
    """
    Multiplies two numbers.
    """
    return x * y


def my_load():
    """
    Loads parameters saved in a .json file.
    """
    with open(ROOT_DIR + "/data/params.json", "rb") as f:
        data = json.load(f)
    return data


