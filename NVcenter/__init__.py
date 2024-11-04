__version__ = "0.1.0"
__project__ = "NVcenter"

import pathlib
import os

ROOT_DIR = str(pathlib.Path(__file__).absolute().parent.parent)
DATA_DIR = os.path.join(ROOT_DIR, __project__, "data")