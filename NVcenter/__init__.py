__version__ = "0.1.0"

PROJECT_NAME = "NVcenter"
VERBOSE = True

import pathlib
import os

ROOT_DIR = str(pathlib.Path(__file__).absolute().parent.parent)
DATA_DIR = os.path.join(ROOT_DIR, PROJECT_NAME, "data")

from .utils import *
from .helpers import *
from .spin_config import *
from .spin import *
from .two_spin_system import *
from .system import *
from .pulses import *