__version__ = "0.1.0"

PROJECT_NAME = "NVcenter"
VERBOSE = True

import pathlib
import os

ROOT_DIR = str(pathlib.Path(__file__).absolute().parent.parent)
DATA_DIR = os.path.join(ROOT_DIR, PROJECT_NAME, "data")

from .utils import *

import matplotlib as mpl
mpl.rcdefaults()
try:
    mpl.style.use("qDNA-default")
except OSError:
    print("Could not load qDNA-default style. Using seaborn-v0_8-paper style instead.")
    mpl.style.use("seaborn-v0_8-paper")

from .helpers import *
from .spin_bath import *
from .spin import *
from .two_spin_system import *
from .spins import *
from .hamiltonian import *
from .pulse import *
from .environment import *
from .visualization import *

from .suter import *
