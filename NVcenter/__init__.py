__version__ = "0.1.0"

PROJECT_NAME = "NVcenter"

import pathlib
import os

ROOT_DIR = str(pathlib.Path(__file__).absolute().parent.parent)
DATA_DIR = os.path.join(ROOT_DIR, PROJECT_NAME, "data")

from .utils import *

from .spin_bath import *
from .spin import *
from .spins import *
from .hamiltonian import *
from .evolution import *
from .environment2 import *

from .pulse import *
from .environment import *

from .visualization import *
from .export import *
from .suter import *

from .brecht import *
