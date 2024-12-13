import os
import shutil
import time
from functools import wraps

import matplotlib as mpl
import matplotlib.pyplot as plt

from .. import ROOT_DIR, PROJECT_NAME

# -------------------------------------------------

def install_style():
    """
    Install the custom Matplotlib style for the QuantumDNA project.
    This function copies the custom Matplotlib style file `qDNA-default.mplstyle`
    from the project's directory to the Matplotlib configuration directory's
    `stylelib` folder. If the `stylelib` folder does not exist, it will be created.
    After copying the style file, the Matplotlib style library is reloaded to
    apply the new style.
    """

    config_dir = mpl.get_configdir()
    stylelib_dir = os.path.join(config_dir, "stylelib")
    os.makedirs(stylelib_dir, exist_ok=True)
    mplstyle_path = os.path.join(ROOT_DIR, PROJECT_NAME, "qDNA-default.mplstyle")
    shutil.copy(mplstyle_path, stylelib_dir)
    plt.style.reload_library()


# install_style()


def timeit(f):
    """A decorator that measures the execution time of a function.

    Parameters
    ----------
    f : function
        The function whose execution time is to be measured.

    Returns
    -------
    function
        A wrapped function that prints the execution time when called.

    Examples
    --------
    >>> @timeit
    >>> def example_function():
    >>>     # function implementation
    >>>     pass
    """

    @wraps(f)
    def timed(*args, **kw):
        start = time.time()
        result = f(*args, **kw)
        end = time.time()
        print(f"Time: {round(end-start,2)} s")
        return result

    return timed

