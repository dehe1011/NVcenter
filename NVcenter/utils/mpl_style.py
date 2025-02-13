import os
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt

from .. import ROOT_DIR, PROJECT_NAME

# -------------------------------------------------


def install_mpl_style():
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
    mplstyle_path = os.path.join(ROOT_DIR, PROJECT_NAME, "NVcenter-default.mplstyle")
    shutil.copy(mplstyle_path, stylelib_dir)
    plt.style.reload_library()

    mpl.rcdefaults()
    mpl.style.use("NVcenter-default")
