from itertools import combinations
import numpy as np

from . import DEFAULTS
from .spin import Spin

# -------------------------------------------------


class Spins:
    """
    A class to represent a system of spins, including both register and bath spins, and to configure
    the system and mean-field parts based on different approximation levels.

    Parameters
    ----------
    register_config : list of tuples
        Configuration of the register spins. Each tuple contains (spin_type, position, initial_spin).
    bath_config : list of tuples
        Configuration of the bath spins. Each tuple contains (spin_type, position, initial_spin).
    approx_level : str
        Approximation level for the system. Can be 'no_bath', 'full_bath', 'gCCE0', 'gCCE1', or 'gCCE2'.

    Important Attributes
    --------------------
    system_spins_list : list
        List of system spins for different approximation levels.
    mf_spins_list : list
        List of mean-field spins for different approximation levels.
    """

    def __init__(self, register_config, bath_config, approx_level):
        self.approx_level = approx_level

        # register porperties
        self.register_config = register_config
        self.register_num_spins = len(self.register_config)
        self.register_spin_types, self.register_spin_pos, self.register_init_spin, _ = list(zip(*self.register_config))

        # bath properties
        self.bath_config = bath_config
        self.bath_num_spins = len(self.bath_config)
        if self.bath_num_spins > 0:
            self.bath_spin_types, self.bath_spin_pos, self.bath_init_spin, _ = list(zip(*self.bath_config))

        # list of instances of Spin class (for each spin in the register and bath)
        self.register_spins = [Spin(*spin_config) for spin_config in self.register_config]
        if self.bath_num_spins > 0:
            self.bath_spins = [Spin(*spin_config) for spin_config in self.bath_config]

        # indices of bath spins that are simulated exactly in the corresponding gCCE approximation
        # we exclude P1 centers too far apart to interact significantly
        if self.bath_num_spins > 0:
            self.idx_gCCE1 = list(range(self.bath_num_spins))
            self.gCCE2_distance = DEFAULTS["gCCE2_distance"]
            self.idx_gCCE2 = self.get_idx_gCCE2()

        # lists of lists of instances of Spin class (for each spin in each system and mean-field configuration)
        self.system_spins_list, self.mf_spins_list = self.get_spins_lists()
        self.num_systems = len(self.system_spins_list)
        self.system_num_spins = len(self.system_spins_list[0])
        self.mf_num_spins = len(self.mf_spins_list[0])

    # -------------------------------------------------

    def get_idx_gCCE2(self):
        """Returns indices of interacting P1 centers in the bath with a distance less than 55nm
        since for larger distances the interaction can be neglected and does not give a contribution
        to the gCCE2 approximation."""

        idx_gCCE2 = list(combinations(range(self.bath_num_spins), 2))
        if "P1" not in self.bath_spin_types:
            return idx_gCCE2
        else:
            idx_gCCE2 = []
            for i, j in combinations(range(self.bath_num_spins), 2):
                distance_NV = np.linalg.norm(
                    np.array(self.bath_spin_pos[i])
                    - np.array(self.register_spin_pos[0])
                )
                distance_P1 = np.linalg.norm(
                    np.array(self.bath_spin_pos[i]) - np.array(self.bath_spin_pos[j])
                )
                if distance_NV < self.gCCE2_distance and distance_P1 < self.gCCE2_distance:
                    idx_gCCE2.append((i, j))
            return idx_gCCE2

    def get_config_lists(self):
        """Returns the system and mean-field configurations."""

        system_config_list, mf_config_list = {}, {}

        if self.approx_level == "no_bath":
            system_config_list[self.approx_level] = self.register_config
            mf_config_list[self.approx_level] = []

        if self.approx_level == "full_bath":
            system_config_list[self.approx_level] = (
                self.register_config + self.bath_config
            )
            mf_config_list[self.approx_level] = []

        if self.approx_level == "gCCE0":
            system_config_list[self.approx_level] = self.register_config
            mf_config_list[self.approx_level] = self.bath_config

        if self.approx_level == "gCCE1":
            for i in self.idx_gCCE1:
                system_config_list[f"gCCE1_{i}"] = self.register_config + [
                    self.bath_config[i]
                ]
                mf_config_list[f"gCCE1_{i}"] = (
                    self.bath_config[:i] + self.bath_config[i + 1 :]
                )

        if self.approx_level == "gCCE2":
            for i, j in self.idx_gCCE2:
                i, j = min(i, j), max(i, j)
                system_config_list[f"gCCE2_{i}_{j}"] = (
                    self.register_config + [self.bath_config[i]] + [self.bath_config[j]]
                )
                mf_config_list[f"gCCE2_{i}_{j}"] = (
                    self.bath_config[:i]
                    + self.bath_config[i + 1 : j]
                    + self.bath_config[j + 1 :]
                )

        return system_config_list, mf_config_list

    def get_spins_lists(self):
        """Returns the system and mean-field spins needed to set up the Hamiltonian for different approximation levels."""
        
        system_spins_list, mf_spins_list = [], []

        if self.approx_level == "no_bath":
            system_spins_list.append(self.register_spins)
            mf_spins_list.append([])

        if self.approx_level == "full_bath":
            system_spins_list.append(self.register_spins + self.bath_spins)
            mf_spins_list.append([])

        if self.approx_level == "gCCE0":
            system_spins_list.append(self.register_spins)
            mf_spins_list.append(self.bath_spins)

        if self.approx_level == "gCCE1":
            for i in self.idx_gCCE1:
                system_spins_list.append(self.register_spins + [self.bath_spins[i]])
                mf_spins_list.append(self.bath_spins[:i] + self.bath_spins[i + 1 :])

        if self.approx_level == "gCCE2":
            for i, j in self.idx_gCCE2:
                i, j = min(i, j), max(i, j)
                system_spins_list.append(
                    self.register_spins + [self.bath_spins[i]] + [self.bath_spins[j]]
                )
                mf_spins_list.append(
                    self.bath_spins[:i]
                    + self.bath_spins[i + 1 : j]
                    + self.bath_spins[j + 1 :]
                )

        return system_spins_list, mf_spins_list
