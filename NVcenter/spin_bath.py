import random
import os
import json
from itertools import product
import numpy as np

from . import CONST
from .helpers import spherical_to_cartesian

# -------------------------------------------------


class SpinBath:
    """
    A class to construct random spin bath configurations (spin positions and spin initial states) of a
    selected spin type in a given volume with given abundancy.

    Examples
    --------
    SpinBath('C13', 0.02e-2, 2e-9, 4.2e-9) # Dominik Fig. 4
    SpinBath('P1', 26e-9, 30e-9, 80e-9) # Dominik Fig. 5

    Parameters
    ----------
    spin_type : str
        Type of the spin (e.g., 'P1' or 'C13').
    abundancy : float
        Abundance of the impurity spins.
    rmin : float
        Minimum radius of the spin bath.
    rmax : float
        Maximum radius of the spin bath.
    seed : int, optional
        Seed for random number generation (default is 123).
    init_state_idx : int, optional
        Index for the initial state of the bath spins (default is 0).

    Improtant Attributes
    --------------------
    config : list
        Configuration of the spin bath.
    """

    def __init__(self, spin_type, abundancy, rmin, rmax, seed=123, init_state_idx=0, lamor_seed=123):	
        self.abundancy = abundancy
        self.rmin = rmin
        self.rmax = rmax
        self.spin_type = spin_type
        self.seed = seed
        self.lamor_seed = lamor_seed
        self.init_state_idx = init_state_idx

        # number of spins
        self.volume = 4 / 3 * np.pi * (self.rmax**3 - self.rmin**3)
        self.num_spins = (
            self.calc_num_spins()
        )  # expected number of impurity spins in the bath

        # spin positions
        self.spin_pos = self.choose_spin_pos()

        # spin types
        self.spin_types = [self.spin_type] * self.num_spins

        # initial spin
        self.init_states = choose_init_states(self.init_state_idx, self.seed, self.num_spins)

        # kwargs
        self.kwargs = [{}] * self.num_spins
        if self.spin_type == "P1":
            self.kwargs = choose_lamor_disorders(self.lamor_seed, self.num_spins)

        # spin config
        self.config = list(
            zip(self.spin_types, self.spin_pos, self.init_states, self.kwargs)
        )

    # ------------------------------------------------------------

    def calc_num_spins(self):
        """Calculates the number of bath spins in a given volume. Equals the expectation value of the binomial distribution (n*p)."""

        a_C = CONST["a_C"]  # lattice constant for carbon
        V_unit = a_C**3  # volume of the unit cell
        N_unit = CONST["N_unit"]  # number of carbon atoms per unit cell
        n = N_unit / V_unit  # density of carbon atoms
        num_C = self.volume * n  # number of carbon atoms
        return int(self.abundancy * num_C)

    # random choices: spin positions, bath initial states and Lamor disorders (for the P1 centers)
    def choose_spin_pos(self):
        """Returns random positions of impurity spins in cartesian coordinates with a given volume."""
        random.seed(self.seed)
        r_vals = [
            random.uniform(self.rmin**3, self.rmax**3) ** (1 / 3)
            for _ in range(self.num_spins)
        ]
        theta_vals = [random.uniform(0, np.pi) for _ in range(self.num_spins)]
        phi_vals = [random.uniform(0, 2 * np.pi) for _ in range(self.num_spins)]
        return [
            spherical_to_cartesian(r, phi, theta)
            for r, theta, phi in zip(r_vals, theta_vals, phi_vals)
        ]
    
# -------------------------------------------------

def choose_init_states(init_state_idx, seed, num_spins):
    """Returns the initial state of the bath spins."""
    random.seed(seed + 1e9)
    states = list(product([0, 1], repeat=num_spins))
    random.shuffle(states)
    return states[init_state_idx]

def choose_lamor_disorders(seed, num_spins):
    """Returns the disorder in the Lamor frequencies of P1 centers due to the hyperfine coupling between nitrogen nuclear spin and the electron 
    (that couples to the NV center). This effect depends on the nitrogen spin and P1 center delocalization axis (due to the Jahn-Teller effect).
    """
    random.seed(seed + 2e9)
    axes = ["111", "-111", "1-11", "11-1"]
    nitrogen_spins = [-1, 0, 1]
    axis_choice = random.choices(axes, k=num_spins)
    nitrogen_spin_choice = random.choices(nitrogen_spins, k=num_spins)
    return [
        {"nitrogen_spin": nitrogen_spin_choice[i], "axis": axis_choice[i]}
        for i in range(num_spins)
    ]

# -------------------------------------------------

def save_spin_baths(
    filename, directory, spin_type, abundancy, rmin, rmax, num_baths, num_init_states
):
    """Save spin bath configurations as a JSON file."""

    spin_configs = {}
    spin_configs["Configurations"] = {}
    for seed in range(num_baths):
        for init_state_idx in range(num_init_states):
            spin_bath = SpinBath(
                spin_type,
                abundancy,
                rmin,
                rmax,
                seed=seed,
                init_state_idx=init_state_idx,
            )
            spin_configs["Configurations"][
                f"config_{seed}_{init_state_idx}"
            ] = spin_bath.config
    spin_configs["Metadata"] = {
        "abundancy": abundancy,
        "rmin": rmin,
        "rmax": rmax,
        "num_baths": num_baths,
        "num_init_state": num_init_states,
    }

    # Save the nested dictionary as a JSON file
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename + ".json")
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(spin_configs, file, indent=4)

    return spin_configs


def load_spin_baths(filename, directory, load_metadata=False):
    """Load spin bath configurations from a JSON file."""

    filepath = os.path.join(directory, filename + ".json")
    with open(filepath, "r", encoding="utf-8") as file:
        spin_configs = json.load(file)
    if load_metadata:
        return spin_configs["Configurations"], spin_configs["Metadata"]
    return spin_configs["Configurations"]
