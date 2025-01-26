import random
import os
import json
import numpy as np

from . import CONST
from .helpers import spherical_to_cartesian

# ------------------------------------------------------------

def get_spin_bath(spin_type, abundancy, rmin, rmax, pos_seed=123, init_state_seed=123, lamor_seed=123):
    """Return a random spin bath configuration."""

    num_spins = calc_num_spins(abundancy, rmin, rmax)
    spin_pos = choose_spin_pos(pos_seed, num_spins, rmin, rmax)
    spin_types = [spin_type] * num_spins
    init_states = choose_init_states(init_state_seed, num_spins)

    kwargs = [{}] * num_spins
    if spin_type == "P1":
        kwargs = choose_lamor_disorders(lamor_seed, num_spins)

    config = list(zip(spin_types, spin_pos, init_states, kwargs))
    return config

# ------------------------------------------------------------

def calc_num_spins(abundancy, rmin, rmax):
    """Calculates the number of bath spins in a given volume. Equals the expectation value of the binomial distribution (n*p)."""

    a_C = CONST["a_C"]  # lattice constant for carbon
    V_unit = a_C**3  # volume of the unit cell
    N_unit = CONST["N_unit"]  # number of carbon atoms per unit cell
    n = N_unit / V_unit  # density of carbon atoms
    volume = 4 / 3 * np.pi * (rmax**3 - rmin**3)
    num_C = volume * n  # number of carbon atoms

    return int(abundancy * num_C)


# random choices: spin positions, bath initial states and Lamor disorders (for the P1 centers)
def choose_spin_pos(seed, num_spins, rmin, rmax):
    """Returns random positions of impurity spins in cartesian coordinates with a given volume."""
    random.seed(seed)
    
    r_vals, theta_vals, phi_vals = [], [], []
    for _ in range(num_spins):
        r_vals.append(random.uniform(rmin**3, rmax**3) ** (1 / 3))
        theta_vals.append(random.uniform(0, np.pi))
        phi_vals.append(random.uniform(0, 2 * np.pi))

    spin_pos = []
    for r, theta, phi in zip(r_vals, theta_vals, phi_vals):
        spin_pos.append( spherical_to_cartesian(r, phi, theta) )
    
    return spin_pos


def choose_init_states(seed, num_spins):
    """Returns the initial state of the bath spins."""
    random.seed(seed)
    return random.choices([0, 1], k=num_spins)


def choose_lamor_disorders(seed, num_spins):
    """Returns the disorder in the Lamor frequencies of P1 centers due to the hyperfine coupling between nitrogen nuclear spin and the electron 
    (that couples to the NV center). This effect depends on the nitrogen spin and P1 center delocalization axis (due to the Jahn-Teller effect).
    """
    random.seed(seed)
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
    filename, directory, spin_type, abundancy, rmin, rmax, num_baths, num_init_states):
    """Save spin bath configurations as a JSON file."""

    spin_configs = [ [None] * num_init_states for _ in range(num_baths) ]

    for bath_idx in range(num_baths):
        for init_state_idx in range(num_init_states):
            spin_bath = get_spin_bath(
                spin_type,
                abundancy,
                rmin,
                rmax,
                pos_seed=bath_idx,
                init_state_seed=init_state_idx,
            )
            spin_configs[bath_idx][init_state_idx] = spin_bath

    metadata = {
        "abundancy": abundancy,
        "rmin": rmin,
        "rmax": rmax,
        "num_baths": num_baths,
        "num_init_state": num_init_states,
    }

    save_spin_baths(spin_configs, metadata, directory, filename)

    return spin_configs

def save_spin_baths(spin_configs, metadata, directory, filename):
    spin_configs = {"Configurations": spin_configs, "Metadata": metadata}

    # Save the nested dictionary as a JSON file
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename + ".json")
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(spin_configs, file, indent=4)


def load_spin_baths(filename, directory, load_metadata=False):
    """Load spin bath configurations from a JSON file."""

    filepath = os.path.join(directory, filename + ".json")
    with open(filepath, "r", encoding="utf-8") as file:
        spin_configs = json.load(file)
    if load_metadata:
        return spin_configs["Configurations"], spin_configs["Metadata"]
    return spin_configs["Configurations"]
