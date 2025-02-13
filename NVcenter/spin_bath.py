import random
import os
import json
import numpy as np

from . import CONST
from .utils import spherical_to_cartesian, cylindrical_to_cartesian

# ------------------------------------------------------------


def get_spin_bath_density(
    spin_type,
    density,
    shape,
    rmin,
    rmax,
    pos_seed=123,
    init_state_seed=123,
    lamor_seed=123,
):
    """Returns a random spin bath configuration."""

    num_spins = calc_num_spins_density(density, shape, rmin, rmax)
    spin_pos = choose_spin_pos(pos_seed, num_spins, shape, rmin, rmax)
    spin_types = [spin_type] * num_spins
    init_states = choose_init_states(init_state_seed, num_spins)

    kwargs = [{}] * num_spins
    if spin_type == "P1":
        kwargs = choose_lamor_disorders(lamor_seed, num_spins)

    config = list(zip(spin_types, spin_pos, init_states, kwargs))
    return config


def get_spin_bath_abundancy(
    spin_type, abundancy, rmin, rmax, pos_seed=123, init_state_seed=123, lamor_seed=123
):
    """Returns a random spin bath configuration."""

    num_spins = calc_num_spins_abundancy(abundancy, rmin, rmax)
    spin_pos = choose_spin_pos(pos_seed, num_spins, "sphere", rmin, rmax)
    spin_types = [spin_type] * num_spins
    init_states = choose_init_states(init_state_seed, num_spins)

    kwargs = [{}] * num_spins
    if spin_type == "P1":
        kwargs = choose_lamor_disorders(lamor_seed, num_spins)

    config = list(zip(spin_types, spin_pos, init_states, kwargs))
    return config


# ------------------------------------------------------------


def calc_spin_baths_density(
    spin_type, density, shape, rmin, rmax, num_baths, num_init_states
):
    """Creates bath configurations."""

    spin_configs = [[None] * num_init_states for _ in range(num_baths)]

    for bath_idx in range(num_baths):
        for init_state_idx in range(num_init_states):
            spin_bath = get_spin_bath_density(
                spin_type,
                density,
                shape,
                rmin,
                rmax,
                pos_seed=bath_idx,
                init_state_seed=init_state_idx,
            )
            spin_configs[bath_idx][init_state_idx] = spin_bath

    metadata = {
        "density": density,
        "shape": shape,
        "rmin": rmin,
        "rmax": rmax,
        "num_baths": num_baths,
        "num_init_state": num_init_states,
    }

    return spin_configs, metadata


def calc_spin_baths_abundancy(
    spin_type, abundancy, rmin, rmax, num_baths, num_init_states
):
    """Creates bath configurations."""

    spin_configs = [[None] * num_init_states for _ in range(num_baths)]

    for bath_idx in range(num_baths):
        for init_state_idx in range(num_init_states):
            spin_bath = get_spin_bath_abundancy(
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

    return spin_configs, metadata


# ------------------------------------------------------------


def calc_num_spins_density(density, shape, rmin, rmax):
    """Calculates the number of bath spins in a circle or sphere from a given density."""

    if shape == "sphere":
        volume = 4 / 3 * np.pi * (rmax**3 - rmin**3)
        return volume * density

    elif shape == "circle":
        area = np.pi * (rmax**2 - rmin**2)
        return int(area * density)


def calc_num_spins_abundancy(abundancy, rmin, rmax):
    """Calculates the number of bath spins in a sphere from a given abundancy.
    Equals the expectation value of the binomial distribution (n*p)."""

    V_unit = CONST["a_C"] ** 3  # volume of the unit cell
    N_unit = CONST["N_unit"]  # number of carbon atoms per unit cell
    density_C = N_unit / V_unit
    density = abundancy * density_C

    num_spins = calc_num_spins_density(density, "sphere", rmin, rmax)
    return num_spins


# ------------------------------------------------------------


def choose_spin_pos(seed, num_spins, shape, rmin, rmax):
    """Returns random positions of bath spins in cartesian coordinates within a sphere or circle."""

    if shape == "sphere":
        return choose_spin_pos_sphere(seed, num_spins, rmin, rmax)
    elif shape == "circle":
        return choose_spin_pos_circle(seed, num_spins, rmin, rmax)


def choose_spin_pos_sphere(seed, num_spins, rmin, rmax):
    """Returns random positions of bath spins in cartesian coordinates within a sphere."""

    random.seed(seed)
    r_list = [np.cbrt(random.uniform(rmin**3, rmax**3)) for _ in range(num_spins)]
    phi_list = [float(random.uniform(0, 2 * np.pi)) for _ in range(num_spins)]
    theta_list = [float(random.uniform(0, np.pi)) for _ in range(num_spins)]

    spin_pos = []
    for r, phi, theta in zip(r_list, phi_list, theta_list):
        spin_pos.append(spherical_to_cartesian(r, phi, theta))
    return spin_pos


def choose_spin_pos_circle(seed, num_spins, rmin, rmax):
    """Returns random positions of bath spins in cartesian coordinates within a circle."""

    random.seed(seed)
    r_list = [np.sqrt(random.uniform(rmin**2, rmax**2)) for _ in range(num_spins)]
    phi_list = [float(random.uniform(0, 2 * np.pi)) for _ in range(num_spins)]
    z_list = [15e-9] * num_spins

    spin_pos = []
    for r, phi, z in zip(r_list, phi_list, z_list):
        spin_pos.append(cylindrical_to_cartesian(r, phi, z))
    return spin_pos


# ------------------------------------------------------------


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


def save_spin_baths(spin_configs, metadata, directory, filename):
    """Save spin bath configurations to a JSON file."""

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


# -------------------------------------------------
