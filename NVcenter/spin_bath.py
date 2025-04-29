import os
from itertools import product
import json
import numpy as np
import matplotlib.pyplot as plt

from . import CONST, DEFAULTS
from .utils import spherical_to_cartesian, cylindrical_to_cartesian

# ------------------------------------------------------------

class SpinBath:
    def __init__(self, spin_type, **kwargs):
        self.spin_type = spin_type
        self.kwargs = kwargs
        self.bath_kwargs = kwargs.get("bath_kwargs", {})

        # calculate the shape and dimensions of the spin bath
        self.shape = self.kwargs.get("shape", DEFAULTS["shape"])
        self.rmin = self.kwargs.get("rmin", DEFAULTS["rmin"])
        self.rmax = self.kwargs.get("rmax", DEFAULTS["rmax"])

        # calculate the density of the spin bath
        self.density = self.kwargs.get("density")
        self.abundancy = self.kwargs.get("abundancy")
        self.density_C = CONST["N_unit"]  / CONST["a_C"] ** 3
        if not self.density:
            assert self.abundancy, "Either density or abundancy should be provided."
            self.shape = "sphere" # abundancy only defined for a sphere
            self.density = self.abundancy * self.density_C

        self.depth = self.kwargs.get("depth", DEFAULTS["depth"])
        self.sample = self.kwargs.get("sample", DEFAULTS["sample"])
        self.num_spins = self.kwargs.get("num_spins", DEFAULTS["num_spins"])

        self.seed_num_spins = self.kwargs.get("seed_num_spins", 123)
        self.seed_spin_pos = self.kwargs.get("seed_spin_pos", 123)
        self.seed_init_states = self.kwargs.get("seed_init_states", 123)
        self.seed_lamor_disorders = self.kwargs.get("seed_lamor_disorders", 123)

    # -------------------------------------------------
        
    def calc_num_spins(self):
        """Calculates the number of bath spins in a circle or sphere from a given density.
        Important: the unit should be nm! """

        # set the seed for reproducibility
        np.random.seed(self.seed_num_spins)

        num_spins = 0
        if self.shape == "sphere":
            volume = 4 / 3 * np.pi * (self.rmax**3 - self.rmin**3)
            if self.sample: 
                n, p = volume * 1e+27, self.density / 1e+27
                num_spins = np.random.binomial(n, p)
            else: 
                # expectation value of the binomial distribution (n*p)
                num_spins = int(volume * self.density) 

        elif self.shape == "circle":
            area = np.pi * (self.rmax**2 - self.rmin**2)
            if self.sample:
                n, p = area *1e+18, self.density / 1e+18
                num_spins = np.random.binomial(n, p)
            else: 
                # expectation value of the binomial distribution (n*p)
                num_spins = int(area * self.density) 

        return num_spins
    
    # -------------------------------------------------

    def choose_spin_pos(self):
        """Returns random positions of bath spins in cartesian coordinates within a sphere or circle."""

        # set the seed for reproducibility
        np.random.seed(self.seed_spin_pos)

        if self.shape == "sphere":
            return self.choose_spin_pos_sphere()
        elif self.shape == "circle":
            return self.choose_spin_pos_circle()

    def choose_spin_pos_sphere(self):
        """Returns random positions of bath spins in cartesian coordinates within a sphere."""

        r_list = [np.cbrt(np.random.uniform(self.rmin**3, self.rmax**3)) for _ in range(self.num_spins)]
        phi_list = [float(np.random.uniform(0, 2 * np.pi)) for _ in range(self.num_spins)]
        theta_list = [float(np.random.uniform(0, np.pi)) for _ in range(self.num_spins)]

        spin_pos = []
        for r, phi, theta in zip(r_list, phi_list, theta_list):
            spin_pos.append(spherical_to_cartesian(r, phi, theta))
        return spin_pos

    def choose_spin_pos_circle(self):
        """Returns random positions of bath spins in cartesian coordinates within a circle."""

        r_list = [np.sqrt(np.random.uniform(self.rmin**2, self.rmax**2)) for _ in range(self.num_spins)]
        phi_list = [float(np.random.uniform(0, 2 * np.pi)) for _ in range(self.num_spins)]
        z_list = [self.depth] * self.num_spins

        spin_pos = []
        for r, phi, z in zip(r_list, phi_list, z_list):
            spin_pos.append(cylindrical_to_cartesian(r, phi, z))
        return spin_pos
    
    # -------------------------------------------------

    def choose_init_states(self):
        """Returns the initial states of bath spins."""

        # set the seed for reproducibility
        np.random.seed(self.seed_init_states)
        return np.random.choice([0, 1], size=self.num_spins).tolist()
    
    # -------------------------------------------------

    def choose_lamor_disorders(self):
        """Returns the disorder in the Lamor frequencies of P1 centers due to the hyperfine coupling between nitrogen nuclear spin and the electron
        (that couples to the NV center). This effect depends on the nitrogen spin and P1 center delocalization axis (due to the Jahn-Teller effect).
        """

        np.random.seed(self.seed_lamor_disorders)
        axes = ["111", "-111", "1-11", "11-1"]
        nitrogen_spins = [-1, 0, 1]
        axis_choice = np.random.choice(axes, size=self.num_spins)
        nitrogen_spin_choice = np.random.choice(nitrogen_spins, size=self.num_spins)
        return [
            {"nitrogen_spin": nitrogen_spin_choice[i], "axis": axis_choice[i]}
            for i in range(self.num_spins)
        ]
        
    # -------------------------------------------------

    def get_spin_bath(self, init_states=None):
        """Returns a random spin bath configuration."""

        # calculate the number of spins in the spin bath
        self.num_spins = self.calc_num_spins()

        spin_types = [self.spin_type] * self.num_spins
        spin_pos = self.choose_spin_pos()
        if init_states is None: 
            init_states = self.choose_init_states()
        else:
            assert(len(init_states) == self.num_spins), f"The number of initial states should be equal to the number of spins {self.num_spins}."

        # Keyword arguments
        kwargs_list = [self.bath_kwargs] * self.num_spins
        if self.spin_type == "P1":
            kwargs_P1_list = self.choose_lamor_disorders()
            for kwargs, kwargs_P1 in zip(kwargs_list, kwargs_P1_list):
                kwargs.update(kwargs_P1)

        config = list(zip(spin_types, spin_pos, init_states, kwargs_list))
        return config

    def calc_spin_baths(self, num_baths, num_init_states, all_init_states=False):
        """Creates bath configurations."""

        # Initialize the list of bath configurations
        spin_configs = []

        for bath_idx in range(num_baths):
            self.seed_spin_pos = bath_idx

            if all_init_states:
                init_states_list = list(product([0,1], repeat=self.calc_num_spins()))
                for init_states in init_states_list:
                    spin_bath = self.get_spin_bath(init_states)
                    spin_configs.append(spin_bath)
            else:
                for init_state_idx in range(num_init_states):
                    self.seed_init_states = init_state_idx
                    spin_bath = self.get_spin_bath()
                    spin_configs.append(spin_bath)

        metadata = self.kwargs
        metadata.update({"num_baths": num_baths, "num_init_states": num_init_states})

        return spin_configs, metadata

# -------------------------------------------------

def cut_spin_bath(bath_configs, cutoff):
    filtered_bath_configs = []
    for bath_config in bath_configs:
        indices_to_remove = []
        positions = list(zip(*bath_config))[1]
        for i, pos in enumerate(positions):
            if np.linalg.norm(pos) >= cutoff:
                indices_to_remove.append(i)
        
        filtered_bath_config = [value for index, value in enumerate(bath_config) if index not in indices_to_remove]
        filtered_bath_configs.append(filtered_bath_config)
 
    return filtered_bath_configs

def calc_bath_polarization(bath_configs):

    counter = 0
    for bath_config in bath_configs:
        init_state = list(zip(*bath_config))[2]
        counter += np.sum(np.array(init_state)-1/2)
    
    return float(counter) / len(bath_configs)

def visualize_spin_bath(bath_configs, metadata):
    
    # bath_configs = bath_configs_nested
    theta = np.linspace(0, 2 * np.pi, 300) 
    x_circle = metadata["rmax"] * np.cos(theta)
    y_circle = metadata["rmax"] * np.sin(theta)

    fig, ax = plt.subplots(figsize=(5,5))

    for bath_config in bath_configs:
        if bath_config == []:
            continue
        coordinates = np.array(list(zip(*bath_config))[1])
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        ax.plot(x *1e9, y*1e9, '.', markersize=8)
    ax.plot(x_circle*1e9, y_circle*1e9, color='k', alpha=0.5, linewidth=2)

    ax.set_xlabel('x [nm]')
    ax.set_ylabel('y [nm]')
    return fig, ax


def save_spin_baths(spin_configs, metadata, directory, filename):
    """Save spin bath configurations to a JSON file."""

    spin_configs = {"Configurations": spin_configs, "Metadata": metadata}

    # Save the nested dictionary as a JSON file
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename + ".json")
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(spin_configs, file, indent=3)


def load_spin_baths(filename, directory, load_metadata=False):
    """Load spin bath configurations from a JSON file."""

    filepath = os.path.join(directory, filename + ".json")
    with open(filepath, "r", encoding="utf-8") as file:
        spin_configs = json.load(file)
    if load_metadata:
        return spin_configs["Configurations"], spin_configs["Metadata"]
    return spin_configs["Configurations"]

# -------------------------------------------------
