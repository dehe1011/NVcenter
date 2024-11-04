from itertools import product, combinations
import random
import numpy as np
import qutip as q

from src import spherical_to_cartesian

class SpinConfig:
    """ This class gives the register and bath configurations (tuple of: spin_type, spin_pos, init_spin, kwargs) as well as system and mean-field configurations for different bath approximations.
    
    Notes:
        abundancy not in % or ppb > use 1e-2 or 1e-9 instead
        
    Examples: 
        IV. A.: SpinConfig([(0.87e-9, 0, 0.19e-9)] , 'C13', 0.02e-2, 2e-9, 4.2e-9)
        IV. A.: SpinConfig([(0.87e-9, 0, 0.19e-9)] , 'P1', 26e-9, 30e-9, 80e-9)
    """
    
    def __init__(self, register_C13_pos, bath_spin_type, abundancy, rmin, rmax, 
                 bath_spin_pos_seed=123, bath_init_state_idx=123, bath_P1_seed=123):

        # constructor arguments
        self.register_C13_pos = register_C13_pos
        self.bath_spin_type = bath_spin_type
        self.abundancy = abundancy
        self.rmin = rmin
        self.rmax = rmax
        self.bath_spin_pos_seed = bath_spin_pos_seed
        self.bath_init_state_idx = bath_init_state_idx # goes from 0 to 2**self.bath_num_spins-1
        self.bath_P1_seed = bath_P1_seed + 100  # change the seed to avoid any unwanted correlations in the random numbers, there are 12*self.bath_num_spins possible configurations

        # carbon properties 
        self.a_C = 3.567e-10 # lattice constant for carbon
        self.V_unit = self.a_C**3 # volume of the unit cell
        self.N_unit = 8 # number of carbon atoms per unit cell
        self.n = self.N_unit/self.V_unit # density of carbon atoms

        # regsiter properties
        self.register_volume = 4/3 * np.pi * self.rmin**3
        self.register_num_C = self.calc_num_C(self.register_volume) # number of C-12 atoms in the given volume 
        self.register_num_spins = self.calc_num_spins(self.register_volume) # expected number of impurity spins in the regsiter
        self.register_num_spins = 1 + len(self.register_C13_pos) # number of register spins given by the input

        # register configuration
        self.register_spin_types = ['NV'] + ['C13'] * len(self.register_C13_pos)
        self.register_spin_pos = [(0, 0, 0)] + self.register_C13_pos
        self.register_init_spin = [0, 0, 0, 0] # this means the zeroth basis element: m_S=-1 for the NV center and m_S=-1/2 for the C-13 
        self.register_kwargs = [{}] * self.register_num_spins
        self.register_configs = list(zip(self.register_spin_types, self.register_spin_pos, self.register_init_spin, self.register_kwargs))
        
        # bath properties 
        self.bath_volume = 4/3 * np.pi * (self.rmax**3 - self.rmin**3)
        self.bath_num_C = self.calc_num_C(self.bath_volume) # number of C-12 atoms in the given volume 
        self.bath_num_spins = self.calc_num_spins(self.bath_volume) # expected number of impurity spins in the bath

        # bath configuration
        self.bath_spin_types = [self.bath_spin_type] * self.bath_num_spins
        self.bath_spin_pos = self.choose_bath_spin_pos()
        self.bath_init_spin = self.choose_bath_init_states()
        self.bath_kwargs = [{}] * self.bath_num_spins
        if self.bath_spin_type == 'P1':
            self.bath_kwargs = self.choose_lamor_disorders()
        self.bath_configs = list(zip(self.bath_spin_types, self.bath_spin_pos, self.bath_init_spin, self.bath_kwargs))

        # system and mean-field configurations
        self.idx_gCCE1 = list(range(self.bath_num_spins))
        self.idx_gCCE2 = list(combinations(range(self.bath_num_spins), 2))
        if self.bath_spin_type == 'P1':
            self.idx_gCCE2 = self.get_idx_gCCE2()
        self.system_configs, self.mf_configs = self.get_configs()
        self.bath_approx_levels = list(self.system_configs.keys())
   
    # ------------------------------------------------------------
    
    # expected number of spins
    def calc_num_C(self, volume):
        """ Calculates the number of C-12 and C-13 atoms in a given volume. """
        return volume * self.n

    def calc_num_spins(self, volume):
        """ Calculates the number of bath spins in a given volume. Equals the expectation value of the binomial distribution (n*p). """  
        return int(self.abundancy * self.calc_num_C(volume))

    # ------------------------------------------------------------

    # random choices: spin positions, bath initial states and Lamor disorders (for the P1 centers)
    def choose_bath_spin_pos(self):
        """ Returns random positions of impurity spins in cartesian coordinates with a given volume. """
        random.seed(self.bath_spin_pos_seed)
        r_vals = [random.uniform(self.rmin**3, self.rmax**3)**(1/3) for _ in range(self.bath_num_spins)]
        theta_vals = [random.uniform(0, np.pi) for _ in range(self.bath_num_spins)]
        phi_vals = [random.uniform(0, 2 * np.pi) for _ in range(self.bath_num_spins)]
        return [spherical_to_cartesian(r, phi, theta) for r, theta, phi in zip(r_vals, theta_vals, phi_vals)]
        
    def choose_lamor_disorders(self):
        """ Returns the disorder in the Lamor frequencies of P1 centers due to the hyperfine coupling between nitrogen nuclear spin and the electron (that couples to the NV center). 
        This effect depends on the nitrogen spin and P1 center delocalization axis (due to the Jahn-Teller effect).  """
        random.seed(self.bath_P1_seed)
        axes = ['111', '-111', '1-11', '11-1']
        nitrogen_spins = [-1, 0, 1]
        axis_choice = random.choices(axes, k=self.bath_num_spins)
        nitrogen_spin_choice = random.choices(nitrogen_spins, k=self.bath_num_spins)
        return [{'nitrogen_spin': nitrogen_spin_choice[i], 'axis': axis_choice[i]} for i in range(self.bath_num_spins)]

    def choose_bath_init_states(self):
        """ Returns the initial state of the bath spins. """
        random.seed(123)
        bath_states = list(product([0, 1], repeat=self.bath_num_spins))
        random.shuffle(bath_states)
        return bath_states[self.bath_init_state_idx]

    # ------------------------------------------------------------

    # split register and bath into system and mean-field part
    def get_idx_gCCE2(self, max_distance=55e-9):
        """ Returns indices of interacting P1 centers in the bath with a distance less than 55nm (for larger distances the interaction can be neglected and does not give a contribution to the gCCE2 approximation. """
        idx_gCCE2 = []
        if self.bath_spin_type == 'P1':
            for i in range(self.bath_num_spins):
                distance_NV = np.linalg.norm(np.array(self.bath_spin_pos[i]))
                if distance_NV < max_distance:
                    for j in range(self.bath_num_spins):
                        distance_P1 = np.linalg.norm(np.array(self.bath_spin_pos[i]) - np.array(self.bath_spin_pos[j]))
                        if distance_P1 < max_distance and i!=j and (j, i) not in idx_gCCE2: # comment out this condition to consider all possible combinations
                            idx_gCCE2.append((i, j))
        
        return idx_gCCE2

    def get_configs(self):
        """ Returns the system and mean-field configurations needed to set up the hamiltonian for different approximation levels. """
        system_configs, mf_configs = {}, {}
        
        system_configs['no_bath'] = self.register_configs
        mf_configs['no_bath'] = []
        
        system_configs['full_bath'] = self.register_configs + self.bath_configs
        mf_configs['full_bath'] = []
        
        system_configs['gCCE0'] = self.register_configs
        mf_configs['gCCE0'] = self.bath_configs
        
        for i in self.idx_gCCE1:
            system_configs[f'gCCE1_{i}'] = self.register_configs + [self.bath_configs[i]]
            mf_configs[f'gCCE1_{i}'] = self.bath_configs[:i] + self.bath_configs[i+1:]
            
        for i, j in self.idx_gCCE2:
            i, j = min(i,j), max(i,j)
            system_configs[f'gCCE2_{i}_{j}'] = self.register_configs + [self.bath_configs[i]] + [self.bath_configs[j]]
            mf_configs[f'gCCE2_{i}_{j}'] = self.bath_configs[:i] + self.bath_configs[i+1:j] + self.bath_configs[j+1:]
            
        return system_configs, mf_configs
