import numpy as np 
import qutip as q

from .helpers import get_dipolar_matrix, calc_H_int, adjust_space_dim
from .spin import Spin


class System:
    r""" This class constructs the Hamiltonian (with interaction) and initial state from a given list of spin configurations for the system and mean-field part and diagonalizes the Hamiltonian. 
    
    Note:
        I think in this class we set $\hbar=1$ (not $h=1$) in the dipolar interaction. This is inconsistant and deviated by a factor 0f $2\pi$.
        Either multiply the Spin Hamiltonians by this factor (preferred, leading to angular frequencies) or divide the interaction strength (leading to frequencies)
    """
    
    def __init__(self, system_configs, mf_configs):
        self.system_configs = system_configs
        self.mf_configs = mf_configs
        self.system_num_spins = len(self.system_configs)
        self.system_spins = [Spin(*system_config) for system_config in system_configs]
        self.S = [[adjust_space_dim(self.system_num_spins, op, i) for op in spin.S] for i, spin in enumerate(self.system_spins)]
        self.I = self.S[0][0]
        self.mf_spins = [Spin(*mf_config) for mf_config in mf_configs]

        self.H_system = self.calc_H_system()
        self.H_mf = self.calc_H_mf()
        self.H = self.H_system + self.H_mf
        self.init_state = q.tensor([spin.init_state for spin in self.system_spins])
        self.eigv, self.eigs = np.linalg.eigh(self.H.full())

    def calc_H_system(self):
        H = 0
        for i, spin1 in enumerate(self.system_spins):
            H += adjust_space_dim(self.system_num_spins, spin1.H, i)
            for j, spin2 in enumerate(self.system_spins):
                if j>i:
                    S1 = self.S[i]
                    S2 = self.S[j]
                    H += calc_H_int(S1, S2, spin1.spin_pos, spin2.spin_pos, spin1.gamma, spin2.gamma)
                    gamma1, gamma2 = spin1.gamma, spin2.gamma
        return H

    def calc_H_mf(self):
        H = 0 
        for i, system_spin in enumerate(self.system_spins):
            for mf_spin in self.mf_spins:
                Sz = self.S[i][3]
                Ez = mf_spin.init_spin - 1/2 # 0,1 -> -1/2, 1/2
                H += Ez * get_dipolar_matrix(system_spin.spin_pos, mf_spin.spin_pos, system_spin.gamma, mf_spin.gamma)[2,2] * Sz
        return H
