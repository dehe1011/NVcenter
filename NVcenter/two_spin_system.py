import numpy as np
import qutip as q
import matplotlib.pyplot as plt

from .helpers import get_dipolar_matrix, calc_H_int
from .spin import Spin


class TwoSpinSystem:
    """ This class is designed to investigate the dynanamics between two spins interacting via the dipolar coupling of their magnetic moments. 
    The class contains two plotting routines to plot the spin observables (Sx, Sy, Sz) and the population of the spins states for each spin. 
    
    Note:
        This works even for the full NV center with spin 1 (not only for the spin subspace of the NV center qubit).
    """
    
    def __init__(self, config_spin1, config_spin2, times):
        # constructor arguments
        self.config_spin1 = config_spin1
        self.config_spin2 = config_spin2
        self.times = times

        self.spin1 = Spin(*self.config_spin1)
        self.spin2 = Spin(*self.config_spin2)
        self.S1 = [q.tensor(op, q.qeye(self.spin2.spin_dim)) for op in self.spin1.S]
        self.S2 = [q.tensor(q.qeye(self.spin1.spin_dim), op) for op in self.spin2.S]

        self.dipolar_matrix = get_dipolar_matrix(self.spin1.spin_pos, self.spin2.spin_pos, self.spin1.gamma, self.spin2.gamma)
        self.init_state = q.tensor(self.spin1.init_state, self.spin2.init_state)
        self.H = self.calc_H()
        self.H1, self.H2 = q.ptrace(self.H, [0]), q.ptrace(self.H, [1])
        self.result = self.calc_dynamics()
        self.observable_dict = self.calc_observable_dict()
        self.spin_pops = self.calc_spin_pops()
        
    def calc_H(self):
        H = 0
        H += q.tensor(self.spin1.H, q.qeye(self.spin2.spin_dim))
        H += q.tensor(q.qeye(self.spin1.spin_dim), self.spin2.H)
        H += calc_H_int(self.S1, self.S2, self.spin1.spin_pos, self.spin2.spin_pos, self.spin1.gamma, self.spin2.gamma)
        return H

    def calc_dynamics(self):
        result = q.mesolve(self.H, self.init_state, self.times).states
        return result

    def calc_observable_dict(self):
        observable_keys = ['S1x', 'S1y', 'S1z', 'S2x', 'S2y', 'S2z']
        observable_vals = [[q.expect(dm, op) for dm in self.result] for op in self.S1[1:]+self.S2[1:]]
        return dict(zip(observable_keys, observable_vals))

    def calc_spin_pops(self):
        spin1_pop = np.array([q.ptrace(dm, [0]).diag() for dm in self.result])
        spin2_pop = np.array([q.ptrace(dm, [1]).diag() for dm in self.result])
        return spin1_pop, spin2_pop

    def plot_observables(self, observable_keys, return_ax=False):        
        fig, ax = plt.subplots()
        for observable_key in observable_keys:
            observable_val = self.observable_dict[observable_key]
            ax.plot(self.times, np.real(observable_val), label=observable_key)
        ax.set_ylim(-1.02, 1.02)
        ax.legend()
        if return_ax: return ax

    def plot_pops(self, return_ax=False):
        spin1_pop, spin2_pop = self.spin_pops
        fig, ax = plt.subplots()
        for i in range(self.spin1.spin_dim):
            ax.plot(self.times, spin1_pop[:, i], label=f"{self.spin1.spin_type}_{i}")
        for i in range(self.spin2.spin_dim):
            ax.plot(self.times, spin2_pop[:, i], label=f"{self.spin2.spin_type}_{i}")
        ax.legend()
        if return_ax: return ax
