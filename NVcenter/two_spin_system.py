import numpy as np
import qutip as q
import matplotlib.pyplot as plt

from .helpers import get_dipolar_matrix, calc_H_int
from .spin import Spin

# -------------------------------------------

class TwoSpinSystem:
    """ This class is designed to investigate the dynanamics between two spins interacting via the dipolar coupling of their magnetic moments. 
    The class contains two plotting routines to plot the spin observables (Sx, Sy, Sz) and the population of the spins states for each spin. 
    
    Note:
        The time should be given in microseconds.
        This works even for the full NV center with spin 1 (not only for the spin subspace of the NV center qubit).
    """
    
    def __init__(self, config_spin1, config_spin2, time):
        # Create instances of the Spin class
        self.spin1 = Spin(*config_spin1)
        self.spin2 = Spin(*config_spin2)

        # Create spin operators in the bigger Hilbert space of two spins
        self.S1 = [q.tensor(op, q.qeye(self.spin2.spin_dim)) for op in self.spin1.S]
        self.S2 = [q.tensor(q.qeye(self.spin1.spin_dim), op) for op in self.spin2.S]

        # Initial state
        self.init_state = q.tensor(self.spin1.init_state, self.spin2.init_state)

        # Calculate Hamiltonians 
        self.dipolar_matrix = get_dipolar_matrix(self.spin1.spin_pos, self.spin2.spin_pos, self.spin1.gamma, self.spin2.gamma)
        self.H_int = calc_H_int(self.S1, self.S2, self.dipolar_matrix)
        self.H = self._calc_H()
        self.H1  = self.spin1.H # alternative: q.ptrace(self.H, [0])
        self.H2 = self.spin2.H # alternative: q.ptrace(self.H, [1])

        # Simulation time
        self.times = np.linspace(0, time, int(time*1e6)*10)

        # Calculate Time Evolution
        self.result = None
        self.observable_dict = None
        self.spin_pops = None        

    def _calc_H(self):
        """Calculate the Hamiltonian of the two-spin system."""
        H1 = q.tensor(self.spin1.H, q.qeye(self.spin2.spin_dim))
        H2 = q.tensor(q.qeye(self.spin1.spin_dim), self.spin2.H)
        return H1 + H2 + self.H_int


    def calc_dynamics(self):
        """Calculate the time evolution of the two-spin system using qutip.mesolve."""
        if self.result is None:
            self.result = q.mesolve(self.H, self.init_state, self.times).states
        return self.result


    def calc_observable_dict(self):
        """Calculate the expectation values of the spin operators for each time step. Used in plot_observables()."""
        self.calc_dynamics()
        if self.observable_dict is None:
            observable_keys = ['S1x', 'S1y', 'S1z', 'S2x', 'S2y', 'S2z']
            observable_vals = [[q.expect(dm, op) for dm in self.result] for op in self.S1[1:]+self.S2[1:]]
            self.observable_dict = dict(zip(observable_keys, observable_vals))
        return self.observable_dict


    def calc_spin_pops(self):
        """Calculate the populations of the spin states for each time step. Used in plot_pops()."""
        self.calc_dynamics()
        if self.spin_pops is None:
            spin1_pop = np.array([q.ptrace(dm, [0]).diag() for dm in self.result])
            spin2_pop = np.array([q.ptrace(dm, [1]).diag() for dm in self.result])
            self.spin_pops = spin1_pop, spin2_pop
        return self.spin_pops


    def plot_observables(self, observable_keys, return_ax=False): 
        """Plot the expectation values of the spin operators."""       
        _, ax = plt.subplots()
        self.calc_observable_dict()

        # plotting
        for observable_key in observable_keys:
            observable_val = self.observable_dict[observable_key]
            ax.plot(self.times, np.real(observable_val), label=observable_key)

        # plot settings
        ax.set_xlabel('Time (us)')
        ax.legend()
        if return_ax: return ax


    def plot_pops(self, return_ax=False):
        """Plot the populations of the spin states."""
        _, ax = plt.subplots()
        self.calc_spin_pops()

        # plotting
        spin1_pop, spin2_pop = self.spin_pops
        for i in range(self.spin1.spin_dim):
            ax.plot(self.times, spin1_pop[:, i], label=f"{self.spin1.spin_type}_{i}")
        for i in range(self.spin2.spin_dim):
            ax.plot(self.times, spin2_pop[:, i], label=f"{self.spin2.spin_type}_{i}")

        # plot settings
        ax.set_xlabel('Time (us)')
        ax.legend()
        if return_ax: return ax
