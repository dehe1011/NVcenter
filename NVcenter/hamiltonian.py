import qutip as q

from .spins import Spins
from .helpers import adjust_space_dim, get_dipolar_matrix, calc_H_int

# -------------------------------------------------


class Hamiltonian(Spins):
    """
    A class to calculate the Spin Hamiltonians and initial states for all given pairs of system and mean-field spins.

    Parameters
    ----------
    register_config : list
        Configuration for the register spins.
    bath_config : list
        Configuration for the bath spins.
    approx_level : str
        Approximation level for the Hamiltonian calculation.

    Attributes
    ----------
    spin_ops : list
        Spin operators I, Sx, Sy, and Sz in the correct Hilbert space dimension.
    register_init_state : qutip.Qobj
        Initial state of the register.
    bath_thermal_state : qutip.Qobj
        Totally mixed thermal state of the bath.
    init_states : list
        System states for a given register state.
    matrices : list
        Hamiltonian matrices for the system and mean-field spins.
    """

    def __init__(self, register_config, bath_config, approx_level):
        """Calculates the Spin Hamiltonians and initial states for all given pairs of system and mean-field spins."""

        super().__init__(register_config, bath_config, approx_level)

        self.spin_ops = self.calc_system_spin_ops()
        self.register_init_state = self.calc_register_init_state()
        self.bath_thermal_state = self.calc_bath_thermal_state()
        self.init_states = self.calc_system_states(self.register_init_state)
        self.matrices = self.calc_matrices()

    def calc_system_spin_ops(self):
        """Returns the spin operators I, Sx, Sy and Sz in the correct dimension."""

        system_spins = self.system_spins_list[0]
        system_num_spins = len(system_spins)
        return [
            [adjust_space_dim(system_num_spins, op, i) for op in spin.S]
            for i, spin in enumerate(system_spins)
        ]

    def calc_register_init_state(self):
        """Calculated the initial state of the register."""
        return q.tensor(
            [register_spin.init_state for register_spin in self.register_spins]
        )

    def calc_bath_thermal_state(self):
        """Calculates the totally mixed thermal state of the bath."""
        return q.tensor([q.qeye(2) for _ in range(self.bath_num_spins)])

    def calc_system_states(self, register_state):
        """Returns the system states for a given register state."""

        if self.approx_level == "no_bath":
            return [register_state]

        if self.approx_level == "full_bath":
            return [q.tensor(register_state, self.bath_thermal_state)]

        if self.approx_level == "gCCE0":
            return [register_state]

        states = []
        for system_spins in self.system_spins_list:
            bath_state = q.tensor(
                [
                    system_spin.init_state
                    for system_spin in system_spins[self.register_num_spins :]
                ]
            )
            states.append(q.tensor(register_state, bath_state))
        return states

    def calc_matrices(self):
        """Returns the full Hamiltonian matrices for the system and mean-field spins
        for each pair of system and mean-field."""

        matrices = []
        for system_spins, mf_spins in zip(self.system_spins_list, self.mf_spins_list):
            H_system = self.calc_H_system(system_spins)
            H_mf = self.calc_H_mf(system_spins, mf_spins)
            matrices.append(H_system + H_mf)
        return matrices

    def calc_H_system(self, system_spins):
        """Returns the Hamiltonian for the system spins."""

        system_num_spins = len(system_spins)
        H = 0
        for i, spin1 in enumerate(system_spins):
            H += adjust_space_dim(system_num_spins, spin1.H, i)
            for j, spin2 in enumerate(system_spins):
                if j > i:
                    spin_op1 = self.spin_ops[i]
                    spin_op2 = self.spin_ops[j]
                    dipolar_matrix = get_dipolar_matrix(
                        spin1.spin_pos, spin2.spin_pos, spin1.gamma, spin2.gamma
                    )
                    H += calc_H_int(spin_op1, spin_op2, dipolar_matrix)
        return H

    def calc_H_mf(self, system_spins, mf_spins):
        """Returns the Hamiltonian for the mean-field spins (in the system Hilbert space)."""

        H = 0
        for i, system_spin in enumerate(system_spins):
            for mf_spin in mf_spins:
                Sz = self.spin_ops[i][3]
                Ez = mf_spin.init_spin - 1 / 2  # 0,1 -> -1/2, 1/2
                H += (
                    Ez
                    * get_dipolar_matrix(
                        system_spin.spin_pos,
                        mf_spin.spin_pos,
                        system_spin.gamma,
                        mf_spin.gamma,
                    )[2, 2]
                    * Sz
                )
        return H
