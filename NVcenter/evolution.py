import time
import copy

import numpy as np
import qutip as q
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator

from . import DEFAULTS
from .hamiltonian import Hamiltonian
from .spin import get_spin_matrices
from .utils import calc_fidelity, calc_logarithmic_negativity, spherical_to_cartesian

# -------------------------------------------------


def get_cnot_gate(num_qubits, control, target):
    """Returns the CNOT gate for a given number of qubits, control and target qubit."""

    qc = QuantumCircuit(num_qubits)
    qc.cx(num_qubits - 1 - control, num_qubits - 1 - target)
    dims = [[2] * num_qubits, [2] * num_qubits]
    cnot_gate = Operator(qc).data
    return q.Qobj(cnot_gate, dims=dims)


def get_hada_gate(num_qubits, target):
    """Returns the Hadamard gate for a given number of qubits and target qubit."""

    qc = QuantumCircuit(num_qubits)
    qc.h(num_qubits - 1 - target)
    dims = [[2] * num_qubits, [2] * num_qubits]
    hada_gate = Operator(qc).data
    return q.Qobj(hada_gate, dims=dims)


# -------------------------------------------------


class Evolution(Hamiltonian):
    """This class constructs the unitary gates and quantum states."""

    def __init__(self, register_config, **kwargs):
        self.t0 = time.time()  # print("Time elapsed:", time.time() - self.t0)

        self._register_config = register_config
        self.kwargs = kwargs

        # Keyword arguments
        self._bath_config = kwargs.get("bath_config", [])
        self._approx_level = kwargs.get("approx_level", DEFAULTS["approx_level"])

        # Initialize the Hamiltonian parent class
        super().__init__(self.register_config, **self.kwargs)

        # Keyword arguments
        self.old_register_states = kwargs.get(
            "old_register_states", [self.register_init_state]
        )
        self._gate_props_list = kwargs.get("gate_props_list", [])
        self.target = kwargs.get("target", self.register_identity)
        self.dyn_dec = kwargs.get("dyn_dec", DEFAULTS["dyn_dec"])
        self.rabi_frequency = kwargs.get("rabi_frequency", DEFAULTS["rabi_frequency"])
        self.dm_offset = kwargs.get("dm_offset", DEFAULTS["dm_offset"])
        self.fidelity_offset = kwargs.get(
            "fidelity_offset", DEFAULTS["fidelity_offset"]
        )
        self.num_hahn_echos = kwargs.get("num_hahn_echos", DEFAULTS["num_hahn_echos"])

        self.initalize(**kwargs)

        # Save the eigensystem of the free Hamiltonian
        self.eigv_free, self.eigs_free = None, None
        # Save the unitary gates
        self.gates_time = None
        # Save the quantum states
        self.states = None

    # -------------------------------------------------

    @property
    def register_config(self):  # pylint: disable=missing-function-docstring
        return self._register_config

    @register_config.setter
    def register_config(self, new_register_config):
        if new_register_config != self._register_config:
            self._register_config = new_register_config
            super().__init__(
                self._register_config,
                **self.kwargs,
            )  # reinitialize the Hamiltonian

        # Save the eigensystem of the free Hamiltonian
        self.eigv_free, self.eigs_free = None, None
        # Save the unitary gates
        self.gates_time = None
        # Save the quantum states
        self.states = None

    @property
    def bath_config(self):  # pylint: disable=missing-function-docstring
        return self._bath_config

    @bath_config.setter
    def bath_config(self, new_bath_config):
        if new_bath_config != self._bath_config:
            self._bath_config = new_bath_config
            self.kwargs["bath_config"] = new_bath_config
            super().__init__(
                self._register_config,
                **self.kwargs,
            )  # reinitialize the Hamiltonian

            # Save the eigensystem of the free Hamiltonian
            self.eigv_free, self.eigs_free = None, None
            # Save the unitary gates
            self.gates_time = None
            # Save the quantum states
            self.states = None

    @property
    def approx_level(self):  # pylint: disable=missing-function-docstring
        return self._approx_level

    @approx_level.setter
    def approx_level(self, new_approx_level):

        if new_approx_level != self._approx_level:
            self._approx_level = new_approx_level
            self.kwargs["approx_level"] = new_approx_level
            super().__init__(
                self.register_config,
                **self.kwargs,
            )  # reinitialize the Hamiltonian

            # Save the eigensystem of the free Hamiltonian
            self.eigv_free, self.eigs_free = None, None
            # Save the unitary gates
            self.gates_time = None
            # Save the quantum states
            self.states = None

    @property
    def gate_props_list(self):  # pylint: disable=missing-function-docstring
        return self._gate_props_list

    @gate_props_list.setter
    def gate_props_list(self, new_gate_props_list):

        if new_gate_props_list != self._gate_props_list:
            self._gate_props_list = new_gate_props_list
            self.initalize(**self.kwargs)

    @property
    def t_list(self):  # pylint: disable=missing-function-docstring
        return self._t_list

    @t_list.setter
    def t_list(self, new_t_list):
        if new_t_list != self._t_list:
            self._t_list = new_t_list

            if isinstance(self._t_list, str):
                if self._t_list == "final":
                    self._t_list = [self.total_gate_time]

                if self._t_list == "auto":
                    self._t_list = self.get_t_list()

    # -------------------------------------------------

    def initalize(self, **kwargs):
        """Initializes the evolution object."""

        # List of the time needed for each gate
        self.gate_time_list = [
            gate_props[1].get("t", 0) for gate_props in self.gate_props_list
        ]

        # List of the cumulated time for each gate
        self.cum_gate_time_list = np.cumsum(self.gate_time_list).tolist()

        # Total time for all gates
        self.total_gate_time = sum(self.gate_time_list)

        # List of time steps for the evolution
        self._t_list = kwargs.get("t_list", DEFAULTS["t_list"])
        if isinstance(self._t_list, str):
            if self._t_list == "final":
                self._t_list = [self.total_gate_time]

            if self._t_list == "auto":
                self._t_list = self.get_t_list()

    def get_t_list(self, stepsize=0.5e-6):
        """elper function to get the time list for a pulse sequence."""

        # Create a list of time steps for the evolution
        t_list = np.arange(0, self.total_gate_time + stepsize, stepsize).tolist()

        # Add time steps shortly before and after each gate for better resolution
        t_cum = self.cum_gate_time_list
        gate_time_before = list(t_cum[:-1])
        t_list.extend(gate_time_before)
        gate_time_after = [gate_time + 1e-9 for gate_time in t_cum[:-1]]
        t_list.extend(gate_time_after)

        return np.real(sorted(t_list))

    # -------------------------------------------------

    def _get_finished_gates_and_left_time(self, threshold_time):
        """Returns the number of finished gates and the time left at a given threshold time."""

        finished_gates, left_time = 0, 0
        for i, cum_gate_time in enumerate(self.cum_gate_time_list):
            if cum_gate_time <= threshold_time:
                finished_gates = i + 1
            else:
                break

        if finished_gates == 0:
            left_time = threshold_time
        else:
            left_time = threshold_time - self.cum_gate_time_list[finished_gates - 1]

        return finished_gates, left_time

    def _get_partial_gate_props_list(self, finished_gates, left_time):
        """Returns the gate properties list for a given number of finished gates and left time."""

        if finished_gates == 0:
            return [("free_evo", dict(t=left_time, dyn_dec=False))]

        # Create a list of the finished gates
        partial_gate_props_list = copy.deepcopy(
            self.gate_props_list[:finished_gates]
        )  # Deep copy

        # Add gates if there is time left
        if left_time != 0:

            # Add a free evolution after the last gate if all gates are finished
            if finished_gates == len(self.gate_props_list):
                hahn_time = float(left_time / (self.num_hahn_echos + 1))
                after_gates = [("free_evo", dict(t=hahn_time, dyn_dec=False))]
                for _ in range(self.num_hahn_echos):
                    after_gates.append(
                        ("inst_rot", dict(alpha=np.pi, phi=0, theta=np.pi / 2))
                    )
                    after_gates.append(("free_evo", dict(t=hahn_time, dyn_dec=False)))
                partial_gate_props_list.extend(after_gates)

            # Add a fraction of the next gate if not all gates are finished
            else:
                current_gate = copy.deepcopy(
                    self.gate_props_list[finished_gates]
                )  # Deep copy
                current_gate[1]["t"] = float(left_time)
                partial_gate_props_list.append(current_gate)

        return partial_gate_props_list

    def get_gate_props_list_time(self, t_list):
        """Returns the gate properties list for each time step in a given time list."""

        # If there are no gates return a free evolution
        if self.gate_props_list == []:
            return [("free_evo", dict(t=0.0, dyn_dec=False))]

        gate_props_list_time = []

        for t in t_list:

            # Find the number of finished gates and the left time
            threshold_time = t
            finished_gates, left_time = self._get_finished_gates_and_left_time(
                threshold_time
            )

            # Get the gate properties list for the given number of finished gates and left time
            partial_gate_props_list = self._get_partial_gate_props_list(
                finished_gates, left_time
            )

            gate_props_list_time.append(partial_gate_props_list)
        return gate_props_list_time

    # ------------------------------------------------

    def calc_H_rot(self, omega, phi, theta=np.pi / 2):
        """Returns a Hamiltonian that rotates the first register spin (NV center) with the Lamor
        frequency around an axis determined by spherical angles."""

        # unit vector of the rotation axis in spherical coordinates
        n = np.array([spherical_to_cartesian(1, phi, theta)])

        # Pauli matrices (multiplied by a factor 1/2)
        spin_matrices = get_spin_matrices(1 / 2)[1:]

        # Hamiltonian for the rotation
        H_rot = omega * np.sum(n * spin_matrices)

        # Adjust the space dimension of the Hamiltonian with the rotation on the first register spin (NV)
        H_rot = self.adjust_space_dim(self.system_num_spins, H_rot, 0)
        return H_rot.to(data_type="CSR") 

    def calc_U_rot(self, alpha, phi, theta=np.pi / 2):
        """Returns the unitary gate that rotates the first register spin (NV center) by an
        angle alpha around an axis determined by spherical angles.

        Examples
        --------
        XGate = self.calc_U_rot(np.pi, 0, theta=np.pi/2) # -1j X
        HGate = self.calc_U_rot(np.pi, 0, theta=np.pi/4) # -1j H
        """

        t = 1  # arbitrary value bacuse it cancels
        omega = alpha / t
        H_rot = self.calc_H_rot(omega, phi, theta=theta)
        U_rot = (-1j * t * H_rot).expm()
        return U_rot.to(data_type="CSR")

    def calc_eigensystem(self):
        """Saves the eigensystem of the free Hamiltonian and of the rotation
        if the pulses are not instantaneous."""

        # Return the eigensystem if it has already been calculated
        if self.eigv_free is not None and self.eigs_free is not None:
            return self.eigv_free, self.eigs_free

        # calculate free Hamiltonian matrices
        matrices = self.calc_matrices()

        # calculate eigensystem of the free Hamiltonian matrices
        eigv_free, eigs_free = [], []
        for matrix in matrices:
            matrix *= 2 * np.pi  # convert to angular frequency
            eigv, eigs = np.linalg.eigh(matrix.full())
            eigv_free.append(eigv)
            eigs_free.append(eigs)

        self.eigv_free, self.eigs_free = eigv_free, eigs_free
        return self.eigv_free, self.eigs_free

    def calc_U_time(self, eigv, eigs, t):
        """Returns the unitary gate for the time evolution given the eigenenergies and eigenstates of an Hamiltonian."""

        U_time = eigs @ np.diag(np.exp(-1j * eigv * t)) @ eigs.conj().T
        U_time = q.Qobj(U_time, dims=self.system_dims)
        return U_time.to(data_type="CSR")

    # ---------------------------------------------------

    def _get_free_evo(self, gate_params, i):
        """Returns the unitary gate for a free evolution."""

        eigv_free, eigs_free = self.calc_eigensystem()
        eigv, eigs = eigv_free[i], eigs_free[i]

        t = gate_params.get("t", 0)

        # Free evolution with dynamical decoupling pulse
        dyn_dec = gate_params.get("dyn_dec", self.dyn_dec)
        if dyn_dec:
            dyn_dec_time = gate_params.get("dyn_dec_time", t / 2)
            difference = t - dyn_dec_time
            if difference >= 0:
                before_gate = self.calc_U_time(eigv, eigs, dyn_dec_time)
                # TODO: the DD pulse is assumed to be instantaneous
                XGate = self.calc_U_rot(np.pi, 0, theta=np.pi / 2)
                after_gate = self.calc_U_time(eigv, eigs, difference)
                gate = before_gate * XGate * after_gate
            else:
                gate = self.calc_U_time(eigv, eigs, t)

        # Free evolution without dynamical decoupling pulse
        else:
            gate = self.calc_U_time(eigv, eigs, t)
        return gate

    def _get_inst_rot(self, gate_params):
        """Returns the unitary gate for an instantaneous rotation."""

        # Calculate the rotation gate
        alpha = gate_params.get("alpha", 0)
        phi = gate_params.get("phi", 0)
        theta = gate_params.get("theta", np.pi / 2)
        gate = self.calc_U_rot(alpha, phi, theta=theta)
        return gate

    def _get_cont_rot(self, gate_params, matrix):
        """Returns the unitary gate for a continuous Rabi rotation."""

        # Calculate the rotation Hamiltonian
        rabi_frequency = 2 * np.pi * self.rabi_frequency  # convert to angular frequency
        phi = gate_params.get("phi", 0)
        theta = gate_params.get("theta", np.pi / 2)
        rot_matrix = self.calc_H_rot(rabi_frequency, phi, theta=theta)

        # Calculate the time evolution gate (with the rotation)
        eigv, eigs = np.linalg.eigh((matrix + rot_matrix).full())
        t = gate_params.get("t", 0)
        gate = self.calc_U_time(eigv, eigs, t)
        return gate

    def get_single_gates(self, gate_props):
        """Returns the single gates for a given gate properties."""

        gate_type, gate_params = gate_props
        assert gate_type in ["free_evo", "inst_rot", "cont_rot"]

        single_gates = np.zeros(self.num_systems, dtype=object)

        matrices = self.calc_matrices()

        for i, matrix in enumerate(matrices):
            matrix *= 2 * np.pi  # convert to angular frequency

            if gate_type == "free_evo":
                single_gates[i] = self._get_free_evo(gate_params, i)

            if gate_type == "inst_rot":
                single_gates[i] = self._get_inst_rot(gate_params)

            if gate_type == "cont_rot":
                single_gates[i] = self._get_cont_rot(gate_params, matrix)

        return single_gates

    def get_gates(self, gate_props_list=None):
        """Returns the gates for a given list of gate properties."""

        if gate_props_list is None:
            gate_props_list = self.gate_props_list

        # Initialize the gates
        gates = np.ones(self.num_systems, dtype=object)

        # Multiplication of gates from right to left
        gate_props_list_inv = gate_props_list[::-1]
        for gate_props in gate_props_list_inv:
            single_gates = self.get_single_gates(gate_props)
            gates *= single_gates

        return gates

    # ---------------------------------------------------

    def get_gates_time(self, t_list=None):
        """Returns the gates for each time step in a given time list."""

        # Return the gates if they have already been calculated
        # if self.gates_time is not None:
        #     return self.gates_time

        if t_list is None:
            t_list = self.t_list

        # Initialize the gates
        gates_time = np.zeros((len(t_list), self.num_systems), dtype=object)

        # Get the gate properties list for each time step
        gate_props_list_time = self.get_gate_props_list_time(t_list)

        # Calculate the gates for each time step
        for i, gate_props_list in enumerate(gate_props_list_time):
            gates = self.get_gates(gate_props_list)
            gates_time[i, :] = gates

        self.gates_time = gates_time
        return self.gates_time

    # ---------------------------------------------------

    def get_states(self, t_list=None, old_register_states=None):
        """Returns the quantum states for each time step and each initial state."""

        # Return the states if they have already been calculated
        # if self.states is not None:
        #     return self.states

        if t_list is None:
            t_list = self.t_list
        if old_register_states is None:
            old_register_states = self.old_register_states

        # Initialize the states
        states = np.zeros(
            (len(old_register_states), len(t_list), self.num_systems), dtype=object
        )

        # Calculate the inital states for each system
        system_init_states_init = [
            self.calc_system_init_states(register_state)
            for register_state in old_register_states 
        ]

        # Calculate the gates for each time step
        gates_time = self.get_gates_time(t_list)

        # Loop over initial states, time steps and systems
        for i, system_init_states in enumerate(system_init_states_init):
            for j, gates in enumerate(gates_time):
                for k, gate in enumerate(gates):

                    # Evolve the initial state with the gate
                    new_state = gate * system_init_states[k] * gate.dag()

                    # Tarce out and remove the bath spins from the state
                    new_state = q.ptrace(new_state, range(self.register_num_spins))

                    # Add the offset to the density matrix for numerical stability
                    new_state += (
                        q.Qobj(np.ones(new_state.shape), dims=new_state.dims)
                        * self.dm_offset
                    )
                    states[i, j, k] = new_state

        self.states = states
        return self.states

    def get_values(self, observable, t_list=None, old_register_states=None):
        """Returns the values of an observable for each time step and each initial state."""

        # Get the quantum states
        states = self.get_states(t_list=t_list, old_register_states=old_register_states)
        shape = states.shape

        # Initialize the values
        values = np.zeros(states.size)

        states = states.flatten()
        for i, state in enumerate(states):
            if observable == "fidelity":
                fidelity = calc_fidelity(state, self.target)
                values[i] = fidelity + self.fidelity_offset
            elif observable == "log_neg":
                values[i] = calc_logarithmic_negativity(state)
            else:
                values[i] = q.expect(state, observable)

        values = values.reshape(shape)
        return values

    # -------------------------------------------------
