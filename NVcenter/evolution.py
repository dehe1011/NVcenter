import time
import copy

import numpy as np
import qutip as q
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator

from . import DEFAULTS
from .hamiltonian import Hamiltonian, adjust_space_dim
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

    def __init__(self, register_config, **kwargs):
        self.t0 = time.time()
        self._register_config = register_config

        # Keyword arguments
        self._bath_config = kwargs.get("bath_config", [])
        self._approx_level = kwargs.get("approx_level", DEFAULTS["approx_level"])

        # Initialize the Hamiltonian parent class
        super().__init__(self.register_config, **kwargs)

        # Keyword arguments
        self.old_register_states = kwargs.get(
            "old_register_states", [self.register_init_state]
        )
        self.target = kwargs.get("target", self.register_identity)
        self._gate_props_list = kwargs.get(
            "gate_props_list", DEFAULTS["gate_props_list"]
        )
        self.dynamical_decoupling = kwargs.get(
            "dynamical_decoupling", DEFAULTS["dynamical_decoupling"]
        )

        self.rabi_frequency = kwargs.get("rabi_frequency", DEFAULTS["rabi_frequency"])
        self.dm_offset = kwargs.get("dm_offset", DEFAULTS["dm_offset"])
        self.num_hahn_echos = kwargs.get("num_hahn_echos", DEFAULTS["num_hahn_echos"])

        self.init_gate_times(**kwargs)

        # Save the unitary gates
        self.gates_time = None
        # Save the quantum states
        self.states = None

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

    @property
    def gate_props_list(self):  # pylint: disable=missing-function-docstring
        return self._gate_props_list

    @gate_props_list.setter
    def gate_props_list(self, new_gate_props_list):

        if new_gate_props_list != self._gate_props_list:
            self._gate_props_list = new_gate_props_list
            self.init_gate_times()

    # -------------------------------------------------

    def init_gate_times(self, **kwargs):
        self.gate_time_list = [
            gate_props[1].get("t", 0) for gate_props in self.gate_props_list
        ]
        self.cum_gate_time_list = [
            sum(self.gate_time_list[: i + 1]) for i in range(len(self.gate_time_list))
        ]
        self.total_gate_time = sum(self.gate_time_list)

        self._t_list = kwargs.get("t_list", "final")
        if isinstance(self._t_list, str):
            if self._t_list == "final":
                self._t_list = [self.total_gate_time]

            if self._t_list == "auto":
                self._t_list = self.get_t_list()
        self.eigv_free, self.eigs_free = None, None

    def get_t_list(self, stepsize=0.5e-6):
        """Helper function to get the time list for a pulse sequence."""

        t_cum = self.cum_gate_time_list
        t_list = []
        t_list.extend(np.arange(0, t_cum[-1], stepsize))
        pulse_time_before = list(t_cum[:-1])
        pulse_time_after = [pulse_time + 1e-9 for pulse_time in t_cum[:-1]]
        t_list.extend(pulse_time_before)
        t_list.extend(pulse_time_after)
        t_list.append(t_cum[-1])

        return np.real(sorted(t_list))

    def get_gate_props_list_time(self, t_list):

        if self.cum_gate_time_list == []:
            return [("free_evo", dict(t=0.0, dynamical_decoupling=False))]

        gate_props_list_time = []
        t_cum = self.cum_gate_time_list
        left_time, finished_gates = 0, 0
        for t in t_list:
            if t == 0:
                gate_props_list_time.append(
                    [("free_evo", dict(t=0.0, dynamical_decoupling=False))]
                )
                continue

            # Find the number of finished gates and the left time
            threshold = t
            for i, time in enumerate(t_cum):
                if time <= threshold:
                    finished_gates = i + 1
                else:
                    break

            if finished_gates == 0:
                left_time = threshold
            else:
                left_time = threshold - t_cum[finished_gates - 1]

            # Create a list of the finished gates
            new_gate_props_list = copy.deepcopy(
                self.gate_props_list[:finished_gates]
            )  # Deep copy

            if left_time != 0:
                # add free evolution after the last gate if all gates are finished
                if finished_gates == len(self.gate_props_list):
                    hahn_time = float(left_time / (self.num_hahn_echos + 1))
                    after_gates = [
                        ("free_evo", dict(t=hahn_time, dynamical_decoupling=False))
                    ]
                    for _ in range(self.num_hahn_echos):
                        after_gates.append(
                            ("inst_rot", dict(alpha=np.pi, phi=0, theta=np.pi / 2))
                        )
                        after_gates.append(
                            ("free_evo", dict(t=hahn_time, dynamical_decoupling=False))
                        )
                    new_gate_props_list.extend(after_gates)

                # add a fraction of the next gate if there is time left
                else:
                    current_gate = copy.deepcopy(
                        self.gate_props_list[finished_gates]
                    )  # Deep copy of current_gate
                    current_gate[1]["t"] = float(left_time)
                    new_gate_props_list.append(current_gate)

            gate_props_list_time.append(new_gate_props_list)
        return gate_props_list_time

    # ------------------------------------------------

    def calc_H_rot(self, omega, phi, theta=np.pi / 2):
        """Returns a Hamiltonian that rotates the first register spin (NV center) with the Lamor
        frequency around an axis determined by spherical angles."""

        n = np.array([spherical_to_cartesian(1, phi, theta)])
        H_rot = omega * np.sum(
            n * get_spin_matrices(1 / 2)[1:]
        )  # factor 1/2 times Pauli matrices
        H_rot = adjust_space_dim(self.system_num_spins, H_rot, 0)
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
        return (-1j * t * H_rot).expm()

    def calc_U_time(self, eigv, eigs, t):
        """Returns the unitary gate for the time evolution given the eigenenergies and eigenstates of an Hamiltonian."""

        U_time = eigs @ np.diag(np.exp(-1j * eigv * t)) @ eigs.conj().T
        U_time = q.Qobj(
            U_time, dims=[[2] * self.system_num_spins, [2] * self.system_num_spins]
        )
        return U_time.to(data_type="CSR")

    # ---------------------------------------------------

    def calc_eigensystem(self):
        """Saves the eigensystem of the free Hamiltonian and of the rotation
        if the pulses are not instantaneous."""
        matrices = self.calc_matrices()

        eigv_free, eigs_free = [], []
        for matrix in matrices:
            matrix *= 2 * np.pi  # convert to angular frequency
            eigv, eigs = np.linalg.eigh(matrix.full())
            eigv_free.append(eigv)
            eigs_free.append(eigs)

        self.eigv_free, self.eigs_free = eigv_free, eigs_free
        return self.eigv_free, self.eigs_free

    def get_single_gates(self, gate_props):

        gate_type, gate_params = gate_props
        assert gate_type in ["free_evo", "inst_rot", "cont_rot"]

        single_gates = np.zeros(self.num_systems, dtype=object)

        matrices = self.calc_matrices()

        for i, matrix in enumerate(matrices):
            matrix *= 2 * np.pi  # convert to angular frequency

            if gate_type == "free_evo":
                if self.eigv_free is None or self.eigs_free is None:
                    self.calc_eigensystem()
                eigv, eigs = self.eigv_free[i], self.eigs_free[i]

                t = gate_params.get("t", 0)
                dynamical_decoupling = gate_params.get(
                    "dynamical_decoupling", self.dynamical_decoupling
                )

                if dynamical_decoupling:
                    dyn_dec_time = gate_params.get("dyn_dec_time", t / 2)
                    difference = t - dyn_dec_time
                    if difference >= 0:
                        before_gate = self.calc_U_time(eigv, eigs, dyn_dec_time)
                        after_gate = self.calc_U_time(eigv, eigs, difference)
                        # TODO: since the DD takes a finite time for continous drive one must be careful
                        # XGate = self.calc_U_time(eigv_rot, eigs_rot, 1/(2*self.rabi_frequency))
                        XGate = self.calc_U_rot(np.pi, 0, theta=np.pi / 2)
                        gate = before_gate * XGate * after_gate
                    else:
                        gate = self.calc_U_time(eigv, eigs, t)

                else:
                    gate = self.calc_U_time(eigv, eigs, t)
                single_gates[i] = gate

            if gate_type == "inst_rot":
                alpha = gate_params.get("alpha", 0)
                phi, theta = gate_params.get("phi", 0), gate_params.get(
                    "theta", np.pi / 2
                )
                gate = self.calc_U_rot(alpha, phi, theta=theta)
                single_gates[i] = gate

            if gate_type == "cont_rot":
                rabi_frequency = (
                    2 * np.pi * self.rabi_frequency
                )  # convert to angular frequency
                phi, theta = gate_params.get("phi", 0), gate_params.get(
                    "theta", np.pi / 2
                )
                rot_matrix = self.calc_H_rot(rabi_frequency, phi, theta=theta)
                eigv, eigs = np.linalg.eigh((matrix + rot_matrix).full())

                t = gate_params.get("t", 0)
                gate = self.calc_U_time(eigv, eigs, t)
                single_gates[i] = gate
        return single_gates

    def get_gates(self, gate_props_list=None):

        if gate_props_list is None:
            gate_props_list = self.gate_props_list

        gates = np.ones(self.num_systems, dtype=object)
        gate_props_list_inv = gate_props_list[
            ::-1
        ]  # because multiplication happens right to left

        for i, gate_props in enumerate(gate_props_list_inv):
            single_gates = self.get_single_gates(gate_props)
            gates *= single_gates

        return gates

    def get_gates_time(self, t_list=None):

        # Return the gates if they have already been calculated
        # if self.gates_time is not None:
        #     return self.gates_time

        self.calc_eigensystem()

        if t_list is None:
            t_list = self.t_list
        gate_props_list_time = self.get_gate_props_list_time(t_list)

        gates_time = np.zeros((len(t_list), self.num_systems), dtype=object)
        for i, gate_props_list in enumerate(gate_props_list_time):
            gates = self.get_gates(gate_props_list)
            gates_time[i][:] = gates

        self.gates_time = gates_time
        return self.gates_time

    # ---------------------------------------------------

    def get_states(self, t_list=None, old_register_states=None):

        # Return the states if they have already been calculated
        # if self.states is not None:
        #     return self.states

        if t_list is None:
            t_list = self.t_list
        if old_register_states is None:
            old_register_states = self.old_register_states

        states = np.zeros(
            (len(old_register_states), len(t_list), self.num_systems), dtype=object
        )

        system_init_states_init = [
            self.calc_system_init_states(register_state)
            for register_state in old_register_states
        ]
        gates_time = self.get_gates_time(t_list)

        for i, system_init_states in enumerate(system_init_states_init):
            for j, gates in enumerate(gates_time):
                for k, gate in enumerate(gates):
                    new_state = q.ptrace(
                        gate * system_init_states[k] * gate.dag(),
                        range(self.register_num_spins),
                    )
                    new_state += (
                        q.Qobj(np.ones(new_state.shape), dims=new_state.dims)
                        * self.dm_offset
                    )
                    states[i, j, k] = new_state

        self.states = states
        return self.states

    def get_values(self, observable, t_list=None, old_register_states=None):

        states = self.get_states(t_list=t_list, old_register_states=old_register_states)
        shape = states.shape
        states = states.flatten()

        values = np.zeros(states.size)

        for i, state in enumerate(states):
            if observable == "fidelity":
                values[i] = calc_fidelity(state, self.target)
            elif observable == "log_neg":
                values[i] = calc_logarithmic_negativity(state)
            else:
                values[i] = q.expect(state, observable)

        values = values.reshape(shape)
        return values
