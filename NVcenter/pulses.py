import numpy as np 
import qutip as q

from .helpers import calc_fidelity 
from .system import System

# -------------------------------------------


class Pulses:
    r""" This class 

    Notes:
        For the optimization the isolated register is considered (neglection of bath spins).  
    """
    
    def __init__(self, spin_config, pulse_seq, target_state, bath_approx_level, dynamical_decoupling=False):
        # constructor arguments
        self.spin_config = spin_config
        self.pulse_seq = pulse_seq
        self.target_state = target_state
        self.bath_approx_level = bath_approx_level
        self.dynamical_decoupling = dynamical_decoupling

        # System
        self.system = System(self.spin_config.system_configs[self.bath_approx_level], self.spin_config.mf_configs[self.bath_approx_level] )
        if self.bath_approx_level == "full_bath":
            self.init_state = q.tensor([q.qeye(2)]*self.system.system_num_spins)
        else:
            self.init_state = self.system.init_state
        
        # Pulse sequence
        self.num_pulses = (len(self.pulse_seq)-1)//3
        self.time_list = self.pulse_seq[:self.num_pulses+1]
        self.theta_list = self.pulse_seq[self.num_pulses+1:2*self.num_pulses+1]
        self.phi_list = self.pulse_seq[2*self.num_pulses+1:]
        self.pulses = [(self.theta_list[i], self.phi_list[i], self.time_list[i]) for i in range(self.num_pulses)]
        self.pulse_time = sum(self.time_list)

        # pulse interval
        self.pulse_interval = 0
        self.final_state = self.calc_final_state()
        self.final_fidelity = self.calc_final_fidelity()

    # -------------------------------------------
    
    # Construct unitary pulse matrix
    def calc_U_rot(self, theta, phi):
        # RGate in qiskit: qc.r()
        Sx = self.system.S[0][1] # S[0] are the spin matrices of the NV center in the full system Hilbert space
        Sy = self.system.S[0][2]
        return (-1j*theta/np.sqrt(2) * ( Sx * np.cos(phi) + Sy * np.sin(phi) )).expm()

    def calc_U_time(self, time):
        U_time = self.system.eigs @ np.diag(np.exp(-1j * self.system.eigv * time)) @ self.system.eigs.conj().T
        return q.Qobj(U_time, dims=[[2]*self.system.system_num_spins, [2]*self.system.system_num_spins])

    def calc_U_pi(self):
        return self.calc_U_rot(np.pi, 0)
        # return np.sqrt(2)*self.system.S[0][1] # X-gate on the NV center, just differs by a global phase from the previous implementation

    def get_S_pulse(self):
        pulse_list = [ self.calc_U_time(self.time_list[0]) ]
        for i in range(self.num_pulses):
            U_rot =  self.calc_U_rot(self.theta_list[i], self.phi_list[i])
            pulse_list.append(U_rot)
            if not self.dynamical_decoupling: 
                U_time = self.calc_U_time(self.time_list[i+1])
            else:
                U_time = self.calc_U_time(self.time_list[i+1]/2) * self.calc_U_pi() * self.calc_U_time(self.time_list[i+1]/2)
            pulse_list.append(U_time)

        pulse_list = pulse_list[:self.pulse_interval] # for smaller times
        S_pulse = self.system.I
        for pulse in pulse_list[::-1]:
            S_pulse *= pulse
        return S_pulse

    # -------------------------------------------
    # intermediate_states 

    def calc_intermediate_state(self, t):
        """ TODO: resolve time between decoupling pulses """
        # If time exceeds or matches pulse time, calculate for the post-pulse regime
        
        if t >= self.pulse_time:
            self.pulse_interval = self.num_pulses * 2 + 1
            unitary = self.calc_U_time(t - self.pulse_time) * self.get_S_pulse()
            intermediate_state = unitary * self.init_state * unitary.dag()
            return q.ptrace(intermediate_state, np.arange(self.spin_config.register_num_spins) )
    
        # Otherwise, find the correct pulse interval
        cumulative_time = 0
        for i, time_interval in enumerate(self.time_list):
            cumulative_time += time_interval
            if t < cumulative_time:
                t_remaining = t - (cumulative_time - time_interval)
                self.pulse_interval = 2*i
                unitary = self.calc_U_time(t_remaining) * self.get_S_pulse()
                intermediate_state = unitary * self.init_state * unitary.dag()  
                return q.ptrace(intermediate_state, np.arange(self.spin_config.register_num_spins) )

    def calc_state_evo(self, times):
        return [self.calc_intermediate_state(t) for t in times]

    def calc_final_state(self):
        return self.calc_intermediate_state(self.pulse_time)

    def calc_intermediate_fidelity(self, t):
        return calc_fidelity(self.calc_intermediate_state(t), self.target_state)

    def calc_fidelity_evo(self, times):
        return [self.calc_intermediate_fidelity(t) for t in times]

    def calc_final_fidelity(self):
        return calc_fidelity(self.final_state, self.target_state)
