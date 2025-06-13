import time
import random
from itertools import product
import multiprocessing

from tqdm import tqdm
import numpy as np
import gymnasium as gym
import qutip as q

from . import DEFAULTS
from .evolution import Evolution
from .utils import calc_fidelity, estimate_fidelity

# ----------------------------------------------------------------------

def get_observation(state):
    """Helper function to get the observation of a quantum state."""

    reshaped_state = state.full().flatten()
    return np.concatenate([np.real(reshaped_state), np.imag(reshaped_state)])

def get_state(observation, dims):
    """Helper function to get the quantum state from an observation."""

    length = len(observation)
    real_part = observation[: length // 2]
    imag_part = observation[length // 2 :]
    return q.Qobj(real_part, dims=dims) + 1j * q.Qobj(imag_part, dims=dims)

def chunk_list(lst, chunk_size=1000):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def check_state(states):
    """Function to check if the states are normalized and hermitian."""

    for state in states:
        norm = np.sum(state.diag())
        normalized = np.isclose(norm, 1, atol=5e-2)
        if not normalized:
            print("State not normalized.", norm)
        sx = q.expect(state, q.sigmax())
        sy = q.expect(state, q.sigmay())
        sz = q.expect(state, q.sigmaz())
        if abs(sx) > (1 + 1e-3):
            print("Sigma X: ", sx)
        if abs(sy) > (1 + 1e-3):
            print("Sigma Y: ", sy)
        if abs(sz) > (1 + 1e-3):
            print("Sigma Z: ", sz)
        hermitian = np.allclose(state.full(), state.dag().full(), atol=1e-5)
        if not hermitian:
            print("State not hermitian.")
    print("State check done.")

def check_fidelity(fidelities):
    """Function to check if the fidelities are between 0 and 1."""

    for fidelity in fidelities:
        if not 0 - 1e-1 <= fidelity <= 1 + 1e-1:
            print("Fidelity not between 0 and 1.", fidelity)
    print("Fidelity check done.")

# ----------------------------------------------------------------------

class Environment2(Evolution, gym.Env):
    def __init__(self, register_config, **kwargs):
        super().__init__(register_config, **kwargs)
        gym.Env.__init__(self)
        self.t0 = time.time()  # print("Time elapsed:", time.time() - self.t0)

        # keyword arguments
        self.bath_configs = kwargs.get("bath_configs", [[]])
        self.num_bath_configs = len(self.bath_configs)
        self.env_approx_level = kwargs.get(
            "env_approx_level", DEFAULTS["env_approx_level"]
        )

        # Machine Learning parameters
        self.step_kwargs = {
            'max_steps': 4,
            'infidelity_threshold': 0.01,
            'instant_pulses': True,
            'single_shot_fidelity': False,
        }

        self.count = 0
        self.fidelity = 0
        self.info = {"Fidelity": self.fidelity}
        self.reward = 0
        self.current_state = self.register_init_state

        # new code 
        self.U = q.qeye(2**self.register_num_spins)
        self.U.dims = self.register_dims
        self.observation = get_observation(self.U)
        # old code 
        # self.observation = get_observation(self.current_state)

        self.done = False

        # action and observation space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-5, high=5, shape=(33,), dtype=np.float64
        )

    # -------------------------------------------------

    def step(self, action):
        """Apply the pulse sequence to the current state of the environment and
        return the new observation and reward."""

        # reset if episode is done (either fidelity threshold is reached or max_steps is reached)
        if self.done:
            if DEFAULTS["verbose"]:
                print("episode done")
            self.reset()
        else:
            # assert self.action_space.contains(action)
            self.count += 1

        # convert action parameters to gates
        action = [(x + 1) / 2 for x in action]
        if self.step_kwargs['instant_pulses']:
            self.gate_props_list = [
                ("free_evo", dict(t=2e-6 * action[0])),
                (
                    "inst_rot",
                    dict(alpha=2 * np.pi * action[1], phi=2 * np.pi * action[2]),
                ),
            ]
        else:
            self.gate_props_list = [
                ("free_evo", dict(t=1e-6 * action[0])),
                ("cont_rot", dict(t=1e-6 * action[1], phi=2 * np.pi * action[2])),
            ]

        # choose a random bath configuration
        i = random.randint(0, self.num_bath_configs - 1)
        bath_configs_idx = [i]

        # magnetic field 
        Bz = action[3] * 500e-4  # T
        kwargs = {"Bz": Bz}

        # fidelity
        self.old_register_states = [self.current_state]
        self.current_state = self.calc_states(bath_configs_idx=bath_configs_idx, kwargs=kwargs)[0, 0]
        rho, rho_target = self.current_state, self.target
        self.fidelity = calc_fidelity(rho, rho_target)

        # Alternative: Cluster expansion on the fidelity. 
        # TODO: How to determine the old register state needed for the fidelity calculation?
        # self.fidelity = self.calc_values('fidelity', bath_configs_idx=bath_configs_idx)[0,0]

        # observation (unitary and normalized count)
        self.approx_level = 'no_bath'
        self.U = self.get_gates(gate_props_list=self.gate_props_list)[0] * self.U
        normalized_count = self.count/self.step_kwargs['max_steps']
        observation = get_observation(self.U)
        observation = np.concatenate([observation, [normalized_count]])

        # Alternative: current state 
        # self.observation = get_observation(self.current_state) 

        # Alternative: one-hot encoding of the count
        # observation = np.zeros(self.step_kwargs['max_steps'] + 1)
        # observation[self.count] = 1
        self.observation = observation

        # done
        if self.count == self.step_kwargs['max_steps']:
            self.done = True
        if 1-self.fidelity < self.step_kwargs['infidelity_threshold']:
            self.done = True

        # reward
        if self.done:
            if self.step_kwargs['single_shot_fidelity']:
                self.reward = estimate_fidelity(self.register_num_spins, 1, 1, rho, rho_target)
            else:
                self.reward = self.fidelity # -np.log(1 - self.fidelity)
        else:
            self.reward = 0

        # info
        self.info = {"Fidelity": self.fidelity}

        return (self.observation, self.reward, self.done, self.done, self.info)

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state before any pulses were applied. Resets everything changed by the step function."""

        gym.Env.reset(self, seed=seed, options=options)

        self.count = 0
        self.current_state = self.register_init_state

        # new code 
        self.U = q.qeye(2**self.register_num_spins)
        self.U.dims = self.register_dims
        observation = get_observation(self.U)
        observation = np.concatenate([observation, [0]])
        self.observation = observation

        self.fidelity = 0
        self.info = {"Fidelity": self.fidelity}
        self.reward = 0
        self.done = False

        return self.observation, self.info

    def render(self):
        return gym.Env.render(self)
    
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _to_qutip(self, dims, states):
        """Converts the states to qutip objects."""

        for init_idx, time_idx in product(*map(range, dims)):
            state = states[init_idx, time_idx]
            states[init_idx, time_idx] = q.Qobj(state, dims=self.register_dims)
        return states
    
    def _comp_gCCE0_states(self, dims, gCCE_states, states_gCCE0):

        for init_idx, time_idx in product(*map(range, dims)):
            gCCE_states[init_idx, time_idx] = states_gCCE0[init_idx, time_idx, 0].full().copy()
        return gCCE_states
    
    def _comp_gCCE1_states(self, dims, gCCE_states, states_gCCE0, states_gCCE1):

        for init_idx, time_idx in product(*map(range, dims)):
            for i in self.idx_gCCE1:
                gCCE_states[init_idx, time_idx] *= (
                    states_gCCE1[init_idx, time_idx, i].full()
                    / states_gCCE0[init_idx, time_idx, 0].full()
                )
        return gCCE_states
        
    def _comp_gCCE2_states(self, dims, gCCE_states, states_gCCE0, 
                           states_gCCE1, states_gCCE2):
        
        for init_idx, time_idx in product(*map(range, dims)):
            for i, idx_gCCE2 in enumerate(self.idx_gCCE2):
                j, k = idx_gCCE2
                gCCE_states[init_idx, time_idx] *= (
                    states_gCCE2[init_idx, time_idx, i].full()
                    * states_gCCE0[init_idx, time_idx, 0].full()
                ) / (
                    states_gCCE1[init_idx, time_idx, j].full()
                    * states_gCCE1[init_idx, time_idx, k].full()
                )
        return gCCE_states
    
    def get_gCCE_states(self, t_list, old_register_states):
        """Function to compute the bath quantities with different gCCE orders."""

        # Initialize the gCCE states
        num_old_register_states = len(old_register_states)
        num_t_list = len(t_list)
        dims = (num_old_register_states, num_t_list)
        gCCE_states = np.zeros(dims, dtype=object)

        # gCCE0
        self.approx_level = "gCCE0"
        states_gCCE0 = self.get_states(t_list, old_register_states)
        gCCE_states = self._comp_gCCE0_states(dims, gCCE_states, states_gCCE0)

        if self.env_approx_level == "gCCE0":
            states = self._to_qutip(dims, gCCE_states)
            return states

        # gCCE1
        self.approx_level = "gCCE1"
        states_gCCE1 = self.get_states(t_list, old_register_states)
        gCCE_states = self._comp_gCCE1_states(dims, gCCE_states, 
                                              states_gCCE0, states_gCCE1)

        if self.env_approx_level == "gCCE1":
            states = self._to_qutip(dims, gCCE_states)
            return states

        # gCCE2
        self.approx_level = "gCCE2"
        states_gCCE2 = self.get_states(t_list, old_register_states)
        gCCE_states = self._comp_gCCE2_states(dims, gCCE_states, states_gCCE0, 
                                              states_gCCE1, states_gCCE2)

        if self.env_approx_level == "gCCE2":
            states = self._to_qutip(dims, gCCE_states)
            return states

    # ------------------------------------------------------------------

    def _comp_gCCE0_values(self, gCCE_values, values_gCCE0):

        gCCE_values = values_gCCE0[:, :, 0].copy()
        return gCCE_values

    def _comp_gCCE1_values(self, gCCE_values, values_gCCE0, values_gCCE1):

        for i in self.idx_gCCE1:
            gCCE_values[:, :] *= values_gCCE1[:, :, i] / values_gCCE0[:, :, 0]
        return gCCE_values

    def _comp_gCCE2_values(self, gCCE_values, values_gCCE0, 
                           values_gCCE1, values_gCCE2):
        
        for i, idx_gCCE2 in enumerate(self.idx_gCCE2):
            j, k = idx_gCCE2
            gCCE_values[:, :] *= (values_gCCE2[:, :, i] * values_gCCE0[:, :, 0]) / (
                values_gCCE1[:, :, j] * values_gCCE1[:, :, k]
            )
        return gCCE_values

    def get_gCCE_values(self, observable, t_list, old_register_states):

        # Initialize the gCCE values
        num_old_register_states = len(old_register_states)
        num_t_list = len(t_list)
        dims = (num_old_register_states, num_t_list)
        gCCE_values = np.zeros(dims)

        # gCCE0
        self.approx_level = "gCCE0"
        values_gCCE0 = self.get_values(observable, t_list, old_register_states)
        gCCE_values = self._comp_gCCE0_values(gCCE_values, values_gCCE0)
        if self.env_approx_level == "gCCE0":
            return gCCE_values

        # gCCE1
        self.approx_level = "gCCE1"
        values_gCCE1 = self.get_values(observable, t_list, old_register_states)
        gCCE_values = self._comp_gCCE1_values(gCCE_values, values_gCCE0, 
                                              values_gCCE1)
        if self.env_approx_level == "gCCE1":
            return gCCE_values

        # gCCE2
        self.approx_level = "gCCE2"
        values_gCCE2 = self.get_values(observable, t_list, old_register_states)
        gCCE_values = self._comp_gCCE2_values(gCCE_values, values_gCCE0, 
                                              values_gCCE1, values_gCCE2)
        if self.env_approx_level == "gCCE2":
            return gCCE_values

    # ------------------------------------------------------------------

    def wrap_get_states(self, t_list, old_register_states):
        if self.env_approx_level == "no_bath":
            self.approx_level = "no_bath"
            states = self.get_states(t_list, old_register_states)[:, :, 0]

        elif self.env_approx_level == "full_bath":
            self.approx_level = "full_bath"
            states = self.get_states(t_list, old_register_states)[:, :, 0]

        else:
            states = self.get_gCCE_states(t_list, old_register_states)

        if self.verbose:
            pass
            # check_state(states.flatten())
        
        return states
    
    def wrap_get_values(self, observable, t_list, old_register_states):
        if self.env_approx_level == "no_bath":
            self.approx_level = "no_bath"
            values = self.get_values(observable, t_list, old_register_states)[:, :, 0]

        elif self.env_approx_level == "full_bath":
            self.approx_level = "full_bath"
            values = self.get_values(observable, t_list, old_register_states)[:, :, 0]

        else:
            values = self.get_gCCE_values(observable, t_list, old_register_states)
        
        if self.verbose and observable == "fidelity":
            pass
            # check_fidelity(values.flatten())

        return values

    def calc_quantity(self, idx, quantity, t_list, old_register_states, kwargs):
        """Function to compute bath quantities in parallel."""

        if kwargs is not None:
            bath_config = self.bath_configs[idx].copy()
            for i, spin in enumerate(bath_config):
                spin = list(spin)
                spin[3] = kwargs
                spin = tuple(spin)
                bath_config[i] = spin
            self.bath_config = bath_config

        if quantity == "states": 
            return self.wrap_get_states(t_list, old_register_states)
        
        else:
            observable = quantity
            return self.wrap_get_values(observable, t_list, old_register_states)
 
    def calc_quantites(self, quantity, bath_configs_idx, t_list, old_register_states, kwargs):

        if kwargs is not None:
            register_config = self.register_config.copy()
            for i, spin in enumerate(register_config):
                spin = list(spin) 
                spin[3] = kwargs
                spin = tuple(spin)
                register_config[i] = spin
            self.register_config = register_config

        if bath_configs_idx is None:
            bath_configs_idx = range(self.num_bath_configs)
        if t_list is None:
            t_list = self.t_list
        if old_register_states is None:
            old_register_states = self.old_register_states

        if self.env_approx_level == "no_bath":
            return self.calc_quantity(0, quantity, t_list, old_register_states, kwargs)

        disable_tqdm = True # True
        message = "Sampling over spin baths..."

        quantites_baths = np.zeros((len(old_register_states), len(t_list)), dtype=object)
        args = [(i, quantity, t_list, old_register_states, kwargs) for i in bath_configs_idx]

        if not self.parallelization:
            for i in tqdm(range(len(bath_configs_idx)), 
                          desc=message, 
                          disable=disable_tqdm
                        ):
                quantites_baths += self.calc_quantity(*args[i])

        else:
            with multiprocessing.Pool(processes=self.num_cpu) as pool:
                # for i, chunk in enumerate(chunk_list(bath_configs_idx, chunk_size=10000)):

                results = list(
                    tqdm( 
                        pool.imap(self.calc_quantity, args, chunksize=len(bath_configs_idx)//100),
                        total=len(bath_configs_idx),
                        desc=message,
                        disable=disable_tqdm,
                    )
                )
                quantites_baths += sum(results)

        return 1 / len(bath_configs_idx) * quantites_baths

    def calc_states(self, t_list=None, old_register_states=None, bath_configs_idx=None, kwargs=None):
        return self.calc_quantites("states", bath_configs_idx, t_list, old_register_states, kwargs)

    def calc_values(self, observable, t_list=None, old_register_states=None, bath_configs_idx=None, kwargs=None):
        return self.calc_quantites(observable, bath_configs_idx, t_list, old_register_states, kwargs)

# ----------------------------------------------------------------------
