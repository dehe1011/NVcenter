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
from .utils import calc_fidelity

# -------------------------------------------------

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

# -------------------------------------------------

class Environment2(Evolution, gym.Env):
    def __init__(self, register_config, **kwargs):

        super().__init__(register_config, **kwargs)
        gym.Env.__init__(self)

        self.t0 = time.time()

        self.bath_configs = kwargs.get("bath_configs", [[]] )
        self.num_bath_configs = len(self.bath_configs)
        self.env_approx_level = kwargs.get(
            "env_approx_level", DEFAULTS["env_approx_level"]
        )

        self.max_steps = 4
        self.infidelity_threshold = 0.1
        self.dims = self.register_init_state.dims

        self.count = 0
        self.fidelity = 0
        self.info = {"Fidelity": self.fidelity}
        self.reward = 0
        self.current_state = self.register_init_state
        self.observation = get_observation(self.current_state)
        self.done = False

        self.action_space = gym.spaces.Box(low=-1, high= 1, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-5, high=5, shape=(32,), dtype=np.float64)

# -------------------------------------------------

    def step(self, action, instant_pulses=True):
        """Apply the pulse sequence to the current state of the environment and
        return the new observation and reward."""

        if self.done:
            if DEFAULTS["verbose"]:
                print("episode done")
            self.reset()
        else:
            self.count += 1

        action = [(x+1)/2 for x in action]

        if instant_pulses:
            self.gate_props_list = [('free_evo', dict(t=1e-6*action[0])), ('inst_rot', dict(alpha=2*np.pi*action[1], phi=2*np.pi*action[2] ) ) ]
        else: 
            self.gate_props_list = [('free_evo', dict(t=1e-6*action[0])), ('cont_rot', dict(t=1e-6*action[1], phi= 2*np.pi*action[2] ) ) ]

        self.old_register_states = [self.current_state]

        i = random.randint(0, self.num_bath_configs-1)
        self.bath_config = self.bath_configs[i]
        self.current_state = self.calc_states()[0,0]
        # self.fidelity = self.calc_values('fidelity', old_register_states=old_register_states)[0,0]
        self.current_state = q.Qobj(self.current_state, dims =self.target.dims)
        self.fidelity = calc_fidelity(self.current_state, self.target)

        self.observation = get_observation(self.current_state)

        # done
        if self.count == self.max_steps:
            self.done = True
        # if 1-self.fidelity < self.infidelity_threshold:
        #     self.done = True

        # reward
        if self.done:
            self.reward = -np.log(1 - self.fidelity)  # -0.05*self.count
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
        self.observation = get_observation(self.current_state)
        self.fidelity = 0
        self.info = {"Fidelity": self.fidelity}
        self.reward = 0
        self.done = False

        return self.observation, self.info
    
    def render(self):
        return gym.Env.render(self)

# -------------------------------------------------

    def get_no_bath(self, quantity, old_register_states, t_list):
        self.bath_config = []
        self.approx_level = self.env_approx_level

        if quantity == 'states':
            return self.get_states(old_register_states=old_register_states, t_list=t_list)[:,:,0]
        
        else:
            observable = quantity
            return self.get_values(observable, old_register_states=old_register_states, t_list=t_list)[:,:,0]

    def get_full_bath(self, quantity, old_register_states, t_list):
        self.approx_level = "full_bath"
        if quantity == 'states':
            return self.get_states(old_register_states=old_register_states, t_list=t_list)[:,:,0]
        
        else:
            observable = quantity
            return self.get_values(observable, old_register_states=old_register_states, t_list=t_list)[:,:,0]

    def _to_qutip(self, num_old_register_states, num_t_list, gCCE_states):
        for init_idx, time_idx in product(range(num_old_register_states), range(num_t_list)):
            gCCE_states[init_idx, time_idx] = q.Qobj(gCCE_states[init_idx, time_idx], dims=self.dims)
        return gCCE_states
    
    def get_gCCE_states(self, old_register_states, t_list):

        self.approx_level = "gCCE0"
        states_gCCE0 = self.get_states(old_register_states=old_register_states, t_list=t_list)

        num_old_register_states, num_t_list = states_gCCE0.shape[:2]
        gCCE_states = np.zeros((num_old_register_states, num_t_list), dtype=object) # important

        for init_idx, time_idx in product(range(num_old_register_states), range(num_t_list)):
            gCCE_states[init_idx, time_idx] = states_gCCE0[init_idx, time_idx,0].full()

        if self.env_approx_level == "gCCE0":
            return self._to_qutip(num_old_register_states, num_t_list, gCCE_states)
        
        self.approx_level = "gCCE1"
        states_gCCE1 = self.get_states(old_register_states=old_register_states, t_list=t_list)

        for init_idx, time_idx in product(range(num_old_register_states), range(num_t_list)):
            for i in self.idx_gCCE1:
                gCCE_states[init_idx, time_idx] *= states_gCCE1[init_idx, time_idx, i].full() / states_gCCE0[init_idx, time_idx,0].full()

        if self.env_approx_level == "gCCE1":
            return self._to_qutip(num_old_register_states, num_t_list, gCCE_states)

        self.approx_level = "gCCE2"
        states_gCCE2 = self.get_states(old_register_states=old_register_states, t_list=t_list)

        for init_idx, time_idx in product(range(num_old_register_states), range(num_t_list)):
            for i, idx_gCCE2 in enumerate(self.idx_gCCE2):
                j, k = idx_gCCE2
                gCCE_states[init_idx, time_idx] *= (states_gCCE2[init_idx, time_idx,i].full() * states_gCCE0[init_idx, time_idx, 0].full()) / (states_gCCE1[init_idx, time_idx,j].full() * states_gCCE1[init_idx, time_idx,k].full())
  
        if self.env_approx_level == "gCCE2":
            return self._to_qutip(num_old_register_states, num_t_list, gCCE_states)
        
    def get_gCCE_values(self, observable, old_register_states, t_list):

        self.approx_level = "gCCE0"
        values_gCCE0 = self.get_values(observable, old_register_states=old_register_states, t_list=t_list)

        gCCE_values = values_gCCE0[:, :,0]

        if self.env_approx_level == "gCCE0":
            return gCCE_values
        
        self.approx_level = "gCCE1"
        values_gCCE1 = self.get_values(observable, old_register_states=old_register_states, t_list=t_list)

        for i in self.idx_gCCE1:
            gCCE_values[:, :] *= values_gCCE1[:, :,i] / values_gCCE0[:, :,0]

        if self.env_approx_level == "gCCE1":
            return gCCE_values
        
        self.approx_level = "gCCE2"
        values_gCCE2 = self.get_values(observable, old_register_states=old_register_states, t_list=t_list)

        for i, idx_gCCE2 in enumerate(self.idx_gCCE2):
            j, k = idx_gCCE2
            gCCE_values[:, :] *= (values_gCCE2[:, :,i] * values_gCCE0[:, :,0]) / (values_gCCE1[:, :,j] * values_gCCE1[:, :,k])

        if self.env_approx_level == "gCCE2":
            return gCCE_values

# -------------------------------------------------

    def _calc_quantity(self, i, quantity, old_register_states, t_list):
        """Function to compute bath quantities in parallel."""
        self.bath_config = self.bath_configs[i]

        if self.env_approx_level == "full_bath":
            return self.get_full_bath(quantity, old_register_states, t_list)

        if self.env_approx_level in ["gCCE0", "gCCE1", "gCCE2"]:
            if quantity == 'states':
                return self.get_gCCE_states(old_register_states, t_list)
            else:
                observable = quantity
                return self.get_gCCE_values(observable, old_register_states, t_list)
        
        return 0  # Default return if no computation was done
    
    def _calc_quantity_wrapper(self, args):
        """Wrapper function for multiprocessing"""
        i, quantity, old_register_states, t_list = args
        return self._calc_quantity(i, quantity, old_register_states, t_list)
    
    def calc_quantites(self, quantity, old_register_states, t_list):

        if t_list is None:
            t_list = self.t_list
        if old_register_states is None:
            old_register_states = self.old_register_states

        if self.env_approx_level == "no_bath":
            return self.get_no_bath(quantity, old_register_states, t_list)
        
        disable_tqdm = not self.verbose
        message = f"Calculating new states for {self.env_approx_level}"

        quantites_baths = np.zeros((len(old_register_states), len(t_list)), dtype=object)

        if not self.parallelization:
            for i in tqdm(range(self.num_bath_configs), desc=message, disable=disable_tqdm):
                quantites_baths += self._calc_quantity(i, quantity, old_register_states, t_list)

        else:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                args = [(i, quantity, old_register_states, t_list) for i in range(self.num_bath_configs)]
                results = list(tqdm(pool.imap(self._calc_quantity_wrapper, args), total=self.num_bath_configs, desc=message, disable=disable_tqdm))
            quantites_baths = sum(results)
        
        return 1 / self.num_bath_configs * quantites_baths
    
    def calc_states(self, old_register_states=None, t_list=None):
        return self.calc_quantites('states', old_register_states, t_list)

    def calc_values(self, observable, old_register_states=None, t_list=None):
        return self.calc_quantites(observable, old_register_states, t_list)
    
# -------------------------------------------------


