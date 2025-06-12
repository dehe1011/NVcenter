# ----------------------------------------------------------------------
# Hyperparameter optimization guide
# ----------------------------------------------------------------------


# 1. Create and activate a virtual environment:
# ----------------------------------------------------------------------
# python -m venv .venv
# .venv/scripts/activate


# 2. Install the required packages:
# ----------------------------------------------------------------------
# git clone https://github.com/dehe1011/NVcenter
# cd NVcenter
# pip install -e .


# 3. Modify rl_zoo3 hyperparameters:
# ----------------------------------------------------------------------
# .venv/Lib/site-packages/rl_zoo3/hyperparams/ppo.yml

# ClusterExpansion-v1:
#   normalize: true
#   n_envs: 1
#   policy: 'MlpPolicy'
#   n_timesteps: 100000
#   batch_size: 256
#   n_steps: 2048
#   gamma: 0.9996
#   learning_rate: 0.00005
#   ent_coef: 0.001
#   clip_range: 0.2
#   n_epochs: 10
#   gae_lambda: 0.95
#   max_grad_norm: 0.5
#   vf_coef: 0.5


# 4. Register the custom environment:
# ----------------------------------------------------------------------
# .venv/Lib/site-packages/rl_zoo3/import_envs.py

# import qutip as q
# C13_pos = (8.728883757198979e-10, 0.0, 1.8558998769620693e-10) # Dominik
# register(
#     id='ClusterExpansion-v1',
#     entry_point='NVcenter:custom_envs:ClusterExpansion',
#     kwargs= {"register_config": [('NV', (0, 0, 0), 0, {}), ('C13', C13_pos, 0, {})],
#              "target": 0.5 * q.tensor(q.ket2dm(q.basis(2,0)), q.ket2dm( q.basis(2,0) + q.basis(2,1) )),
#              }, 
# )

# 5. Create an alias environment class in this file
# ----------------------------------------------------------------------
# NVcenter/custom_envs.py
from NVcenter import Environment2

class ClusterExpansion(Environment2):
    pass

# 6. Run the training script (in the NVcenter folder):
# ----------------------------------------------------------------------
# python -m rl_zoo3.train --algo ppo --env ClusterExpansion-v1 -n 50000 -optimize --n-trials 1000 --n-jobs 2 --sampler random --pruner median
# ----------------------------------------------------------------------
