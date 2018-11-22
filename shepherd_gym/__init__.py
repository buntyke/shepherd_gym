"""shepherd_gym - Gym environment implementation of dog shepherding task"""

from gym.envs.registration import register
from shepherd_gym.shepherd_sim import ShepherdSim
from shepherd_gym.models.dog_heuristic import dog_heuristic_model

register(
    id='Shepherd-v0',
    entry_point='shepherd_gym.envs:ShepherdEnv',
)

register(
    id='Shepherd-v1',
    entry_point='shepherd_gym.envs:ShepherdFixedEnv',
)