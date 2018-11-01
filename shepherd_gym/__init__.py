"""shepherd_gym - Gym environment implementation of dog shepherding task"""

from gym.envs.registration import register
from shepherd_gym.shepherd_sim import ShepherdSim

register(
    id='Shepherd-v0',
    entry_point='shepherd_gym.envs:ShepherdEnv',
)
