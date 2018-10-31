"""shepherd_gym - Gym environment implementation of dog shepherding task"""

from gym.envs.registration import register

register(
    id='Shepherd-v0',
    entry_point='shepherd_gym.envs:ShepherdEnv',
)
