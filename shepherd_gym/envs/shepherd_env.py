#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate the dog-sheep shepherding environment.

Each episode requires the dog to shepherd the sheep to the goal.
"""

# core modules
import gym
import numpy as np
import logging.config
from gym import spaces

class ShepherdEnv(gym.Env):
    """
    Define the shepherding environment.

    The environment treats the dog as the agent and the sheep as a part of the environment.
    """

    def __init__(self):
        self.__version__ = '0.1.0'
        logging.info(f'ShepherdEnv - Version {self.__version__}')