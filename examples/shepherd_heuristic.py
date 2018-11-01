#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the shepherd gym env with dog heuristic model.
"""

# core modules
import gym
import numpy as np
import matplotlib.pyplot as plt

# shepherd_gym functions
import shepherd_gym
from shepherd_gym.models.dog_heuristic import dog_heuristic_model

def main():
    # run the simulation several times
    n_trials = 10

    # create the environment
    shepherd_env = gym.make('Shepherd-v0')

    # render the simulation
    shepherd_env.render()
    
    for n in range(n_trials):
        # reset the environment
        (state,reward,finish,info) = shepherd_env.reset()
        
        # run the main simulation
        while not finish:
            # get the dog's action
            action = dog_heuristic_model(state, info)

            # execute the action and update the state
            (state, reward, finish, info) = shepherd_env.step(action)

        print(f'Finish simulation: {n}')

if __name__=='__main__':
    main()