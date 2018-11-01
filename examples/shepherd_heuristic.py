#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the shepherd gym env with dog heuristic model.
"""

# core modules
import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ipython debugging
from IPython.terminal.debugger import set_trace as keyboard

# shepherd_gym functions
import shepherd_gym
from shepherd_gym.models.dog_heuristic import dog_heuristic_model

def main():
    # setup argument parser


    # run the simulation several times
    n_trials = 1
    render_sim = True
    plot_dataset = True
    store_dataset = False

    # choose the type of model
    model = 'Heuristic'
    exp_name = 'Heuristic'

    # create the environment
    shepherd_env = gym.make('Shepherd-v0')

    # render the simulation
    if render_sim:
        shepherd_env.render()

    keyboard()

    # state space dimensionality
    n_state = shepherd_env.observation_space.shape[0]

    # initialize list to store dataset
    dataset = []
    n_samples = []

    for n in range(n_trials):

        # initialize trial data variable
        trial_data = np.zeros((0,2*n_state+2))

        # reset the environment
        (state, reward, finish, info) = shepherd_env.reset()
        
        # run the main simulation
        while not finish:
            # get the dog's action
            if model == 'Heuristic':
                action = dog_heuristic_model(state, info)
            else:
                action = np.random.randint(8)

            # execute the action and update the state
            (new_state, reward, finish, info) = shepherd_env.step(action)

            # append to variable
            sample = np.hstack((state,np.array([action,reward]),new_state))
            trial_data = np.vstack((trial_data,sample[None,:]))

            # update state
            state = new_state

        # append to the dataset
        dataset.append(trial_data)
        n_samples.append(trial_data.shape[0])

        # information
        print(f'Finish simulation: {n}')
    
    # stop the simulations
    shepherd_env.close()

    # store experience to files
    if store_dataset:
        pass

    # plot the generated rewards
    if plot_dataset:
        fig = plt.figure()

        for n in range(n_trials):
            x_data = np.arange(n_samples[n])
            plt.plot(x_data,dataset[n][:,n_state+1],'-',linewidth=2,label=f'Ep {n}')
        
        plt.xlabel('# Steps')
        plt.ylabel('Reward r(t)')
        plt.title(f'{model} Model')
        plt.legend(bbox_to_anchor=(1.2, 1.0))

        plt.tight_layout()
        plt.show()


if __name__=='__main__':
    main()