#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the shepherd gym env with dog heuristic model.
"""

# core modules
import os
import gym
import shutil
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
    parser = argparse.ArgumentParser(description='Test heuristic model with shepherd env')

    parser.add_argument('-e' ,'--experiment', default='heuristic', type=str, help='name of the experiment')
    parser.add_argument('-m', '--model', default='heuristic', type=str, help='name of model to use')

    parser.add_argument('-n', '--ntrials', default=5, type=int, help='number of episodes')
    parser.add_argument('-s', '--seed', default=41, type=int, help='seed value for reproducibility')

    parser.add_argument('--store', action='store_true', default=False, help='flag to store experience')
    parser.add_argument('--noplot', dest='plot', action='store_false', default=True, help='flag to plot rewards')
    parser.add_argument('--norender', dest='render', action='store_false', default=True, help='flag for rendering sim')
	
    # parse arguments and assign variables
    args = parser.parse_args()

    exp_name = args.experiment
    model = args.model

    seed = args.seed
    n_trials = args.ntrials

    render_sim = args.render
    plot_dataset = args.plot
    store_dataset = args.store

    # create folder to save generated dataset
    if plot_dataset or store_dataset:
        data_path = '../data'
        if not os.path.isdir(data_path):
            os.mkdir(data_path)
        
        exp_path = f'{data_path}/{exp_name}'
        if os.path.isdir(exp_path):
            shutil.rmtree(exp_path, ignore_errors=True)
        os.mkdir(exp_path)

    # create the environment
    shepherd_env = gym.make('Shepherd-v0')
    shepherd_env.seed(seed)
    shepherd_env.print_info = True

    # render the simulation
    if render_sim:
        shepherd_env.render()

    # state space dimensionality
    n_state = shepherd_env.observation_space.shape[0]

    # initialize list to store dataset
    dataset = []
    n_samples = []

    for n in range(n_trials):

        # initialize trial data variable
        trial_data = np.zeros((0,2*n_state+5))

        # reset the environment
        state = shepherd_env.reset()

        # run the main simulation
        (state,_,finish,info) = shepherd_env.step(0)
        while not finish:
            # get the dog's action
            if model == 'heuristic':
                action, int_goal, dog_mode = dog_heuristic_model(state, info)
            else:
                dog_mode = -1.0
                int_goal = np.array([0.0,0.0])
                action = np.random.randint(8)

            # execute the action and update the state
            (new_state, reward, finish, info) = shepherd_env.step(action)

            # append to variable
            sample = np.hstack((state,int_goal,np.array([dog_mode,action,reward]),new_state))
            trial_data = np.vstack((trial_data,sample[None,:]))

            # update state
            state = new_state

        # check for failure
        if trial_data[-1,n_state+4] == -10.0:
            print('Fail!')

        # append to the dataset
        dataset.append(trial_data)
        n_samples.append(trial_data.shape[0])

        # information
        print(f'Finish simulation: {n+1}')
    
    # stop the simulations
    shepherd_env.close()

    # store experience to files
    if store_dataset:
        for n in range(n_trials):
            np.savetxt(f'{exp_path}/trial{n+1}', dataset[n], 
                       fmt='%.3f', delimiter=',')

    # plot the generated rewards
    if plot_dataset:
        plt.figure()

        for n in range(n_trials):
            if n%4 == 0:
                x_data = np.arange(n_samples[n])
                plt.plot(x_data,dataset[n][:,n_state+4],'-',linewidth=2,label=f'Ep {n+1}')
        
        plt.xlabel('# steps')
        plt.ylabel('reward r(t)')
        plt.title(f'{model} model')
        plt.legend(bbox_to_anchor=(1.2, 1.0))

        plt.tight_layout()
        plt.savefig(f'{exp_path}/rewards.png')

        plt.show()


if __name__=='__main__':
    main()