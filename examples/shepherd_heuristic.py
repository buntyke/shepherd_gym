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
    parser = argparse.ArgumentParser(description='Heuristic model with shepherd')

    parser.add_argument('-e' , '--experiment', default='heuristic', type=str, 
                        help='name of the experiment')
    parser.add_argument('-m',  '--model', default='heuristic', type=str, 
                        help='model to use: heuristic, random')
    parser.add_argument('-c', '--continuous', default=False, action='store_true',
                        help='flag to create continuous action env')

    parser.add_argument('-n', '--ntrials', default=5, type=int, 
                        help='number of episodes')
    parser.add_argument('-s', '--seed', default=41, type=int, 
                        help='seed value for reproducibility')
    parser.add_argument('--nsheep', default=100, type=int, 
                        help='number of sheep')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='flag to evaluate heuristic model')
    parser.add_argument('--store', action='store_true', default=False, 
                        help='flag to store experience')
    parser.add_argument('--noplot', dest='plot', action='store_false', 
                        default=True, help='flag to plot rewards')
    parser.add_argument('--norender', dest='render', action='store_false', 
                        default=True, help='flag for rendering sim')
    parser.add_argument('--wrap', dest='wrap', action='store_true', 
                        default=False, help='flag to wrap sampler')

    # parse arguments and assign variables
    args = parser.parse_args()

    exp_name = args.experiment
    model = args.model

    seed = args.seed
    n_sheep = args.nsheep
    n_trials = args.ntrials

    render_sim = args.render
    plot_dataset = args.plot
    store_dataset = args.store

    # create folder to save generated dataset
    if plot_dataset or store_dataset:
        data_path = '../data'
        if not os.path.isdir(data_path):
            os.mkdir(data_path)

        exp_path = '{}/{}'.format(data_path, exp_name)
        if os.path.isdir(exp_path):
            shutil.rmtree(exp_path, ignore_errors=True)
        os.mkdir(exp_path)

    # create the environment
    if args.continuous:
        env_name = 'ShepherdCont-v0'
    else:
        env_name = 'Shepherd-v0'

    shepherd_env = gym.make(env_name)
    if args.wrap:
        shepherd_env = shepherd_gym.wrappers.SamplerWrapper(shepherd_env,
                        demo_path='../data/curriculum',
                        increment_freq=2, initial_window_width=5,
                        window_increment=5)
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
    n_attempts = 0

    n = 0
    while n < n_trials:

        # initialize trial data variable
        trial_data = np.zeros((0,2*n_state+5))

        # reset the environment
        state = shepherd_env.reset()

        # run the main simulation
        (state,_,finish,info) = shepherd_env.step(0)
        while not finish:
            # get the dog's action
            if model == 'heuristic':
                action, int_goal, dog_mode = dog_heuristic_model(state, info, 
                                                                 args.continuous)
            else:
                dog_mode = -1.0
                int_goal = np.array([0.0,0.0])
                action = np.random.randint(8)

            # execute the action and update the state
            (new_state, reward, finish, info) = shepherd_env.step(action)

            # render the simulation
            if render_sim:
                shepherd_env.render(mode='detailed',subgoal=int_goal)

            # append to variable
            sample = np.hstack((state,int_goal,
                            np.array([dog_mode,action,reward]),new_state))
            trial_data = np.vstack((trial_data,sample[None,:]))

            # update state
            state = new_state

        # update attempts
        n_attempts += 1

        # check for failure
        if trial_data.shape[0] == 0 or shepherd_env.target_distance>1.0:
            print('Fail!')
        else:
            # else append to the dataset
            n += 1
            dataset.append(trial_data)
            n_samples.append(trial_data.shape[0])

        # information
        print('Finish simulation: {}'.format(n))

        # break for valid attempts
        if n_attempts==n_trials and args.evaluate:
            break

    # stop the simulations
    shepherd_env.close()

    # evaluate the success rate
    success_rate = n/n_trials   
    print('success_rate:', success_rate)
    
    with open('success_rates.csv', 'a') as success_rates_file:
        data = '{}, {}, {}, {}\n'.format(seed, n_trials, n_sheep, success_rate)
        success_rates_file.write(data)

    # store experience to files
    if store_dataset:
        for n in range(n_trials):
            np.savetxt('{}/trial{}'.format(exp_path,n+1), dataset[n], 
                       fmt='%.3f', delimiter=',')

    # plot the generated rewards
    if plot_dataset:
        plt.figure()

        for n in range(n_trials):
            if n%4 == 0:
                x_data = np.arange(n_samples[n])
                plt.plot(x_data, dataset[n][:,n_state+4],'-', 
                         linewidth=2, label='Ep {}'.format(n+1))
        
        plt.xlabel('# steps')
        plt.ylabel('reward r(t)')
        plt.title('{} model'.format(model))
        plt.legend(bbox_to_anchor=(1.2, 1.0))

        plt.tight_layout()
        plt.savefig('{}/rewards.png'.format(exp_path))

        plt.show()

if __name__=='__main__':
    main()