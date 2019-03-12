#!/usr/bin/env python

# import libraries
import os
import gym
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

# import utility classes
from shepherd_gym.models.imitation_utils import Policy
from shepherd_gym.models.dog_heuristic import dog_heuristic_model

# IPython debugging
from IPython.terminal.debugger import set_trace as keyboard

def main():

    # setup argument parsing
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('-e', '--experiment', type=str, required=True, 
                        help='name of experiment')

    # optional arguments
    parser.add_argument('-m', '--model', type=str, default='heuristic', 
                        help='model for evaluation: heuristic, model name')
    parser.add_argument('-n', '--episodes', type=int, default=10,
                        help='number of episodes for evaluation')
    parser.add_argument('--env', type=str, default='Shepherd-v0',
                        help='environment for testing model')
    parser.add_argument('--nocuda', action='store_true', default=False, 
                        help='flag to switch to cpu')
    parser.add_argument('--seed', type=int, default=42, 
                        help='seed value for reproducibility')
    parser.add_argument('--results', type=str, default='../results/imitation/',
                        help='path to results folder')
    parser.add_argument('--store', action='store_true', default=False, 
                        help='flag to store experience')
    parser.add_argument('--norender', dest='render', action='store_false', 
                        default=True, help='flag for rendering sim')

    # parse arguments
    args = parser.parse_args()

    # initialize torch environment
    seed = args.seed
    is_cuda = not args.nocuda

    torch.manual_seed(seed)
    np.random.seed(seed)

    if is_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # initialize variables
    model = args.model
    exp = args.experiment
    n_episodes = args.episodes
    store_dataset = args.store
    result_path = args.results
    experiment = args.experiment

    # create folder to save generated dataset
    if store_dataset:
        data_path = '../data'
        if not os.path.isdir(data_path):
            os.mkdir(data_path)

        exp_path = '{}/{}'.format(data_path, exp)
        if not os.path.isdir(exp_path):
            os.mkdir(exp_path)

    # create new environment
    env = gym.make(args.env)
    if args.render:
        env.render()

    # state space dimensionality
    n_state = env.observation_space.shape[0]

    print('Experiment: ',experiment)
    print('### Initialization Done ###')

    # load the il model
    if model != 'heuristic':
        # define network and optimizer
        policy = Policy(drop_rate=0.0)
        if is_cuda:
            policy.cuda()

        policy.load_state_dict(torch.load(result_path+model+'/model.pt'))
        policy.eval()

        print('### Network Created ###')

    # loop over episodes
    dataset = []
    success_trials = 0

    for n in range(n_episodes):
        done = False
        state = env.reset()
        trial_data = np.zeros((0, 2*n_state+5))
        
        # run until episode terminates
        (state,_,done,info) = env.step(0)
        while not done:
            if model != 'heuristic':
                state_var = torch.from_numpy(state[None,:].astype(np.float32))

                if is_cuda:
                    state_var = state_var.cuda()
                state_var = Variable(state_var)

                # get output action
                output, int_goal, mode = policy(state_var)
                pred = F.softmax(output, dim=1).data.max(1, keepdim=True)[1]
                action = pred.cpu().numpy()[0,0]

                # get intermediate goal and predicted mode
                int_goal = int_goal.detach().cpu().numpy()[0]
                int_goal = int_goal.astype(np.float64)

                mode = F.softmax(mode, dim=1).data.max(1, keepdim=True)[1]
                dog_mode = mode.cpu().numpy()[0,0]
                
            else:
                action, int_goal, dog_mode = dog_heuristic_model(state, info, False)

            # perform action and get state
            new_state, reward, done, info = env.step(action)

            # check for success
            if done and info['s']:
                print('Success!')
                success_trials += 1.0

            # save the sample dataset
            sample = np.hstack((state,int_goal,
                            np.array([dog_mode,action,reward]),new_state))
            trial_data = np.vstack((trial_data,sample[None,:]))

            # update state variable
            state = new_state

            # render simulation 
            if args.render:
                env.render(mode='detailed',subgoal=int_goal)

        dataset.append(trial_data)
    # shutdown env
    env.close()

    # save the results
    if store_dataset:
        for n in range(n_episodes):
            np.savetxt('{}/{}_{}'.format(exp_path,model,n+1), dataset[n], 
                        fmt='%.3f', delimiter=',')

    print('### Testing Completed ###')
    print(f'Sucess Rate: {success_trials/n_episodes}')

if __name__=='__main__':
    main()    