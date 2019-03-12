#!/usr/bin/env python

# import libraries
import os
import pickle
import argparse
import numpy as np

def main():

    # setup argument parser
    parser = argparse.ArgumentParser(description='Script for preprocess')
    parser.add_argument('-d','--datapath',type=str,default='../data/heuristic',
                        help='path to dataset')
    parser.add_argument('-m','--mode',type=str,default='il',
                        help='mode to preprocess dataset: il, dump')
    parser.add_argument('-n','--nepisodes',type=int,default=1000,
                        help='number of episodes to parse')
    parser.add_argument('-p','--percenttest',type=float,default=0.05, 
                        help='percentage of dataset as test')
    parser.add_argument('--maxepisodes',type=int,default=1000,
                        help='maximum number of episodes in folder')

    # parser arguments
    args = parser.parse_args()

    data_mode = args.mode
    n_episodes = args.nepisodes
    n_test = int(args.percenttest*n_episodes)
    data_path = '../data/{}'.format(args.datapath)

    # variables for dataset
    n_goal = 2
    n_mode = 1
    n_state = 10
    n_action = 1

    if data_mode == 'il':
        # initialize dataset dictionary
        dataset = {'state':np.zeros((0,n_state)),
                'action':np.zeros((0,n_action)),
                'goal':np.zeros((0,n_goal)),
                'mode':np.zeros((0,n_mode)),
                'lengths':np.zeros((n_episodes))}
    else:
        # initialize dataset list
        dataset = []

    # loop over dataset files
    ep_inds = np.random.choice(args.maxepisodes, n_episodes)
    for n in range(n_episodes):
        tmp = np.loadtxt('{}/trial{}'.format(data_path,ep_inds[n]+1),delimiter=',')
        
        if data_mode == 'il':
            dataset['lengths'][n] = tmp.shape[0]
            dataset['state'] = np.vstack((dataset['state'], tmp[:,:n_state]))
            dataset['mode'] = np.vstack((dataset['mode'], tmp[:,n_state+2][:,None]))
            dataset['goal'] = np.vstack((dataset['goal'], tmp[:,n_state:n_state+2]))
            dataset['action'] = np.vstack((dataset['action'], 
                                        tmp[:,n_state+3][:,None]))
        else:
            ep = {'observations': tmp[:,:n_state], 'actions': tmp[:,n_state+3][:,None]}
            dataset.append(ep)

    if data_mode == 'il':
        # get stats
        print(dataset['state'].shape, dataset['action'].shape,
            dataset['goal'].shape, dataset['mode'].shape)

        # compute n_test and n_train
        dataset['n_train'] = int(np.sum(dataset['lengths'][:-n_test]))
        dataset['n_test'] = int(np.sum(dataset['lengths'][-n_test:]))

        print(dataset['n_train'],dataset['n_test'],dataset['state'].shape[0])

        # save dataset to file
        np.savez('{}/dataset.npz'.format(data_path), state=dataset['state'],
                action=dataset['action'], goal=dataset['goal'], 
                mode=dataset['mode'], n_test=dataset['n_test'],
                n_train=dataset['n_train'])
    else:
        with open('{}/dataset.pkl'.format(data_path), 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()