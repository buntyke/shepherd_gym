#!/usr/bin/env python

# import libraries
import os
import argparse
import numpy as np

def main():

    # setup argument parser
    parser = argparse.ArgumentParser(description='Script for preprocess')
    parser.add_argument('-d','--datapath',type=str,default='../data/heuristic',
                        help='path to dataset')
    parser.add_argument('-n','--nepisodes',type=int,default=1000,
                        help='number of episodes to parse')
    parser.add_argument('-p','--percenttest',type=float,default=0.05, 
                        help='percentage of dataset as test')

    # parser arguments
    args = parser.parse_args()
    data_path = args.datapath
    n_episodes = args.nepisodes
    n_test = int(args.percenttest*n_episodes)

    # variables for dataset
    n_goal = 2
    n_mode = 1
    n_state = 10
    n_action = 1

    # initialize dataset dictionary
    dataset = {'state':np.zeros((0,n_state)),
               'action':np.zeros((0,n_action)),
               'goal':np.zeros((0,n_goal)),
               'mode':np.zeros((0,n_mode)),
               'lengths':np.zeros((n_episodes))}

    # loop over dataset files
    for n in range(n_episodes):
        tmp = np.loadtxt('{}/trial{}'.format(data_path,n+1),delimiter=',')
        dataset['lengths'][n] = tmp.shape[0]
        dataset['state'] = np.vstack((dataset['state'], tmp[:,:n_state]))
        dataset['mode'] = np.vstack((dataset['mode'], tmp[:,n_state+2][:,None]))
        dataset['goal'] = np.vstack((dataset['goal'], tmp[:,n_state:n_state+2]))
        dataset['action'] = np.vstack((dataset['action'], 
                                       tmp[:,n_state+3][:,None]))

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

if __name__ == '__main__':
    main()