#!/usr/bin/env python

# import libraries
import os
import pickle
import argparse
import numpy as np

def main():

    # setup argument parser
    parser = argparse.ArgumentParser(description='Preprocessing for curriculum learning')
    parser.add_argument('-d','--datapath',type=str,
                        default='../data/heuristic',
                        help='path to dataset')
    parser.add_argument('-n','--nepisodes',type=int,
                        default=1000,
                        help='number of episodes to parse')

    # parser arguments
    args = parser.parse_args()
    data_path = args.datapath
    n_episodes = args.nepisodes

    # variables for dataset
    n_state = 10

    # initialize dataset dictionary
    dataset = {}

    # loop over dataset files
    for n in range(n_episodes):
        dataset[n] = {}
        tmp = np.loadtxt(f'{data_path}/trial{n+1}',delimiter=',')
        dataset[n] = tmp[:,:n_state]

    # save dataset to file
    with open(f'{data_path}/curriculum.npz', 'wb') as f:
        pickle.dump(dataset,f)

if __name__ == '__main__':
    main()