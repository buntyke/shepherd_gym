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
    c_id = 0
    min_seq_length = 1000
    store_seq_length = 100
    for n in range(n_episodes):
        tmp = np.loadtxt(f'{data_path}/trial{n+1}',delimiter=',')
        seq_length = tmp.shape[0]

        if seq_length >= store_seq_length:
            dataset[c_id] = tmp[:,:n_state]
            c_id += 1

            if seq_length < min_seq_length:
                min_seq_length = seq_length
                print(f'Min: {min_seq_length}')
        else:
            print(f'Skip: {seq_length}')

    # trim all sequences
    for n in range(c_id):
        seq_length = dataset[n].shape[0]
        idx = np.linspace(0, seq_length, min_seq_length, 
                          endpoint=False, dtype=np.int32)
        dataset[n] = dataset[n][idx,:]

    # save dataset to file
    with open(f'{data_path}/curriculum.npz', 'wb') as f:
        pickle.dump(dataset,f)

if __name__ == '__main__':
    main()