#!/usr/bin/env python

# evaluate.py: Evaluate of trained behavioral cloning models
# Author: Nishanth Koganti
# Date: 2018/03/05

# import libraries
import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from utils import DemoDataset, Policy, Tester

def main():
    
    # setup argument parsing
    parser = argparse.ArgumentParser(description='Behavioral Cloning Evaluation')
    parser.add_argument('-e', '--experiment', type=str, required=True, 
                        help='name of experiment')
        
    # optional arguments
    parser.add_argument('--cuda', action='store_false', default=True, 
                        help='flag to switch to cpu')
    parser.add_argument('--data', type=str, default='../data/heuristic',
                        help='path to dataset folder')
    parser.add_argument('--results', type=str, default='results/imitation/', 
                        help='path to results folder')

    parser.add_argument('--lossWeights', nargs='+', type=float, 
                        help='weights for loss function', default=[1.0,0.5,0.5])

    # parse arguments
    args = parser.parse_args()

    isCuda = args.cuda
    experiment = args.experiment

    if isCuda:
        torch.backends.cudnn.deterministic = True

    dataPath = args.data
    resultPath = args.results
    lossWeights = args.lossWeights

    print('Experiment: ',experiment)
    print('### Initialization Done ###')

    # load the dataset
    datasetFile = f'{dataPath}/dataset.npz'

    with np.load(datasetFile) as dataset:
        nTest = dataset['nTest']
        nTrain = dataset['nTrain']

        # obtain the state, action, goal and mode matrices
        testStates = dataset['state'][-nTest:,:].astype(np.float32)
        trainStates = dataset['state'][:nTrain,:].astype(np.float32)

        testAction = (dataset['action'][-nTest:,:]).astype(np.int64)
        trainAction = (dataset['action'][:nTrain,:]).astype(np.int64)

        testGoal = dataset['goal'][-nTest:,:].astype(np.float32)
        trainGoal = dataset['goal'][:nTrain,:].astype(np.float32)

        testMode = (dataset['mode'][-nTest:,:]).astype(np.int64)
        trainMode = (dataset['mode'][:nTrain,:]).astype(np.int64)

    # prepare dataset and dataloaders
    trainData = {'state': trainStates, 'action': trainAction, 
                'goal': trainGoal, 'mode': trainMode}
    trainDataset = DemoDataset(trainData)
    testData = {'state': testStates, 'action': testAction, 
                'goal': testGoal, 'mode': testMode}
    testDataset = DemoDataset(testData)

    if isCuda:
        trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=False)
        testLoader = DataLoader(testDataset, batch_size=64, shuffle=False)
    else:
        trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=False)
        testLoader = DataLoader(testDataset, batch_size=64, shuffle=False)

    print('### Data Loaded ###')

    # define network and optimizer
    policy = Policy()
    if isCuda:
        policy.cuda()

    policy.load_state_dict(torch.load(resultPath+experiment+'/model.pt'))

    print('### Network Created ###')

    # define tester class and run
    tester = Tester(experiment, policy, trainLoader, testLoader, batchSize=64,
                    isCuda=isCuda, lossWeights=lossWeights, resultPath=resultPath)
    results = tester.evaluate()

    # save results to experiment file
    np.save(resultPath+experiment+'/results.npy', results)

    print('### Evaluation Completed ###')

if __name__=='__main__':
    main()