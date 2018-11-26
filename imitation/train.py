#!/usr/bin/env python

# train.py: Implementation of behavioral cloning
# Author: Nishanth Koganti
# Date: 2018/03/05

# import libraries
import os
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# import utility classes
from utils import DemoDataset, RandomSampler, Policy, Trainer

def main():

    # setup argument parsing
    parser = argparse.ArgumentParser(description='Behavioral Cloning')
    parser.add_argument('-e', '--experiment', type=str, required=True, 
                        help='name of experiment')
        
    # optional arguments
    parser.add_argument('--nocuda', action='store_true', default=False, 
                        help='flag to switch to cpu')
    parser.add_argument('--seed', type=int, default=42, 
                        help='seed value for reproducibility')
    parser.add_argument('--dataSeed', type=int, default=42, 
                        help='data loader seed value')
    parser.add_argument('--data', type=str, default='../data/heuristic', 
                        help='path to dataset folder')
    parser.add_argument('--results', type=str, 
                        default='results/imitation/',
                        help='path to results folder')

    parser.add_argument('--epochs', type=int, default=30, 
                        help='number of epochs')
    parser.add_argument('--dropRate', type=float, default=0.0, 
                        help='dropout rate for network')
    parser.add_argument('--lossWeights', nargs='+', type=float, 
                        default=[1.0,0.1,0.1], 
                        help='weights for loss function')
    parser.add_argument('--learnRate', type=float, default=1e-3, 
                        help='learning rate for optimizer')

    # parse arguments
    args = parser.parse_args()

    # initialize torch environment
    seed = args.seed
    dataSeed = args.dataSeed
    isCuda = not args.nocuda
    
    torch.manual_seed(seed)
    np.random.seed(dataSeed)

    if isCuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # initialize variables
    dataPath = args.data
    resultPath = args.results
    experiment = args.experiment

    nEpochs = args.epochs
    dropRate = args.dropRate
    learnRate = args.learnRate
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

    sampler = RandomSampler(trainDataset)
    if isCuda:
        trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=False, sampler=sampler)
        testLoader = DataLoader(testDataset, batch_size=64, shuffle=False)
    else:
        trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=False, sampler=sampler)
        testLoader = DataLoader(testDataset, batch_size=64, shuffle=False)
    
    print('### Data Loaded ###')

    # define network and optimizer
    policy = Policy(dropRate=dropRate)
    if isCuda:
        policy.cuda()

    optimizer = optim.Adam(policy.parameters(), lr=learnRate)

    print('### Network Created ###')

    # define trainer class and run
    trainer = Trainer(experiment, policy, trainLoader, testLoader, optimizer, 
                    nEpochs=nEpochs, isCuda=isCuda, lossWeights=lossWeights,
                    resultPath=resultPath)
    trainer.run()

    print('### Training Completed ###')

if __name__=='__main__':
    main()