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
from shepherd_gym.models.imitation_utils import DemoDataset, RandomSampler, \
        Policy, Trainer

def main():

    # setup argument parsing
    parser = argparse.ArgumentParser(description='Behavioral Cloning')
    parser.add_argument('-e', '--experiment', type=str, required=True, 
                        help='name of experiment')

    # optional arguments
    parser.add_argument('-m', '--mode', type=str, default='train', 
                        help='mode for algorithm')
    parser.add_argument('-n', '--episodes', type=int, default=10,
                        help='number of episodes for evaluation')
    parser.add_argument('--display', action='store_true', default=False,
                        help='flag to enable rendering')
    parser.add_argument('--env', type=str, default='Shepherd-v0',
                        help='environment for testing model')
    parser.add_argument('--nocuda', action='store_true', default=False, 
                        help='flag to switch to cpu')
    parser.add_argument('--seed', type=int, default=42, 
                        help='seed value for reproducibility')
    parser.add_argument('--dataseed', type=int, default=42, 
                        help='data loader seed value')
    parser.add_argument('--data', type=str, default='../data/heuristic', 
                        help='path to dataset folder')
    parser.add_argument('--results', type=str, default='../results/imitation/',
                        help='path to results folder')

    parser.add_argument('--epochs', type=int, default=30, 
                        help='number of epochs')
    parser.add_argument('--droprate', type=float, default=0.0, 
                        help='dropout rate for network')
    parser.add_argument('--lossweights', nargs='+', type=float, 
                        default=[1.0,0.1,0.1], 
                        help='weights for loss function')
    parser.add_argument('--learnrate', type=float, default=1e-3, 
                        help='learning rate for optimizer')

    # parse arguments
    args = parser.parse_args()

    # initialize torch environment
    seed = args.seed
    data_seed = args.dataseed
    is_cuda = not args.nocuda

    torch.manual_seed(seed)
    np.random.seed(data_seed)

    if is_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # initialize variables
    mode = args.mode
    data_path = args.data
    result_path = args.results
    experiment = args.experiment

    n_epochs = args.epochs
    drop_rate = args.droprate
    n_episodes = args.episodes
    learn_rate = args.learnrate
    loss_weights = args.lossweights

    print('Experiment: ',experiment)
    print('### Initialization Done ###')

    if mode == 'train':
        # load the dataset
        dataset_file = '{}/dataset.npz'.format(data_path)

        with np.load(dataset_file) as dataset:
            n_test = dataset['n_test']
            n_train = dataset['n_train']

            # obtain the state, action, goal and mode matrices
            test_states = dataset['state'][-n_test:,:].astype(np.float32)
            train_states = dataset['state'][:n_train,:].astype(np.float32)

            test_action = (dataset['action'][-n_test:,:]).astype(np.int64)
            train_action = (dataset['action'][:n_train,:]).astype(np.int64)

            test_goal = dataset['goal'][-n_test:,:].astype(np.float32)
            train_goal = dataset['goal'][:n_train,:].astype(np.float32)

            test_mode = (dataset['mode'][-n_test:,:]).astype(np.int64)
            train_mode = (dataset['mode'][:n_train,:]).astype(np.int64)

        # prepare dataset and dataloaders
        train_data = {'state': train_states, 'action': train_action, 
                    'goal': train_goal, 'mode': train_mode}
        train_dataset = DemoDataset(train_data)
        test_data = {'state': test_states, 'action': test_action, 
                    'goal': test_goal, 'mode': test_mode}
        test_dataset = DemoDataset(test_data)

        sampler = RandomSampler(train_dataset)
        if is_cuda:
            train_loader = DataLoader(train_dataset, batch_size=64, 
                                      shuffle=False, sampler=sampler)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size=64, 
                                      shuffle=False, sampler=sampler)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        print('### Data Loaded ###')

    # define network and optimizer
    policy = Policy(drop_rate=drop_rate)
    if is_cuda:
        policy.cuda()

    if mode == 'train':
        optimizer = optim.Adam(policy.parameters(), lr=learn_rate)

    print('### Network Created ###')

    if mode == 'train':
        # define trainer class and run
        trainer = Trainer(experiment, policy, train_loader, test_loader, 
                          optimizer, n_epochs=n_epochs, is_cuda=is_cuda, 
                          loss_weights=loss_weights, result_path=result_path)
        trainer.run()
        print('### Training Completed ###')

    # create new environment
    env = gym.make(args.env)
    if args.display:
        env.render()

    # load trained model
    policy.load_state_dict(torch.load(result_path+experiment+'/model.pt'))
    policy.eval()

    # loop over episodes
    for _ in range(n_episodes):
        done = False
        state = env.reset()
        
        # run until episode terminates
        while not done:
            state_var = torch.from_numpy(state[None,:].astype(np.float32))

            if is_cuda:
                state_var = state_var.cuda()    
            state_var = Variable(state_var)
            
            # get output action
            output,_,_ = policy(state_var)
            pred = F.softmax(output, dim=1).data.max(1, keepdim=True)[1]
            action = pred.cpu().numpy()

            # perform action and get state
            state,_,done,_ = env.step(action)

    # shutdown env
    env.close()
    print('### Testing Completed ###')

if __name__=='__main__':
    main()