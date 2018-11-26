#!/usr/bin/env python

# test.py: Implementation of Unity ML-agents interface
# Author: Nishanth Koganti
# Date: 2018/03/05

# import libraries
import gym
import argparse
import numpy as np
import shepherd_gym

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils import Policy

def main():
    
    # setup argument parsing
    parser = argparse.ArgumentParser(description='BC with VR dataset')
    
    # required arguments
    parser.add_argument('-e', '--experiment', type=str, help='name of experiment', default='baseline')
    parser.add_argument('-n', '--episodes', type=int, help='number of episodes for evaluation', default=10)

    # optional arguments
    parser.add_argument('--env', type=str, help='path to unity env', default='Shepherd-v0')
    parser.add_argument('--display', action='store_true', help='display mode for game', default=False)
    parser.add_argument('--results', type=str, help='results folder path', default='results/imitation/')
    parser.add_argument('--seed', type=int, help='seed value for reproducibility', default=42)
    parser.add_argument('--nocuda', action='store_true', help='flag to switch to cpu', default=False)

    # parse arguments
    args = parser.parse_args()
    
    # initialize game parameters
    seed = args.seed
    np.random.seed(seed)
    isCuda = not args.nocuda
    experiment = args.experiment

    if isCuda:
        torch.backends.cudnn.deterministic = True    

    resultPath = args.results
    nEpisodes = args.episodes

    print('Experiment: ',experiment)
    print('### Initialization Done ###')

    # create new unity environment
    env = gym.make(args.env)
    if args.display:
        env.render()

    # load the model from directory
    policy = Policy()
    if isCuda:
        policy.cuda()

    policy.load_state_dict(torch.load(resultPath+experiment+'/model.pt'))
    policy.eval()

    # loop over episodes
    for _ in range(nEpisodes):
        done = False
        state = env.reset()
        
        # run until episode terminates
        while not done:
            stateVar = torch.from_numpy(state[None,:].astype(np.float32))

            if isCuda:
                stateVar = stateVar.cuda()    
            stateVar = Variable(stateVar)
            
            # get output action
            output,_,_ = policy(stateVar)
            pred = F.softmax(output, dim=1).data.max(1, keepdim=True)[1]
            action = pred.cpu().numpy()

            # perform action and get state
            state,_,done,_ = env.step(action)

    # shutdown env
    env.close()
    print('### Testing Completed ###')
    
if __name__=='__main__':
    main()