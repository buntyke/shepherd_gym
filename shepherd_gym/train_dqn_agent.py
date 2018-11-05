#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import gym
import gym

# import torch utilities
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# ipython debugging
from IPython.terminal.debugger import set_trace as keyboard

# import other dependencies
import math
import random
import argparse
import numpy as np
from itertools import count
from collections import namedtuple

# import the agent
from shepherd_gym.models.dqn_agent import DQN

# setup GPU usage
use_cuda = torch.cuda.is_available()
ByteTensor = torch.cuda.ByteTensor if use_cuda \
            else torch.ByteTensor
LongTensor = torch.cuda.LongTensor if use_cuda \
            else torch.LongTensor
FloatTensor = torch.cuda.FloatTensor if use_cuda \
            else torch.FloatTensor

# transitions tuple used for experience replay
Transition = namedtuple('Transition', ('state', 
                'action', 'next_state', 'reward'))

# replay memory class to store memory and sample from it
class ReplayMemory(object):

    def __init__(self, capacity):
        # list with transitions
        self.memory = []

        # capacity of the buffer
        self.capacity = capacity

        # position in the buffer
        self.position = 0

    def push(self, *args):
        # function to save a transition

        # append to list if not full
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        # add transition at memory
        self.memory[self.position] = Transition(*args)

        # update position in a cyclic manner
        self.position = (self.position+1) % self.capacity

    def sample(self, batch_size):
        # sample from the memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        # utility function to get current length
        return len(self.memory)

# define class to train network
class DQNAgent():

    def __init__(self, use_cuda = True):

        # initialize gym environment
        self.env = gym.make('Shepherd-v0')

        self.env.render()
        self.n_actions = 8

        # hyper parameters for training
        self.gamma = 0.9
        self.eps_end = 0.05
        self.eps_start = 0.9
        self.eps_decay = 1000
        self.batch_size = 128

        # use gpu
        self.use_cuda = use_cuda

        # define dqn network
        self.model = DQN()
        if self.use_cuda:
            self.model.cuda()

        # define the memory buffer
        self.memory = ReplayMemory(10000)

        # define optimizer for model training
        self.optimizer = optim.RMSprop(self.model.parameters())

        # variables for training network
        self.last_sync = 0
        self.steps_done = 0
        self.eps_durations = []

    # function for clean finish
    def end_train(self):
        print('Complete')
        self.env.close()

    def select_action(self, state, greedy=False):
        # obtain random sample
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                       math.exp(-1. * self.steps_done/self.eps_decay)

        # increament counter
        self.steps_done += 1

        # greedy action
        if sample > eps_threshold or greedy == True:
            return self.model(Variable(state).type(FloatTensor)).data.max(1)[1].view(1,1)
        else:
            return LongTensor([[random.randrange(self.n_actions)]])

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)

        # transpose the batch 
        batch = Transition(*zip(*transitions))

        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        # compute Q(s_t, a) - the model computes Q(s_t)
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                            batch.next_state)))

        # get non_final_next states
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)

        # compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(self.batch_size).type(FloatTensor))
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0]

        # now, we don't want to mess up the loss with a volatile flag
        next_state_values_no_grad = next_state_values.detach()

        # compute the expected Q values
        expected_state_action_values = (next_state_values_no_grad * self.gamma) + reward_batch
        expected_state_action_values = FloatTensor(expected_state_action_values).unsqueeze(1)
        
        # compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # optimize the model with gradient clamping
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

def main():

    # define command line arguments
    parser = argparse.ArgumentParser(description='Script to train dqn agent for shepherd env')

    # add arguments
    parser.add_argument('--episodes', type=int, default=100)
    args = parser.parse_args()

    greedy = False
    n_episodes = args.episodes

    # initialize trainer
    agent = DQNAgent()

    # perform training
    for ep in range(n_episodes):
        # initialize the environment and state
        (state,reward,done,info) = agent.env.reset()
        state = FloatTensor([state])

        # run final episode with greedy policy without exploration
        if ep == n_episodes-1:
            greedy = True

        for t in count():
            # select and perform an action
            action = agent.select_action(state, greedy=greedy)
            tmp_action = action.clone()
            next_state, reward, done, info = agent.env.step(tmp_action[0, 0].cpu().numpy())

            reward = FloatTensor(reward)
            next_state = FloatTensor([next_state])
            
            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            agent.optimize_model()

            # append duration on finish and break
            if done:
                agent.eps_durations.append(t + 1)
                break

    # make sure to have smooth finish
    agent.end_train()

if __name__ =="__main__":
    main()
