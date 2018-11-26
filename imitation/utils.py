#!/usr/bin/env python

# utils.py: Utility classes in pytorch for behavioral cloning
# Author: Nishanth Koganti
# Date: 2018/03/05

# import libraries
import numpy as np
from tensorboardX import SummaryWriter

# import torch dependencies
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset

# ipython debugging
from IPython.terminal.debugger import set_trace as keyboard

# class implementation to load dataset
class DemoDataset(Dataset):
    """Dataset class to handle demonstration data"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data['state'].shape[0]

    def __getitem__(self, idx):
        sample = {}
        for key in self.data.keys():
            sample[key] = self.data[key][idx]
        return sample

# class implementation to shuffle dataset    
class RandomSampler(Sampler):
    """Sampler class to shuffle dataset"""

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(np.random.permutation(len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)

# class implementation of policy network
class Policy(nn.Module):
    """Policy class for behavioral cloning"""

    def __init__(self, nActions=8, nGoal=2, nModes=2, nState=10, dropRate=0.0):
        super(Policy, self).__init__()

        self.dense1 = nn.Linear(nState,256)
        self.drop1 = nn.Dropout(dropRate)
        self.dense2 = nn.Linear(256,256)
        self.drop2 = nn.Dropout(dropRate)
        self.dense3 = nn.Linear(256,64)
        self.drop3 = nn.Dropout(dropRate)

        self.output1 = nn.Linear(64,nActions)
        self.output2 = nn.Linear(64,nGoal)
        self.output3 = nn.Linear(64,nModes)

    def forward(self, state):

        output = F.relu(self.drop1(self.dense1(state)))
        output = F.relu(self.drop2(self.dense2(output)))
        output = F.relu(self.drop3(self.dense3(output)))

        action = self.output1(output)
        goal = self.output2(output)
        mode = self.output3(output)
        return action, goal, mode
    
# class implementation for training and test evaluation
class Trainer():
    """Trainer utility class for performing behavioral cloning"""
    
    def __init__(self, experiment, policy, trainLoader, testLoader, optimizer,  
                 resultPath='results/imitation/', nEpochs=20, isCuda=True, logInterval=50, 
                 lossWeights=[1.0,0.1,0.1], batchSize=64):
        
        # initialize variables
        self.isCuda = isCuda
        self.nEpochs = nEpochs
        self.batchSize = batchSize
        self.logInterval = logInterval
        self.lossWeights = lossWeights
        
        # initialize policy network and data loaders
        self.policy = policy
        self.optimizer = optimizer
        self.testLoader = testLoader
        self.trainLoader = trainLoader

        # experiment details
        self.experiment = experiment
        self.resultPath = resultPath

        # initialize run variables
        self.epoch = 0
        self.globalStep = 0
        self.writer = SummaryWriter(log_dir=self.resultPath+self.experiment)

    def run(self):

        self.test()
        for self.epoch in range(1, self.nEpochs+1):
            self.train()
            self.test()
            print(f'#### Epoch {self.epoch} ####')

        self.writer.close()
        torch.save(self.policy.state_dict(), self.resultPath+self.experiment+'/model.pt')

    def train(self):
        correct = 0
        self.policy.train()

        for batch_idx, sample in enumerate(self.trainLoader):
            if self.isCuda:
                for key in sample.keys():
                    sample[key] = sample[key].cuda()
            for key in sample.keys():
                sample[key] = Variable(sample[key])

            # perform forward pass
            self.optimizer.zero_grad()
            action, goal, mode = self.policy(sample['state'])

            # setup loss function
            goalLoss = F.mse_loss(goal, sample['goal'])
            modeLoss = F.cross_entropy(mode, sample['mode'][:,0])
            actLoss = F.cross_entropy(action, sample['action'][:,0])
            loss = self.lossWeights[0]*actLoss + self.lossWeights[1]*goalLoss + self.lossWeights[2]*modeLoss

            # get index of max log-probability
            pred = F.softmax(action, dim=1).data.max(1, keepdim=True)[1] 
            correct += pred.eq(sample['action'].data.view_as(pred)).cpu().sum()

            # perform backpropogation
            loss.backward()
            self.optimizer.step()            
            self.globalStep += 1

            if batch_idx % self.logInterval == 0:
                self.writer.add_scalars('loss', {'train':loss}, self.globalStep)
                self.writer.add_scalars('actLoss', {'train':actLoss}, self.globalStep)
                self.writer.add_scalars('goalLoss', {'train':goalLoss}, self.globalStep)
                self.writer.add_scalars('modeLoss', {'train':modeLoss}, self.globalStep)
        self.writer.add_scalars('accuracy', {'train':100.*correct/len(self.trainLoader.dataset)}, self.globalStep)

    def test(self):
        correct = 0
        testActLoss = 0
        testGoalLoss = 0
        testModeLoss = 0
        self.policy.eval()

        for sample in self.testLoader:
            if self.isCuda:
                for key in sample.keys():
                    sample[key] = sample[key].cuda()
                for key in sample.keys():
                    sample[key] = Variable(sample[key])
        
            action, goal, mode = self.policy(sample['state'])

            # setup loss function
            testGoalLoss += F.mse_loss(goal, sample['goal']).data
            testModeLoss += F.cross_entropy(mode, sample['mode'][:,0]).data
            testActLoss += F.cross_entropy(action, sample['action'][:,0]).data

            # get index of max log-probability
            pred = F.softmax(action, dim=1).data.max(1, keepdim=True)[1] 
            correct += pred.eq(sample['action'].data.view_as(pred)).cpu().sum()

        testLoss = self.lossWeights[0]*testActLoss + self.lossWeights[1]*testGoalLoss + self.lossWeights[2]*testModeLoss        
        
        testLoss /= len(self.testLoader.dataset)/self.batchSize
        testActLoss /= len(self.testLoader.dataset)/self.batchSize
        testGoalLoss /= len(self.testLoader.dataset)/self.batchSize
        testModeLoss /= len(self.testLoader.dataset)/self.batchSize
        
        self.writer.add_scalars('loss', {'test':testLoss}, self.globalStep)
        self.writer.add_scalars('actLoss', {'test':testActLoss}, self.globalStep)
        self.writer.add_scalars('goalLoss', {'test':testGoalLoss}, self.globalStep)
        self.writer.add_scalars('modeLoss', {'test':testModeLoss}, self.globalStep)
        self.writer.add_scalars('accuracy', {'test':100.*correct/len(self.testLoader.dataset)}, self.globalStep)
        
# class implementation for training and test evaluation
class Tester():
    """Tester utility class for performing behavioral cloning"""
    
    def __init__(self, experiment, policy, trainLoader, testLoader, 
                 resultPath='results/imitation/', isCuda=True, 
                 lossWeights=[1.0,0.1,0.1], batchSize=64):
        
        # initialize variables
        self.isCuda = isCuda
        self.batchSize = batchSize
        self.lossWeights = lossWeights

        # initialize policy network and data loaders
        self.policy = policy
        self.testLoader = testLoader
        self.trainLoader = trainLoader

        # experiment details
        self.experiment = experiment
        self.resultPath = resultPath

    def evaluate(self):
        self.policy.eval()
        loaders = {'train': self.trainLoader, 'test': self.testLoader}

        # training dataset evaluation
        results = {}
        for dataType, loader in loaders.items():
            # initialize variables
            loss = 0.0
            correct = 0
            actLoss = 0.0
            goalLoss = 0.0
            modeLoss = 0.0
            results[dataType] = {}

            nSamples = len(loader.dataset)
            actionData = np.zeros((0, 1))
            goalData = np.zeros((0, 2))
            modeData = np.zeros((0, 1))

            # loop over batches
            for sample in loader:
                if self.isCuda:
                    for key in sample.keys():
                        sample[key] = sample[key].cuda()
                for key in sample.keys():
                    sample[key] = Variable(sample[key])

                # perform forward pass
                action, goal, mode = self.policy(sample['state'])

                # setup loss function
                goalLoss += F.mse_loss(goal, sample['goal']).data.cpu().numpy()
                actLoss += F.cross_entropy(action, sample['action'][:,0]).data.cpu().numpy()
                modeLoss += F.cross_entropy(mode, sample['mode'][:,0]).data.cpu().numpy()

                # get index of max log-probability
                modePred = F.softmax(mode, dim=1).data.max(1, keepdim=True)[1]
                actionPred = F.softmax(action, dim=1).data.max(1, keepdim=True)[1] 
                correct += actionPred.eq(sample['action'].data.view_as(actionPred)).cpu().sum()

                actionData = np.concatenate((actionData, actionPred.cpu().numpy()), axis=0)
                goalData = np.concatenate((goalData, goal.data.cpu().numpy()), axis=0)
                modeData = np.concatenate((modeData, modePred.cpu().numpy()), axis=0)

            # compute overall loss
            loss = self.lossWeights[0]*actLoss + self.lossWeights[1]*goalLoss + self.lossWeights[2]*modeLoss        

            loss /= nSamples/self.batchSize
            actLoss /= nSamples/self.batchSize
            goalLoss /= nSamples/self.batchSize
            modeLoss /= nSamples/self.batchSize
            accuracy = 100.0*correct/nSamples
            
            # compile results
            results[dataType]['loss'] = loss
            results[dataType]['actLoss'] = actLoss
            results[dataType]['goalLoss'] = goalLoss
            results[dataType]['modeLoss'] = modeLoss
            results[dataType]['accuracy'] = accuracy    
        return results