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

    def __init__(self, n_actions=8, n_goal=2, n_modes=2, 
                 n_state=10, drop_rate=0.0):
        super(Policy, self).__init__()

        self.dense1 = nn.Linear(n_state,256)
        self.drop1 = nn.Dropout(drop_rate)
        self.dense2 = nn.Linear(256,256)
        self.drop2 = nn.Dropout(drop_rate)
        self.dense3 = nn.Linear(256,64)
        self.drop3 = nn.Dropout(drop_rate)

        self.output1 = nn.Linear(64,n_actions)
        self.output2 = nn.Linear(64,n_goal)
        self.output3 = nn.Linear(64,n_modes)

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
    
    def __init__(self, experiment, policy, train_loader, test_loader, optimizer,  
                 result_path='../results/imitation/', n_epochs=20, is_cuda=True, 
                 log_interval=50, loss_weights=[1.0,0.1,0.1], batch_size=64):

        # initialize variables
        self.is_cuda = is_cuda
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.loss_weights = loss_weights
        
        # initialize policy network and data loaders
        self.policy = policy
        self.optimizer = optimizer
        self.test_loader = test_loader
        self.train_loader = train_loader

        # experiment details
        self.experiment = experiment
        self.result_path = result_path

        # initialize run variables
        self.epoch = 0
        self.global_step = 0
        self.writer = SummaryWriter(log_dir=self.result_path+self.experiment)

    def run(self):

        self.test()
        for self.epoch in range(1, self.n_epochs+1):
            self.train()
            self.test()
            print(f'#### Epoch {self.epoch} ####')

        self.writer.close()
        torch.save(self.policy.state_dict(), 
                   self.result_path+self.experiment+'/model.pt')

    def train(self):
        correct = 0
        self.policy.train()

        for batch_idx, sample in enumerate(self.train_loader):
            if self.is_cuda:
                for key in sample.keys():
                    sample[key] = sample[key].cuda()
            for key in sample.keys():
                sample[key] = Variable(sample[key])

            # perform forward pass
            self.optimizer.zero_grad()
            action, goal, mode = self.policy(sample['state'])

            # setup loss function
            goal_loss = F.mse_loss(goal, sample['goal'])
            mode_loss = F.cross_entropy(mode, sample['mode'][:,0])
            act_loss = F.cross_entropy(action, sample['action'][:,0])
            loss = self.loss_weights[0]*act_loss \
                    + self.loss_weights[1]*goal_loss \
                    + self.loss_weights[2]*mode_loss

            # get index of max log-probability
            pred = F.softmax(action, dim=1).data.max(1, keepdim=True)[1] 
            correct += pred.eq(sample['action'].data.view_as(pred)).cpu().sum()

            # perform backpropogation
            loss.backward()
            self.optimizer.step()            
            self.global_step += 1

            if batch_idx % self.log_interval == 0:
                self.writer.add_scalars('loss', {'train':loss}, self.global_step)
                self.writer.add_scalars('actLoss', {'train':act_loss}, self.global_step)
                self.writer.add_scalars('goalLoss', {'train':goal_loss}, self.global_step)
                self.writer.add_scalars('modeLoss', {'train':mode_loss}, self.global_step)
        self.writer.add_scalars('accuracy', {'train':100.*correct/len(self.train_loader.dataset)}, 
                                self.global_step)

    def test(self):
        correct = 0
        test_act_loss = 0
        test_goal_loss = 0
        test_mode_loss = 0
        self.policy.eval()

        for sample in self.test_loader:
            if self.is_cuda:
                for key in sample.keys():
                    sample[key] = sample[key].cuda()
                for key in sample.keys():
                    sample[key] = Variable(sample[key])
        
            action, goal, mode = self.policy(sample['state'])

            # setup loss function
            test_goal_loss += F.mse_loss(goal, sample['goal']).data
            test_mode_loss += F.cross_entropy(mode, sample['mode'][:,0]).data
            test_act_loss += F.cross_entropy(action, sample['action'][:,0]).data

            # get index of max log-probability
            pred = F.softmax(action, dim=1).data.max(1, keepdim=True)[1] 
            correct += pred.eq(sample['action'].data.view_as(pred)).cpu().sum()

        test_loss = self.loss_weights[0]*test_act_loss + self.loss_weights[1]*test_goal_loss \
                    + self.loss_weights[2]*test_mode_loss        

        test_loss /= len(self.test_loader.dataset)/self.batch_size
        test_act_loss /= len(self.test_loader.dataset)/self.batch_size
        test_goal_loss /= len(self.test_loader.dataset)/self.batch_size
        test_mode_loss /= len(self.test_loader.dataset)/self.batch_size
        
        self.writer.add_scalars('loss', {'test':test_loss}, self.global_step)
        self.writer.add_scalars('actLoss', {'test':test_act_loss}, self.global_step)
        self.writer.add_scalars('goalLoss', {'test':test_goal_loss}, self.global_step)
        self.writer.add_scalars('modeLoss', {'test':test_mode_loss}, self.global_step)
        self.writer.add_scalars('accuracy', {'test':100.*correct/len(self.test_loader.dataset)}, 
                                self.global_step)