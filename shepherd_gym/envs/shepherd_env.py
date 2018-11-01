#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate the dog-sheep shepherding environment.

Each episode requires the dog to shepherd the sheep to the goal.
"""

# suppress runtime warnings
import warnings
warnings.filterwarnings("ignore")

# core modules
import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

class ShepherdEnv(gym.Env):
    """
    Define the shepherding environment.
    The environment treats the dog as the agent and the sheep as a part of the environment.

    State: 
    1) Position of center of mass (x,y)
    2) Position of farthest sheep (x,y)
    3) Position of target (x,y)
    4) Position of dog (x,y)
    5) Radius of sheep (r)
    6) Distance to target (d)

    Action:
    1) Increment in position of dog (x,y)

    Reward: 
    1) Negative of farthest sheep distance to com (d_f)
    2) Negative of com distance to target (d_t)
    """

    def __init__(self, num_sheep=25):
        
        # initialize observation space
        obs_low = np.array([-1000.0,-1000.0,-1000.0,-1000.0,-1000.0,-1000.0,-1000.0])
        obs_high = np.array([1000.0,1000.0,1000.0,1000.0,1000.0,1000.0,1000.0])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # initialize action space
        self.action_space = spaces.Discrete(9)

        # limit episode length
        self.MAX_STEPS = 1000

        # create buffer and episode variable
        self.curr_step = -1
        self.curr_episode = -1

        # radius for sheep to be considered as collected by dog
        self.dog_collect_radius = 2.0

        # weight multipliers for sheep forces
        self.com_term = 1.05
        self.noise_term = 0.3
        self.inertia_term = 0.5
        self.repulsion_dog_term = 1.0
        self.repulsion_sheep_term = 2.0

        # constants used to update environment
        self.delta_sheep_pose = 1.0
        self.dog_repulsion_dist = 70.0
        self.sheep_repulsion_dist = 2.0

        # assign number of sheep
        self.num_sheep = num_sheep

        # flag to show simulation, false by default
        self.show_sim = False

    def step(self, action):
        """
        The dog takes a step in the environment

        Parameters
        ----------
        action : float array

        Returns
        -------
        ob, reward, episode_over, info : tuple
            observation (float array) : 
                observation after dog position is updated.
            reward (float) : 
                amount of reward achieved by dog in the previous step.
            episode_over (bool) : 
                flag that indicates if the environment is reset or not.
            info (dict) :
                useful information about the environment for debugging.
        """
        
        self.curr_step += 1
        self._take_action(action)

        ob = self._get_state()
        reward = self._get_state()

        info = {'n':self.num_sheep}

        if self.curr_step >= self.MAX_STEPS or self.target_distance <= 1.0:
            self.finish = True

        if self.show_sim and self.curr_step%5 == 0:
            plt.clf()

            plt.scatter(self.target[0], self.target[1], c='g', s=40, label='Goal')
            plt.scatter(self.dog_pose[0], self.dog_pose[1], c='r', s=50, label='Dog')
            plt.scatter(self.sheep_poses[:,0], self.sheep_poses[:,1], c='b', s=50, label='Sheep')
            
            plt.title('Shepherding')
            plt.xlim([-300,300])
            plt.ylim([-300,300])
            plt.legend()
            plt.draw()
            plt.pause(0.01)

        return ob, reward, self.finish, info

    def reset(self):
        """
        Reset the environment and return the init state

        Returns
        -------
        observation (float array) : initial observation after reset.
        """

        # initialize gym env variables
        self.finish = False
        self.curr_step = -1
        self.curr_episode += 1

        # initialize target position
        self.target = np.random.uniform(-10.0,10.0,size=(2))

        # initialize sheep positions
        init_sheep_pose = np.random.uniform(-200.0,200.0,size=(2))
        self.sheep_poses = (np.random.uniform(-50.0,50.0, size=(self.num_sheep,2))) \
                           + init_sheep_pose[None,:]
        self.sheep_com = self.sheep_poses.mean(axis=0)

        # get the farthest sheep and radius of the sheep
        dist_to_com = np.linalg.norm((self.sheep_poses - self.sheep_com[None,:]), axis=1)
        self.farthest_sheep = self.sheep_poses[np.argmax(dist_to_com),:]
        self.radius_sheep = np.array([np.max(dist_to_com)])

        # update distance to target
        self.target_distance = np.linalg.norm(self.target - self.sheep_com)

        # initialize dog position
        init_dog_pose = init_sheep_pose + 75.0*(2*np.random.randint(2,size=(2))-1)
        self.dog_pose = init_dog_pose

        # initialize inertia
        self.inertia = np.ones((self.num_sheep, 2))

        # get the state, reward, finish, info
        state = self._get_state()
        reward = self._get_reward()
        info = {'n':self.num_sheep}

        return (state,reward,self.finish,info)

    def seed(self, seed):
        """Function to set the seed of env"""

        random.seed(seed)
        np.random.seed

    def _take_action(self, action):
        """Update position of dog based on action and env"""

        increment = np.array([0.0,0.0])
        if action == 0:
            increment[0] = 1.5
        elif action == 1:
            increment[0] = 1.225
            increment[1] = 1.225
        elif action == 2:
            increment[1] = 1.5
        elif action == 3:
            increment[0] = -1.225
            increment[1] = 1.225
        elif action == 4:
            increment[0] = -1.5
        elif action == 5:
            increment[0] = -1.225
            increment[1] = -1.225
        elif action == 6:
            increment[1] = -1.5
        elif action == 7:
            increment[0] = 1.225
            increment[1] = -1.225
        else:
            print('NOP!')
        
        self.dog_pose += increment
        self._update_environment()

    def _update_environment(self):
        """Update environment based on new position of dog"""

        # compute a distance matrix
        distance_matrix = np.zeros((self.num_sheep,self.num_sheep))
        for i in range(self.num_sheep):
            for j in range(i):
                dist = np.linalg.norm(self.sheep_poses[i,:] - self.sheep_poses[j,:])
                distance_matrix[i,j] = dist
                distance_matrix[j,i] = dist

        # find the sheep which are within 2 meters distance
        xvals, yvals = np.where((distance_matrix < self.sheep_repulsion_dist) & (distance_matrix != 0))
        interact = np.hstack((xvals[:,None],yvals[:,None]))

        # compute the repulsion forces within sheep
        repulsion_sheep = np.zeros((self.num_sheep,2))

        for val in range(self.num_sheep):
            iv = interact[interact[:,0] == val,1]
            transit = self.sheep_poses[val,:][None,:] - self.sheep_poses[iv,:]
            transit /= np.linalg.norm(transit, axis=1, keepdims=True)
            repulsion_sheep[val,:] = np.sum(transit, axis=0)

        repulsion_sheep /= np.linalg.norm(repulsion_sheep, axis=1, keepdims=True)
        repulsion_sheep[np.isnan(repulsion_sheep)] = 0

        # find sheep near dog
        dist_to_dog = np.linalg.norm((self.sheep_poses - self.dog_pose[None,:]), axis=1)
        sheep_inds = np.where(dist_to_dog < self.dog_repulsion_dist)
        near_sheep = sheep_inds[0]
        num_near_sheep = near_sheep.shape[0]

        # repulsion from dog
        repulsion_dog = np.zeros((self.num_sheep,2))
        repulsion_dog[near_sheep,:] = self.sheep_poses[near_sheep,:] - self.dog_pose[None,:]
        repulsion_dog /= np.linalg.norm(repulsion_dog, axis=1, keepdims=True)
        repulsion_dog[np.isnan(repulsion_dog)] = 0

        # attraction to COM
        attraction_com = np.zeros((self.num_sheep,2))
        attraction_com[near_sheep,:] = self.sheep_com[None,:] - self.sheep_poses[near_sheep,:]
        attraction_com /= np.linalg.norm(attraction_com, axis=1, keepdims=True)
        attraction_com[np.isnan(attraction_com)] = 0

        # error term
        noise = np.random.randn(self.num_sheep,2)
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)

        # compute sheep motion direction
        self.inertia = self.inertia_term*self.inertia + self.com_term*attraction_com + \
                self.repulsion_sheep_term*repulsion_sheep + self.repulsion_dog_term*repulsion_dog + \
                self.noise_term*noise
        
        # normalize the inertia terms
        self.inertia /= np.linalg.norm(self.inertia, axis=1, keepdims=True)
        self.inertia[np.isnan(self.inertia)] = 0

        # find new sheep position
        self.sheep_poses += self.delta_sheep_pose*self.inertia
        self.sheep_com = np.mean(self.sheep_poses,axis=0)

        # get the farthest sheep and radius of the sheep
        dist_to_com = np.linalg.norm((self.sheep_poses - self.sheep_com[None,:]), axis=1)
        self.farthest_sheep = self.sheep_poses[np.argmax(dist_to_com),:]
        self.radius_sheep = np.array([np.max(dist_to_com)])

        # update distance to target
        self.target_distance = np.linalg.norm(self.target - self.sheep_com)

    def _get_reward(self):
        """Return reward based on action of the dog"""

        # compute reward depending on the radius and distance to target
        reward = -(self.target_distance+self.radius_sheep)
        return reward

    def _get_state(self):
        """Return state based on action of the dog"""

        # stack all variables and return state array
        state = np.hstack((self.sheep_com, self.farthest_sheep, 
                    self.target, self.dog_pose, self.radius_sheep, 
                    self.target_distance))
        return state

    def render(self, mode='human', close=False):

        if mode == 'human':
            # create a figure
            self.fig = plt.figure()
            plt.ion()
            plt.show()

            # set the flag for plotting
            self.show_sim = True

        return