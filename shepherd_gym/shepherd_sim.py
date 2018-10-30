#!/usr/bin/env python
# -*- coding: utf-8 -*-

# suppress runtime warnings
import warnings
warnings.filterwarnings("ignore")

# import libraries
import numpy as np
import matplotlib.pyplot as plt

# class implementation of shepherding
class ShepherdSim:

    def __init__(self, num_sheep=25, init_radius=100):

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

        # initialize target position
        self.target = np.array([0,0])

        # initialize sheep positions
        init_sheep_pose = np.random.uniform(-200.0,200.0,size=(2))
        self.sheep_poses = (np.random.uniform(-50.0,50.0, size=(self.num_sheep,2))) \
                           + init_sheep_pose[None,:]
        self.sheep_com = self.sheep_poses.mean(axis=0)

        # initialize dog position
        init_dog_pose = init_sheep_pose + 75.0*(2*np.random.randint(2,size=(2))-1)
        self.dog_pose = init_dog_pose

        # initialize inertia
        self.inertia = np.ones((self.num_sheep, 2))

    # main function to perform simulation
    def run_simulation(self):

        # start the simulation
        print('Start simulation')

        # initialize counter for plotting
        counter = 0

        # initialize matplotlib figure
        fig = plt.figure()
        plt.ion()
        plt.show()

        # main loop for simulation
        while np.linalg.norm(self.target - self.sheep_com) > 1.0:
            # update counter variable
            counter += 1

            # get the new dog position
            self.dog_heuristic_model()

            # find new inertia
            self.update_environment()

            # plot every 5th frame
            if counter%5 == 0:
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

        # complete execution
        print('Finish simulation')

    # function to find new inertia for sheep
    def update_environment(self):
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

    # function to get new position of dog
    def dog_heuristic_model(self):

        # check if sheep are within field
        field = self.dog_collect_radius*(self.num_sheep**(2/3))
        dist_to_com = np.linalg.norm((self.sheep_poses - self.sheep_com[None,:]), axis=1)

        is_within_field = False
        if np.max(dist_to_com) < field:
            is_within_field = True

        # determine the dog position
        if is_within_field:
            # perform herding
            
            # compute the direction
            direction = (self.sheep_com - self.target)
            direction /= np.linalg.norm(direction)

            # compute the factor
            factor = self.dog_collect_radius*(np.sqrt(self.num_sheep))

            # get intermediate herding goal
            int_goal = self.sheep_com + (direction*factor)
        else:
            # perform collecting

            # get the farthest sheep
            dist_to_com = np.linalg.norm((self.sheep_poses - self.sheep_com[None,:]), axis=1)
            farthest_sheep = self.sheep_poses[np.argmax(dist_to_com),:]            

            # compute the direction            
            direction = (farthest_sheep - self.sheep_com)
            direction /= np.linalg.norm(direction)

            # compute the distance factor
            factor = self.dog_collect_radius

            # get intermediate collecting goal
            int_goal = farthest_sheep + (direction*factor)

        # find distances of dog to sheep
        dist_to_dog = np.linalg.norm((self.sheep_poses - self.dog_pose[None,:]), axis=1)

        # solve for size of step
        if np.all(dist_to_dog > 3*self.dog_collect_radius):
            mag = 1.5
        else:
            mag = 0.3*self.dog_collect_radius

        # compute increments in x,y components
        direction = int_goal-self.dog_pose
        factor = mag/np.linalg.norm(direction)
        increment = direction*factor

        # update position
        self.dog_pose = self.dog_pose+increment

def main():
    shepherd_env = ShepherdSim()
    shepherd_env.run_simulation()

if __name__=='__main__':
    main()