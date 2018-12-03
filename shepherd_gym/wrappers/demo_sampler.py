import os
import pickle
import numpy as np

from shepherd_gym.wrappers import Wrapper

class SamplerWrapper(Wrapper):
    env = None

    def __init__(self, env, demo_path,
                 increment_freq=100, 
                 initial_window_width=10,
                 window_increment=10):

        # inherit from base wrapper class
        super().__init__(env)

        # load demo dataset
        self.demo_path = demo_path
        with open(f'{self.demo_path}/curriculum.npz','rb') as f:
            self.demo_data = pickle.load(f)

        # number of trajectories
        self.num_traj = len(self.demo_data)

        # initialize number of demos sampled
        self.demo_sampled = 0

        # initialize sampling variables
        self.increment_freq = increment_freq
        self.window_size = initial_window_width
        self.window_increment = window_increment

    def reset(self):

        # get a state sample
        state = self.sample()
        return self.env.reset_from_state(state)

    def sample(self):

        # get a random episode index
        ep_ind = np.random.choice(self.num_traj)
        states = self.demo_data[ep_ind]

        # sample uniformly
        eps_len = states.shape[0]
        index = np.random.randint(max(eps_len - self.window_size, 0), eps_len)
        state = states[index]

        # increment window size
        self.demo_sampled += 1
        if self.demo_sampled >= self.increment_freq:
            if self.window_size < eps_len:
                self.window_size += self.window_increment 
            self.demo_sampled = 0

        # return the state
        return state