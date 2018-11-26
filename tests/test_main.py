#!/usr/bin/env python
# -*- coding: utf-8 -*-

# core modules
import gym
import unittest
import numpy as np
import shepherd_gym

class Environments(unittest.TestCase):

    def test_env(self):
        env = gym.make('Shepherd-v0')
        env.seed(0)
        env.render()
        env.reset()
        for act in range(9):
            env.step(act)
        env.close()

    def test_fixed_env(self):
        env = gym.make('Shepherd-v1')
        env.seed(0)
        env.render()
        env.reset()
        for act in range(9):
            env.step(act)
        env.close()

    def test_heuristic(self):
        env = gym.make('Shepherd-v0')
        env.reset()

        (state,_,finish,info) = env.step(0)
        while not finish:
            action = shepherd_gym.dog_heuristic_model(state,info)
            (state,_,finish,info) = env.step(action)

    def test_sim(self):
        sim = shepherd_gym.ShepherdSim()
        sim.run_simulation()
