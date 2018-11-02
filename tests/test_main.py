#!/usr/bin/env python
# -*- coding: utf-8 -*-

# core modules
import gym
import unittest
import shepherd_gym

class Environments(unittest.TestCase):

    def test_env(self):
        env = gym.make('Shepherd-v0')
        env.seed(0)
        env.render()
        env.reset()
        env.step(0)
        env.close()

    def test_heuristic(self):
        env = gym.make('Shepherd-v0')
        env.seed(0)
        env.render()
        (state,_,_,info) = env.reset()
        action = shepherd_gym.dog_heuristic_model(state,info)
        env.step(action)
        env.close()

    def test_sim(self):
        sim = shepherd_gym.ShepherdSim()
        sim.run_simulation()
