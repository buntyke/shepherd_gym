#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implement dog heuristic model proposed in ??.

The dog switches between two behaviors of collecting and herding.
"""

# core modules
import numpy as np

def dog_heuristic_model(state, info):

    # initialize dog collect radius
    dog_collect_radius = 2.0

    # get number of sheep in env
    num_sheep = info['n']

    # filter different variables
    sheep_com = state[:2]
    farthest_sheep = state[2:4]
    target = state[4:6]
    dog_pose = state[6:8]
    radius_sheep = state[8]
    target_distance = state[9]

    # check if sheep are within field
    field = dog_collect_radius*(num_sheep**(2/3))

    is_within_field = False
    if radius_sheep < field:
        is_within_field = True

    # determine the dog position
    if is_within_field:
        # perform herding
        
        # compute the direction
        direction = (sheep_com - target)
        direction /= np.linalg.norm(direction)

        # compute the factor
        factor = dog_collect_radius*(np.sqrt(num_sheep))

        # get intermediate herding goal
        int_goal = sheep_com + (direction*factor)
    else:
        # perform collecting

        # compute the direction            
        direction = (farthest_sheep - sheep_com)
        direction /= np.linalg.norm(direction)

        # compute the distance factor
        factor = dog_collect_radius

        # get intermediate collecting goal
        int_goal = farthest_sheep + (direction*factor)

    # compute increments in x,y components
    direction = int_goal-dog_pose
    direction /= np.linalg.norm(direction)

    # discretize actions
    theta = np.arctan2(direction[1], direction[0])*180/np.pi

    action = 8
    if theta <= 22.5 and theta >= -22.5:
        action = 0
    elif theta <= 67.5 and theta > 22.5:
        action = 1
    elif theta <= 112.5 and theta > 67.5:
        action = 2
    elif theta <= 157.5 and theta > 112.5:
        action = 3
    elif theta < -157.5 or theta > 157.5:
        action = 4
    elif theta >= -157.5 and theta < -112.5:
        action = 5
    elif theta >= -112.5 and theta < -67.5:
        action = 6
    elif theta >= -67.5 and theta < -22.5:
        action = 7
    else:
        print('Error!')

    return action