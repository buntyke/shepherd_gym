shepherd_gym
============

[![Build Status](https://travis-ci.org/buntyke/shepherd_gym.png)](https://travis-ci.org/buntyke/shepherd_gym)
[![codecov.io](https://codecov.io/github/buntyke/shepherd_gym/coverage.svg?branch=master)](https://codecov.io/github/buntyke/shepherd_gym?branch=master)

Gym environment implementation of dog shepherding task

Installation
------------

* Package `stable-baselines` requires the following dependencies
  ```
  $ sudo apt update
  $ sudo apt install -y cmake libopenmpi-dev python3-dev zlib1g-dev
  ```

* Install machine learning dependencies for examples folder:
  ```
  $ pip install -r requirements.txt
  ```

* The library can be installed by running:
  ```
  $ pip install -e .
  ```

Usage
-----

This package has several scripts:

* To test the dog heuristic model with the shepherd gym env:
  ```
  $ python examples/shepherd_heuristic.py
  ```

  The program supports several command line arguments. Check using:
  ```
  $ python examples/shepherd_heuristic.py -h
  ```

Imitation Learning
------------------

Follow the following sequence to experiment with imitation learning

* Generate training dataset using the shepherd env:
  ```
  $ cd examples
  $ python shepherd_heuristic.py -e heuristic -n 1000 --store --norender --noplot
  ```
  This should create the shepherding data for 1000 trials in the `data/heuristic` folder.

* Preprocess the training dataset into a pickle file:
  ```
  $ python dataset_process.py -d heuristic
  ```
  This should create a pickle file with processed dataset in `data/heuristic` folder.

* Training imitation learning model:
  ```
  $ python shepherd_imitation.py -e heuristic 
  ```
  This will train a policy network using the expert dataset and store in `results` folder.

* View training performance using tensorboard:
  ```
  $ cd ../results/imitation
  $ tensorboard --logdir=.
  ```
  Open a webbrowser and check the URL: `localhost:6006`.

* Check training performance of model:
  ```
  $ python shepherd_imitation.py -e heuristic -m eval 

* Test performance of imitation learning model:
  ```
  $ cd examples
  $ python shepherd_imitation.py -e heuristic -m test --display
  ```
  This should render the environment window showing performance of IL agent.

Curriculum Learning
-------------------

* Generate training dataset using heuristic envrionment:
  ```
  $ cd examples
  $ python shepherd_heuristic.py -e curriculum -n 1000 --store --norender --noplot
  ```
  This should create the shepherding data for 1000 trials in the `data/curriculum` folder.

* Generate the curriculum for training dataset:
  ```
  $ python dataset_curriculum.py -d ../data/curriculum -n 1000
  ```
  This should create a pickle file with processed dataset. All the 
  trajectories have a fixed length of 100 for uniform sampling.

* Train reinforcement learning model using curriculum wrapper:
  ```
  $ python shepherd_curriculum.py -d ../data/curriculum -m train -t 1000000
  ```
  This should train a neural network policy using PPO and the 
  curriculum generated from the pickle file. 

* The performance of curriculum learning is not so good. It is
  still a work in progress.

Simulations
-----------

Heuristic model simulation (generated using matplotlib):

![heuristic model simulation](images/heuristic.gif)

Rewards for heuristic model (rewards keep increasing):

![heuristic model rewards](images/heuristic_rewards.png)

Rewards for random model (rewards remain low):

![random model rewards](images/random_rewards.png)

Requirements
------------
* gym>=0.10.8 
* numpy>=1.15.0
* matplotlib>=2.2.2

Compatibility
-------------

* python>=3.5 
* Note: There are some issues with python 3.7 due to stable-baselines dependency. However, it will work without that dependency.

Authors
-------

`shepherd_gym` was written by `Clark Kendrick Go, Nishanth Koganti`.
