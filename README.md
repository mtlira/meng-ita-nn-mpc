# README (Writing underway)
Code repository used for AIAA's article "Imitation Learning of MPC for Fault Tolerant Control Allocation of an Octorotor"
### The codes are currently messy in terms of readability and organization. We are soon going to clean, refactor and organize it, as well as write an explanatory README. Feel free to send an e-mail asking for any clarifications at matheusribeiro10@gmail.com. We appreciate your understanding!

- parameters folder: parameters for the octorotor and the simulations
- multirotor.py: Multirotor's nonlinear CoM dynamics and auxiliary methods
- linearize.py: State-space linearization and discretization methods
- mpc.py: Implementation of the MPC formulation
- simulate_mpc.py: script that generates the MPC simulation dataset
- dataset_handler.py: Methods to group, split and normalize the simulation dataset
- neural_network.py: Neural Network class with auxiliary methods
- hyperparameter_tuning.py: Perform the neural network's hyperparameter tuning using Optuna + K-fold
