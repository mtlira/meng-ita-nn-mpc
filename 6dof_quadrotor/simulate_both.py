import numpy as np
import trajectory_handler
import multirotor
from neural_network import *
import torch
from scipy.integrate import odeint
from plots import DataAnalyser
import pandas as pd
import restriction_handler
from simulate_mpc import simulate_mpc, wrap_metadata
import time
from pathlib import Path

# Script for generating batches of comparative simulations between the neural network and the MPC.

#use_optuna_model = True

### MULTIROTOR PARAMETERS ###
from parameters.octorotor_parameters import m, g, I_x, I_y, I_z, l, b, d, num_rotors, thrust_to_weight

### Create model of multirotor ###
multirotor_model = multirotor.Multirotor(m, g, I_x, I_y, I_z, b, l, d, num_rotors, thrust_to_weight)

num_neurons_hidden_layers = 128 # TODO: AUTOMATIZAR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


### SIMULATION PARAMETERS ###
from parameters.simulation_parameters import time_step, T_sample, N, M, gain_scheduling, include_phi_theta_reference, include_psi_reference
q_neuralnetwork = 3 # Number of MPC outputs (x, y z)
num_inputs = 205 - num_rotors # TODO: AUTOMATIZAR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Initial condition
X0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

# Input and state values at the equilibrium condition
omega_eq = multirotor_model.get_omega_eq_hover()
omega_squared_eq = omega_eq**2

# Trajectory
tr = trajectory_handler.TrajectoryHandler()

def simulate_mpc_nn(X0, multirotor_model, N, M, num_inputs, q_neuralnetwork, omega_squared_eq, dataset_mother_folder, weights_file_name, time_step, T_sample, T_simulation, trajectory, trajectory_type, restriction, restriction_metadata, output_weights, control_weights, \
                    gain_scheduling, disturb_input, num_neurons_hidden_layers, optuna_version, trajectory_metadata = None):
    
    global dataset_dataframe
    global dataset_id
    global trajectory_id
    t_samples = np.arange(0, T_simulation, T_sample)
    analyser = DataAnalyser()
    simulation_save_path = f'{dataset_mother_folder}comparative_simulations/{trajectory_type}/{str(dataset_id)}/'

    simulator = NeuralNetworkSimulator(multirotor_model, N, M, num_inputs, num_rotors, q_neuralnetwork, omega_squared_eq, time_step)

    x_nn, u_nn, omega_nn, nn_metadata, omega_squared_nn = simulator.simulate_neural_network(X0, dataset_mother_folder, weights_file_name, t_samples, trajectory, optuna_version=optuna_version,\
                                num_neurons_hidden_layers=num_neurons_hidden_layers, restriction=restriction, disturb_input=disturb_input, clip=True)
    
    mpc_success, mpc_metadata, simulation_data = simulate_mpc(X0, time_step, T_sample, T_simulation, trajectory, restriction, output_weights, control_weights, gain_scheduling,\
                                disturb_input=disturb_input, plot=False)
    x_mpc, u_mpc, omega_mpc, _ = simulation_data if simulation_data is not None else [None, None, None, None]


    simulation_metadata = wrap_metadata(dataset_id, trajectory_type, T_simulation, T_sample, N, M, mpc_success, mpc_metadata, restriction_metadata, disturbed_inputs=disturb_input, trajectory_id=trajectory_id)
    Path(simulation_save_path).mkdir(parents=True, exist_ok=True)

    for nn_key in list(nn_metadata.keys()):
        if nn_key not in list(simulation_metadata.keys()):
            simulation_metadata[nn_key] = nn_metadata[nn_key]
    simulation_metadata['radius (m)'] = trajectory_metadata['radius'] if trajectory_metadata is not None else 'nan'
    simulation_metadata['period (s)'] = trajectory_metadata['period'] if trajectory_metadata is not None else 'nan'

    if x_mpc is not None and x_nn is not None:
        simulation_metadata['inter_position_RMSe'] = analyser.RMSe(x_nn[:,9:], x_mpc[:,9:])
        for i, u_rmse in enumerate(analyser.RMSe_control(omega_mpc, omega_nn)):
            simulation_metadata[f'RMSe_u{i}'] = u_rmse
    else:
        simulation_metadata['inter_position_RMSe'] = 'nan'  
        for i in range(num_rotors):
            simulation_metadata[f'RMSe_u{i}'] = 'nan'
    
    pd.DataFrame(simulation_metadata).to_csv(simulation_save_path + 'stats.csv', index=False)
    dataset_dataframe = pd.concat([dataset_dataframe, pd.DataFrame(simulation_metadata)])
    if dataset_id % 1 == 0: dataset_dataframe.to_csv(dataset_mother_folder + 'dataset_metadata.csv', sep=',', index=False)

    legend = ['MPC', 'Neural Network', 'Trajectory'] if x_mpc is not None else ['Neural Network', 'Trajectory']

    if trajectory_type == 'spiral_toroid': dict_toroid = {'R':7, 'r':3, 'n_winds':10,'period':100}
    else: dict_toroid = None

    #if x_nn is not None: analyser.plot_omega_squared(omega_squared_nn, t_samples[:np.shape(omega_squared_nn)[0]], simulation_save_path)
    if x_nn is not None and x_mpc is not None: 
        analyser.plot_states(x_mpc, t_samples[:np.shape(x_nn)[0]], X_lin=x_nn, trajectory=trajectory[:len(t_samples)], u_vector=[u_mpc, u_nn], omega_vector=[omega_mpc, omega_nn], legend=legend, equal_scales=True, save_path=simulation_save_path, plot=False, pdf=True, extra_dict = dict_toroid,alpha=0.77)
        analyser.plot_states_shrink(x_mpc, t_samples[:np.shape(x_nn)[0]], X_lin=x_nn, trajectory=trajectory[:len(t_samples)], u_vector=[u_mpc, u_nn], omega_vector=[omega_mpc, omega_nn], legend=legend, equal_scales=True, save_path=simulation_save_path, plot=False, pdf=True, extra_dict = dict_toroid,alpha=0.77)
    elif x_mpc is not None: analyser.plot_states(x_mpc, t_samples[:np.shape(x_mpc)[0]], trajectory=trajectory[:len(t_samples)], u_vector=[u_mpc], omega_vector=[omega_mpc], legend=['MPC', 'Trajectory'], equal_scales=True, save_path=simulation_save_path, plot=False, pdf=True)
    elif x_nn is not None:  analyser.plot_states(x_nn, t_samples[:np.shape(x_nn)[0]], trajectory=trajectory[:len(t_samples)], u_vector=[u_nn], omega_vector=[omega_nn], legend=['NN', 'Trajectory'], equal_scales=True, save_path=simulation_save_path, plot=False, pdf=True)

    if x_mpc is not None and x_nn is not None:
        np.save(simulation_save_path + 'x_mpc.npy', x_mpc)
        np.save(simulation_save_path + 'omega_mpc.npy', omega_mpc)

        np.save(simulation_save_path + 'x_nn.npy', x_nn)
        np.save(simulation_save_path + 'omega_nn.npy', omega_nn)

    dataset_id += 1



def simulate_2_nns(X0, multirotor_model, N, M, num_inputs, q_neuralnetwork, omega_squared_eq, dataset_mother_folder, weights_file_name, time_step, T_sample, T_simulation, trajectory, trajectory_type, restriction, restriction_metadata, output_weights, control_weights, \
                    gain_scheduling, disturb_input, num_neurons_hidden_layers, optuna_version, trajectory_metadata = None):
    # Simulate clipped and unclipped NNs
    
    global dataset_dataframe_nn
    global dataset_id
    t_samples = np.arange(0, T_simulation, T_sample)
    analyser = DataAnalyser()
    simulation_save_path = f'{dataset_mother_folder}comparative_simulations/{trajectory_type}/{str(dataset_id)}/'

    simulator = NeuralNetworkSimulator(multirotor_model, N, M, num_inputs, num_rotors, q_neuralnetwork, omega_squared_eq, time_step)

    x_nn, u_nn, omega_nn, nn_metadata, omega_squared_nn = simulator.simulate_neural_network(X0, dataset_mother_folder, weights_file_name, t_samples, trajectory, optuna_version=optuna_version,\
                                num_neurons_hidden_layers=num_neurons_hidden_layers, restriction=restriction, disturb_input=disturb_input, clip=True)
    
    x_nn2, u_nn2, omega_nn2, nn_metadata2, omega_squared_nn2 = simulator.simulate_neural_network(X0, dataset_mother_folder, weights_file_name, t_samples, trajectory, optuna_version=optuna_version,\
                                num_neurons_hidden_layers=num_neurons_hidden_layers, restriction=restriction, disturb_input=disturb_input, clip=False)
    
    simulation_metadata = nn_metadata
    for key in list(nn_metadata2.keys()):
        simulation_metadata[f'{key}_2'] = [nn_metadata2[key]]
    
    Path(simulation_save_path).mkdir(parents=True, exist_ok=True)

    #for nn_key in list(nn_metadata.keys()):
    #    if nn_key not in list(simulation_metadata.keys()):
    #        simulation_metadata[nn_key] = nn_metadata[nn_key]
    simulation_metadata['radius (m)'] = trajectory_metadata['radius'] if trajectory_metadata is not None else 'nan'
    simulation_metadata['period (s)'] = trajectory_metadata['period'] if trajectory_metadata is not None else 'nan'

    if x_nn is not None and x_nn2 is not None:
        simulation_metadata['inter_position_RMSe'] = analyser.RMSe(x_nn[:,9:], x_nn2[:,9:])
        for i, u_rmse in enumerate(analyser.RMSe_control(omega_nn, omega_nn2)):
            simulation_metadata[f'RMSe_u{i}'] = u_rmse
    else:
        simulation_metadata['inter_position_RMSe'] = 'nan'  
        for i in range(num_rotors):
            simulation_metadata[f'RMSe_u{i}'] = 'nan'
    print('simulation_metadata\n',simulation_metadata)
    dataset_dataframe_nn = pd.concat([dataset_dataframe_nn, pd.DataFrame(simulation_metadata)])
    if dataset_id % 1 == 0: dataset_dataframe_nn.to_csv(dataset_mother_folder + 'dataset_metadata_nn.csv', sep=',', index=False)

    legend = ['Clipped NN', 'Unclipped NN', 'Trajectory'] if x_nn is not None else ['Neural Network', 'Trajectory']

    if trajectory_type == 'spiral_toroid': dict_toroid = {'R':7, 'r':3, 'n_winds':10,'period':100}
    else: dict_toroid = None

    if x_nn is not None: analyser.plot_omega_squared([omega_squared_nn, omega_squared_nn2], t_samples[:np.shape(omega_squared_nn)[0]], simulation_save_path)
    if x_nn is not None and x_nn2 is not None: 
        analyser.plot_states_shrink(x_nn, t_samples[:np.shape(x_nn)[0]], X_lin=x_nn2, trajectory=trajectory[:len(t_samples)], u_vector=[u_nn, u_nn2], omega_vector=[omega_nn, omega_nn2], legend=legend, equal_scales=True, save_path=simulation_save_path, plot=False, pdf=True, extra_dict = dict_toroid,alpha=1)
        analyser.plot_states(x_nn, t_samples[:np.shape(x_nn)[0]], X_lin=x_nn2, trajectory=trajectory[:len(t_samples)], u_vector=[u_nn, u_nn2], omega_vector=[omega_nn, omega_nn2], legend=legend, equal_scales=True, save_path=simulation_save_path, plot=False, pdf=True, extra_dict = dict_toroid, alpha=1)
    #elif x_mpc is not None: analyser.plot_states(x_mpc, t_samples[:np.shape(x_mpc)[0]], trajectory=trajectory[:len(t_samples)], u_vector=[u_mpc], omega_vector=[omega_mpc], legend=['MPC', 'Trajectory'], equal_scales=True, save_path=simulation_save_path, plot=False, pdf=True)
    #elif x_nn is not None:  analyser.plot_states(x_nn, t_samples[:np.shape(x_nn)[0]], trajectory=trajectory[:len(t_samples)], u_vector=[u_nn], omega_vector=[omega_nn], legend=['NN', 'Trajectory'], equal_scales=True, save_path=simulation_save_path, plot=False, pdf=True)

    if x_nn is not None and x_nn2 is not None:
        np.save(simulation_save_path + 'x_nn.npy', x_nn)
        np.save(simulation_save_path + 'omega_nn.npy', omega_nn)

        np.save(simulation_save_path + 'x_nn_unclipped.npy', x_nn2)
        np.save(simulation_save_path + 'omega_mpc.npy', omega_nn2)

    dataset_id += 1

def simulate_mpc_nn_2(X0, multirotor_model, N, M, num_inputs, q_neuralnetwork, omega_squared_eq, dataset_mother_folder, weights_file_name, time_step, T_sample, T_simulation, trajectory, trajectory_type, restriction, restriction_metadata, output_weights, control_weights, \
                    gain_scheduling, disturb_input, num_neurons_hidden_layers, optuna_version, trajectory_metadata = None, restrictions_unconstrained=None):
    # Simulate Constrained MPC, NN and ANGLE-Unconstrained MPC.
    global dataset_dataframe
    global dataset_id
    global trajectory_id
    t_samples = np.arange(0, T_simulation, T_sample)
    analyser = DataAnalyser()
    simulation_save_path = f'{dataset_mother_folder}comparative_simulations/{trajectory_type}/{str(dataset_id)}/'

    simulator = NeuralNetworkSimulator(multirotor_model, N, M, num_inputs, num_rotors, q_neuralnetwork, omega_squared_eq, time_step)

    x_nn, u_nn, omega_nn, nn_metadata, omega_squared_nn = simulator.simulate_neural_network(X0, dataset_mother_folder, weights_file_name, t_samples, trajectory, optuna_version=optuna_version,\
                                num_neurons_hidden_layers=num_neurons_hidden_layers, restriction=restriction, disturb_input=disturb_input, clip=True)
    
    mpc_success, mpc_metadata, simulation_data = simulate_mpc(X0, time_step, T_sample, T_simulation, trajectory, restriction, output_weights, control_weights, gain_scheduling,\
                                disturb_input=disturb_input, plot=False)
    x_mpc, u_mpc, omega_mpc, _ = simulation_data if simulation_data is not None else [None, None, None, None]

    mpc_success_2, mpc_metadata_2, simulation_data_2 = simulate_mpc(X0, time_step, T_sample, T_simulation, trajectory, restrictions_unconstrained, output_weights, control_weights, gain_scheduling,\
                                disturb_input=disturb_input, plot=False)
    x_mpc_2, u_mpc_2, omega_mpc_2, _ = simulation_data_2 if simulation_data_2 is not None else [None, None, None, None]


    simulation_metadata = wrap_metadata(dataset_id, trajectory_type, T_simulation, T_sample, N, M, mpc_success, mpc_metadata, restriction_metadata, disturbed_inputs=disturb_input, trajectory_id=trajectory_id)
    Path(simulation_save_path).mkdir(parents=True, exist_ok=True)

    if mpc_success_2:
        for key in list(mpc_metadata_2.keys()):
            simulation_metadata[f'{key}_2'] = mpc_metadata_2[key]

    for nn_key in list(nn_metadata.keys()):
        if nn_key not in list(simulation_metadata.keys()):
            simulation_metadata[nn_key] = nn_metadata[nn_key]
    simulation_metadata['radius (m)'] = trajectory_metadata['radius'] if trajectory_metadata is not None else 'nan'
    simulation_metadata['period (s)'] = trajectory_metadata['period'] if trajectory_metadata is not None else 'nan'

    

    #if x_mpc is not None and x_nn is not None:
    #    simulation_metadata['inter_position_RMSe'] = analyser.RMSe(x_nn[:,9:], x_mpc[:,9:])
    #    for i, u_rmse in enumerate(analyser.RMSe_control(omega_mpc, omega_nn)):
    #        simulation_metadata[f'RMSe_u{i}'] = u_rmse
    #else:
    #    simulation_metadata['inter_position_RMSe'] = 'nan'  
    #    for i in range(num_rotors):
    #        simulation_metadata[f'RMSe_u{i}'] = 'nan'
    
    dataset_dataframe = pd.concat([dataset_dataframe, pd.DataFrame(simulation_metadata)])
    if dataset_id % 1 == 0: dataset_dataframe.to_csv(dataset_mother_folder + 'dataset_metadata.csv', sep=',', index=False)

    legend = ['MPC', 'Neural Network', 'Trajectory'] if x_mpc is not None else ['Neural Network', 'Trajectory']

    if trajectory_type == 'spiral_toroid': dict_toroid = {'R':7, 'r':3, 'n_winds':10,'period':100}
    else: dict_toroid = None

    np.save(simulation_save_path + 'x_mpc.npy', x_mpc)
    np.save(simulation_save_path + 'omega_mpc.npy', omega_mpc)

    np.save(simulation_save_path + 'x_mpc_unconstrained.npy', x_mpc_2)
    np.save(simulation_save_path + 'omega_mpc_unconstrained.npy', omega_mpc_2)

    np.save(simulation_save_path + 'x_nn.npy', x_nn)
    np.save(simulation_save_path + 'omega_nn.npy', omega_nn)

    #if x_nn is not None: analyser.plot_omega_squared(omega_squared_nn, t_samples[:np.shape(omega_squared_nn)[0]], simulation_save_path)
    if x_nn is not None and x_mpc is not None and x_mpc_2 is not None: analyser.plot_3d([x_mpc[:,9:], x_nn[:,9:], x_mpc_2[:,9:]], ['tab:blue','tab:orange','violet'], ['Constrained MPC', 'Neural Network', 'Unconstrained MPC'], t_samples[:np.shape(x_nn)[0]], trajectory[:len(t_samples)], True, simulation_save_path)

    if x_nn is not None and x_mpc is not None: analyser.plot_states(x_mpc, t_samples[:np.shape(x_nn)[0]], X_lin=x_nn, trajectory=trajectory[:len(t_samples)], u_vector=[u_mpc, u_nn], omega_vector=[omega_mpc, omega_nn], legend=legend, equal_scales=True, save_path=simulation_save_path, plot=False, pdf=True, extra_dict = dict_toroid, alpha=0.7)
    
    elif x_mpc is not None: analyser.plot_states(x_mpc, t_samples[:np.shape(x_mpc)[0]], trajectory=trajectory[:len(t_samples)], u_vector=[u_mpc], omega_vector=[omega_mpc], legend=['MPC', 'Trajectory'], equal_scales=True, save_path=simulation_save_path, plot=False, pdf=True)
    
    elif x_nn is not None:  analyser.plot_states(x_nn, t_samples[:np.shape(x_nn)[0]], trajectory=trajectory[:len(t_samples)], u_vector=[u_nn], omega_vector=[omega_nn], legend=['NN', 'Trajectory'], equal_scales=True, save_path=simulation_save_path, plot=False, pdf=True)
    dataset_id += 1

# def simulate_mpc_nn_3(X0, multirotor_model, N, M, num_inputs, q_neuralnetwork, omega_squared_eq, dataset_mother_folder, weights_file_name, time_step, T_sample, T_simulation, trajectory, trajectory_type, restriction, restriction_metadata, output_weights, control_weights, \
#                     gain_scheduling, disturb_input, num_neurons_hidden_layers, optuna_version, restrictions_thrust_unconstrained, trajectory_metadata = None):
#     # Simulate Constrained MPC, NN and THRUST-Unconstrained MPC.
#     global dataset_dataframe
#     global dataset_id
#     t_samples = np.arange(0, T_simulation, T_sample)
#     analyser = DataAnalyser()
#     simulation_save_path = f'{dataset_mother_folder}comparative_simulations/{trajectory_type}/{str(dataset_id)}/'

#     simulator = NeuralNetworkSimulator(multirotor_model, N, M, num_inputs, num_rotors, q_neuralnetwork, omega_squared_eq, time_step)

#     x_nn, u_nn, omega_nn, nn_metadata, omega_squared_nn = simulator.simulate_neural_network(X0, dataset_mother_folder, weights_file_name, t_samples, trajectory, optuna_version=optuna_version,\
#                                 num_neurons_hidden_layers=num_neurons_hidden_layers, restriction=restriction, disturb_input=disturb_input, clip=False)
    
#     mpc_success, mpc_metadata, simulation_data = simulate_mpc(X0, time_step, T_sample, T_simulation, trajectory, restriction, output_weights, control_weights, gain_scheduling,\
#                                 disturb_input=disturb_input, plot=False)
#     x_mpc, u_mpc, omega_mpc, _ = simulation_data if simulation_data is not None else [None, None, None, None]

#     _, _, simulation_data_2 = simulate_mpc(X0, time_step, T_sample, T_simulation, trajectory, restrictions_thrust_unconstrained, output_weights, control_weights, gain_scheduling,\
#                                 disturb_input=disturb_input, plot=False)
#     x_mpc_2, u_mpc_2, omega_mpc_2, _ = simulation_data_2 if simulation_data_2 is not None else [None, None, None, None]


#     simulation_metadata = wrap_metadata(dataset_id, trajectory_type, T_simulation, T_sample, N, M, mpc_success, mpc_metadata, restriction_metadata, disturbed_inputs=disturb_input)
#     Path(simulation_save_path).mkdir(parents=True, exist_ok=True)

#     for nn_key in list(nn_metadata.keys()):
#         if nn_key not in list(simulation_metadata.keys()):
#             simulation_metadata[nn_key] = nn_metadata[nn_key]
#     simulation_metadata['radius (m)'] = trajectory_metadata['radius'] if trajectory_metadata is not None else 'nan'
#     simulation_metadata['period (s)'] = trajectory_metadata['period'] if trajectory_metadata is not None else 'nan'

#     if x_mpc is not None and x_nn is not None:
#         simulation_metadata['inter_position_RMSe'] = analyser.RMSe(x_nn[:,9:], x_mpc[:,9:])
#         for i, u_rmse in enumerate(analyser.RMSe_control(omega_mpc, omega_nn)):
#             simulation_metadata[f'RMSe_u{i}'] = u_rmse
#     else:
#         simulation_metadata['inter_position_RMSe'] = 'nan'  
#         for i in range(num_rotors):
#             simulation_metadata[f'RMSe_u{i}'] = 'nan'
    
#     dataset_dataframe = pd.concat([dataset_dataframe, pd.DataFrame(simulation_metadata)])
#     if dataset_id % 1 == 0: dataset_dataframe.to_csv(dataset_mother_folder + 'dataset_metadata.csv', sep=',', index=False)

#     legend = ['MPC', 'Neural Network', 'Trajectory'] if x_mpc is not None else ['Neural Network', 'Trajectory']

#     if trajectory_type == 'spiral_toroid': dict_toroid = {'R':7, 'r':3, 'n_winds':10,'period':100}
#     else: dict_toroid = None

#     np.save(simulation_save_path + 'x_mpc.npy', x_mpc)
#     np.save(simulation_save_path + 'omega_mpc.npy', omega_mpc)

#     np.save(simulation_save_path + 'x_mpc_unconstrained.npy', x_mpc_2)
#     np.save(simulation_save_path + 'omega_mpc_unconstrained.npy', omega_mpc_2)

#     np.save(simulation_save_path + 'x_nn.npy', x_nn)
#     np.save(simulation_save_path + 'omega_nn.npy', omega_nn)

#     #if x_nn is not None: analyser.plot_omega_squared(omega_squared_nn, t_samples[:np.shape(omega_squared_nn)[0]], simulation_save_path)
#     if x_nn is not None and x_mpc is not None and x_mpc_2 is not None: analyser.plot_3d([x_mpc[:,9:], x_nn[:,9:], x_mpc_2[:,9:]], ['tab:blue','tab:orange','violet'], ['Constrained MPC', 'Neural Network', 'Unconstrained MPC'], t_samples[:np.shape(x_nn)[0]], trajectory[:len(t_samples)], True, simulation_save_path)

#     if x_nn is not None and x_mpc is not None: analyser.plot_states(x_mpc, t_samples[:np.shape(x_nn)[0]], X_lin=x_nn, trajectory=trajectory[:len(t_samples)], u_vector=[u_mpc, u_nn], omega_vector=[omega_mpc, omega_nn], legend=legend, equal_scales=True, save_path=simulation_save_path, plot=False, pdf=True, extra_dict = dict_toroid, alpha=0.7)
    
#     elif x_mpc is not None: analyser.plot_states(x_mpc, t_samples[:np.shape(x_mpc)[0]], trajectory=trajectory[:len(t_samples)], u_vector=[u_mpc], omega_vector=[omega_mpc], legend=['MPC', 'Trajectory'], equal_scales=True, save_path=simulation_save_path, plot=False, pdf=True)
    
#     elif x_nn is not None:  analyser.plot_states(x_nn, t_samples[:np.shape(x_nn)[0]], trajectory=trajectory[:len(t_samples)], u_vector=[u_nn], omega_vector=[omega_nn], legend=['NN', 'Trajectory'], equal_scales=True, save_path=simulation_save_path, plot=False, pdf=True)
#     dataset_id += 1
        
def simulate_batch(trajectory_type, args_vector, restrictions_vector, disturb_input, mode,checkpoint_id = None, relax_constraint=None):
    global dataset_id
    global trajectory_id
    global total_simulations
    global failed_simulations
    global dataset_dataframe
    global dataset_dataframe_nn
    global dataset_mother_folder
    global weights_file_name

    total_simulations = len(args_vector) * len(restrictions_vector)
    tr = trajectory_handler.TrajectoryHandler()
    for args in args_vector:
        trajectory_id += 1
        for restrictions, output_weights, control_weights, restriction_metadata in restrictions_vector:
            if checkpoint_id is None or dataset_id >= checkpoint_id:
                trajectory = tr.generate_trajectory(trajectory_type, args, include_psi_reference, include_phi_theta_reference)
                #folder_name = f'{trajectory_type}/' + str(dataset_id)
                trajectory_metadata = None
                if trajectory_type in ['circle_xy', 'circle_xz', 'lissajous_xy']:
                    trajectory_metadata = {'radius': args[1], 'period': round(2*np.pi/args[0], 1)}
                # Simulation without disturbances
                print(f'{trajectory_type} Simulation {dataset_id}/{total_simulations}')
                T_simulation = args[-1]
                if mode == 'constrained_mpc_and_nn':
                    simulate_mpc_nn(X0, multirotor_model, N, M, num_inputs, q_neuralnetwork, omega_squared_eq, dataset_mother_folder, weights_file_name, time_step, T_sample, T_simulation, trajectory, trajectory_type, restrictions, restriction_metadata, output_weights, control_weights, gain_scheduling, disturb_input, num_neurons_hidden_layers, optuna_version, trajectory_metadata)                
                elif mode == 'compare_clipped_unclipped_nns':
                    simulate_2_nns(X0, multirotor_model, N, M, num_inputs, q_neuralnetwork, omega_squared_eq, dataset_mother_folder, weights_file_name, time_step, T_sample, T_simulation, trajectory, trajectory_type, restrictions, restriction_metadata, output_weights, control_weights, gain_scheduling, disturb_input, num_neurons_hidden_layers, optuna_version, trajectory_metadata)
                elif mode == 'constrained_and_unconstrained_mpc_and_nn':
                    #if restrictions_vector_unconstrained is None: raise Exception('restrictions_uncontrained_vector is None')
                    restriction_thrust_unconstrained, _, _, _ = rst.restriction(restriction_metadata['operation_mode'], restriction_metadata['rotors_idx'], relax_constraint)
                    simulate_mpc_nn_2(X0, multirotor_model, N, M, num_inputs, q_neuralnetwork, omega_squared_eq, dataset_mother_folder, weights_file_name, time_step, T_sample, T_simulation, trajectory, trajectory_type, restrictions, restriction_metadata, output_weights, control_weights, gain_scheduling, disturb_input, num_neurons_hidden_layers, optuna_version, trajectory_metadata, restriction_thrust_unconstrained)

            else:
                dataset_id += 1
    
    # Reset id
    dataset_id = 1
    trajectory_id = 0

if __name__ == '__main__':
    nn_weights_folder = 'training_results/'
    dataset_mother_folder = nn_weights_folder
    weights_file_name = 'model_weights.pth'
    optuna_version = 'v6_database_v5_smooth_kfold'
    disturb_input = False

    if Path(dataset_mother_folder + 'dataset_metadata.csv').is_file():
        dataset_dataframe = pd.read_csv(dataset_mother_folder + 'dataset_metadata.csv', sep=',')
    else:
        dataset_dataframe = pd.DataFrame({})
    if Path(dataset_mother_folder + 'dataset_metadata_nn.csv').is_file():
        dataset_dataframe_nn = pd.read_csv(dataset_mother_folder + 'dataset_metadata_nn.csv', sep=',')
    else:
        dataset_dataframe_nn = pd.DataFrame({})

    dataset_id = 1
    trajectory_id = 0
    total_simulations = 0
    analyser = DataAnalyser()
    rst = restriction_handler.Restriction(multirotor_model, T_sample, N, M)

    # Choose which trajectories to generate by setting as True
    run_circle_xy = False
    run_circle_xz = False
    run_point = False
    run_lissajous_xy = False
    run_line = False
    run_circle_xy_performance = False
    fault_2rotors = False
    one_example = True
    reasonable_traj_dataset_normal_and_single_failure = False
    single_rotor_combination = False

    restriction_vector = [rst.restriction('normal')]
    restriction_fault = [rst.restriction('total_failure', [0])]
    restriction_fault_2 = [rst.restriction('total_failure', [0,1])]
    restriction_mixed = [rst.restriction('normal'), rst.restriction('total_failure', [0])]

    if run_circle_xy:
        args = tr.generate_circle_xy_trajectories()
        simulate_batch('circle_xy', args, restriction_fault, False)

    if run_circle_xy_performance:
        args = tr.generate_circle_xy_performance_analysis()
        simulate_batch('circle_xy', args, restriction_fault, False, mode='constrained_and_unconstrained_mpc_and_nn',relax_constraint='angle')

    if run_circle_xz:
        args = tr.generate_circle_xz_trajectories()
        simulate_batch('circle_xz', args, restriction_vector, False)

    if run_lissajous_xy:
        args = tr.generate_lissajous_xy_trajectories()
        simulate_batch('lissajous_xy', args, restriction_fault, False)

    if run_line:
        args = tr.generate_line_trajectories(41)
        simulate_batch('line', args, restriction_vector, False, checkpoint_id=21)

    if fault_2rotors:
        restrictions_2failures = rst.restrictions_2_rotor_faults()
        args = [[0, 0, 0, 60]]
        simulate_batch('point_failure', args, restrictions_2failures, disturb_input=False,mode='constrained_mpc_and_nn')

    if single_rotor_combination:
        restrictions_combination_1rf = rst.restrictions_1_rotor_fault()

        args_circle = [[2*np.pi/13, 5, 26]]
        simulate_batch('circle_xy', args_circle, restrictions_combination_1rf, disturb_input = False, mode = 'constrained_mpc_and_nn')

        args = [[2*np.pi/10, 3, 30]]
        simulate_batch('lissajous_xy', args, restrictions_combination_1rf, disturb_input = False,mode='constrained_mpc_and_nn')

    if reasonable_traj_dataset_normal_and_single_failure:
        restriction_mixed = [rst.restriction('normal'), rst.restriction('total_failure', [0])]
        args = tr.generate_circle_xy_trajectories_FIXED()
        simulate_batch('circle_xy', args, restriction_mixed, disturb_input=False, mode='constrained_mpc_and_nn')

        args_lissajous = tr.generate_lissajous_xy_trajectories_FIXED()
        simulate_batch('lissajous_xy', args_lissajous, restriction_mixed, disturb_input=False, mode='constrained_mpc_and_nn')

        args_line = tr.generate_line_trajectories(80)
        simulate_batch('line', args_line, restriction_mixed, disturb_input=False, mode='constrained_mpc_and_nn',)

    # Simulate one example only
    if one_example:
        #restriction_fault = [rst.restriction('total_failure', [0])]
        #args = [[0, 0, -15, 20]]
        #simulate_batch('point', args, restriction_fault, disturb_input = False)

        args = [[2*np.pi/10, 3, 30]]
        simulate_batch('lissajous_xy', args, restriction_fault, disturb_input = False, mode='constrained_mpc_and_nn') 

        #args_circle = [[2*np.pi/13, 5, 26]]
        #args_temp = [[2*np.pi/14, 6.5, 14*1.25]]
        #simulate_batch('circle_xy', args_circle, restriction_vector, disturb_input = False, mode = 'constrained_mpc_and_nn')

        # x, y, z, 
        #args_point = [[0, 0, 0, 15]]
        #simulate_batch('point', args_point, restriction_fault_2, disturb_input = False, mode='constrained_mpc_and_nn')

        #args_line = [[4,4,4, 1000, 25]]
        #simulate_batch('line', args_line, restriction_vector, False)

        #args = [[2*np.pi/25, 2.5, 50]]
        #simulate_batch('lissajous_3d', args, restriction_vector, disturb_input = False, mode = 'constrained_mpc_and_nn')

        #args = [[2*np.pi/10 , 0, 0.25, 0.5, 30]]
        #simulate_batch('helicoidal', args, restriction_fault, disturb_input = True,compare_nns=True)

       # args = [[7, 3, 2*np.pi/100, 10, 100]]
        #simulate_batch('spiral_toroid', args, restriction_vector, disturb_input=False,mode='constrained_mpc_and_nn')

        # Maximum thrust constraint handling ####################
        # trajectory_type='point'
        # z_ref = -np.arange(0.5,40,0.5)
        # args = np.array([np.zeros(len(z_ref)),
        #                  np.zeros(len(z_ref)),
        #                  z_ref,
        #                  2*np.absolute(z_ref)]).transpose()
        
        
        # restriction_test = [rst.restriction('normal')]
        # T_simulation = args[-1]
        # simulate_batch('point', args, restriction_test, False, 'constrained_and_unconstrained_mpc_and_nn', None, ['thrust'])
        ###########################################################


    dataset_dataframe.to_csv(dataset_mother_folder + 'dataset_metadata.csv', sep=',', index=False)
    dataset_dataframe_nn.to_csv(dataset_mother_folder + 'dataset_metadata_nn.csv', sep=',', index=False)