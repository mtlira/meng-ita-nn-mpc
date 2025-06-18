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

#use_optuna_model = True

### MULTIROTOR PARAMETERS ###
from parameters.octorotor_parameters import m, g, I_x, I_y, I_z, l, b, d, num_rotors, thrust_to_weight

### Create model of multirotor ###
multirotor_model = multirotor.Multirotor(m, g, I_x, I_y, I_z, b, l, d, num_rotors, thrust_to_weight)

num_neurons_hidden_layers = 128 # TODO: AUTOMATIZAR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#nn_weights_folder = 'dataset_canon/canon_N_90_M_10_hover_only/global_dataset/'
#weights_file_name = 'model_weights.pth'

### SIMULATION PARAMETERS ###
from parameters.simulation_parameters import time_step, T_sample, N, M, gain_scheduling, include_phi_theta_reference, include_psi_reference
q_neuralnetwork = 3 # Number of MPC outputs (x, y z)
num_inputs = 205 - num_rotors # TODO: AUTOMATIZAR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#T_simulation = 30 #Total simulation time
#t_samples = np.arange(0,T_simulation, T_sample)
#t_samples_extended = np.arange(0,2*T_simulation, T_sample)

# Initial condition
X0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

# Trajectory type
#trajectory_type = 'lissajous_xy'
#disturb_input = False
#use_optuna_model = True

# Input and state values at the equilibrium condition
#u_eq = [m*g, 0, 0, 0]
omega_eq = multirotor_model.get_omega_eq_hover()
omega_squared_eq = omega_eq**2
print('omega_squared_eq',omega_squared_eq)

# Trajectory
#w=2*np.pi*1/10
tr = trajectory_handler.TrajectoryHandler()

# if trajectory_type == 'circle_xy':
#     r_tracking = tr.circle_xy(w, 5, T_simulation,include_psi_reference, include_phi_theta_reference)

# if trajectory_type == 'lissajous_xy':
#     r_tracking = tr.lissajous_xy(w, 2, T_simulation,include_psi_reference, include_phi_theta_reference)

# if trajectory_type == 'circle_xz':
#     r_tracking = tr.circle_xz(w, 5, T_simulation, include_psi_reference, include_phi_theta_reference)

# if trajectory_type == 'point':
#     r_tracking = tr.point(0, 0, 0, T_simulation,include_psi_reference, include_phi_theta_reference)

# if trajectory_type == 'line':
#     r_tracking = tr.line(1, 1, -1, 20, T_simulation, include_psi_reference, include_phi_theta_reference)

#if trajectory_type == 'helicoidal':
#    r_tracking = tr.helicoidal(w,T_simulation, include_psi_reference, include_phi_theta_reference)

def simulate_mpc_nn(X0, multirotor_model, N, M, num_inputs, q_neuralnetwork, omega_squared_eq, dataset_mother_folder, weights_file_name, time_step, T_sample, T_simulation, trajectory, trajectory_type, restriction, restriction_metadata, output_weights, control_weights, \
                    gain_scheduling, disturb_input, num_neurons_hidden_layers, optuna_version, trajectory_metadata = None):
    
    global dataset_dataframe
    global dataset_id
    t_samples = np.arange(0, T_simulation, T_sample)
    analyser = DataAnalyser()
    simulation_save_path = f'{dataset_mother_folder}comparative_simulations/{trajectory_type}/{str(dataset_id)}/'

    simulator = NeuralNetworkSimulator(multirotor_model, N, M, num_inputs, num_rotors, q_neuralnetwork, omega_squared_eq, time_step)

    x_nn, u_nn, omega_nn, nn_metadata = simulator.simulate_neural_network(X0, dataset_mother_folder, weights_file_name, t_samples, trajectory, optuna_version=optuna_version,\
                                num_neurons_hidden_layers=num_neurons_hidden_layers, restriction=restriction)
    
    mpc_success, mpc_metadata, simulation_data = simulate_mpc(X0, time_step, T_sample, T_simulation, trajectory, restriction, output_weights, control_weights, gain_scheduling,\
                                disturb_input=disturb_input, plot=False)
    x_mpc, u_mpc, omega_mpc, _ = simulation_data if simulation_data is not None else [None, None, None, None]


    simulation_metadata = wrap_metadata(dataset_id, trajectory_type, T_simulation, T_sample, N, M, mpc_success, mpc_metadata, restriction_metadata, disturbed_inputs=disturb_input)
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
    
    dataset_dataframe = pd.concat([dataset_dataframe, pd.DataFrame(simulation_metadata)])
    if dataset_id % 1 == 0: dataset_dataframe.to_csv(dataset_mother_folder + 'dataset_metadata.csv', sep=',', index=False)

    legend = ['MPC', 'Neural Network', 'Trajectory'] if x_mpc is not None else ['Neural Network', 'Trajectory']
    if x_nn is not None and x_mpc is not None: analyser.plot_states(x_mpc, t_samples[:np.shape(x_nn)[0]], X_lin=x_nn, trajectory=trajectory[:len(t_samples)], u_vector=[u_mpc, u_nn], omega_vector=[omega_mpc, omega_nn], legend=legend, equal_scales=True, save_path=simulation_save_path, plot=False, pdf=True)
    elif x_mpc is not None: analyser.plot_states(x_mpc, t_samples[:np.shape(x_mpc)[0]], trajectory=trajectory[:len(t_samples)], u_vector=[u_mpc], omega_vector=[omega_mpc], legend=['MPC', 'Trajectory'], equal_scales=True, save_path=simulation_save_path, plot=False, pdf=True)
    elif x_nn is not None:  analyser.plot_states(x_nn, t_samples[:np.shape(x_nn)[0]], trajectory=trajectory[:len(t_samples)], u_vector=[u_nn], omega_vector=[omega_nn], legend=['NN', 'Trajectory'], equal_scales=True, save_path=simulation_save_path, plot=False, pdf=True)
    dataset_id += 1
        
def simulate_batch(trajectory_type, args_vector, restrictions_vector, disturb_input, checkpoint_id = None):
    global dataset_id
    global total_simulations
    global failed_simulations
    global dataset_dataframe
    global dataset_mother_folder
    global weights_file_name

    total_simulations = len(args_vector) * len(restrictions_vector)
    tr = trajectory_handler.TrajectoryHandler()
    for args in args_vector:
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
                simulate_mpc_nn(X0, multirotor_model, N, M, num_inputs, q_neuralnetwork, omega_squared_eq, dataset_mother_folder, weights_file_name, time_step, T_sample, T_simulation, trajectory, trajectory_type, restrictions, restriction_metadata, output_weights, control_weights, gain_scheduling, disturb_input, num_neurons_hidden_layers, optuna_version, trajectory_metadata)

            else:
                dataset_id += 1
    
    # Reset id
    dataset_id = 1

if __name__ == '__main__':
    nn_weights_folder = 'training_results/test_v6_smooth/'
    dataset_mother_folder = nn_weights_folder
    weights_file_name = 'model_weights.pth'
    optuna_version = 'v6_database_v5_smooth_kfold'
    disturb_input = False
    if Path(dataset_mother_folder + 'dataset_metadata.csv').is_file():
        dataset_dataframe = pd.read_csv(dataset_mother_folder + 'dataset_metadata.csv', sep=',')
    else:
        dataset_dataframe = pd.DataFrame({})
    dataset_id = 1
    total_simulations = 0
    analyser = DataAnalyser()
    rst = restriction_handler.Restriction(multirotor_model, T_sample, N, M)

    run_circle_xy = False
    run_circle_xz = False
    run_point = False
    run_lissajous_xy = False
    run_line = False
    run_circle_xy_performance = False
    fault_2rotors = False
    one_example = True

    restriction_vector = [rst.restriction('normal')]
    restriction_mixed = [rst.restriction('normal'), rst.restriction('total_failure', [0])]
    restriction_fault = [rst.restriction('total_failure', [1])]
    restriction_fault_2 = [rst.restriction('total_failure', [0,1])]

    if run_circle_xy:
        args = tr.generate_circle_xy_trajectories()
        simulate_batch('circle_xy', args, restriction_vector, False)

    if run_circle_xy_performance:
        args = tr.generate_circle_xy_performance_analysis()
        simulate_batch('circle_xy', args, restriction_vector, False)

    if run_circle_xz:
        args = tr.generate_circle_xz_trajectories()
        simulate_batch('circle_xz', args, restriction_vector, False)

    if run_lissajous_xy:
        args = tr.generate_lissajous_xy_trajectories()
        simulate_batch('lissajous_xy', args, restriction_vector, False)

    if run_line:
        args = tr.generate_line_trajectories(41)
        simulate_batch('line', args, restriction_vector, False, checkpoint_id=21)

    if fault_2rotors:
        restrictions_2failures = rst.restrictions_2_rotor_faults()
        args = [[0, 0, 0, 20]]
        simulate_batch('point_failure', args, restrictions_2failures, disturb_input=False)


    # Simulate one example only
    if one_example:
        #restriction_fault = [rst.restriction('total_failure', [0])]
        #args = [[0, 0, -15, 20]]
        #simulate_batch('point', args, restriction_fault, disturb_input = False)

        #args = [[2*np.pi/10, 3, 30]]
        #simulate_batch('lissajous_xy', args, restriction_mixed, disturb_input = False)

        #args_circle = [[2*np.pi/10, 5, 30]]
        #simulate_batch('circle_xy', args_circle, restriction_mixed, disturb_input = False)

        # x, y, z, 
        #args_point = [[0, 0, 0, 20]]
        #simulate_batch('point_failure', args_point, restriction_fault_2, disturb_input = False)

        args = [[2*np.pi/25, 2.5, 50]]
        simulate_batch('lissajous_3d', args, restriction_vector, disturb_input = False)

        args = [[2*np.pi/10 , 0, 0.25, 0.5, 30]]
        simulate_batch('helicoidal', args, restriction_vector, disturb_input = False)

    dataset_dataframe.to_csv(dataset_mother_folder + 'dataset_metadata.csv', sep=',', index=False)