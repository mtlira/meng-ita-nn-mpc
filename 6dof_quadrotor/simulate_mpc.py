from scipy.integrate import odeint
from scipy import signal
#from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import sympy as sp
from pid import PID
from plots import *
from linearize import *
import control as ct
import lqr as lqr
import mpc as mpc
import multirotor
import trajectory_handler
from pathlib import Path
from datetime import datetime
from restriction_handler import Restriction
import pandas as pd

### MULTIROTOR PARAMETERS ###
from parameters.octorotor_parameters import m, g, I_x, I_y, I_z, l, b, d, num_rotors, thrust_to_weight

# Simulation trajectory
#trajectory_type = 'circle_xy' # maybe delete

### Create model of multirotor ###
model = multirotor.Multirotor(m, g, I_x, I_y, I_z, b, l, d, num_rotors, thrust_to_weight)

### SIMULATION PARAMETERS ###
from parameters.simulation_parameters import time_step, T_sample, N, M, include_phi_theta_reference, include_psi_reference, gain_scheduling


# Input and state values at the equilibrium condition
u_eq = [m*g, 0, 0, 0]
omega_eq = model.get_omega_eq_hover()
X_eq = np.zeros(12)

# Initial condition
X0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

### Linearization of the model
A, B, C = model.linearize()

tr = trajectory_handler.TrajectoryHandler()
analyser = DataAnalyser()

# Parameters
failed_simulations = 0
total_simulations = 0
dataset_id = 1

simulation_metadata = {
    'sim_id': [],
    'trajectory_type': [],
    'disturbed_inputs': [],
    'simulation_time (s)': [],
    'time_sample (s)': [],
    'N': [],
    'M': [],
    'success': [],
    'RMSe': [],
    'execution_time (s)': [],
    'operation_mode': [],
    'failed_rotors': [],
    'ang_speed_percentages (%)': [],
    'min_phi (rad)': [],
    'max_phi (rad)': [],
    'mean_phi (rad)': [],
    'std_phi (rad)': [],
    'min_theta (rad)': [],
    'max_theta (rad)': [],
    'mean_theta (rad)': [],
    'std_theta (rad)': [],
    'min_psi (rad)': [],
    'max_psi (rad)': [],
    'mean_psi (rad)': [],
    'std_psi (rad)': [],
}
simulation_metadata = {}
dataset_dataframe = pd.DataFrame(simulation_metadata)
#dataset_dataframe['success'] = dataset_dataframe['success'].astype(bool)
#dataset_dataframe['disturbed_inputs'] = dataset_dataframe['disturbed_inputs'].astype(bool)

# MPC Implementation

def simulate_mpc(X0, time_step, T_sample, T_simulation, trajectory, restrictions, output_weights, control_weights,\
                  gain_scheduling, dataset_name=None, folder_name=None, disturb_input=False, plot=True):
    '''
    Executes a control simulation with MPC given the time step, time sample, total simulation time and the desired trajectory.\n
    dataset_name: name of dataset folder the current simulation will belong to\n
    folder_name: name of the folder that will store the simulation results\n
    (obs: the folder hierarchy is dataset_name/folder_name/)
    '''
    if dataset_name is None or folder_name is None: generate_dataset = False
    else: generate_dataset = True

    t_samples = np.arange(0,T_simulation, T_sample)

    Bw = B @ model.Gama
    try:
        phi_grid = np.array([0])
        theta_grid = np.array([0])
        gain_MPC = mpc.GainSchedulingMPC(model, phi_grid, theta_grid, M, N, time_step, T_sample, output_weights, \
            control_weights, restrictions, include_psi_reference, include_phi_theta_reference)
        x_mpc_rotors, u_rotors, omega_vector, NN_dataset, metadata = \
            gain_MPC.simulate_future_rotors(model, X0, t_samples, trajectory, gain_scheduling, generate_dataset=generate_dataset, \
                                            disturb_input=disturb_input)
                    
    except Exception as error:
        x_mpc_rotors, u_rotors, omega_vector, NN_dataset, metadata = None, None, None, None, None
        print('exception:', error)

    simulation_data = (x_mpc_rotors, u_rotors, omega_vector, NN_dataset)
    if x_mpc_rotors is not None:
        save_path = 'simulations/{}/{}/'.format(dataset_name, folder_name) if dataset_name is not None else None
        #now = datetime.now()
        #current_time = now.strftime("%m_%d_%Hh-%Mm")
        if dataset_name is not None:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            #np.savetxt(save_path + "dataset.csv", NN_dataset, delimiter=",")
            np.save(save_path + 'dataset.npy', NN_dataset.astype(np.float32))
        analyser.plot_states(x_mpc_rotors, t_samples[:np.shape(x_mpc_rotors)[0]], trajectory=trajectory, u_vector=u_rotors, omega_vector=omega_vector, equal_scales=True, legend=['MPC', 'Trajectory'], save_path=save_path, plot=plot)
        return True, metadata, simulation_data
    return False, None, None

def simulate_batch(trajectory_type, args_vector, restrictions_vector, simulate_disturbances, dataset_save_path, checkpoint_id = None):
    # 3. Simulation of POINT trajectories
    global dataset_id
    global total_simulations
    global failed_simulations
    global dataset_dataframe

    total_simulations = len(args_vector) * len(restrictions_vector)
    if simulate_disturbances: total_simulations *= 2
    tr = trajectory_handler.TrajectoryHandler()
    for args in args_vector:
        for restrictions, output_weights, control_weights, restrictions_metadata in restrictions_vector:
            if checkpoint_id is None or dataset_id >= checkpoint_id:
                trajectory = tr.generate_trajectory(trajectory_type, args, include_psi_reference, include_phi_theta_reference)
                folder_name = f'{trajectory_type}/' + str(dataset_id)

                # Simulation without disturbances
                print(f'{trajectory_type} Simulation {dataset_id}/{total_simulations}')
                T_simulation = args[-1]
                simulation_success, simulation_metadata, _ = simulate_mpc(X0, time_step, T_sample, T_simulation, trajectory, \
                    restrictions, output_weights, control_weights, gain_scheduling, dataset_name, folder_name, disturb_input = simulate_disturbances)

                #if not simulation_success:
                    #restrictions_hover = restrictions.copy()
                    #restrictions_hover['y_max'][3:5] /= 2
                    #restrictions_hover['y_min'] = -restrictions_hover['y_max'][3:5]
                #    output_weights_hover = np.copy(output_weights)
                #    output_weights_hover[3:5] *= 4 # Divide delta_y_max by 2 for phi and theta
                #    simulation_success, simulation_metadata, _ = simulate_mpc(X0, time_step, T_sample, T_simulation, trajectory, restrictions, output_weights_hover, control_weights, dataset_name, folder_name, disturb_input = False, gain_scheduling=True)

                simulation_metadata = wrap_metadata(dataset_id, trajectory_type, T_simulation, T_sample, N, M, simulation_success, \
                                                    simulation_metadata, restrictions_metadata, False)

                if not simulation_success: failed_simulations += 1
                dataset_id += 1
                folder_name = f'{trajectory_type}/' + str(dataset_id)
                dataset_dataframe = pd.concat([dataset_dataframe, simulation_metadata])

                # Simulation with disturbances
                # if simulate_disturbances:
                #     print(f'{trajectory_type} Simulation {dataset_id}/{total_simulations}')
                #     simulation_success, simulation_metadata, _ = \
                #         simulate_mpc(X0, time_step, T_sample, T_simulation, trajectory, restrictions, output_weights, \
                #                      control_weights, gain_scheduling, dataset_name, folder_name, disturb_input = True)
                    
                #     #if not simulation_success:
                #     #    #restrictions_hover = restrictions.copy()
                #     #    #restrictions_hover['y_max'][3:5] /= 2
                #     #    #restrictions_hover['y_min'] = -restrictions_hover['y_max'][3:5]
                #     #    output_weights_hover = np.copy(output_weights)
                #     #    output_weights_hover[3:5] *= 4 # Divide delta_y_max by 2 for phi and theta
                #     #    simulation_success, simulation_metadata = simulate_mpc(X0, time_step, T_sample, T_simulation, trajectory, restrictions, output_weights_hover, control_weights, dataset_name, folder_name, disturb_input = True)


                #     simulation_metadata = wrap_metadata(dataset_id, trajectory_type, T_simulation, T_sample, N, M, simulation_success, simulation_metadata, \
                #         restrictions_metadata, True)

                #     if not simulation_success: failed_simulations += 1
                #     dataset_id += 1
                #     dataset_dataframe = pd.concat([dataset_dataframe, simulation_metadata])
                if int(dataset_id) % 2 <= 1:
                    dataset_dataframe.to_csv(dataset_save_path, sep=',', index = False)
            else:
                dataset_id += 1
                if simulate_disturbances: dataset_id += 1
    # Reset_id
    dataset_id = 1

def generate_dataset(dataset_name = None):
    if dataset_name is None:
        now = datetime.now()
        current_time = now.strftime("%m_%d_%Hh-%Mm")
        dataset_name = current_time
    dataset_save_path = f'simulations/{dataset_name}/dataset_metadata.csv'
    
    generate_point = False
    generate_circle_xy = False
    generate_circle_xz = False
    generate_line = False
    generate_lissajous_xy = True
    simulate_fault_tolerance = False
    run_circle_xy_performance = False
    
    rst = Restriction(model, T_sample, N, M)

    global dataset_dataframe
    
    # 1. Restrictions vector
    restrictions_performance = rst.restrictions_performance()
    restrictions_fault_tolerance = rst.restrictions_fault_tolerance()
    restriction_normal = [rst.restriction('normal')]

    # 2. Generation of trajectory batches and simulation of batches
    if generate_point:
        T_simulation_point = 10
        num_points = 45
        points_args = tr.generate_point_trajectories(num_points, T_simulation_point)
        simulate_batch('point', points_args, restrictions_performance, simulate_disturbances = True,dataset_save_path=dataset_save_path,checkpoint_id=10)
    
    if generate_circle_xy:
        circle_xy_args = tr.generate_circle_xy_trajectories()
        simulate_batch('circle_xy', circle_xy_args, restrictions_performance, simulate_disturbances = True, dataset_save_path=dataset_save_path)

    if generate_line:
        num_lines = 10
        line_args = tr.generate_line_trajectories(num_lines)
        simulate_batch('line', line_args, restrictions_performance, simulate_disturbances = True, dataset_save_path=dataset_save_path, checkpoint_id=19)

    if generate_circle_xz:
        circle_xz_args = tr.generate_circle_xz_trajectories()
        simulate_batch('circle_xz', circle_xz_args, restrictions_performance, simulate_disturbances = True, dataset_save_path=dataset_save_path)

    if generate_lissajous_xy:
        lissajous_xy_args = tr.generate_lissajous_xy_trajectories()
        simulate_batch('lissajous_xy', lissajous_xy_args, restriction_normal, simulate_disturbances = True, dataset_save_path=dataset_save_path)

    if simulate_fault_tolerance:
        fault_tolerance_args = [[0, 0, 0, 20], [0,0,1,20],[0,0,-1, 20], [0, 0.5, 0.5, 20]]
        simulate_batch('point_failure', fault_tolerance_args, restrictions_fault_tolerance, simulate_disturbances = True, dataset_save_path = dataset_save_path, checkpoint_id=6)

    if run_circle_xy_performance:
        restriction_vector = [rst.restriction('normal')]
        args = tr.generate_circle_xy_performance_analysis()
        simulate_batch('circle_xy', args, restriction_vector, True, dataset_save_path)

    # Save dataset
    if len(dataset_dataframe) > 2:
        dataset_dataframe.to_csv(dataset_save_path, sep=',', index = False)
    
    print(f'Failed simulations: {failed_simulations}/{total_simulations}') # TODO: deixar mais generico (nao so pontos)

def wrap_metadata(dataset_id, trajectory_type, T_simulation, T_sample, N, M, simulation_success, simulation_metadata, \
                  restrictions_metadata, disturbed_inputs, trajectory_id):
    simulation_metadata = {
        'sim_id': dataset_id,
        'trajectory_id': trajectory_id,
        'trajectory_type': trajectory_type,
        'disturbed_inputs': disturbed_inputs,
        'simulation_time (s)': T_simulation,
        'time_sample (s)': T_sample,
        'N': N,
        'M': M,
        'mpc_success': simulation_success,
        'num_iterations': simulation_metadata['num_iterations'] if simulation_success else 'nan',   
        'mpc_RMSe': simulation_metadata['mpc_RMSe'] if simulation_success else 'nan',
        'mpc_execution_time (s)': simulation_metadata['mpc_execution_time'] if simulation_success else 'nan',
        'operation_mode': restrictions_metadata['operation_mode'],
        'failed_rotors': restrictions_metadata['failed_rotors'],
        'ang_speed_percentages (%)': restrictions_metadata['ang_speed_percentages'],
        'mpc_min_phi (rad)': simulation_metadata['mpc_min_phi'] if simulation_success else 'nan',
        'mpc_max_phi (rad)': simulation_metadata['mpc_max_phi'] if simulation_success else 'nan',
        'mpc_mean_phi (rad)': simulation_metadata['mpc_mean_phi'] if simulation_success else 'nan',
        'mpc_std_phi (rad)': simulation_metadata['mpc_std_phi'] if simulation_success else 'nan',
        'mpc_min_theta (rad)': simulation_metadata['mpc_min_theta'] if simulation_success else 'nan',
        'mpc_max_theta (rad)': simulation_metadata['mpc_max_theta'] if simulation_success else 'nan',
        'mpc_mean_theta (rad)': simulation_metadata['mpc_mean_theta'] if simulation_success else 'nan',
        'mpc_std_theta (rad)': simulation_metadata['mpc_std_theta'] if simulation_success else 'nan',
        'mpc_min_psi (rad)': simulation_metadata['mpc_min_psi'] if simulation_success else 'nan',
        'mpc_max_psi (rad)': simulation_metadata['mpc_max_psi'] if simulation_success else 'nan',
        'mpc_mean_psi (rad)': simulation_metadata['mpc_mean_psi'] if simulation_success else 'nan',
        'mpc_std_psi (rad)': simulation_metadata['mpc_std_psi'] if simulation_success else 'nan',
        'mpc_max_omega0': simulation_metadata['mpc_max_omega0'] if simulation_success else 'nan',
        'mpc_max_omega1': simulation_metadata['mpc_max_omega1'] if simulation_success else 'nan',
        'mpc_max_omega2': simulation_metadata['mpc_max_omega2'] if simulation_success else 'nan',
        'mpc_max_omega3': simulation_metadata['mpc_max_omega3'] if simulation_success else 'nan',
        'mpc_max_omega4': simulation_metadata['mpc_max_omega4'] if simulation_success else 'nan',
        'mpc_max_omega5': simulation_metadata['mpc_max_omega5'] if simulation_success else 'nan',
        'mpc_max_omega6': simulation_metadata['mpc_max_omega6'] if simulation_success else 'nan',
        'mpc_max_omega7': simulation_metadata['mpc_max_omega7'] if simulation_success else 'nan',
    }
    simulation_metadata = pd.DataFrame([simulation_metadata])
    return simulation_metadata

if __name__ == '__main__':
    try:
        now = datetime.now()
        current_time = now.strftime("%m_%d_%Hh-%Mm")
        dataset_name = current_time
        generate_dataset(dataset_name)
    except Exception as error:
        print('There was an error:', error)
        save_path = f'simulations/{dataset_name}/'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        dataset_dataframe.to_csv(save_path + '/dataset_metadata.csv', sep=',', index = False)
 