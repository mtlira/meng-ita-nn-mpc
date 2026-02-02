import os
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from plots import DataAnalyser
import time
from scipy.integrate import odeint
import pickle
import multirotor
from mpc import add_input_disturbance
from parameters.octorotor_parameters import num_rotors
from parameters.octorotor_parameters import m, g, I_x, I_y, I_z, l, b, d, thrust_to_weight, num_rotors
from parameters.simulation_parameters import T_sample

model = multirotor.Multirotor(m, g, I_x, I_y, I_z, b, l, d, num_rotors, thrust_to_weight)

### 1. Dataset class ###

class ControlAllocationDataset(Dataset):
    '''Class to be used if the dataset is a single CSV file'''
    def __init__(self, dataset_path, num_outputs, transform=None):
        self.dataframe = pd.read_csv(dataset_path, header = None).astype('float32')
        self.transform = transform
        self.num_outputs = num_rotors
        self.num_inputs = np.shape(self.dataframe)[1] - num_outputs
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input = self.dataframe.iloc[idx, 0:self.num_inputs]
        output = self.dataframe.iloc[idx, self.num_inputs:]

        sample = {'input': np.array(input), 'output': np.array(output)}

        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample

class ControlAllocationDataset_Binary(Dataset):
    '''Class to be used if the total dataset is split into multiple npy files\n
    mother_folder_path: Path of the folder that contains all the npy files that make up the total dataset'''
    def __init__(self, mother_folder_path, has_header, num_outputs):
        self.normalization_file_name = 'normalization_data.csv'
        self.mother_folder_path = mother_folder_path
        self.has_header = has_header
        self.num_outputs = num_outputs
        # Creating list of all the csv dataset paths
        #self.npy_files = []
        self.dataset = []
        print('Loading dataset indexes...')
        for subdir, _, files in os.walk(self.mother_folder_path):
            for file in files:
                if file == 'dataset.npy':
                    #self.npy_files.append(os.path.join(subdir, file))
                    self.dataset.append(np.load(os.path.join(subdir, file)))
        self.dataset = np.concatenate(self.dataset, axis = 0)
        self.num_inputs = len(self.dataset[0]) - num_outputs
        
        print(f'\tLoaded {len(self.dataset)} samples')
        print(f'\tSample length: {len(self.dataset[0])}')
        print(f'\tDataset size: {self.dataset.nbytes / 1024**2} MB')
        print('Interity check:')
        for i in range(len(self.dataset)):
            if len(self.dataset[i]) != len(self.dataset[0]):
                print(f'corrupted row: i={i}, {len(self.dataset[i])} samples')
        print('\t There are no corrupted data')

        self.normalize()
        self.dataset = self.dataset.astype(np.float32)

    def get_dataset(self):
        return self.dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        input = self.dataset[idx, :self.num_inputs]
        output = self.dataset[idx, self.num_inputs:]
        dict = {'input': torch.tensor(input, dtype=torch.float32),
                'output': torch.tensor(output, dtype=torch.float32)}
        return dict
    
    def normalize(self):
        # TEMP - REMOVER DEPOIS
        #self.dataset[:,196:] -= model.get_omega_eq_hover()[0]**2

        print('Normalizing the Dataset...')
        if not os.path.isfile(self.mother_folder_path + self.normalization_file_name):
            self.mean = np.mean(self.dataset, axis = 0)
            self.std = np.std(self.dataset, axis = 0)
            data = np.concatenate(([self.mean], [self.std]), axis = 0)
            np.savetxt(self.mother_folder_path + 'normalization_data.csv', data, delimiter=",")
        
        else:
            print('\tNormalization dataset file already exists - loading mean and std')
            normalization_df = pd.read_csv(self.mother_folder_path + 'normalization_data.csv', header = None)
            self.mean = np.array([normalization_df.iloc[0, :]])
            self.std = np.array([normalization_df.iloc[1, :]])

        self.dataset = (self.dataset - self.mean) / self.std

class ControlAllocationDataset_Binary_Short(Dataset):
    '''Class to be used if the total dataset is split into multiple CSV files\n
    mother_folder_path: Path of the folder that contains all the CSV files that make up the total dataset'''
    def __init__(self, dataset: np.ndarray, num_outputs):
        self.dataset = dataset.astype(np.float32)
        print('Dataset type:', type(dataset), type(dataset[0]), type(dataset[0][0]))

        self.num_outputs = num_outputs
        self.num_inputs = len(self.dataset[0]) - num_outputs

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        input = self.dataset[idx, :self.num_inputs]
        output = self.dataset[idx, self.num_inputs:]
        dict = {'input': torch.from_numpy(input),
                'output': torch.from_numpy(output)}
        return dict

class ControlAllocationDataset_Split(Dataset):
    '''Class to be used if the total dataset is split into multiple CSV files\n
    mother_folder_path: Path of the folder that contains all the CSV files that make up the total dataset'''
    def __init__(self, mother_folder_path, has_header, num_outputs):
        self.normalization_file_name = 'normalization_data.csv'
        self.mother_folder_path = mother_folder_path
        self.has_header = has_header
        self.num_outputs = num_outputs
        # Creating list of all the csv dataset paths
        self.csv_files = []
        print('Loading dataset indexes...')
        for subdir, _, files in os.walk(self.mother_folder_path):
            for file in files:
                if 'dataset.csv' in file and 'metadata' not in file and 'normalization' not in file and '04_21_21h-19m' in subdir: # Skips metadata datasets
                    self.csv_files.append(os.path.join(subdir, file))

        
        self.file_index = [] # Maps global idx to file + local row index
        for file_path in self.csv_files:
            n_rows = sum(1 for _ in open(file_path))
            if has_header: n_rows -= 1
            self.file_index.extend([(file_path, i) for i in range(n_rows)])
        
        self.num_inputs = np.shape(pd.read_csv(self.csv_files[0], skiprows = 0, nrows = 1, header = 0 if has_header else None))[1] - self.num_outputs
        print('\tNumber of samples:', len(self.file_index))
        print('\tNumber of inputs:',self.num_inputs)
        print('\tNumber of outputs',self.num_outputs)
        
        if not os.path.isfile(mother_folder_path + self.normalization_file_name):
            self.normalize()
        
        normalized_data = pd.read_csv(mother_folder_path + self.normalization_file_name, sep = ',', header = 0 if has_header else None)
        self.mean = normalized_data.iloc[0, :].values.squeeze()
        self.std = normalized_data.iloc[1, :].values.squeeze()
    def __len__(self):
        return len(self.file_index)
    
    def __getitem__(self, idx):
        '''Given a global idx (i.e, regarding the global dataset as a whole), get a sample. Performs the translation from the global idx to the individual file and row the sample is located'''
        file_path, row_idx = self.file_index[idx]
        skiprows = row_idx
        if self.has_header:
            skiprows += 1
        # Read just the specific row
        df = pd.read_csv(file_path, skiprows = skiprows, nrows = 1, header = 0 if self.has_header else None)
        data = df.values.squeeze() # Converts DataFrame to array

        input_normalized = (data[:self.num_inputs] - self.mean[:self.num_inputs]) / self.std[:self.num_inputs]
        output_normalized = (data[self.num_inputs:] - self.mean[self.num_inputs:]) / self.std[self.num_inputs:]

        sample = {
            'input': torch.tensor(input_normalized, dtype=torch.float32),
            'output': torch.tensor(output_normalized, dtype=torch.float32)
            }
        return sample
    
    def normalize(self):
        print('Normalizing the Dataset')
        sum = np.zeros(self.num_inputs + self.num_outputs)
        sum_squared = np.zeros(self.num_inputs + self.num_outputs)
        num_samples = 0
        delta_squared = 0

        # Mean
        for file_path in self.csv_files:
            df = pd.read_csv(file_path, header = 0 if self.has_header else None)
            num_samples += len(df)
            sum += df.sum(axis = 0)
        
        mean = (sum / num_samples)

        # Std
        for file_path in self.csv_files:
            df = pd.read_csv(file_path, header = 0 if self.has_header else None)
            delta_squared += ((df - mean)**2).sum(axis = 0)
            #sum_squared += (df**2).sum()
        
        std = np.sqrt(delta_squared / (num_samples-1))
        #std = np.sqrt(sum_squared/num_samples - mean**2)
        norm_data = pd.concat([mean, std], ignore_index = True, axis = 1).T
        norm_data.to_csv(self.mother_folder_path + self.normalization_file_name, header = True if self.has_header else False, index = False)

        return mean, std


### 2. Neural Network class ###
class NeuralNetwork(nn.Module):
    def __init__(self, version:str, num_inputs:int, num_outputs:int):
        super().__init__()
        # if version == 'no_optuna':
        #     self.layer_stack = nn.Sequential(
        #         nn.Linear(in_features = num_inputs, out_features = num_hidden_layers),
        #         nn.LeakyReLU(negative_slope=0.01), # inplace ?
        #         nn.Linear(in_features = num_hidden_layers, out_features = num_hidden_layers),
        #         nn.LeakyReLU(negative_slope=0.01),
        #         nn.Linear(in_features=num_hidden_layers, out_features = num_outputs)
        #     )
        if version == 'v0':
            self.layer_stack = nn.Sequential(
                nn.Linear(in_features = num_inputs, out_features = 128),
                nn.LeakyReLU(negative_slope=0.04242861410346148), # before: 0.01
                #nn.Dropout(0.09382298344626222),
                nn.Linear(in_features = 128, out_features = 64),
                nn.LeakyReLU(negative_slope=0.04242861410346148),
                #nn.Dropout(0.21326313772325148),
                nn.Linear(in_features=64, out_features = num_outputs)
            )
        elif version == 'v1':
            self.layer_stack = nn.Sequential(
                nn.Linear(in_features = num_inputs, out_features = 1058),
                nn.LeakyReLU(negative_slope=0.010027561298), # before: 0.01
                #nn.Dropout(0.09382298344626222),
                nn.Linear(in_features = 1058, out_features = 1145),
                nn.LeakyReLU(negative_slope=0.010027561298),
                #nn.Dropout(0.21326313772325148),
                nn.Linear(in_features=1145, out_features = num_outputs)
            )
        elif version == 'v2':
            self.layer_stack = nn.Sequential(
                nn.Linear(in_features = num_inputs, out_features = 1276),
                nn.LeakyReLU(negative_slope=0.036692291), # before: 0.01
                #nn.Dropout(0.09382298344626222),
                nn.Linear(in_features = 1276, out_features = 482),
                nn.LeakyReLU(negative_slope=0.036692291),
                #nn.Dropout(0.21326313772325148),
                nn.Linear(in_features = 482, out_features = 77),
                nn.LeakyReLU(negative_slope=0.036692291), # before: 0.01
                nn.Linear(in_features=77, out_features = num_outputs)
            )
            self.optimizer = 'RMSprop'
            self.opt_leaning_rate = 0.0001397094036
            self.l2_lambda = 7.55128976623e-5

        elif version == 'v3':
            self.layer_stack = nn.Sequential(
                nn.Linear(in_features = num_inputs, out_features = 1363),
                nn.LeakyReLU(negative_slope=0.0108732857), # before: 0.01
                #nn.Dropout(0.09382298344626222),
                nn.Linear(in_features = 1363, out_features = 1205),
                nn.LeakyReLU(negative_slope=0.0108732857),
                #nn.Dropout(0.21326313772325148),
                nn.Linear(in_features = 1205, out_features = 252),
                nn.LeakyReLU(negative_slope=0.0108732857),
                nn.Linear(in_features = 252, out_features = 1502),
                nn.LeakyReLU(negative_slope=0.0108732857), # before: 0.01
                nn.Linear(in_features=1502, out_features = num_outputs)
            )
            self.optimizer = 'RMSprop'
            self.opt_leaning_rate = 0.0001001708
            self.l2_lambda = 1.49497837616e-6
        
        elif version == 'v4':
            self.layer_stack = nn.Sequential(
                nn.Linear(in_features = num_inputs, out_features = 1276),
                nn.LeakyReLU(negative_slope=0.036692291), # before: 0.01
                #nn.Dropout(0.09382298344626222),
                nn.Linear(in_features = 1276, out_features = 482),
                nn.LeakyReLU(negative_slope=0.036692291),
                #nn.Dropout(0.21326313772325148),
                nn.Linear(in_features = 482, out_features = 77),
                nn.LeakyReLU(negative_slope=0.036692291), # before: 0.01
                nn.Linear(in_features=77, out_features = num_outputs)
            )
            self.optimizer = 'RMSprop'
            self.opt_leaning_rate = 0.0001397094036
            self.l2_lambda = 7.55128976623e-5
        
        elif version == 'v6_database_v5_smooth_kfold':
            n_layer0, n_layer1, n_layer2, n_layer3 = (665, 507, 1104, 1953)
            lrelu_negative_slope = 0.08473559984569866
            self.layer_stack = nn.Sequential(
                nn.Linear(in_features = num_inputs, out_features = n_layer0),
                nn.LeakyReLU(negative_slope=lrelu_negative_slope),
                nn.Linear(in_features = n_layer0, out_features = n_layer1),
                nn.LeakyReLU(negative_slope=lrelu_negative_slope),
                nn.Linear(in_features = n_layer1, out_features = n_layer2),
                nn.LeakyReLU(negative_slope=lrelu_negative_slope),
                nn.Linear(in_features=n_layer2, out_features = n_layer3),
                nn.LeakyReLU(negative_slope=lrelu_negative_slope),
                nn.Linear(in_features=n_layer3, out_features = num_outputs),
            )
            self.optimizer = 'RMSprop'
            self.opt_leaning_rate = 0.00010277806356991367
            self.l2_lambda = 5.9126412923258794e-05
        else:
            raise Exception('Neural network version incorrect.')

    def forward(self, x):
        logits = self.layer_stack(x)
        return logits

### 3. Early Stopper class ###
class EarlyStopper():
    def __init__(self, drift_patience, plateau_patience, drift_percentage, save_path=None):
        self.drift_patience = drift_patience
        self.plateau_patience = plateau_patience
        self.drift_percentage = drift_percentage
        self.minimum_loss = np.inf
        self.tolerance_break_counter = 0
        self.plateau_counter = 0 # counts how many consecutive losses remain within the range of minimum_loss and minimum_loss*(1 + drift_percentage)
        self.save_path = save_path

    def stop_early(self, loss, model):
        if loss < self.minimum_loss:
            # Reset counters
            self.tolerance_break_counter = 0 
            self.plateau_counter = 0

            # Update loss
            self.minimum_loss = loss

            # Save model's weights in save_path
            if self.save_path is not None:
                torch.save(model.state_dict(), self.save_path + 'model_weights.pth')
            
        elif loss > self.minimum_loss*(1 + self.drift_percentage):
            self.tolerance_break_counter += 1
        
        else:
            self.plateau_counter += 1

        if self.tolerance_break_counter > self.drift_patience or self.plateau_counter > self.plateau_patience:
            return True
        else:
            return False

class NeuralNetworkSimulator(object):
    def __init__(self, model, N, M, num_inputs, num_rotors, q_eff, u_ref, time_step):
        self.model = model
        self.N = N
        self.M = M
        self.num_inputs = num_inputs
        self.num_rotors = num_rotors
        self.q_eff = q_eff # Number of effective reference coordinates for the neural network: 3 (x, y, z)
        self.u_ref = u_ref # input around which the model was linearized (omega_squared_eq)
        self.time_step = time_step

    def simulate_neural_network(self, X0, nn_weights_folder, file_name, t_samples, trajectory, optuna_version, restriction, disturb_input, clip:bool):
        analyser = DataAnalyser()
        nn_weights_path = nn_weights_folder + file_name
        omega_squared_eq = self.u_ref

        # Disturb logic (state disturbance)
        disturb_frequency = 0.1

        # 1. Load Neural Network model
        #device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        #print(f"Using {device} device")
        
        nn_model = NeuralNetwork(optuna_version,self.num_inputs, num_rotors).to('cpu')
        nn_model.load_state_dict(torch.load(nn_weights_path, weights_only=True, map_location=torch.device('cpu')))
        nn_model.eval()

        x_k = X0

        X_vector = [X0]
        u_vector = []
        omega_vector = []
        omega_squared_vector = []
            
        normalization_df = pd.read_csv(nn_weights_folder + 'normalization_data.csv', header = None)
        omega_max = restriction['u_max'] + omega_squared_eq
        #clip_max_omega = np.copy(max(omega_max)*np.ones(num_rotors))
        clip_max_omega = np.array([0 if omega < 0.1 else omega for omega in omega_max])
        failed_rotors = [{'indice': i, 'value': max(omega_squared_eq), 'reached_zero': False} for i, omega in enumerate(omega_max) if omega < 0.01]
        omega_squared_previous = np.copy(omega_squared_eq)
        for i in range(len(clip_max_omega)):
            if clip_max_omega[i] == 0: omega_squared_previous[i] = 0


        ## DEBUG (REMOVE LATER) ##
        #min_omega_squared = np.inf
        ## DEBUG (REMOVE LATER) ##

        # TEMP - LIMITAÇÂO DE DELTA U#########################################################
        omega_eq = self.model.get_omega_eq_hover()
        alpha = self.model.angular_acceleration
        delta_u_max = (2*omega_eq + alpha*T_sample)*alpha * T_sample
        alpha = -alpha
        delta_u_min = (2*omega_eq + alpha*T_sample)*alpha * T_sample
        ####################################################################################

        mean_input = normalization_df.iloc[0, :self.num_inputs].to_numpy()
        std_input = normalization_df.iloc[1, :self.num_inputs].to_numpy()

        mean_output = normalization_df.iloc[0, self.num_inputs:].to_numpy()
        std_output = normalization_df.iloc[1, self.num_inputs:].to_numpy()
        
        reached_zero = False

        # Control loop
        execution_time = 0
        waste_time = 0
        start_time = time.perf_counter()

        for k in range(0, len(t_samples)-1): # TODO: confirmar se é -1 mesmo:
            # Mount input tensor to feed NN
            nn_input = np.array([])

            ref_N = trajectory[k:k+self.N, 0:3].reshape(-1) # TODO validar se termina em k+N-1 ou em k+N
            if np.shape(ref_N)[0] < self.q_eff*self.N:
                #print('kpi',N - int(np.shape(ref_N)[0]/q))
                ref_N = np.concatenate((ref_N, np.tile(trajectory[-1, :3].reshape(-1), self.N - int(np.shape(ref_N)[0]/self.q_eff))), axis = 0) # padding de trajectory[-1] em ref_N quando trajectory[k+N] ultrapassa ultimo elemento

            # Calculating reference values relative to multirotor's current position at instant k
            position_k = np.tile(x_k[9:], self.N).reshape(-1)
            ref_N_relative = ref_N - position_k

            # Clarification: u is actually (u - ueq) and delta_u is (u-ueq)[k] - (u-ueq)[k-1] in this MPC formulation (i.e., u is in reference to u_eq, not 0)
            nn_input = np.concatenate((nn_input, x_k[0:9], ref_N_relative, restriction['u_max'] + omega_squared_eq), axis = 0)

            # Normalization of the input
            #for i_column in range(num_inputs):
            #    mean = normalization_df.iloc[0, i_column]
            #    std = normalization_df.iloc[1, i_column]
            #    nn_input[i_column] = (nn_input[i_column] - mean)/std
            #mean_input = normalization_df.iloc[0, :self.num_inputs]
            #std_input = normalization_df.iloc[1, :self.num_inputs]
            nn_input = np.array((nn_input - mean_input) / std_input)

            nn_input = nn_input.astype('float32')

            # Get NN output
            omega_squared = nn_model(torch.from_numpy(nn_input)).detach().numpy()

            # De-normalization of the output
            #for i_output in range(num_outputs):
            #    mean = normalization_df.iloc[0, num_inputs + i_output]
            #    std = normalization_df.iloc[1, num_inputs + i_output]
            #    delta_omega_squared[i_output] = mean + std*delta_omega_squared[i_output]
            #mean = normalization_df.iloc[0, self.num_inputs:]
            #std = normalization_df.iloc[1, self.num_inputs:]
            omega_squared = (mean_output + std_output*omega_squared)

            ## DEBUG (REMOVE LATER) ##
            #debug_omega_squared = omega_squared_eq + delta_omega_squared
            #if np.min(debug_omega_squared) < min_omega_squared:
            #    min_omega_squared = np.min(debug_omega_squared)
            ## DEBUG (REMOVE LATER) ##

            # Applying multirotor restrictions
            # Fixing infinitesimal values out that violate the constraints
            #TEMP - LIMITACAO DE DELTA OMEGA###############
            ###############################################
            if clip:
                omega_squared = np.clip(omega_squared, np.max([omega_squared_previous + delta_u_min, np.zeros(num_rotors)],axis=0), np.min([omega_squared_previous + delta_u_max, clip_max_omega],axis=0)) # Safe
                
                # for rotor in failed_rotors:
                #     if not rotor['reached_zero']:
                #         rotor['value'] += delta_u_min[0]
                #         if rotor['value'] <= 0.01:
                #             rotor['reached_zero'] = True
                #             rotor['value'] = 0
                #         omega_squared[rotor['indice']] = rotor['value']
                #     else:
                #         omega_squared[rotor['indice']] = 0
                            
            omega_squared_previous = np.copy(omega_squared)

            # omega**2 --> u
            #print('omega_squared',omega_squared)
            u_k = self.model.Gama @ (omega_squared)

            # Apply INPUT disturbance if enabled
            if disturb_input and k>0:
                waste_start_time = time.perf_counter()
                probability = np.random.rand()
                if probability > disturb_frequency:
                    u_k = add_input_disturbance(u_k, model)
                waste_end_time = time.perf_counter()
                waste_time += waste_end_time - waste_start_time

            f_t_k, t_x_k, t_y_k, t_z_k = u_k # Attention for u_eq (solved the problem)
            t_simulation = np.arange(t_samples[k], t_samples[k+1], self.time_step)

            # Update plant control (update x_k)
            # x[k+1] = f(x[k], u[k])
            x_k = odeint(self.model.f2, x_k, t_simulation, args = (f_t_k, t_x_k, t_y_k, t_z_k))
            x_k = x_k[-1]

            if np.linalg.norm(x_k[9:12] - trajectory[k, :3]) > 60: #or np.max(np.abs(x_k[0:2])) > 1.75:
                print('Simulation exploded.')
                print(f'x_{k} =',x_k)

                metadata = {
                'nn_success': False,
                'num_iterations': len(t_samples)-1,    
                'nn_execution_time (s)': execution_time,
                'nn_RMSe': 'nan',
                'nn_min_phi (rad)': 'nan',
                'nn_max_phi (rad)': 'nan',
                'nn_mean_phi (rad)': 'nan',
                'nn_std_phi (rad)': 'nan',
                'nn_min_theta (rad)': 'nan',
                'nn_max_theta (rad)': 'nan',
                'nn_mean_theta (rad)': 'nan',
                'nn_std_theta (rad)': 'nan',
                'nn_min_psi (rad)': 'nan',
                'nn_max_psi (rad)': 'nan',
                'nn_mean_psi (rad)': 'nan',
                'nn_std_psi (rad)': 'nan',
                }
                return None, None, None, metadata, None

            waste_start_time = time.perf_counter()
            X_vector.append(x_k)
            u_vector.append(u_k)
            omega_vector.append(np.sqrt(omega_squared))
            omega_squared_vector.append(omega_squared)
            waste_end_time = time.perf_counter()
            waste_time += waste_end_time - waste_start_time
        
        end_time = time.perf_counter()

        ## DEBUG (REMOVE LATER) ##
        #print('min omega squared',min_omega_squared)

        X_vector = np.array(X_vector)
        RMSe = analyser.RMSe(X_vector[:, 9:], trajectory[:len(X_vector), :3])
        execution_time = (end_time - start_time) - waste_time

        min_phi = np.min(X_vector[:,0])
        max_phi = np.max(X_vector[:,0])
        mean_phi = np.mean(X_vector[:,0])
        std_phi = np.std(X_vector[:,0])

        min_theta = np.min(X_vector[:,1])
        max_theta = np.max(X_vector[:,1])
        mean_theta = np.mean(X_vector[:,1])
        std_theta = np.std(X_vector[:,1])

        min_psi = np.min(X_vector[:,2])
        max_psi = np.max(X_vector[:,2])
        mean_psi = np.mean(X_vector[:,2])
        std_psi = np.std(X_vector[:,2])

        min_omega_squared = np.min(np.min(omega_squared_vector, axis = 0))

        metadata = {
            'nn_success': True,
            'num_iterations': len(t_samples)-1,    
            'nn_execution_time (s)': execution_time,
            'nn_RMSe': RMSe,
            'nn_min_phi (rad)': min_phi,
            'nn_max_phi (rad)': max_phi,
            'nn_mean_phi (rad)': mean_phi,
            'nn_std_phi (rad)': std_phi,
            'nn_min_theta (rad)': min_theta,
            'nn_max_theta (rad)': max_theta,
            'nn_mean_theta (rad)': mean_theta,
            'nn_std_theta (rad)': std_theta,
            'nn_min_psi (rad)': min_psi,
            'nn_max_psi (rad)': max_psi,
            'nn_mean_psi (rad)': mean_psi,
            'nn_std_psi (rad)': std_psi,
            'nn_min_omega_squared': min_omega_squared
        }

        omega_vector = np.array(omega_vector)
        omega_squared_vector = np.array(omega_squared_vector)
        for i in range(num_rotors): exec(f'metadata[\'nn_max_omega{i}\'] = np.max(omega_vector[:,{i}])')

        return np.array(X_vector), np.array(u_vector), omega_vector, metadata, omega_squared_vector



if __name__ == '__main__':
    pass
    #Teste
    #teste = ControlAllocationDataset_Split('teste/', False, num_rotors)
    teste = ControlAllocationDataset_Binary('../Datasets/Training datasets - v3', False, num_rotors)
    print('first sample\n',teste.__getitem__(0))
    print(teste.num_inputs,teste.num_outputs)

