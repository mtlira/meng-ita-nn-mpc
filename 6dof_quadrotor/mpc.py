import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import odeint
from scipy.signal import StateSpace, cont2discrete, lsim
import cvxopt
from linearize import discretize
import time
from linearize import *
from plots import DataAnalyser

from parameters.octorotor_parameters import num_rotors

class MPC(object):
    def __init__(self, M, N, A, B, C, time_step, T_sample, output_weights, control_weights, restrictions, u_ref,\
                  include_psi_reference = True, include_phi_theta_reference = True):
        self.M = M
        self.N = N
        self.A = A
        self.B = B
        self.T = time_step # Simulation time step
        self.T_sample = T_sample # Controller time sample
        self.u_ref = u_ref
        
        #print('flag1', restrictions['y_max'])
        _restrictions = restrictions.copy()
        _output_weights = np.copy(output_weights)
        _C = np.copy(C)

        if not include_psi_reference:
            _restrictions['y_max'] = np.delete(_restrictions['y_max'], (5), axis = 0)
            _restrictions['y_min'] = np.delete(_restrictions['y_min'], (5), axis = 0)
            _output_weights =  _output_weights[:-1]
            _C = _C[:-1]
        #print('flag2', restrictions['y_max'])

        if not include_phi_theta_reference:
            _restrictions['y_max'] = np.delete(_restrictions['y_max'], (3,4), axis = 0)
            _restrictions['y_min'] = np.delete(_restrictions['y_min'], (3,4), axis = 0)
            _output_weights =  np.delete(_output_weights, (3,4), axis = 0)
            _C = np.delete(_C, (3,4), axis = 0)
        #print('flag3', restrictions['y_max'])

        self.output_weights = _output_weights
        self.control_weights = control_weights
        self.restrictions = _restrictions
        self.C = _C

        #print('Check:')
        #print('restrictions[y_max] =\n', _restrictions['y_max'])
        #print('C =\n', _C)

        # p - Number of controls
        # q - Number of outputs
        self.p = np.shape(self.B)[1]
        #print('p =',self.p,'Controls')
        self.q = np.shape(self.C)[0]
        #print('q =',self.q,'Outputs')
        self.n_x = np.shape(self.A)[0]

    def initialize_matrices(self):
        p = self.p
        q = self.q
        n_x = self.n_x
        
        # 1. Discretization of the space state

        # Discretização aproximada
        #Ad = np.eye(np.shape(self.A)[0]) + self.A*self.T
        #Bd = self.B*self.T

        # Discretização exata
        Ad, Bd, _ = discretize(self.A, self.B, self.C, self.T_sample)

        # #################################### APAGAR #################################################
        #Ad = self.A
        #Bd = self.B
        # #################################### APAGAR #################################################

        # 2. Initialize A_tilda, B_tilda, C_tild and phi
        A_tilda = np.concatenate((
            np.concatenate((Ad, Bd), axis = 1), # TODO coonfirm if its really Ad and Bd or A and B
            np.concatenate((np.zeros((p, n_x)), np.eye(p)), axis = 1)
            ), axis = 0)

        B_tilda = np.concatenate((Bd, np.eye(p)), axis = 0)

        C_tilda = np.concatenate((self.C, np.zeros((q,p))), axis = 1)

        phi = C_tilda @ A_tilda
        for i in range(2, self.N+1):
            phi = np.concatenate((phi, C_tilda @ np.linalg.matrix_power(A_tilda, i)), axis = 0)

        # 2.5 phi_state
        phi_state = A_tilda
        for i in range(2, self.N+1):
            phi_state = np.concatenate((phi_state, np.linalg.matrix_power(A_tilda, i)), axis = 0)

        # 3. Initialize G
        # First column
        column = C_tilda @ B_tilda
        for i in range(1,self.N):
            column = np.concatenate((column, C_tilda @ np.linalg.matrix_power(A_tilda, i) @ B_tilda), axis = 0)

        # Other columns
        G = np.array(column, copy = True)
        for i in range(1, self.M):
            column = np.roll(column, q, axis = 0)
            column[0:q,0:p] = np.zeros((q,p))
            G = np.concatenate((G, column), axis = 1)

        # 3.5 - State restriction G_state
        # First column
        column = B_tilda
        for i in range(1,self.N):
            column = np.concatenate((column, np.linalg.matrix_power(A_tilda, i) @ B_tilda), axis = 0)

        # Other columns
        G_state = np.array(column, copy = True)
        for i in range(1, self.M):
            column = np.roll(column, (n_x + p), axis = 0)
            column[0:n_x + p,0:p] = np.zeros((n_x + p, p))
            G_state = np.concatenate((G_state, column), axis = 1)

        # Y_state
        Y_state = np.zeros((3, n_x + p))
        Y_state[0][0] = 1
        Y_state[1][1] = 1
        Y_state[2][2] = 1

        #print('dim Y_state block = ',np.shape(Y_state))

        # First column
        column = np.zeros((3*self.N, n_x + p))
        column[0:3,:] = np.array(Y_state, copy = True)

        # Other columns
        Y_state = np.array(column, copy = True)
        for i in range(1, self.N):
            column = np.roll(column, 3, axis = 0)
            #column[0:q,0:p] = np.zeros((q,p))
            Y_state = np.concatenate((Y_state, column), axis = 1)
        #####

        #print('dim Y_state global =',np.shape(Y_state))
        #print('dim G_state =', np.shape(G_state))
        #print('dim G =', np.shape(G))

        # 4. Initialize Q and R
        Q = np.diag(np.full(q, self.output_weights))
        R = np.diag(np.full(p, self.control_weights))

        # Q - First column
        column = np.array(Q, copy = True)
        for i in range(1, self.N):
            column = np.concatenate((column, np.zeros((q,q))), axis = 0)
        
        # Q - Other columns
        Q_super = np.array(column, copy = True)
        for i in range(1, self.N):
            column = np.roll(column, q, axis = 0)
            column[0:q, 0:q] = np.zeros((q,q))
            Q_super = np.concatenate((Q_super, column), axis = 1)
        #print('Q_super=\n',pd.DataFrame(Q_super))
        #print('diag Q_super =',np.diag(Q_super))
        #print('shape Qsuper =',np.shape(Q_super))

        # R - First column
        column = np.array(R, copy = True)
        for i in range(1, self.M):
            column = np.concatenate((column, np.zeros((p,p))), axis = 0)
        
        # R - Other columns
        R_super = np.array(column, copy = True)
        for i in range(1, self.M):
            column = np.roll(column, p, axis = 0)
            column[0:p, 0:p] = np.zeros((p,p))
            R_super = np.concatenate((R_super, column), axis = 1)
        #print('R_super=\n',pd.DataFrame(R_super))
        #print('diag R_super =',np.diag(R_super))
        #print('shape Rsuper =',np.shape(R_super))

        # Gn, Hqp and Aqp
        Gn = Q_super @ G
        Hqp = 2*(np.transpose(G) @ Q_super @ G + R_super)

        # Aqp
            # T_M
        column = np.eye(p)
        for i in range(1,self.M):
            column = np.concatenate((column, np.eye(p)), axis = 0)
        
        T_M = np.array(column, copy = True)
        for i in range(1, self.M):
            column = np.roll(column, p, axis = 0)
            column[0:p, 0:p] = np.zeros((p,p))
            T_M = np.concatenate((T_M, column), axis = 1)

        Aqp = np.eye(p*self.M)
        Aqp = np.concatenate((
            Aqp,
            -np.eye(p*self.M),
            T_M,
            -T_M,
            #G,
            #-G,
            Y_state @ G_state,
            -Y_state @ G_state
            ), axis = 0)

        self.Ad = Ad
        self.Bd = Bd
        self.A_tilda = A_tilda
        self.B_tilda = B_tilda
        self.C_tilda = C_tilda,
        self.phi = phi
        self.phi_state = phi_state
        self.G = G
        self.G_state = G_state
        self.Y_state = Y_state
        self.Gn = Gn
        self.Q_super = Q_super
        self.R_super = R_super
        self.Aqp = Aqp
        self.Hqp = Hqp
    
    
    def simulate_future(self, f_model, X0, t_samples, trajectory, u_eq):
        """
        Output of the MPC is the thrust force and the moments that are the input of the multirotor.
        Takes into account future trajectory reference from trajectory[k] until trajectory[k+N-1]
        """
        p = self.p
        q = self.q

        #u_minus_1 = np.array(u_eq) # TODO: Confirmar se é u_eq ou 0
        u_minus_1 = np.zeros(p)
        x_k = X0
        u_k_minus_1 = u_minus_1
        X_vector = [X0]
        u_vector = []
        Hqp = self.Hqp
        cvxopt.solvers.options['show_progress'] = False

        for k in range(0, len(t_samples)-1): # TODO: confirmar se é -1 mesmo:
            ref_N = trajectory[k:k+self.N].reshape(-1) # TODO validar se termina em k+N-1 ou em k+N
            if np.shape(ref_N)[0] < q*self.N:
                #print('kpi',self.N - int(np.shape(ref_N)[0]/q))
                ref_N = np.concatenate((ref_N, np.tile(trajectory[-1].reshape(-1), self.N - int(np.shape(ref_N)[0]/q))), axis = 0) # padding de trajectory[-1] em ref_N quando trajectory[k+N] ultrapassa ultimo elemento
            #ref_N = np.tile(trajectory[k,:], self.N)
            epsilon_k = np.concatenate((x_k, u_k_minus_1), axis = 0)
            f = self.phi @ epsilon_k
            fqp = 2*np.transpose(self.Gn) @ (f - ref_N)
            
            # bqp
            bqp = np.concatenate((
                np.tile(self.restrictions['delta_u_max'], self.M), # delta_u_max_M
                - np.tile(self.restrictions['delta_u_min'], self.M), # -delta_u_min_M
                np.tile(self.restrictions['u_max'] - u_k_minus_1, self.M),
                np.tile(u_k_minus_1 - self.restrictions['u_min'], self.M),
                np.tile(self.restrictions['y_max'], self.N) - f,
                f - np.tile(self.restrictions['y_min'], self.N)
            ), axis = 0)

            # optimization
            #cost_function = lambda x: 0.5*np.transpose(x) @ Hqp @ x + np.transpose(fqp) @ x

            #constraints_dict = {
            #    'type': 'ineq',
            #    'fun': lambda x: -(self.Aqp @ x - bqp)
            #}
            #opt = {'maxiter': 1000}
            #res = minimize(fun= cost_function, x0=delta_u_initial, constraints=constraints_dict)#, options=opt) ######################### TODO: verificar opt ######################
            #print('Hqp',np.max(np.abs(Hqp - Hqp.T)))
           
            # METODO 2 #######################################
            Hqp = Hqp.astype(np.double)
            fqp = fqp.astype(np.double)
            self.Aqp = self.Aqp.astype(np.double)
            bqp = bqp.astype(np.double)
            res = cvxopt.solvers.qp(cvxopt.matrix(Hqp), cvxopt.matrix(fqp), cvxopt.matrix(self.Aqp), cvxopt.matrix(bqp))
            x = np.array(res['x']).reshape((Hqp.shape[1],))
            ##################################################

            #print('res.x',res.x)
            #print('res2.x',np.array(res2['x']).reshape((Hqp.shape[1],)))
            delta_u_k = np.concatenate((np.eye(p), np.zeros((p, p*(self.M - 1)))), axis = 1) @ x # optimal delta_u_k
            u_k = u_k_minus_1 + delta_u_k # TODO: confirmar se tem esse u_eq

            # Apply control u_k in the multi-rotor
            f_t_k, t_x_k, t_y_k, t_z_k = u_eq + u_k # Attention for u_eq (solved the problem)
            t_simulation = np.arange(t_samples[k], t_samples[k+1], self.T)
            #t_simulation2 = np.arange(0,t_samples[1], self.T)
            #x_k = odeint(f_model, x_k, t_simulation, args = (f_t_k, t_x_k, t_y_k, t_z_k))
            x_k = odeint(f_model, x_k, t_simulation, args = (f_t_k, t_x_k, t_y_k, t_z_k))
            x_k = x_k[-1]
            #if np.linalg.norm(x_k[9:12]) > 10 or np.max(np.abs(x_k[0:2])) > 1.4:
            #    print('Simulation exploded.')
            #    print('x_k =',x_k)
            #    break
            
            X_vector.append(x_k)
            u_k_minus_1 = u_k
            u_vector.append(u_k + u_eq)
            #delta_u_initial = np.tile(delta_u_k,self.M)
        return np.array(X_vector), np.array(u_vector)
    
    def simulate_future_rotors(self, model, X0, t_samples, trajectory, generate_dataset = False, disturb_input = False):
        """
        Output of the MPC are the angular speeds that are the input of the multirotor.
        Takes into account future trajectory reference from trajectory[k] until trajectory[k+N-1]
        """

        analyser = DataAnalyser()

        p = self.p
        q = self.q

        # Disturb logic (state disturbance)
        disturb_frequency = 0.1

        #u_minus_1 = np.array(u_eq) # TODO: Confirmar se é u_eq ou 0
        u_minus_1 = np.zeros(p)
        x_k = X0
        u_k_minus_1 = u_minus_1
        omega_k = model.get_omega_eq_hover()
        #alpha = model.angular_acceleration (Not being used at the moment)
        #alpha_neg = -alpha
        X_vector = [X0]
        u_vector = []
        omega_vector = []
        Hqp = self.Hqp
        cvxopt.solvers.options['show_progress'] = False

        # NN Dataset
        NN_dataset = [] if generate_dataset else None

        execution_time = 0
        waste_time = 0
        start_time = time.perf_counter()
        for k in range(0, len(t_samples)-1): # TODO: confirmar se é -1 mesmo:
            ref_N = trajectory[k:k+self.N].reshape(-1) # TODO validar se termina em k+N-1 ou em k+N
            if np.shape(ref_N)[0] < q*self.N:
                #print('kpi',self.N - int(np.shape(ref_N)[0]/q))
                ref_N = np.concatenate((ref_N, np.tile(trajectory[-1].reshape(-1), self.N - int(np.shape(ref_N)[0]/q))), axis = 0) # padding de trajectory[-1] em ref_N quando trajectory[k+N] ultrapassa ultimo elemento
            #ref_N = np.tile(trajectory[k,:], self.N)
            epsilon_k = np.concatenate((x_k, u_k_minus_1), axis = 0)
            f = self.phi @ epsilon_k
            fqp = 2*np.transpose(self.Gn) @ (f - ref_N)
            
            # Update delta_u restrictions
            #print(self.restrictions['delta_u_min'])
            #self.restrictions['delta_u_max'] = (2*omega_k + alpha*self.T_sample)*alpha * self.T_sample
            #self.restrictions['delta_u_min'] = (2*omega_k + alpha_neg*self.T_sample)*alpha_neg * self.T_sample

            # bqp
            bqp = np.concatenate((
                np.tile(self.restrictions['delta_u_max'], self.M), # delta_u_max_M
                - np.tile(self.restrictions['delta_u_min'], self.M), # -delta_u_min_M
                np.tile(self.restrictions['u_max'] - u_k_minus_1, self.M),
                np.tile(u_k_minus_1 - self.restrictions['u_min'], self.M),
                np.tile(self.restrictions['y_max'], self.N) - f,
                f - np.tile(self.restrictions['y_min'], self.N)
            ), axis = 0)

            # optimization
            #cost_function = lambda x: 0.5*np.transpose(x) @ Hqp @ x + np.transpose(fqp) @ x

            #constraints_dict = {
            #    'type': 'ineq',
            #    'fun': lambda x: -(self.Aqp @ x - bqp)
            #}
            #opt = {'maxiter': 1000}
            #res = minimize(fun= cost_function, x0=delta_u_initial, constraints=constraints_dict)#, options=opt) ######################### TODO: verificar opt ######################
            #print('Hqp',np.max(np.abs(Hqp - Hqp.T)))
           
            # METODO 2 #######################################
            Hqp = Hqp.astype(np.double)
            fqp = fqp.astype(np.double)
            self.Aqp = self.Aqp.astype(np.double)
            bqp = bqp.astype(np.double)
            res = cvxopt.solvers.qp(cvxopt.matrix(Hqp), cvxopt.matrix(fqp), cvxopt.matrix(self.Aqp), cvxopt.matrix(bqp))
            res = np.array(res['x']).reshape((Hqp.shape[1],))
            ##################################################


            #print('res.x',res.x)
            #print('res2.x',np.array(res2['x']).reshape((Hqp.shape[1],)))
            delta_u_k = np.concatenate((np.eye(p), np.zeros((p, p*(self.M - 1)))), axis = 1) @ res # optimal delta_u_k
            u_k = u_k_minus_1 + delta_u_k # TODO: confirmar se tem esse u_eq

            # omega**2 --> u
            omega_squared = np.clip(u_k + self.u_ref, 0, None)
            omega_k = np.sqrt(omega_squared)
            uu_k = model.Gama @ omega_squared

            # Apply INPUT disturbance if enabled
            if disturb_input and k>0:
                waste_start_time = time.perf_counter()
                probability = np.random.rand()
                if probability > disturb_frequency:
                    uu_k = add_input_disturbance(uu_k, model)
                waste_end_time = time.perf_counter()
                waste_time += waste_end_time - waste_start_time               

            # Apply control u_k in the multi-rotor
            f_t_k, t_x_k, t_y_k, t_z_k = uu_k # Attention for u_eq (solved the problem)
            t_simulation = np.arange(t_samples[k], t_samples[k+1], self.T)
            #t_simulation2 = np.arange(0,t_samples[1], self.T)
            #x_k = odeint(f_model, x_k, t_simulation, args = (f_t_k, t_x_k, t_y_k, t_z_k))
            
            # x[k+1] = f(x[k], u[k])
            x_k_old = x_k # Used only to mount nn_sample array
            x_k = odeint(model.f2, x_k, t_simulation, args = (f_t_k, t_x_k, t_y_k, t_z_k))
            x_k = x_k[-1]

            waste_start_time= time.perf_counter()

            if np.linalg.norm(x_k[9:12] - trajectory[k, :3]) > 10: #or np.max(np.abs(x_k[0:2])) > 1.75:
                print('Simulation exploded.')
                print(f'x_{k} =',x_k)
                return None, None, None, None, None
            
            X_vector.append(x_k)
            u_k_minus_1 = u_k
            u_vector.append(uu_k)
            omega_vector.append(omega_k)
            #delta_u_initial = np.tile(delta_u_k,self.M)

            # Storing dataset samples
            if generate_dataset:
                NN_sample = np.array([])

                # Calculating reference values relative to multirotor's current position at instant k
                ref_N_position = trajectory[k:k+self.N, 0:3].reshape(-1) # TODO validar se termina em k+N-1 ou em k+N
                q_neuralnetwork = 3 # For the NN, only x, y and z counts as effective references, since the angle references are all 0
                if np.shape(ref_N_position)[0] < q_neuralnetwork*self.N:
                    ref_N_position = np.concatenate((ref_N_position, np.tile(trajectory[-1, :3].reshape(-1), self.N - int(np.shape(ref_N_position)[0]/q_neuralnetwork))), axis = 0) # padding de trajectory[-1] em ref_N quando trajectory[k+N] ultrapassa ultimo elemento
                position_k = x_k_old[9:]
                position_k = np.tile(position_k, self.N).reshape(-1)
                ref_N_relative = ref_N_position - position_k

                # Clarification: u is actually (u - ueq) and delta_u is (u-ueq)[k] - (u-ueq)[k-1] in this MPC formulation (i.e., u is in reference to u_eq, not 0)
                NN_sample = np.concatenate((NN_sample, x_k_old[0:9], ref_N_relative, self.restrictions['u_max'] + self.u_ref, u_k), axis = 0) #TODO: depois, acrescentar restrições

                NN_dataset.append(NN_sample)

            waste_end_time = time.perf_counter()
            waste_time += waste_end_time - waste_start_time

        end_time = time.perf_counter()
        
        # Metadata
        X_vector = np.array(X_vector)
        execution_time = (end_time - start_time) - waste_time

        position = X_vector[:, 9:]
        #delta_position = trajectory[:len(position),:3] - position
        #RMSe = np.sqrt(np.mean(delta_position**2))
        RMSe = analyser.RMSe(position, trajectory[:len(position),:3])

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

        metadata = {
            'num_iterations': len(t_samples)-1,    
            'mpc_execution_time': execution_time,
            'mpc_RMSe': RMSe,
            'mpc_min_phi': min_phi,
            'mpc_max_phi': max_phi,
            'mpc_mean_phi': mean_phi,
            'mpc_std_phi': std_phi,
            'mpc_min_theta': min_theta,
            'mpc_max_theta': max_theta,
            'mpc_mean_theta': mean_theta,
            'mpc_std_theta': std_theta,
            'mpc_min_psi': min_psi,
            'mpc_max_psi': max_psi,
            'mpc_mean_psi': mean_psi,
            'mpc_std_psi': std_psi,
        }

        return np.array(X_vector), np.array(u_vector), np.array(omega_vector), np.asarray(NN_dataset), metadata
    
    def RMSe(self, position, trajectory):
        delta_position = trajectory - position   # shape (N, 3)
        squared_norms = np.sum(delta_position**2, axis=1)  # Sum over x, y, z for each time step
        RMSe = np.sqrt(np.mean(squared_norms))
        return RMSe

class GainSchedulingMPC(object):
    def __init__(self, model, phi_grid_deg, theta_grid_deg, M, N, time_step, T_sample, output_weights, control_weights,\
                  restrictions, include_psi_reference = True, include_phi_theta_reference = True):
        self.model = model
        self.phi_grid_deg = phi_grid_deg
        self.theta_grid_deg = theta_grid_deg
        self.linear_model = {}
        self.M = M
        self.N = N
        self.time_step = time_step
        self.T_sample = T_sample

        # if not include_psi_reference:
        #     restrictions['y_max'] = restrictions['y_max'][:-1]
        #     restrictions['y_min'] = restrictions['y_min'][:-1]
        #     output_weights = output_weights[:-1]

        # if not include_phi_theta_reference:
        #     restrictions['y_max'] = restrictions['y_max'][:-2]
        #     restrictions['y_min'] = restrictions['y_min'][:-2]
        #     output_weights = output_weights[:-2]

        for phi in phi_grid_deg:
            for theta in theta_grid_deg:
                U = np.array([self.model.m * self.model.g/(np.cos(0*phi*np.pi/180)*np.cos(0*theta*np.pi/180)), 0, 0, 0])
                omega_squared_ref = self.model.get_omega(U)**2
                X = np.array([phi*np.pi/180, theta*np.pi/180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                A, B, C = self.model.linearize(X=X, U=U)
                #if not include_psi_reference: 
                #    C = C[:-1]
                #if not include_phi_theta_reference:
                #    C = C[:-2]
                Bw = B @ self.model.Gama
                self.linear_model[(phi, theta)] = MPC(M, N, A, Bw, C, time_step, T_sample, output_weights, control_weights,\
                     restrictions, omega_squared_ref, include_psi_reference, include_phi_theta_reference)
                self.linear_model[(phi, theta)].initialize_matrices()

    def choose_model(self, phi_rad, theta_rad):
        closest_phi_deg = min(self.phi_grid_deg, key=lambda x:abs(x*np.pi/180 - phi_rad))
        closest_theta_deg = min(self.theta_grid_deg, key=lambda x:abs(x*np.pi/180 - theta_rad))
        #print(closest_phi_deg, closest_theta_deg)
        return self.linear_model[(closest_phi_deg, closest_theta_deg)]

    def simulate_future_rotors(self, model, X0, t_samples, trajectory, gain_scheduling, generate_dataset = False, disturb_input = False):
        """
        Output of the MPC are the angular speeds that are the input of the multirotor.
        Takes into account future trajectory reference from trajectory[k] until trajectory[k+N-1]\n
        Gets the linear model closest to the current operating point regarding pitch and roll angles at every iteration
        """
        analyser = DataAnalyser()

        #p = len(self.linear_model[(0,0)].u_ref)
        #q = len(self.linear_model[(0,0)].C)
        p = self.linear_model[(0,0)].p
        q = self.linear_model[(0,0)].q

        # Disturb logic (state disturbance)
        disturb_frequency = 0.1

        u_minus_1 = np.zeros(p)
        x_k = X0
        u_k_minus_1 = u_minus_1
        
        x_k = X0
        omega_squared_eq = model.get_omega_eq_hover()**2
        omega_k = model.get_omega_eq_hover()
        #alpha = model.angular_acceleration (Not being used at the moment)
        #alpha_neg = -alpha
        X_vector = [X0]
        u_vector = []
        omega_vector = []
        cvxopt.solvers.options['show_progress'] = False

        # NN Dataset
        NN_dataset = [] if generate_dataset else None
        #print('q = ',q)

        # Temp ###
        omega_max = self.linear_model[(0,0)].restrictions['u_max'] + omega_squared_eq
        #clip_max_omega = np.copy(max(omega_max)*np.ones(num_rotors)) # Starting in u_eq
        clip_max_omega = np.array([0 if omega < 0.1 else omega for omega in omega_max])
        failed_rotors = [{'indice': i, 'value': max(omega_squared_eq), 'reached_zero': False} for i, omega in enumerate(omega_max) if omega < 0.01]
        omega_squared_previous = np.copy(omega_squared_eq)
        for i in range(len(clip_max_omega)):
            if clip_max_omega[i] == 0:
                omega_squared_previous[i] = 0
                u_k_minus_1[i] = 0
        ###

        execution_time = 0
        waste_time = 0
        start_time = time.perf_counter()
        for k in range(0, len(t_samples)-1): # TODO: confirmar se é -1 mesmo:
            if gain_scheduling:
                linear_model = self.choose_model(x_k[0], x_k[1])
            else:
                linear_model = self.linear_model[(0,0)]
            ref_N = trajectory[k:k+self.N].reshape(-1) # TODO validar se termina em k+N-1 ou em k+N
            if np.shape(ref_N)[0] < q*self.N:
                ref_N = np.concatenate((ref_N, np.tile(trajectory[-1].reshape(-1), self.N - int(np.shape(ref_N)[0]/q))), axis = 0) # padding de trajectory[-1] em ref_N quando trajectory[k+N] ultrapassa ultimo elemento
            #ref_N = np.tile(trajectory[k,:], self.N)
            epsilon_k = np.concatenate((x_k, u_k_minus_1), axis = 0)
            f = linear_model.phi @ epsilon_k
            f_state = linear_model.phi_state @ epsilon_k
            fqp = 2*np.transpose(linear_model.Gn) @ (f - ref_N)
            
            # Update delta_u restrictions
            #print(self.restrictions['delta_u_min'])
            #self.restrictions['delta_u_max'] = (2*omega_k + alpha*self.T_sample)*alpha * self.T_sample
            #self.restrictions['delta_u_min'] = (2*omega_k + alpha_neg*self.T_sample)*alpha_neg * self.T_sample

            # bqp
            #print('linear_model.restrictions\n', linear_model.restrictions['y_max'])
            #print('f\n', len(f))
            #print('N=',self.N)
            bqp = np.concatenate((
                np.tile(linear_model.restrictions['delta_u_max'], self.M), # delta_u_max_M
                - np.tile(linear_model.restrictions['delta_u_min'], self.M), # -delta_u_min_M
                np.tile(linear_model.restrictions['u_max'] - u_k_minus_1, self.M),
                np.tile(u_k_minus_1 - linear_model.restrictions['u_min'], self.M),
                #np.tile(linear_model.restrictions['y_max'], self.N) - f,
                #f - np.tile(linear_model.restrictions['y_min'], self.N),
                np.tile(linear_model.restrictions['angle_max'], self.N) - linear_model.Y_state @ f_state,
                linear_model.Y_state @ f_state - np.tile(linear_model.restrictions['angle_min'], self.N)
            ), axis = 0)

            # optimization
            #cost_function = lambda x: 0.5*np.transpose(x) @ Hqp @ x + np.transpose(fqp) @ x

            #constraints_dict = {
            #    'type': 'ineq',
            #    'fun': lambda x: -(self.Aqp @ x - bqp)
            #}
            #opt = {'maxiter': 1000}
            #res = minimize(fun= cost_function, x0=delta_u_initial, constraints=constraints_dict)#, options=opt) ######################### TODO: verificar opt ######################
            #print('Hqp',np.max(np.abs(Hqp - Hqp.T)))
           
            # METODO 2 #######################################
            Hqp = linear_model.Hqp.astype(np.double)
            fqp = fqp.astype(np.double)
            linear_model.Aqp = linear_model.Aqp.astype(np.double)
            bqp = bqp.astype(np.double)
            res = cvxopt.solvers.qp(cvxopt.matrix(Hqp), cvxopt.matrix(fqp), cvxopt.matrix(linear_model.Aqp), cvxopt.matrix(bqp))

            temp_counter = 0
            if res['status'] != 'optimal':
                temp_counter += 1
                if temp_counter > 0:
                    print(f'solution at time {k} is {res['status']}')
                    print('solution dict:\n',res)
            res = np.array(res['x']).reshape((Hqp.shape[1],))

            ##################################################
            #print('res.x',res.x)
            #print('res2.x',np.array(res2['x']).reshape((Hqp.shape[1],)))
            delta_u_k = np.concatenate((np.eye(p), np.zeros((p, p*(self.M - 1)))), axis = 1) @ res # optimal delta_u_k
            u_k = u_k_minus_1 + delta_u_k # TODO: confirmar se tem esse u_eq
            omega_squared = u_k + linear_model.u_ref

            # omega**2 --> u
            #omega_squared = np.clip(u_k + linear_model.u_ref, 0, None) # Safe
            omega_squared = np.clip(u_k + linear_model.u_ref, a_min=0, a_max=np.clip(linear_model.restrictions['u_max'] + linear_model.u_ref, 0, None))
            #omega_squared = np.clip(u_k + linear_model.u_ref, 0,  a_max=clip_max_omega) # Temp
            #omega_squared = np.clip(omega_squared, np.max([omega_squared_previous + linear_model.restrictions['delta_u_min'], np.zeros(num_rotors)],axis=0), np.min([omega_squared_previous + linear_model.restrictions['delta_u_max'], clip_max_omega],axis=0))

            # TEMP ###
            # for rotor in failed_rotors:
            #     if not rotor['reached_zero']:
            #         rotor['value'] += linear_model.restrictions['delta_u_min'][0]
            #         if rotor['value'] <= 0.01:
            #             rotor['reached_zero'] = True
            #             rotor['value'] = 0
            #         omega_squared[rotor['indice']] = rotor['value']
            #     else:
            #         omega_squared[rotor['indice']] = 0
            
            omega_squared_previous = np.copy(omega_squared)
                ### ###

            omega_k = np.sqrt(omega_squared)
            uu_k = model.Gama @ omega_squared

            # Apply INPUT disturbance if enabled
            if disturb_input and k>0:
                waste_start_time = time.perf_counter()
                probability = np.random.rand()
                if probability > disturb_frequency:
                    uu_k = add_input_disturbance(uu_k, model)
                waste_end_time = time.perf_counter()
                waste_time += waste_end_time - waste_start_time

            # Apply control u_k in the multi-rotor
            f_t_k, t_x_k, t_y_k, t_z_k = uu_k # Attention for u_eq (solved the problem)
            t_simulation = np.arange(t_samples[k], t_samples[k+1], linear_model.T)
            #t_simulation2 = np.arange(0,t_samples[1], self.T)
            #x_k = odeint(f_model, x_k, t_simulation, args = (f_t_k, t_x_k, t_y_k, t_z_k))

            # x[k+1] = f(x[k], u[k])
            x_k_old = x_k # Used only to mount nn_sample array
            x_k = odeint(model.f2, x_k, t_simulation, args = (f_t_k, t_x_k, t_y_k, t_z_k))
            x_k = x_k[-1]

            waste_start_time = time.perf_counter()

            if np.linalg.norm(x_k[9:12] - trajectory[k, :3]) > 60: #TODO: unificar criterio de saida para todos os metodos
               print('Simulation exploded.')
               print(f'x_{k} =',x_k)
               return None, None, None, None, None
                  
            X_vector.append(x_k)
            u_k_minus_1 = np.copy(u_k)
            u_vector.append(uu_k)
            omega_vector.append(omega_k)
            #delta_u_initial = np.tile(delta_u_k,self.M)

            # Storing dataset samples
            if generate_dataset:
                NN_sample = np.array([])

                # Calculating reference values relative to multirotor's current position at instant k
                ref_N_position = trajectory[k:k+self.N, 0:3].reshape(-1) # TODO validar se termina em k+N-1 ou em k+N
                q_neuralnetwork = 3 # For the NN, only x, y and z counts as effective references, since the angle references are all 0
                if np.shape(ref_N_position)[0] < q_neuralnetwork*self.N:
                    ref_N_position = np.concatenate((ref_N_position, np.tile(trajectory[-1, :3].reshape(-1), self.N - int(np.shape(ref_N_position)[0]/q_neuralnetwork))), axis = 0) # padding de trajectory[-1] em ref_N quando trajectory[k+N] ultrapassa ultimo elemento
                position_k = x_k_old[9:]
                position_k = np.tile(position_k, self.N).reshape(-1)
                ref_N_relative = ref_N_position - position_k

                # Clarification: u is actually (u - ueq) and delta_u is (u-ueq)[k] - (u-ueq)[k-1] in this MPC formulation (i.e., u is in reference to u_eq, not 0)
                NN_sample = np.concatenate((NN_sample, x_k_old[0:9], ref_N_relative, linear_model.restrictions['u_max'] + linear_model.u_ref, omega_squared), axis = 0) #TODO: depois, acrescentar restrições

                NN_dataset.append(NN_sample)

            waste_end_time = time.perf_counter()
            waste_time += waste_end_time - waste_start_time

        end_time = time.perf_counter()
        
        # Metadata
        X_vector = np.array(X_vector)
        execution_time = (end_time - start_time) - waste_time

        position = X_vector[:, 9:]
        #delta_position = trajectory[:len(position),:3] - position
        #RMSe = np.sqrt(np.mean(delta_position**2))
        RMSe = analyser.RMSe(position, trajectory[:len(X_vector),:3])

        #print('RMSe',RMSe, '\nWrong RMSe',RMSe_wrong)

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

        metadata = {
            'num_iterations': len(t_samples)-1,
            'mpc_execution_time': execution_time,
            'mpc_RMSe': RMSe,
            'mpc_min_phi': min_phi,
            'mpc_max_phi': max_phi,
            'mpc_mean_phi': mean_phi,
            'mpc_std_phi': std_phi,
            'mpc_min_theta': min_theta,
            'mpc_max_theta': max_theta,
            'mpc_mean_theta': mean_theta,
            'mpc_std_theta': std_theta,
            'mpc_min_psi': min_psi,
            'mpc_max_psi': max_psi,
            'mpc_mean_psi': mean_psi,
            'mpc_std_psi': std_psi,
        }

        omega_vector = np.array(omega_vector)
        for i in range(num_rotors): exec(f'metadata[\'mpc_max_omega{i}\'] = np.max(omega_vector[:,{i}])')

        return np.array(X_vector), np.array(u_vector), omega_vector, np.asarray(NN_dataset), metadata


def add_input_disturbance(u, model):
    '''
    Adds small disturbances to the state vector X.
    '''

    thrust_range = 0.2*model.m*model.g
    tx_range = 0.05*model.m*model.g*model.l
    ty_range = tx_range
    tz_range = tx_range

    ranges = np.array([
        thrust_range,
        tx_range,
        ty_range,
        tz_range
    ])


    # disturbance in [-range, + range]
    disturbances = 2*ranges*np.random.rand(len(u))  - ranges

    # adding disturbances
    u += disturbances

    #Making sure thrust is not negative
    u[0] = np.clip(u[0], a_min = 0.0, a_max=None)

    return u

    # def discretize(self):
    #     #sys = StateSpace(self.A,self.B,self.C, np.zeros((q,p)))
    #     sys_d = cont2discrete((self.A,self.B,self.C, np.zeros((self.q,self.p))), self.T_sample, 'zoh')
    #     Ad, Bd, Cd, _, _ = sys_d
    #     return Ad, Bd, Cd

if __name__ == '__main__':
    ## TESTE - DELETAR DEPOIS ###########################################################################
    ### MULTIROTOR PARAMETERS ###
    from parameters.octorotor_parameters import m, g, I_x, I_y, I_z, l, b, d, thrust_to_weight, num_rotors

    # ### SIMULATION PARAMETERS ###
    from parameters.simulation_parameters import time_step, T_sample, N, M
    T_simulation = 30
    import multirotor
    import restriction_handler
    model = multirotor.Multirotor(m, g, I_x, I_y, I_z, b, l, d, num_rotors, thrust_to_weight)
    rst = restriction_handler.Restriction(model, T_sample, N, M)

    restriction, output_weights, control_weights, _ = rst.restriction('normal')

    phi_grid = [-15, 0, 15]
    theta_grid = [-15, 0, 15]
    teste = GainSchedulingMPC(model, phi_grid, theta_grid, M, N, time_step, T_sample, output_weights, control_weights, restriction)
    print(teste.linear_model)