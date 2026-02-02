import numpy as np
import itertools

class Restriction(object):
    def __init__(self, model, time_sample, N, M):
        self.model = model
        self.time_sample = time_sample
        self.N = N
        self.M = M


    def restriction(self, operation_mode, rotors_idx=None, relax_constraint=None):
        if operation_mode not in ['normal', 'total_failure', 'partial_failure']:
            raise ValueError('Operation mode not correct')
        
        omega_eq = self.model.get_omega_eq_hover()

        omega_max = np.sqrt(self.model.thrust_to_weight)*omega_eq # Decision: Max thrust force = 2*m*g => omega_max = sqrt(2)*omega_eq
        if relax_constraint is not None and 'thrust' in relax_constraint: omega_max = 50*omega_eq # Relax omega_max constraint
        omega_min = np.zeros(self.model.num_rotors)

        if operation_mode != 'normal':
            # Failure constraint
            for idx in rotors_idx:
                if operation_mode == 'total_failure':
                    omega_max[idx] = 0
                else: # partial failure
                    omega_max[idx] = (0.8*self.random.rand() + 0.1)*omega_eq[idx]

        u_max = omega_max**2 - omega_eq**2
        u_min = omega_min**2 - omega_eq**2

        alpha = self.model.angular_acceleration
        delta_u_max = (2*omega_eq + alpha*self.time_sample)*alpha * self.time_sample
        alpha = -alpha
        delta_u_min = (2*omega_eq + alpha*self.time_sample)*alpha * self.time_sample

        # Order: x, y, z, phi, beta, psi
        y_max = np.array([100, 100, 100, 1, 1, 1.2]) # Tune in case of unsatisfactory simulation results
        #if operation_mode == 'normal' or len(rotors_idx) <= 1: y_max = np.concatenate((y_max, [1000]), axis = 0) # Adding psi = 0 reference in normal operation (In fault mode, yaw control is sacrificed)
        y_min = (-1) * y_max

        angle_max = np.array([0.7, 0.7, 1.2])
        angle_min = np.array([-0.7, -0.7, -1.2])
        #for i in range(15): print('WARNING: TEMPORARY ANGLE CONSTRAINTS BEING USED (0.4, 0.4, 0.6) RAD. FIX IT OR YOUR RESULTS ARE WRONG FOR THE MASTERS!!!!!!')
        if relax_constraint is not None and 'angle' in relax_constraint: # Relax attitude angle constraints
            angle_max = np.array([5, 5, 1000])
            angle_min = np.array([-5, -5, -1000])

        restrictions = {
            'delta_u_max': delta_u_max,
            'delta_u_min': delta_u_min,
            'u_max': u_max,
            'u_min': u_min,
            'y_max': y_max,
            'y_min': y_min,
            'angle_max': angle_max,
            'angle_min': angle_min
        }

        # Order: x, y, z, phi, beta, psi
        delta_y_max = np.array([1, 1, 1, 1, 1, 1.2]) # Tune in case of unsatisfactory simulation results
        #delta_y_max = np.array([1, 1, 1, 0.6, 0.6, 1.2]) # Tune in case of unsatisfactory simulation results
        #if operation_mode == 'normal' or len(rotors_idx) <= 1: delta_y_max = np.concatenate((delta_y_max, [1.2]), axis = 0) # Adding psi = 0 reference in normal operation (In fault mode for more than 1 rotor, yaw control is sacrificed)

        output_weights = 1 / (self.N*delta_y_max**2)
        control_weights = 1 / (self.M*restrictions['delta_u_max']**2)

        # Restrictions Metadata
        if rotors_idx is not None and len(rotors_idx) > 0:
            rotors_idx = np.sort(rotors_idx)
            failed_rotors = ''
            for i in rotors_idx:
                failed_rotors += str(i) + '-'
            failed_rotors = failed_rotors[:-1]
        else:
            failed_rotors = 'nan'

        ang_speed_percentages = omega_max / (np.sqrt(self.model.thrust_to_weight)*self.model.get_omega_eq_hover())
        ang_speed_percentages_string = ''
        for percentage in ang_speed_percentages:
            ang_speed_percentages_string += str(100*round(percentage , 1)) + '-'
        ang_speed_percentages_string = ang_speed_percentages_string[:-1]

        metadata = {
            'operation_mode': operation_mode,
            'failed_rotors': failed_rotors,
            'ang_speed_percentages': ang_speed_percentages_string,
            'rotors_idx': rotors_idx
        }

        return restrictions, output_weights, control_weights, metadata
    
    def restrictions_performance(self):
        restrictions = []

        # Normal operation
        restrictions.append(self.restriction('normal'))

        # Single total failures
        for i in range(self.model.num_rotors):
            restrictions.append(self.restriction('total_failure', [i]))

        return restrictions
    
    def restrictions_fault_tolerance(self):
        restrictions = []

        # Normal operation
        #restrictions.append(self.restriction('normal'))

        # Single total failures
        #for i in range(self.model.num_rotors): # Already accounted for in performance simulations
        #    restrictions.append(self.restriction('total_failure', [i]))

        # 2 total failures of all rotors
        for combination in itertools.combinations(range(self.model.num_rotors), 2):
            restrictions.append(self.restriction('total_failure', combination))

        # 3 total failures of all rotors
        for combination in itertools.combinations(range(self.model.num_rotors), 3):
            restrictions.append(self.restriction('total_failure', combination))

        # 4 total failures of all rotors
        #for combination in itertools.combinations(range(self.model.num_rotors), 4):
        #    restrictions.append(self.restriction('total_failure', combination))
        
        return restrictions

    def restrictions_2_rotor_faults(self):
        restrictions = []

        # 2 total failures of all rotors
        for combination in itertools.combinations(range(self.model.num_rotors), 2):
            restrictions.append(self.restriction('total_failure', combination))
        return restrictions

    def restrictions_1_rotor_fault(self):
        restrictions = []
        for combination in itertools.combinations(range(self.model.num_rotors), 1):
            restrictions.append(self.restriction('total_failure', combination))
        return restrictions
    
    # def generate_restrictions(self):

    #     restrictions = []

    #     # Normal operation
    #     restrictions.append(self.restriction('normal'))

    #     # Single total failures
    #     for i in range(self.model.num_rotors):
    #         restrictions.append(self.restriction('total_failure', [i]))

    #     # 2 total failures of all rotors
    #     for combination in itertools.combinations(range(self.model.num_tools), 2):
    #         restrictions.append(self.restriction('total_failure', combination))

    #     # 3 total failures of all rotors
    #     for combination in itertools.combinations(range(self.model.num_rotors), 3):
    #         restrictions.append(self.restriction('total_failure', combination))

    #     # 4 total failures of all rotors
    #     for combination in itertools.combinations(range(self.model.num_rotors), 4):
    #         restrictions.append(self.restriction('total_failure', combination))

    #     return restrictions

# for i_comb, comb in enumerate(itertools.combinations(range(8), 2)):
#     print('comb', i_comb, comb)