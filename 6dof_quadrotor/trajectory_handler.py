import numpy as np
from parameters.simulation_parameters import T_sample

# x, y and z coordinates are on the upside-down body axis

class TrajectoryHandler(object):
    def __init__(self):
        pass

    def point(self, xp, yp, zp, T_simulation,include_psi_reference = True, include_phi_theta_reference = True):
        t = np.arange(0, T_simulation, T_sample)
        r_point = np.array([xp*np.ones(len(t)),
                    yp*np.ones(len(t)),
                    zp*np.ones(len(t)),
                    0*t,
                    0*t,
                    0*t
                    ]).transpose()
        
        if not include_psi_reference:
            r_point = r_point[:, :-1]

        if not include_phi_theta_reference:
            r_point = r_point[:, :-2]

        return r_point
    
    def line(self, a, b, c, clamp, T_simulation, include_psi_reference = True, include_phi_theta_reference = True):
        """
        Creates a line trajectory defined by [a.t b.t c.t]
        clamp: maximum coordinate value allowed for the trajectory
        """
        t = np.arange(0, T_simulation, T_sample)
        r_line = np.array([a*t, b*t, c*t, 0*t, 0*t, 0*t]).transpose()
        r_line = r_line.clip(min=-clamp, max=clamp)

        if not include_psi_reference:
            r_line = r_line[:, :-1]

        if not include_phi_theta_reference:
            r_line = r_line[:, :-2]

        return r_line  

    def line_transition(self, a, b, c, clamp, T_simulation, include_psi_reference = True, include_phi_theta_reference = True):
        """
        Creates a line trajectory defined by [a.t b.t c.t]
        clamp: maximum coordinate value allowed for the trajectory
        """
        t = np.arange(0, T_simulation, T_sample)
        r_line = np.array([a*t, b*t, (c + 0.2*t)*t, 0*t, 0*t, 0*t]).transpose()
        r_line = r_line.clip(min=-clamp, max=clamp)

        if not include_psi_reference:
            r_line = r_line[:, :-1]

        if not include_phi_theta_reference:
            r_line = r_line[:, :-2]

        return r_line           
    
    def circle_xy(self, w, r, T_simulation, include_psi_reference = True, include_phi_theta_reference = True):
        signal_r = np.random.choice([1,-1])
        #r = signal_r*r
        
        signal_w = np.random.choice([1,-1])
        #w = signal_w*w
        t = np.arange(0, 2*T_simulation, T_sample)
        r_circle_xy = np.array([r*np.sin(w*t),
                       -r + r*np.cos(w*t),
                       np.zeros(len(t)),
                       0*t,
                       0*t,
                       0*t
                       ]).transpose()
        
        if not include_psi_reference:
            r_circle_xy = r_circle_xy[:, :-1] #TODO: CORRIGIR ISSO AQUI, ESTA ERRADO.
        
        if not include_phi_theta_reference:
            r_circle_xy = r_circle_xy[:, :-2]

        return r_circle_xy
    
    def circle_xz(self, w, r, T_simulation, include_psi_reference = True, include_phi_theta_reference = True):
        signal_r = np.random.choice([1,-1])
        r = signal_r*r
        signal_w = np.random.choice([1,-1])
        w = signal_w*w
        t = np.arange(0, T_simulation, T_sample)
        r_circle_xz = np.array([r*np.sin(w*t),
                        np.zeros(len(t)),
                        -r + r*np.cos(w*t),
                        0*t,
                        0*t,
                        0*t
                       ]).transpose()
        
        if not include_psi_reference:
            r_circle_xz = r_circle_xz[:, :-1]

        if not include_phi_theta_reference:
            r_circle_xz = r_circle_xz[:, :-2]

        return r_circle_xz
    
    def lissajous_xy(self, w, r, T_simulation,include_psi_reference = True, include_phi_theta_reference = True):
        signal_r = np.random.choice([1,-1])
        #r = signal_r*r
        r = -r
        w = w
        signal_w = np.random.choice([1,-1])
        #w = signal_w*w
        t = np.arange(0, T_simulation, T_sample)
        r_lissajous_xy = np.array([r*np.sin(w*t + np.pi/2) - r,
                       (r*np.sin(2/3*w*t)),
                       np.zeros(len(t)),
                       0*t,
                       0*t,
                       0*t
                       ]).transpose()    
        
        if not include_psi_reference:
            r_lissajous_xy = r_lissajous_xy[:, :-1]

        if not include_phi_theta_reference:
            r_lissajous_xy = r_lissajous_xy[:, :-2]    

        return r_lissajous_xy
    
    def lissajous_3d(self, w, r, T_simulation,include_psi_reference = True, include_phi_theta_reference = True):
        t = np.arange(0, T_simulation, T_sample)
        r_ = np.array([r*np.sin(3*w*t),
                       r*np.sin(2*w*t),
                       r*np.sin(w*t),
                       0*t,
                       0*t,
                       0*t
                       ]).transpose()
        
        if not include_psi_reference:
            r_ = r_[:, :-1]
        
        if not include_phi_theta_reference:
            r_ = r_[:, :-2]

        return r_
    
    def helicoidal(self, w, r0, alpha, beta, T_simulation, include_psi_reference, include_phi_theta_reference, extend=False):
        if extend: t = np.arange(0, 2*T_simulation, T_sample)
        else: t = np.arange(0, T_simulation, T_sample)
        r_helicoidal = np.array([(r0 + alpha*t)*np.sin(w*t),
                       (-r0 + (r0 + alpha*t)*np.cos(w*t)),
                       -beta*t,
                       0*t,
                       0*t,
                       0*t
                       ]).transpose()
        
        if not include_psi_reference:
            r_helicoidal = r_helicoidal[:, :-1]

        if not include_phi_theta_reference:
            r_helicoidal = r_helicoidal[:, :-2]

        return r_helicoidal

    def helicoidal_znegative(self, w, T_simulation, include_psi_reference = True, include_phi_theta_reference = True):
        t = np.arange(0, T_simulation, T_sample)
        r_helicoidal = np.array([5*(1 + 0.1*t)*np.sin(w*t),
                       (5 - 5*(1 + 0.1*t)*np.cos(w*t)),
                       1*t,
                       ]).transpose()
    
        if not include_psi_reference:
            r_helicoidal = r_helicoidal[:, :-1]

        if not include_phi_theta_reference:
            r_helicoidal = r_helicoidal[:, :-2]

        return r_helicoidal
    
    def spiral_toroid(self, R, r, w, n_winds, T_simulation, include_psi_reference, include_phi_theta_reference):
        t = np.arange(0,2*T_simulation, T_sample) # multiplied by 2 to keep abrupt stop of the octorotor at the end of simulation
        x = (R + r*np.cos(n_winds*w*t))*np.cos(w*t) - (R+r)
        y = (R + r*np.cos(n_winds*w*t))*np.sin(w*t)
        z = r*np.sin(n_winds*w*t)

        traj = np.array([x, y, z, 0*t, 0*t, 0*t]).transpose()

        if not include_psi_reference:
            traj = traj[:, :-1]

        if not include_phi_theta_reference:
            traj = traj[:, :-2]
        return traj
    
    def generate_trajectory(self, trajectory_type, args, include_psi_reference, include_phi_theta_reference):
        if trajectory_type in ['point', 'point_failure']:
            return self.point(args[0], args[1], args[2], args[3], include_psi_reference, include_phi_theta_reference)
        
        if trajectory_type == 'line':
            return self.line(args[0], args[1], args[2], args[3], args[4], include_psi_reference, include_phi_theta_reference)
        
        if trajectory_type == 'line_transition':
            return self.line_transition(args[0], args[1], args[2], args[3], args[4], include_psi_reference, include_phi_theta_reference)
        
        if trajectory_type == 'circle_xy':
            return self.circle_xy(args[0], args[1], args[2], include_psi_reference, include_phi_theta_reference)
        
        if trajectory_type == 'lissajous_xy':
            return self.lissajous_xy(args[0], args[1], args[2], include_psi_reference, include_phi_theta_reference)
        
        if trajectory_type == 'circle_xz':
            return self.circle_xz(args[0], args[1], args[2], include_psi_reference, include_phi_theta_reference)
        
        if trajectory_type == 'helicoidal':
            # w, r0, alpha, beta, T_simulation
            return self.helicoidal(args[0], args[1], args[2], args[3], args[4], include_psi_reference, include_phi_theta_reference, extend=True)
        
        if trajectory_type == 'helicoidal_znegative':
            return self.helicoidal_znegative(args[0], args[1], include_psi_reference, include_phi_theta_reference)
    
        if trajectory_type == 'lissajous_3d':
            return self.lissajous_3d(args[0], args[1], args[2], include_psi_reference, include_phi_theta_reference)
        
        if trajectory_type == 'spiral_toroid':
            return self.spiral_toroid(args[0], args[1], args[2], args[3], args[4], include_psi_reference, include_phi_theta_reference)
        
        raise ValueError('Trajectory type not compatible')
    
    def speed_reference(self, trajectory, t):
        '''
        Calculates the speed components in the Earth reference by derivating the trajectory [x(t) y(t) z(t)] vector. Returns the trajectory vector with the speed references concatenated.
        '''
        speed_ref = [(trajectory[1] - trajectory[0])/(t[1] - t[0])]
        for i in range(1, len(trajectory) - 1):
            drdt = (trajectory[i+1] - trajectory[i-1])/(t[i+1] - t[i-1])
            speed_ref.append(drdt)
        speed_ref.append(speed_ref[-1]) # Repete penultima derivada #TODO: verificar uma forma de melhoria de codigo (prever futuro, etc)
        speed_ref = np.array(speed_ref)
        tr = np.concatenate((speed_ref, trajectory), axis = 1)
        return tr

    def generate_point_trajectories(self, point_numbers, T_simulation):
        points_vector = []

        points_vector.append(np.array([0.0,0.0,0.0, T_simulation]))
        points_vector.append(np.array([0.0,0.0,1.0, T_simulation]))
        points_vector.append(np.array([0.0,0.0,-1.0, T_simulation]))
        points_vector.append(np.array([0.0,0.0,-5.0, T_simulation]))
        points_vector.append(np.array([0.0,0.0,-10.0, T_simulation]))
        points_vector.append(np.array([0.0,0.0,-20.0, T_simulation]))
        points_vector.append(np.array([0.0,0.0,-25.0, T_simulation]))
        points_vector.append(np.array([0.0,0.0,2.0, T_simulation]))
        points_vector.append(np.array([0.0,0.0,-2.0, T_simulation]))
        points_vector.append(np.array([0.0,0.0,5.0, T_simulation]))
        points_vector.append(np.array([0.0,0.0,-15.0, T_simulation]))
        points_vector.append(np.array([1.0,0.0,0.0, T_simulation]))
        points_vector.append(np.array([-1.0,0.0,0.0, T_simulation]))
        points_vector.append(np.array([0.0,1.0,0.0, T_simulation]))
        points_vector.append(np.array([0.0,-1.0,0.0, T_simulation]))

        # Random points
        for i in range(point_numbers - len(points_vector)):
            point = 6*np.random.rand(3) - 3 # Random point inside sphere of 15m radius centered in (0,0,0)
            point = np.concatenate((point, [T_simulation]), axis = 0)
            points_vector.append(point)
        
        return points_vector
    
    def generate_circle_xy_trajectories(self, reasonable_trajectories_config=True):
        if reasonable_trajectories_config:
            short_radius_vector = np.arange(4, 7, 1)
            long_radius_vector = np.arange(5, 10, 0.5)
            short_period_vector = np.arange(1, 10, 1)
            long_period_vector = np.arange(10, 16, 1)
        else:
            short_radius_vector = np.arange(4, 6, 1)
            long_radius_vector = np.arange(5, 10, 0.5)
            short_period_vector = np.arange(1, 8, 1)
            long_period_vector = np.arange(10, 15, 1)

        # Parametros do dataset v3
        # short_radius_vector = np.arange(1, 5, 1)
        # long_radius_vector = np.arange(5, 10, 0.5)
        # short_period_vector = np.arange(1, 8, 1)
        # long_period_vector = np.arange(10, 15, 1)

        w_short_vector = 2*np.pi/short_period_vector
        w_long_vector = 2*np.pi/long_period_vector

        args = []

        for period in short_period_vector:
            for radius in short_radius_vector:
                #args = np.concatenate((args, [[2*np.pi/period, radius, 3*period]]), axis = 0)
                args.append([2*np.pi/period, radius, 3*period])
        for period in long_period_vector:
            for radius in long_radius_vector:
                #args = np.concatenate((args, [[2*np.pi/period, radius, 1.3*period]]), axis = 0)
                args.append([2*np.pi/period, radius, 1.25*period])
        num_circles = len(short_radius_vector) * len(short_period_vector) + len(long_radius_vector) * len(long_period_vector)
        print('num_circles =',num_circles)
        return args
    
    def generate_circle_xz_trajectories(self):
        short_radius_vector = np.arange(3, 5, 1) # Aggressive (1,5,1)
        long_radius_vector = np.arange(4, 7, 1) # Aggressive (5,9,1)
        short_period_vector = np.arange(5, 8, 1) # Aggressive (2,8,2)
        long_period_vector = np.arange(10, 14, 2) # Aggressive (10,14,2)

        w_short_vector = 2*np.pi/short_period_vector
        w_long_vector = 2*np.pi/long_period_vector

        args = []

        for period in short_period_vector:
            for radius in short_radius_vector:
                #args = np.concatenate((args, [[2*np.pi/period, radius, 3*period]]), axis = 0)
                args.append([2*np.pi/period, radius, 3*period])
        for period in long_period_vector:
            for radius in long_radius_vector:
                #args = np.concatenate((args, [[2*np.pi/period, radius, 1.3*period]]), axis = 0)
                args.append([2*np.pi/period, radius, 1.25*period])
        num_circles = len(short_radius_vector) * len(short_period_vector) + len(long_radius_vector) * len(long_period_vector)
        print('num circles xz =',num_circles)
        return args


    def generate_line_trajectories(self, num_lines):
        #coefficients = 7*np.random.rand(num_lines, 3) - 3.5 # Aggressive: [-5,5] (antigo)
        print('WARNING: COEFFICIENTS ALTERED!!!!!!!!!!!!!')
        coefficients = 4*np.random.rand(num_lines, 3) - 2 # Aggressive: [-5,5]
        clamp = 10*np.random.rand(num_lines, 1) + 20
        T_simulation = 30*np.ones((num_lines, 1))

        args = np.concatenate((coefficients, clamp, T_simulation), axis = 1)
        return args
    
    def generate_lissajous_xy_trajectories(self):
        short_radius_vector = np.arange(0.5, 4, 0.5) # aggressive (1, 5, 0.5)
        long_radius_vector = np.arange(5, 7, 1) # aggressive (5, 10, 1)
        short_period_vector = np.arange(5, 9, 1) # aggressive (1, 9, 1)
        long_period_vector = np.arange(10, 12, 1) # Aggresive (10, 14, 1)

        args = []

        for period in short_period_vector:
            for radius in short_radius_vector:
                args.append([2*np.pi/period, radius, 3*period])
        #for period in long_period_vector:
        #    for radius in long_radius_vector:
                #args.append([2*np.pi/period, radius, 3*period])
        #num_circles = len(short_radius_vector) * len(short_period_vector) + len(long_radius_vector) * len(long_period_vector)
        #print('lissajous_xy trajectories =',num_circles)
        return args
    
    def generate_lissajous_xy_trajectories_FIXED(self):
        short_radius_vector = np.arange(0.5, 4, 0.5) # aggressive (1, 5, 0.5)
        long_radius_vector = np.arange(5, 8, 0.5) # aggressive (5, 10, 1)
        short_period_vector = np.arange(5, 9, 0.5) # aggressive (1, 9, 1)
        long_period_vector = np.arange(10, 12, 0.5) # Aggresive (10, 14, 1)

        args = []

        for period in short_period_vector:
            for radius in short_radius_vector:
                args.append([2*np.pi/period, radius, 3*period])
        for period in long_period_vector:
            for radius in long_radius_vector:
                args.append([2*np.pi/period, radius, 3*period])
        #num_circles = len(short_radius_vector) * len(short_period_vector) + len(long_radius_vector) * len(long_period_vector)
        #print('lissajous_xy trajectories =',num_circles)
        return args
    
    def generate_circle_xy_trajectories_FIXED(self, reasonable_trajectories_config=True):
        if reasonable_trajectories_config:
            short_radius_vector = np.arange(4, 7, 1)
            long_radius_vector = np.arange(5, 10, 0.5)
            short_period_vector = np.arange(1, 10, 1)
            long_period_vector = np.arange(10, 16, 1)
        else:
            short_radius_vector = np.arange(4, 6, 1)
            long_radius_vector = np.arange(5, 10, 0.5)
            short_period_vector = np.arange(1, 8, 1)
            long_period_vector = np.arange(10, 15, 1)

        # Parametros do dataset v3
        # short_radius_vector = np.arange(1, 5, 1)
        # long_radius_vector = np.arange(5, 10, 0.5)
        # short_period_vector = np.arange(1, 8, 1)
        # long_period_vector = np.arange(10, 15, 1)

        w_short_vector = 2*np.pi/short_period_vector
        w_long_vector = 2*np.pi/long_period_vector

        args = []

        for period in short_period_vector:
            for radius in short_radius_vector:
                #args = np.concatenate((args, [[2*np.pi/period, radius, 3*period]]), axis = 0)
                args.append([2*np.pi/period, radius, 3*period])
        for period in long_period_vector:
            for radius in long_radius_vector:
                #args = np.concatenate((args, [[2*np.pi/period, radius, 1.3*period]]), axis = 0)
                args.append([2*np.pi/period, radius, 1.25*period])
        num_circles = len(short_radius_vector) * len(short_period_vector) + len(long_radius_vector) * len(long_period_vector)
        print('num_circles =',num_circles)
        return args
    
    def generate_circle_xz_trajectories_FIXED(self):
        short_radius_vector = np.arange(3, 5, 1) # Aggressive (1,5,1)
        long_radius_vector = np.arange(4, 7, 1) # Aggressive (5,9,1)
        short_period_vector = np.arange(5, 8, 1) # Aggressive (2,8,2)
        long_period_vector = np.arange(10, 14, 2) # Aggressive (10,14,2)

        w_short_vector = 2*np.pi/short_period_vector
        w_long_vector = 2*np.pi/long_period_vector

        args = []

        for period in short_period_vector:
            for radius in short_radius_vector:
                #args = np.concatenate((args, [[2*np.pi/period, radius, 3*period]]), axis = 0)
                args.append([2*np.pi/period, radius, 3*period])
        for period in long_period_vector:
            for radius in long_radius_vector:
                #args = np.concatenate((args, [[2*np.pi/period, radius, 1.3*period]]), axis = 0)
                args.append([2*np.pi/period, radius, 1.25*period])
        num_circles = len(short_radius_vector) * len(short_period_vector) + len(long_radius_vector) * len(long_period_vector)
        print('num circles xz =',num_circles)
        return args


    def generate_circle_xy_performance_analysis(self):
        period_vector = np.arange(0.25, 20, 0.25)
        r_vector = 3*np.ones(len(period_vector))
        T_simulation_vector = np.array([max(10, period) for period in period_vector])

        args = np.concatenate(([2*np.pi/period_vector], [r_vector], [T_simulation_vector]), axis = 0).transpose()
        return args
        

if __name__ == '__main__':

    teste = TrajectoryHandler()
    teste.generate_circle_xy_trajectories()
#args = teste.generate_line_trajectories()
#print('argsshape',np.shape(args))

