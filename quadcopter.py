import numpy as np
import math
import scipy.integrate
import time
import datetime
import threading


class Propeller():
    def __init__(self, prop_dia, prop_pitch, thrust_unit='N'):
        self.dia = prop_dia
        self.pitch = prop_pitch
        self.thrust_unit = thrust_unit
        self.speed = 0 #RPM
        self.thrust = 0

    def set_speed(self,speed):
        self.speed = speed
        # From http://www.electricrcaircraftguy.com/2013/09/propeller-static-dynamic-thrust-equation.html
        self.thrust = 4.392e-8 * self.speed * math.pow(self.dia,3.5)/(math.sqrt(self.pitch))
        self.thrust = self.thrust*(4.23e-4 * self.speed * self.pitch)
        if self.thrust_unit == 'Kg':
            self.thrust = self.thrust*0.101972
            
    def get_thrust(self):
        return self.thrust
            
    def get_thrust_unit(self):
        if self.thrust_unit == 'N':
            return 1.0  # Return a numerical value of 1.0 for 'N'
        elif self.thrust_unit == 'kgf':
            return 9.81  # Return a numerical value of 9.81 for 'kgf'
        else:
            return 1.0  # Return a default numerical value of 1.0
        return self.thrust_unit

class Quadcopter():
    # State space representation: [x y z x_dot y_dot z_dot theta phi gamma theta_dot phi_dot gamma_dot]
    # From Quadcopter Dynamics, Simulation, and Control by Andrew Gibiansky
    def __init__(self,quads,gravity=9.81,b=0.0245):
        self.quads = quads
        self.g = gravity
        self.b = b
        self.thread_object = None
        self.ode =  scipy.integrate.ode(self.state_dot).set_integrator('vode',nsteps=500,method='bdf')
        self.time = datetime.datetime.now()
        self.drag_coefficient = 0.5
        self.forces = {}
        q1_props = {
        'position': [0, 0, 0],
        'orientation': [0, 0, 0],
        'L': 0.3,
        'r': 0.1,
        'torques': {},
        'prop_size': [10, 4.5],
        'weight': 1.2,
        'forces': {},  # Initialize forces dictionary
        'prop': Propeller(10, 4.5, thrust_unit='N')  # Create an instance of the Propeller class
        }
        self.quads['q1'] = q1_props
        
        for key in self.quads:
            self.quads[key]['state'] = np.zeros(12)
            self.quads[key]['state'][0:3] = self.quads[key]['position']
            self.quads[key]['state'][6:9] = self.quads[key]['orientation']
            self.quads[key]['m1'] = Propeller(self.quads[key]['prop_size'][0],self.quads[key]['prop_size'][1])
            self.quads[key]['m2'] = Propeller(self.quads[key]['prop_size'][0],self.quads[key]['prop_size'][1])
            self.quads[key]['m3'] = Propeller(self.quads[key]['prop_size'][0],self.quads[key]['prop_size'][1])
            self.quads[key]['m4'] = Propeller(self.quads[key]['prop_size'][0],self.quads[key]['prop_size'][1])
            # From Quadrotor Dynamics and Control by Randal Beard
            ixx=((2*self.quads[key]['weight']*self.quads[key]['r']**2)/5)+(2*self.quads[key]['weight']*self.quads[key]['L']**2)
            iyy=ixx
            izz=((2*self.quads[key]['weight']*self.quads[key]['r']**2)/5)+(4*self.quads[key]['weight']*self.quads[key]['L']**2)
            self.quads[key]['I'] = np.array([[ixx,0,0],[0,iyy,0],[0,0,izz]])
            self.quads[key]['invI'] = np.linalg.inv(self.quads[key]['I'])
        self.run = True
    
        
    def rotation_matrix(self,angles):
        ct = math.cos(angles[0])
        cp = math.cos(angles[1])
        cg = math.cos(angles[2])
        st = math.sin(angles[0])
        sp = math.sin(angles[1])
        sg = math.sin(angles[2])
        R_x = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
        R_y = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        R_z = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R

    def wrap_angle(self,val):
        return( ( val + np.pi) % (2 * np.pi ) - np.pi )

    def state_dot(self, time, state, key):
        state_dot = np.zeros(12)
        # The velocities(t+1 x_dots equal the t x_dots)
        state_dot[0] = self.quads[key]['state'][3]
        state_dot[1] = self.quads[key]['state'][4]
        state_dot[2] = self.quads[key]['state'][5]
        # The acceleration
        x_dotdot = np.array([0,0,-self.quads[key]['weight']*self.g]) + np.dot(self.rotation_matrix(self.quads[key]['state'][6:9]),np.array([0,0,(self.quads[key]['m1'].thrust + self.quads[key]['m2'].thrust + self.quads[key]['m3'].thrust + self.quads[key]['m4'].thrust)]))/self.quads[key]['weight']
        state_dot[3] = x_dotdot[0]
        state_dot[4] = x_dotdot[1]
        state_dot[5] = x_dotdot[2]
        # The angular rates(t+1 theta_dots equal the t theta_dots)
        state_dot[6] = self.quads[key]['state'][9]
        state_dot[7] = self.quads[key]['state'][10]
        state_dot[8] = self.quads[key]['state'][11]
        # The angular accelerations
        omega = self.quads[key]['state'][9:12]
        tau = np.array([self.quads[key]['L']*(self.quads[key]['m1'].thrust-self.quads[key]['m3'].thrust), self.quads[key]['L']*(self.quads[key]['m2'].thrust-self.quads[key]['m4'].thrust), self.b*(self.quads[key]['m1'].thrust-self.quads[key]['m2'].thrust+self.quads[key]['m3'].thrust-self.quads[key]['m4'].thrust)])
        omega_dot = np.dot(self.quads[key]['invI'], (tau - np.cross(omega, np.dot(self.quads[key]['I'],omega))))
        state_dot[9] = omega_dot[0]
        state_dot[10] = omega_dot[1]
        state_dot[11] = omega_dot[2]
        return state_dot
    
    def apply_force(self, quad_id, prop_pitch, speed, force):
        quad = self.quads[quad_id]
        qd = self.get_linear_rate(quad_id)

        # Convert prop_pitch to radians
        prop_pitch_rad = np.radians(prop_pitch)
        # Ensure the 'forces' and 'torques' keys are present in the quad dictionary
        if 'forces' not in quad:
            quad['forces'] = {}
        if 'torques' not in quad:
            quad['torques'] = {}

        # Calculate thrust and torque
        prop_thrust = quad['prop'].thrust
        thrust = prop_thrust * np.cos(prop_pitch_rad)
        torque = thrust * quad['L'] * np.sin(prop_pitch_rad)

        # Apply forces and torques to the quadcopter
        quad['forces']['thrust'] = np.array([0, 0, thrust])  # Thrust force
        quad['forces']['drag'] = -self.drag_coefficient * qd  # Drag force
       # quad['forces']['wind'] = wind_force  # Wind force
        quad['torques']['roll'] = np.array([torque, 0, 0])  # Roll torque
        quad['torques']['pitch'] = np.array([0, -torque, 0])  # Pitch torque
        quad['torques']['yaw'] = np.array([0, 0, 0])  # Yaw torque

    def update(self, dt):
        for key in self.quads:
            quad = self.quads[key]

            # Apply forces if available
            if key in self.forces:
                self.apply_force(key, quad['prop']['prop_pitch'], quad['prop']['speed'], self.forces[key])
                self.forces.pop(key)

            # Calculate acceleration and angular acceleration
            quad_state = quad['state']
            f = np.array([0, 0, -quad['weight'] * self.g])  # Gravity
            f += np.sum(list(quad['forces'].values()), axis=0)  # Total forces
            omega = self.get_angular_rate(key)  # Angular velocity
            qdd = f / quad['weight']  # Linear acceleration

            # Calculate torque
            torques = np.sum(list(quad['torques'].values()), axis=0)
            torques -= np.cross(omega, np.dot(quad['I'], omega))

            quad_state[9:12] += np.dot(dt, np.dot(quad['invI'], torques))  # Angular acceleration
            
            # Integrate angular velocity to obtain orientation
            quad_state[6:9] += np.dot(dt, omega)
            quad_state[3:6] += np.dot(dt, qdd)

            # Update the state using an ODE solver
            self.ode.set_initial_value(quad['state'], 0).set_f_params(key)
            quad['state'] = self.ode.integrate(self.ode.t + dt)

            # Wrap the orientation angles
            quad['state'][6:9] = self.wrap_angle(quad['state'][6:9])

            # Ensure the altitude remains non-negative
            quad['state'][2] = max(0, quad['state'][2])
                        
    def set_motor_speeds(self,quad_name,speeds):
        self.quads[quad_name]['m1'].set_speed(speeds[0])
        self.quads[quad_name]['m2'].set_speed(speeds[1])
        self.quads[quad_name]['m3'].set_speed(speeds[2])
        self.quads[quad_name]['m4'].set_speed(speeds[3])

    def get_position(self,quad_name):
        return self.quads[quad_name]['state'][0:3]

    def get_linear_rate(self,quad_name):
        return self.quads[quad_name]['state'][3:6]

    def get_orientation(self,quad_name):
        return self.quads[quad_name]['state'][6:9]

    def get_angular_rate(self,quad_name):
        return self.quads[quad_name]['state'][9:12]

    def get_state(self,quad_name):
        return self.quads[quad_name]['state']

    def set_position(self,quad_name,position):
        self.quads[quad_name]['state'][0:3] = position

    def set_orientation(self,quad_name,orientation):
        self.quads[quad_name]['state'][6:9] = orientation

    def get_time(self):
        return self.time

    def thread_run(self,dt,time_scaling):
        rate = time_scaling*dt
        last_update = self.time
        while(self.run==True):
            time.sleep(0)
            self.time = datetime.datetime.now()
            if (self.time-last_update).total_seconds() > rate:
                self.update(dt)
                last_update = self.time

    def start_thread(self,dt=0.002,time_scaling=1):
        self.thread_object = threading.Thread(target=self.thread_run,args=(dt,time_scaling))
        self.thread_object.start()

    def stop_thread(self):
        self.run = False