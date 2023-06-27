'--------------------------------IMPORT------------------------------------------------------------'
import quadcopter
import gui
import controller
import signal
import sys
import argparse
import math
import windspeed_direction
import importlib
import numpy as np
import time
from threading import Timer
import matplotlib.pyplot as plt

importlib.reload(windspeed_direction)

'------------------------------------CONSTANTS---------------------------------------------------------'
# Constants
TIME_SCALING = 1.0 # (kleiner is sneller) 1.0 -> Realtime, 0.0 -> Ren zo snel mogelijk
QUAD_DYNAMICS_UPDATE = 0.002 
CONTROLLER_DYNAMICS_UPDATE = 0.005 
run = True



'------------------------------SINGLEPOINT2POINT-----------------------------------------------------------'
def Single_Point2Point():
    # Set goals to go to
    GOALS = [(200, 300, 200)]
    YAWS = [0, 3.14, -1.54, 1.54]
    # Define the quadcopters
    QUADCOPTER = {'q1': {'position': [0, 0, 0], 'orientation': [0, 0, 0], 'L': 0.3, 'r': 0.1, 'prop_size': [10, 4.5],
                        'weight': 1.2}} #1.2
    # Controller parameters
    CONTROLLER_PARAMETERS = {'Motor_limits': [6000, 9000],  
                             'Tilt_limits': [-10, 10],
                             'Yaw_Control_Limits': [-900, 900],
                             'Z_XY_offset': 500,
                             'Linear_PID': {'P': [1000, 1000, 11000], 'I': [0.04, 0.04, 4.5], 'D': [1200, 1200, 9000]}, 
                             'Linear_To_Angular_Scaler': [1, 1, 0],
                             'Yaw_Rate_Scaler': 0.18,
                             'Angular_PID': {'P': [22000, 22000, 1500], 'I': [0, 0, 1.2], 'D': [12000, 12000, 0]},
                             }

    # Catch Ctrl+C to stop threads
    signal.signal(signal.SIGINT, signal_handler)
    # Make objects for quadcopter, gui, and controller
    quad = quadcopter.Quadcopter(QUADCOPTER)
    gui_object = gui.GUI(quads=QUADCOPTER)
    ctrl = controller.Controller_PID_Point2Point(quad.get_state, quad.get_time, quad.set_motor_speeds,
                                                 params=CONTROLLER_PARAMETERS, quad_identifier='q1')
   
    quad.start_thread(dt=QUAD_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
    ctrl.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
    
    windsnelheden = np.load("simulation_windsnelheden.npy")
    windrichtingen = np.load("simulation_windrichtingen.npy")
        
    prop_pitch = 5.0  
    speed = 1000  
       
    iterations = int(60 / QUAD_DYNAMICS_UPDATE)
    Timer(60, stop_simulation).start()
    
    start_time = time.time()
    
    positions = []
    orientations = []
    velocities = []
    times = []
    
    for i in range(iterations):
        current_time = time.time() - start_time
        if current_time >= 60:
            break
        
        for goal, y in zip(GOALS, YAWS):
            ctrl.update_target(goal)
            ctrl.update_yaw_target(y)
            
            gui_object.quads['q1']['position'] = quad.get_position('q1')
            gui_object.quads['q1']['orientation'] = quad.get_orientation('q1')
            gui_object.update()
                        
            current_timestamp = i * CONTROLLER_DYNAMICS_UPDATE * TIME_SCALING
            current_windsnelheid = windsnelheden[int(current_timestamp)]
            current_windrichting = windrichtingen[int(current_timestamp)]
            
            # radialen
            current_windrichting_rad = math.radians(current_windrichting)
            
            # vector
            wind_force_magnitude = current_windsnelheid
            wind_force = wind_force_magnitude * np.array(
                [math.cos(current_windrichting_rad), math.sin(current_windrichting_rad), 0])
            
            # richting
            desired_direction = np.array(goal) - np.array(quad.get_position('q1'))
            
            # adjusten
            adjusted_direction = desired_direction - wind_force
            
            # update
            ctrl.update_target(adjusted_direction)
            
            # kracht
            quad.apply_force('q1', prop_pitch, speed, wind_force)
            ctrl.update()
            
            # checken
            current_position = quad.get_position('q1')
            goal_reached = np.linalg.norm(np.array(current_position) - np.array(goal)) < 0.1
            if goal_reached:
                break  
         
        positions.append(quad.get_position('q1'))
        orientations.append(quad.get_orientation('q1'))
        velocities.append(quad.get_linear_rate('q1'))
        times.append(current_time)
    
    quad.stop_thread()
    ctrl.stop_thread()
    
    plot_motion(positions, orientations,velocities, times)


def stop_simulation():
    global run
    run = False

def signal_handler(signal, frame):
    global run
    run = False
    print('Stopping')
    sys.exit(0)
    


#----------------------------------------------------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Quadcopter Simulator")
    parser.add_argument("--sim", help='single_p2p, multi_p2p or single_velocity', default='single_p2p')
    parser.add_argument("--time_scale", type=float, default=-1.0, help='Time scaling factor. 0.0:fastest,1.0:realtime,>1:slow, ex: --time_scale 0.1')
    parser.add_argument("--quad_update_time", type=float, default=0.0, help='delta time for quadcopter dynamics update(seconds), ex: --quad_update_time 0.002')
    parser.add_argument("--controller_update_time", type=float, default=0.0, help='delta time for controller update(seconds), ex: --controller_update_time 0.005')
    return parser.parse_args()

#----------------------------------------PLOT------------------------------------------------------

def plot_motion(positions, orientations, velocities, times):
    x = [pos[0] for pos in positions]
    y = [pos[1] for pos in positions]
    z = [pos[2] for pos in positions]
    
    roll = [r[0] for r in orientations]
    pitch = [p[1] for p in orientations]
    yaw = [y[2] for y in orientations]    
    vx = [vel[0] for vel in velocities]
    vy = [vel[1] for vel in velocities]
    vz = [vel[2] for vel in velocities]
   
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    # Plot positie
    axs[0].plot(times, x, label='X')
    axs[0].plot(times, y, label='Y')
    axs[0].plot(times, z, label='Z')
    axs[0].set_title('Positie')
    axs[0].set_xlabel('Tijd (s)')
    axs[0].set_ylabel('Positie (m)')
    axs[0].legend()

    # Plot oriëntatie
    axs[1].plot(times, roll, label='Roll')
    axs[1].plot(times, pitch, label='Pitch')
    axs[1].plot(times, yaw, label='Yaw')
    axs[1].set_title('Oriëntatie')
    axs[1].set_xlabel('Tijd (s)')
    axs[1].set_ylabel('Oriëntatie (°)')
    axs[1].legend(loc='lower left')

    # Plot Snelheid
    axs[2].plot(times, vx, label='Vx')
    axs[2].plot(times, vy, label='Vy')
    axs[2].plot(times, vz, label='Vz')
    axs[2].set_title('Snelheid')
    axs[2].set_xlabel('Tijd (s)')
    axs[2].set_ylabel('Snelheid (m/s)')
    axs[2].legend()
   
    plt.tight_layout()   
    plt.show()



#--------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    if args.time_scale>=0: TIME_SCALING = args.time_scale
    if args.quad_update_time>0: QUAD_DYNAMICS_UPDATE = args.quad_update_time
    if args.controller_update_time>0: CONTROLLER_DYNAMICS_UPDATE = args.controller_update_time
    if args.sim == 'single_p2p':
        Single_Point2Point()
    
