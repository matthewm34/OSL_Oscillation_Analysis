# Code Outline for Convolving ZVD Shaper in real time
    # designed to shape the torque only during the swing phase of the OSL

import numpy as np
import matplotlib.pyplot as plt


# Example torque curve (torque_vec) and timesteps_vec (don't need these for real time implimentation, only used for simulation)
x = np.arange(0,256,1)      
torque_vec = np.concatenate((np.ones(len(x)), np.zeros(50)))  # pretend the torque vec is unit step function
timesteps_vec = np.arange(0,306,1)


# Convolve function shaping in real time
def convolve_EI(cur_torque,time_step):
    A_1 = cur_torque * Amplitude_1
    buffer_1[time_delay1 + time_step]= cur_torque * Amplitude_2
    A_2 = buffer_1[time_step]
    buffer_2[time_delay2 + time_step]= cur_torque * Amplitude_3
    A_3 = buffer_2[time_step]

    torque_output = A_1 + A_2 + A_3 #output the torque

    return torque_output
    

# Initialize our ZV Shaper Parameters
Amplitude_1 = 1.05/4
Amplitude_2 = 0.95/2
Amplitude_3 = 1.05/4
time_delay1 = 9  
time_delay2 = 19

# Convolution initialization parameters
time_step = 0                   # first time step of the swing phase should be indexed as 0
buffer_1 = np.zeros(1000)         # make sure buffer is at least of length: max(time_step) + (time_delay)  
buffer_2 = np.zeros(1000) 
shaped_torque_output_vec = []   # only for plotting 


# SIMULATION:
# Simulate the OSl code running and developing a torque each timestep (in real time)
for cur_torque in torque_vec:

    # in OSL code check if (in swing phase) -> then run this code and shape torque
    torque_output = convolve_EI(cur_torque,time_step)   # calculated the shaped torque
    time_step = time_step + 1                        # increase time_step
    shaped_torque_output_vec.append(torque_output)   # only for plotting 


# plotting shaped torque result 
plt.figure(1)
plt.plot(timesteps_vec, shaped_torque_output_vec)
plt.title('Shaped torque')
plt.xlabel('Time Steps')
plt.draw()

