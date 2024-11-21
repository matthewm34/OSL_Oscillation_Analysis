# Code Outline for Convolving ZV Shaper in real time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import useful_functions

# Example torque curve (torque_vec) and timesteps_vec (don't need these for real time implimentation, only used for simulation)
x = np.arange(0,100,1)                           
torque_vec = np.concatenate((np.ones(len(x)), np.zeros(50))) # pretend the torque vec is unit step function
timesteps_vec = np.arange(0,150,1)

# Convolve function shaping in real time
def convolve(cur_torque,time_step):
    A_1 = cur_torque * Amplitude_1
    buffer[time_delay + time_step]= cur_torque * Amplitude_2
    A_2 = buffer[time_step]

    torque_output = A_1 + A_2 #output the torque

    return torque_output
    

# Initialize our ZV Shaper Parameters
Amplitude_1 = .5
Amplitude_2 = .5
time_delay = 9  

# Convolution initialization parameters
time_step = 0 
buffer = np.zeros(1000) # make sure buffer is at least of length: max(time_step) + (time_delay)  
shaped_torque_output_vec = []


# SIMULATION:
# Simulate the OSl code running and developing a torque each timestep (in real time)
for cur_torque in torque_vec:
    
    torque_output = convolve(cur_torque,time_step)
    time_step = time_step + 1
    shaped_torque_output_vec.append(torque_output)


# plotting shaped torque result 
plt.figure(1)
plt.plot(timesteps_vec, shaped_torque_output_vec)
plt.title('Shaped Torque')
plt.xlabel('Time Steps')
plt.draw()
