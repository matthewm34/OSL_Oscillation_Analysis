# Code Outline for Convolving ZVD Shaper in real time
    # designed to shape the torque only during the swing phase of the OSL

import numpy as np
import matplotlib.pyplot as plt


# Example k curve (k_vec) and timesteps_vec (don't need these for real time implimentation, only used for simulation)
x = np.arange(0,256,1)      
k_vec = np.concatenate((np.ones(len(x)), np.zeros(50)))  # pretend the k vec is unit step function
timesteps_vec = np.arange(0,306,1)


# Convolve function shaping in real time
def convolve_EI(cur_k,time_step):
    A_1 = cur_k * Amplitude_1
    buffer_1[time_delay1 + time_step]= cur_k * Amplitude_2
    A_2 = buffer_1[time_step]
    buffer_2[time_delay2 + time_step]= cur_k * Amplitude_3
    A_3 = buffer_2[time_step]

    k_output = A_1 + A_2 + A_3 #output the k

    return k_output
    

# Initialize our ZV Shaper Parameters
Amplitude_1 = .25
Amplitude_2 = .5
Amplitude_3 = .25
time_delay1 = 10  
time_delay2 = 20

# Convolution initialization parameters
time_step = 0                   # first time step of the swing phase should be indexed as 0
buffer_1 = np.zeros(1000)         # make sure buffer is at least of length: max(time_step) + (time_delay)  
buffer_2 = np.zeros(1000) 
shaped_k_output_vec = []   # only for plotting 


# SIMULATION:
# Simulate the OSl code running and developing a k each timestep (in real time)
for cur_k in k_vec:

    # in OSL code check if (in swing phase) -> then run this code and shape k
    k_output = convolve_EI(cur_k,time_step)   # calculated the shaped k
    time_step = time_step + 1                        # increase time_step
    shaped_k_output_vec.append(k_output)   # only for plotting 


# plotting shaped k result 
plt.figure(1)
plt.plot(timesteps_vec, shaped_k_output_vec)
plt.title('Shaped k')
plt.xlabel('Time Steps')
plt.draw()
plt.show()

