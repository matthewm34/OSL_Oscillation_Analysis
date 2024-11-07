'''
*** Description ***

This code aims to find direction of gravity and rotation matrix of sensor
by combining 3-axis Accelerometer and Gyroscope through Madgwick's Algorithm.

*** Writer: Hanjun Kim @ GT EPIC ***
'''

import numpy as np
from math import atan2, atan

def norm2(vector):
    '''
    Calculate euclidean norm of input vector
    '''

    sqrval = []
    for l in range(0,len(vector)):
        sqrval.append(np.square(vector[l]))

    normVal = np.sqrt(np.sum(sqrval))

    return normVal

def qProd(a, b):
    '''
    Product of two input Quaternions
    a, b should be 4x1 vector (Quaternion)
    '''

    prod = np.array([
        [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]],
        [a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]],
        [a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]],
        [a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]
    ])

    return prod

def qConj(q):
    '''
    Conjugate of Quaternion input
    q should be 4x1 vector (Quaternion)
    '''

    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])

    return q_conj

def f(Q_se, S_s, method):
    '''
    Objective function to minimize
    Q_se: Normalized Quaternion of the Earth frame(e) relative to the Sensor frame(s)
    S_s: Sensor Measurement in the Sensor frame (Acceleration)
    '''
    #
    # Q_se = np.array([0,0,0,0]) ### For Test
    # S_s = np.array([[0,0,0]]) ### For Test

    q1 = Q_se[0]
    q2 = Q_se[1]
    q3 = Q_se[2]
    q4 = Q_se[3]

    ax = S_s[0]
    ay = S_s[1]
    az = S_s[2]

    if method == 1:
        objective_function = np.array([
            [2*(q2*q4 - q1*q3) - ax],
            [2*(q1*q2 + q3*q4) - ay],
            [2*(0.5- q2**2 - q3**2) - az]
            ])
    else:
        if method == 2:
            v = np.array([
            [2*(q2*q4 - q1*q3)],
            [2*(q1*q2 + q3*q4)],
            [2*(0.5- q2**2 - q3**2)]
            ])
            objective_function = np.cross(v.T, S_s)
            objective_function = objective_function.T

    return objective_function

def Jacobian_f(Q_se):
    '''
    Jacobian of Q_se
    Q_se: Normalized Quaternion of the Earth frame(e) relative to the Sensor frame(s)
    '''

    q1 = Q_se[0]
    q2 = Q_se[1]
    q3 = Q_se[2]
    q4 = Q_se[3]

    J = np.array([[-2*q3, 2*q4, -2*q1, 2*q2],
                 [2*q2, 2*q1, 2*q4, 2*q3],
                 [0, -4*q2, -4*q3, 0]])

    return J

def AHRS_Madgwick(data_acc, data_gyro, q_prev, beta, sample_period, method):
    '''
    data_acc: Accelerometer data (1x3) should be m/s^2
    data_gyro: Gyrosensor data (1x3), should be rad/second
    q_prev: Previous Quaternion matrix (1x4)
    beta: Learning Rate (0~1)
    Sample Period: 1/fs (For our case, fs = 100)
    '''

    acc_norm = norm2(data_acc)
    S_s = data_acc/acc_norm

    objective_function = f(q_prev, S_s, method)
    J = Jacobian_f(q_prev)

    step = np.matmul(J.T,objective_function)
    step = step/norm2(step)

    gyro_array = np.array([0, data_gyro[0], data_gyro[1], data_gyro[2]])

    if method == 1:
        q_dot = 0.5 * qProd(q_prev, gyro_array) - beta * step
    else:
        if method == 2:
            gyro_array[1] -= 2 * objective_function[0]
            gyro_array[2] -= 2 * objective_function[1]
            gyro_array[3] -= 2 * objective_function[2]

            q_dot = 0.5 * qProd(q_prev, gyro_array)

    q_update = np.zeros(4)
    q_update[0] = q_prev[0] + q_dot[0] * sample_period
    q_update[1] = q_prev[1] + q_dot[1] * sample_period
    q_update[2] = q_prev[2] + q_dot[2] * sample_period
    q_update[3] = q_prev[3] + q_dot[3] * sample_period

    q_update = q_update/norm2(q_update)

    return q_update

def q2Rot(q):
    '''
    Converts Quaternion (1x4) to Rotation Matrix (3x3)
    '''

    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]

    Rmat = np.array([
        [2 * (q1**2) - 1 + 2 * (q2**2), 2 * (q2*q3 + q1*q4), 2 * (q2*q4 - q1*q3)],
        [2 * (q2*q3 - q1*q4), 2 * (q1**2) - 1 + 2 * (q3**2), 2 * (q3*q4 + q1*q2)],
        [2 * (q2*q4 + q1*q3), 2 * (q3*q4 - q1*q2), 2 * (q1**2) - 1 + 2 * (q4**2)]
    ])

    return Rmat

def q2Euler(q):
    '''
    Converts Quaternion (1x4) to Euler Angle (3x1)
    '''

    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]

    roll = atan2(2 * (q3*q4 - q1*q2), 2 * (q1**2) - 1 + 2 * (q4**2))
    pitch = -atan(2 * (q2*q4 + q1*q3) / np.sqrt(1 - (2 * q2*q4 + 2 * q1*q3)**2))
    yaw = atan2(2 * (q2*q3 - q1*q4), 2 * (q1**2) - 1 + 2 * (q2**2))

    EulerMat = np.array([roll, pitch, yaw])

    return EulerMat

def R2Euler(R):
    '''
    Converts Rotation Matrix (3x3) to Euler Angle (3x1)
    '''

    roll = atan2(R[2,1], R[2,2])
    pitch = -atan(R[2,0] / np.sqrt(1-R[2,0]**2))
    yaw = atan2(R[1,0], R[0,0])

    EulerMat = np.array([roll, pitch, yaw])

    return EulerMat

