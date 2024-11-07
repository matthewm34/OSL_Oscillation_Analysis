'''
*** Description ***

.bag file processing pipeline
1. Load .bag file and convert into Python dictionary form
2. Align topics of main interest in same timestamp: SensorData, SensorInfo, Foot & Thigh IMU, FSM
3. Set slope ground truth w.r.t. different definitions (start/end timings are defined differently during MidSwing)
...

*** Writer: Hanjun Kim @ GT EPIC ***
'''

import shutil
from bagpy import bagreader as bRead
import os
import pandas as pd
from useful_functions.align_sensor2ES_v2 import closest_point
import numpy as np
from scipy.interpolate import interp1d as itpd
import matplotlib.pyplot as plt
from useful_functions import AHRS_Functions as AHRS
import math
from useful_functions import Functions_SP as fncs

def checkTopics(rosbagFilename):
    '''
    This function checks topics before .dict file is extracted from .bag files
    rosbagFilename: Exact file location including ".bag"
    ex)rosbagFilename = 'C:/Users/hkim910/Documents/MATLAB/OSL_testData/32.bag'
    '''

    b = bRead(rosbagFilename)

    if os.path.exists(rosbagFilename[:-4] + '/'):  ### This will remove redundant folder created by bRead function
        # print('File or Directory with same name exists...Removing')
        shutil.rmtree(rosbagFilename[:-4] + '/')

    ### Extract entire topics
    topics = b.topic_table.Topics
    return topics

### Current Experimental Data: Use read2var or read2var2
def read2var(bagfilename, topics_include):

    '''
    rosbagDir: directory that includes your .bag files
    ex)rosbagDir = 'C:/Users/hkim910/Documents/MATLAB/OSL_testData/'

    topics_include: list of topics that you want to include in your data file
    ex)topics_include = ['/SensorData', '/fsm/State', '/matlab/ground_truth',
                      '/ml/estimation_filtered', '/ml/estimation_unfiltered']

    ### Example Available Topic list#### (Subject to change)
    0                     /SensorData ##
    1                     /SensorInfo ##
    2     /decision_making/fsm/events ##
    3                      /fsm/State ##
    4                    /fsm/command ##
    5                  /fsm/commandGT ##
    6                    /fsm/context ##
    7                 /fsm/delay_flag ##
    8            /matlab/ground_truth ##
    9                      /ml/enable ##
    10             /ml/enable_command ##
    11        /ml/estimation_filtered ##
    12      /ml/estimation_unfiltered ##
    13        /ml/features_continuous ##
    14          /ml/features_discrete ##
    ### Example Available Topic list####
    '''

    ### Test
    # bagfilename = 'D:\Hanjun\OSL_SlopeEstimation\Ramp_Data_Raw\TF15v2/10.bag'
    # topics_include = checkTopics(bagfilename)
    ### Test


    print('Bag File Processing: ', bagfilename)
    b = bRead(bagfilename)

    if os.path.exists(bagfilename[:-4]): ### This will remove redundant folder created by bRead function
        shutil.rmtree(bagfilename[:-4])

    ### Extract entire topics
    topics = b.topic_table.Topics
    target_topics_list = []

    ### Check if every topic of interest is included in raw data
    for t in range(0, len(topics_include)):
        topic_candidate = topics_include[t]
        # print(topic_candidate)

        if topic_candidate in topics.tolist():
            target_topics_list.append(topic_candidate)

    ##### Extract Raw message list from .bag file
    ### Save messages
    combDat_dict = {}

    for target_topic in target_topics_list:
        msg_list = []

        for msg in b.reader.read_messages(topics=target_topic):

            msg_list.append(msg)

            if len(msg_list) == 1:

                slotlist = msg_list[0][1].__slots__  ### slotlist provides 'key' strings for dictionaries
                dict = {}

            if target_topic == '/SensorInfo':

                if len(msg_list) == 1:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message
                    time_in_float = time_val.to_sec()
                    dict['header'] = [time_in_float]

                    knee_setpoint_orgmsg = [getattr(msg_list[-1][1], 'knee_setpoint')]  ### [-1] extracts latest message
                    ankle_setpoint_orgmsg = [getattr(msg_list[-1][1], 'ankle_setpoint')]  ### [-1] extracts latest message

                    # Crop torque values from msg
                    knee_torque_setpoint = float(str(knee_setpoint_orgmsg).rsplit('torque_setpoint: ')[1].rsplit('\nk')[0])
                    knee_torque_applied = getattr(msg_list[-1][1], 'knee_torque_applied')
                    knee_k = float(str(knee_setpoint_orgmsg).rsplit('k: ')[1].rsplit('\nb')[0])
                    knee_b = float(str(knee_setpoint_orgmsg).rsplit('b: ')[1].rsplit('\nt')[0])
                    knee_theta_eq = float(str(knee_setpoint_orgmsg).rsplit('theta_eq: ')[1].rsplit('\nzk')[0])
                    knee_zk = float(str(knee_setpoint_orgmsg).rsplit('zk: ')[1].rsplit('\nzb')[0])
                    knee_zb = float(str(knee_setpoint_orgmsg).rsplit('zb: ')[1].rsplit('\nztheta_eq')[0])
                    # knee_ztheta_eq = float(str(knee_setpoint_orgmsg).rsplit('ztheta_eq: ')[1].rsplit('\nankle')[0])
                    knee_ztheta_eq = float(str(knee_setpoint_orgmsg).rsplit('ztheta_eq: ')[1].rsplit(']')[0])
                    # knee_theta_mot = float(str(knee_setpoint_orgmsg).rsplit('knee_theta_mot: ')[1].rsplit('\nknee')[0])
                    # knee_thetadot_mot = float(str(knee_setpoint_orgmsg).rsplit('knee_thetadot_mot: ')[1].rsplit(']')[0])

                    ankle_torque_setpoint = float(str(ankle_setpoint_orgmsg).rsplit('torque_setpoint: ')[1].rsplit('\nk')[0])
                    ankle_torque_applied = getattr(msg_list[-1][1], 'ankle_torque_applied')
                    ankle_k = float(str(ankle_setpoint_orgmsg).rsplit('k: ')[1].rsplit('\nb')[0])
                    ankle_b = float(str(ankle_setpoint_orgmsg).rsplit('b: ')[1].rsplit('\nt')[0])
                    ankle_theta_eq = float(str(ankle_setpoint_orgmsg).rsplit('theta_eq: ')[1].rsplit('\nzk')[0])
                    ankle_zk = float(str(ankle_setpoint_orgmsg).rsplit('zk: ')[1].rsplit('\nzb')[0])
                    ankle_zb = float(str(ankle_setpoint_orgmsg).rsplit('zb: ')[1].rsplit('\nztheta_eq')[0])
                    # ankle_ztheta_eq = float(str(ankle_setpoint_orgmsg).rsplit('ztheta_eq: ')[1].rsplit('\nankle')[0])
                    ankle_ztheta_eq = float(str(ankle_setpoint_orgmsg).rsplit('ztheta_eq: ')[1].rsplit(']')[0])
                    # ankle_theta_mot = float(str(ankle_setpoint_orgmsg).rsplit('ankle_theta_mot: ')[1].rsplit('\nankle')[0])
                    # ankle_thetadot_mot = float(str(ankle_setpoint_orgmsg).rsplit('ankle_thetadot_mot: ')[1].rsplit('\nknee')[0])

                    dict['knee_torque_setpoint'] = [knee_torque_setpoint] ### [-1] extracts latest message
                    dict['knee_torque_applied'] = [knee_torque_applied]  ### [-1] extracts latest message
                    dict['knee_k'] = [knee_k]  ### [-1] extracts latest message
                    dict['knee_b'] = [knee_b]  ### [-1] extracts latest message
                    dict['knee_theta_eq'] = [knee_theta_eq]  ### [-1] extracts latest message
                    dict['knee_zk'] = [knee_zk]  ### [-1] extracts latest message
                    dict['knee_zb'] = [knee_zb]  ### [-1] extracts latest message
                    dict['knee_ztheta_eq'] = [knee_ztheta_eq]  ### [-1] extracts latest message
                    # dict['knee_theta_mot'] = [knee_theta_mot]  ### [-1] extracts latest message
                    # dict['knee_thetadot_mot'] = [knee_thetadot_mot]  ### [-1] extracts latest message

                    dict['ankle_torque_setpoint'] = [ankle_torque_setpoint]  ### [-1] extracts latest message
                    dict['ankle_torque_applied'] = [ankle_torque_applied]  ### [-1] extracts latest message
                    dict['ankle_k'] = [ankle_k]  ### [-1] extracts latest message
                    dict['ankle_b'] = [ankle_b]  ### [-1] extracts latest message
                    dict['ankle_theta_eq'] = [ankle_theta_eq]  ### [-1] extracts latest message
                    dict['ankle_zk'] = [ankle_zk]  ### [-1] extracts latest message
                    dict['ankle_zb'] = [ankle_zb]  ### [-1] extracts latest message
                    dict['ankle_ztheta_eq'] = [ankle_ztheta_eq]  ### [-1] extracts latest message
                    # dict['ankle_theta_mot'] = [ankle_theta_mot]  ### [-1] extracts latest message
                    # dict['ankle_thetadot_mot'] = [ankle_thetadot_mot]  ### [-1] extracts latest message


                else:
                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message
                    time_in_float = time_val.to_sec()
                    dict['header'].append(time_in_float)

                    knee_setpoint_orgmsg = [getattr(msg_list[-1][1], 'knee_setpoint')]  ### [-1] extracts latest message
                    ankle_setpoint_orgmsg = [getattr(msg_list[-1][1], 'ankle_setpoint')]  ### [-1] extracts latest message

                    # Crop torque values from msg
                    knee_torque_setpoint = float(str(knee_setpoint_orgmsg).rsplit('torque_setpoint: ')[1].rsplit('\nk')[0])
                    knee_torque_applied = getattr(msg_list[-1][1], 'knee_torque_applied')
                    knee_k = float(str(knee_setpoint_orgmsg).rsplit('k: ')[1].rsplit('\nb')[0])
                    knee_b = float(str(knee_setpoint_orgmsg).rsplit('b: ')[1].rsplit('\nt')[0])
                    knee_theta_eq = float(str(knee_setpoint_orgmsg).rsplit('theta_eq: ')[1].rsplit('\nzk')[0])
                    knee_zk = float(str(knee_setpoint_orgmsg).rsplit('zk: ')[1].rsplit('\nzb')[0])
                    knee_zb = float(str(knee_setpoint_orgmsg).rsplit('zb: ')[1].rsplit('\nztheta_eq')[0])
                    # knee_ztheta_eq = float(str(knee_setpoint_orgmsg).rsplit('ztheta_eq: ')[1].rsplit('\nankle')[0])
                    knee_ztheta_eq = float(str(knee_setpoint_orgmsg).rsplit('ztheta_eq: ')[1].rsplit(']')[0])
                    # knee_theta_mot = float(str(knee_setpoint_orgmsg).rsplit('knee_theta_mot: ')[1].rsplit('\nknee')[0])
                    # knee_thetadot_mot = float(str(knee_setpoint_orgmsg).rsplit('knee_thetadot_mot: ')[1].rsplit(']')[0])

                    ankle_torque_setpoint = float(str(ankle_setpoint_orgmsg).rsplit('torque_setpoint: ')[1].rsplit('\nk')[0])
                    ankle_torque_applied = getattr(msg_list[-1][1], 'ankle_torque_applied')
                    ankle_k = float(str(ankle_setpoint_orgmsg).rsplit('k: ')[1].rsplit('\nb')[0])
                    ankle_b = float(str(ankle_setpoint_orgmsg).rsplit('b: ')[1].rsplit('\nt')[0])
                    ankle_theta_eq = float(str(ankle_setpoint_orgmsg).rsplit('theta_eq: ')[1].rsplit('\nzk')[0])
                    ankle_zk = float(str(ankle_setpoint_orgmsg).rsplit('zk: ')[1].rsplit('\nzb')[0])
                    ankle_zb = float(str(ankle_setpoint_orgmsg).rsplit('zb: ')[1].rsplit('\nztheta_eq')[0])
                    # ankle_ztheta_eq = float(str(ankle_setpoint_orgmsg).rsplit('ztheta_eq: ')[1].rsplit('\nankle')[0])
                    ankle_ztheta_eq = float(str(ankle_setpoint_orgmsg).rsplit('ztheta_eq: ')[1].rsplit(']')[0])
                    # ankle_theta_mot = float(str(ankle_setpoint_orgmsg).rsplit('ankle_theta_mot: ')[1].rsplit('\nankle')[0])
                    # ankle_thetadot_mot = float(str(ankle_setpoint_orgmsg).rsplit('ankle_thetadot_mot: ')[1].rsplit('\nknee')[0])

                    dict['knee_torque_setpoint'].append(knee_torque_setpoint) ### [-1] extracts latest message
                    dict['knee_torque_applied'].append(knee_torque_applied)  ### [-1] extracts latest message
                    dict['knee_k'].append(knee_k)  ### [-1] extracts latest message
                    dict['knee_b'].append(knee_b)  ### [-1] extracts latest message
                    dict['knee_theta_eq'].append(knee_theta_eq)  ### [-1] extracts latest message
                    dict['knee_zk'].append(knee_zk)  ### [-1] extracts latest message
                    dict['knee_zb'].append(knee_zb)  ### [-1] extracts latest message
                    dict['knee_ztheta_eq'].append(knee_ztheta_eq)  ### [-1] extracts latest message
                    # dict['knee_theta_mot'].append(knee_theta_mot)  ### [-1] extracts latest message
                    # dict['knee_thetadot_mot'].append(knee_thetadot_mot)  ### [-1] extracts latest message

                    dict['ankle_torque_setpoint'].append(ankle_torque_setpoint)  ### [-1] extracts latest message
                    dict['ankle_torque_applied'].append(ankle_torque_applied)  ### [-1] extracts latest message
                    dict['ankle_k'].append(ankle_k)  ### [-1] extracts latest message
                    dict['ankle_b'].append(ankle_b)  ### [-1] extracts latest message
                    dict['ankle_theta_eq'].append(ankle_theta_eq)  ### [-1] extracts latest message
                    dict['ankle_zk'].append(ankle_zk)  ### [-1] extracts latest message
                    dict['ankle_zb'].append(ankle_zb)  ### [-1] extracts latest message
                    dict['ankle_ztheta_eq'].append(ankle_ztheta_eq)  ### [-1] extracts latest message
                    # dict['ankle_theta_mot'].append(ankle_theta_mot)  ### [-1] extracts latest message
                    # dict['ankle_thetadot_mot'].append(ankle_thetadot_mot)  ### [-1] extracts latest message

            elif target_topic == '/foot2/imu/data':

                if len(msg_list) == 1:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message

                    time_in_float = time_val.to_sec()

                    dict['header'] = [time_in_float]

                    gyro_orgmsg = [getattr(msg_list[-1][1], 'angular_velocity')]  ### [-1] extracts latest message

                    accel_orgmsg = [getattr(msg_list[-1][1], 'linear_acceleration')]  ### [-1] extracts latest message

                    # Crop torque values from msg

                    gyro_x = float(str(gyro_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])

                    gyro_y = float(str(gyro_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])

                    gyro_z = float(str(gyro_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    acc_x = float(str(accel_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])

                    acc_y = float(str(accel_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])

                    acc_z = float(str(accel_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    dict['foot_gyro_x'] = [gyro_x]

                    dict['foot_gyro_y'] = [gyro_y]

                    dict['foot_gyro_z'] = [gyro_z]

                    dict['foot_acc_x'] = [acc_x]

                    dict['foot_acc_y'] = [acc_y]

                    dict['foot_acc_z'] = [acc_z]


                else:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message

                    time_in_float = time_val.to_sec()

                    dict['header'].append(time_in_float)

                    gyro_orgmsg = [getattr(msg_list[-1][1], 'angular_velocity')]  ### [-1] extracts latest message

                    accel_orgmsg = [

                        getattr(msg_list[-1][1], 'linear_acceleration')]  ### [-1] extracts latest message

                    # Crop torque values from msg

                    gyro_x = float(str(gyro_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])

                    gyro_y = float(str(gyro_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])

                    gyro_z = float(str(gyro_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    acc_x = float(str(accel_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])

                    acc_y = float(str(accel_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])

                    acc_z = float(str(accel_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    dict['foot_gyro_x'].append(gyro_x)

                    dict['foot_gyro_y'].append(gyro_y)

                    dict['foot_gyro_z'].append(gyro_z)

                    dict['foot_acc_x'].append(acc_x)

                    dict['foot_acc_y'].append(acc_y)

                    dict['foot_acc_z'].append(acc_z)

            elif target_topic == '/foot2/nav/filtered_imu/data':

                if len(msg_list) == 1:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message

                    time_in_float = time_val.to_sec()

                    dict['header'] = [time_in_float]

                    foot_orientation_orgmsg = [
                        getattr(msg_list[-1][1], 'orientation')]  ### [-1] extracts latest message

                    # Crop torque values from msg

                    orien_x = float(str(foot_orientation_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])

                    orien_y = float(str(foot_orientation_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])

                    orien_z = float(str(foot_orientation_orgmsg).rsplit('z: ')[1].rsplit('\nw')[0])

                    orien_w = float(str(foot_orientation_orgmsg).rsplit('w: ')[1].rsplit(']')[0])

                    dict['orientation_x'] = [orien_x]

                    dict['orientation_y'] = [orien_y]

                    dict['orientation_z'] = [orien_z]

                    dict['orientation_w'] = [orien_w]


                else:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message

                    time_in_float = time_val.to_sec()

                    dict['header'].append(time_in_float)

                    foot_orientation_orgmsg = [
                        getattr(msg_list[-1][1], 'orientation')]  ### [-1] extracts latest message

                    # Crop torque values from msg

                    orien_x = float(str(foot_orientation_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])

                    orien_y = float(str(foot_orientation_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])

                    orien_z = float(str(foot_orientation_orgmsg).rsplit('z: ')[1].rsplit('\nw')[0])

                    orien_w = float(str(foot_orientation_orgmsg).rsplit('w: ')[1].rsplit(']')[0])

                    dict['orientation_x'].append(orien_x)

                    dict['orientation_y'].append(orien_y)

                    dict['orientation_z'].append(orien_z)

                    dict['orientation_w'].append(orien_w)

            elif target_topic == '/thigh/imu/data':

                if len(msg_list) == 1:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message

                    time_in_float = time_val.to_sec()

                    dict['header'] = [time_in_float]

                    gyro_orgmsg = [getattr(msg_list[-1][1], 'angular_velocity')]  ### [-1] extracts latest message

                    accel_orgmsg = [

                        getattr(msg_list[-1][1], 'linear_acceleration')]  ### [-1] extracts latest message

                    # Crop torque values from msg

                    gyro_x = float(str(gyro_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])

                    gyro_y = float(str(gyro_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])

                    gyro_z = float(str(gyro_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    acc_x = float(str(accel_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])

                    acc_y = float(str(accel_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])

                    acc_z = float(str(accel_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    dict['thigh_gyro_x'] = [gyro_x]

                    dict['thigh_gyro_y'] = [gyro_y]

                    dict['thigh_gyro_z'] = [gyro_z]

                    dict['thigh_acc_x'] = [acc_x]

                    dict['thigh_acc_y'] = [acc_y]

                    dict['thigh_acc_z'] = [acc_z]


                else:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message

                    time_in_float = time_val.to_sec()

                    dict['header'].append(time_in_float)

                    gyro_orgmsg = [getattr(msg_list[-1][1], 'angular_velocity')]  ### [-1] extracts latest message

                    accel_orgmsg = [

                        getattr(msg_list[-1][1], 'linear_acceleration')]  ### [-1] extracts latest message

                    # Crop torque values from msg

                    gyro_x = float(str(gyro_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])

                    gyro_y = float(str(gyro_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])

                    gyro_z = float(str(gyro_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    acc_x = float(str(accel_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])

                    acc_y = float(str(accel_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])

                    acc_z = float(str(accel_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    dict['thigh_gyro_x'].append(gyro_x)

                    dict['thigh_gyro_y'].append(gyro_y)

                    dict['thigh_gyro_z'].append(gyro_z)

                    dict['thigh_acc_x'].append(acc_x)

                    dict['thigh_acc_y'].append(acc_y)

                    dict['thigh_acc_z'].append(acc_z)

            elif target_topic == '/thigh/nav/filtered_imu/data':

                if len(msg_list) == 1:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message

                    time_in_float = time_val.to_sec()

                    dict['header'] = [time_in_float]

                    thigh_orientation_orgmsg = [

                        getattr(msg_list[-1][1], 'orientation')]  ### [-1] extracts latest message

                    # Crop torque values from msg

                    orien_x = float(str(thigh_orientation_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])

                    orien_y = float(str(thigh_orientation_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])

                    orien_z = float(str(thigh_orientation_orgmsg).rsplit('z: ')[1].rsplit('\nw')[0])

                    orien_w = float(str(thigh_orientation_orgmsg).rsplit('w: ')[1].rsplit(']')[0])

                    dict['orientation_x'] = [orien_x]

                    dict['orientation_y'] = [orien_y]

                    dict['orientation_z'] = [orien_z]

                    dict['orientation_w'] = [orien_w]


                else:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message

                    time_in_float = time_val.to_sec()

                    dict['header'].append(time_in_float)

                    thigh_orientation_orgmsg = [

                        getattr(msg_list[-1][1], 'orientation')]  ### [-1] extracts latest message

                    # Crop torque values from msg

                    orien_x = float(str(thigh_orientation_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])

                    orien_y = float(str(thigh_orientation_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])

                    orien_z = float(str(thigh_orientation_orgmsg).rsplit('z: ')[1].rsplit('\nw')[0])

                    orien_w = float(str(thigh_orientation_orgmsg).rsplit('w: ')[1].rsplit(']')[0])

                    dict['orientation_x'].append(orien_x)

                    dict['orientation_y'].append(orien_y)

                    dict['orientation_z'].append(orien_z)

                    dict['orientation_w'].append(orien_w)

            else:

                for slotname in slotlist:
                    if len(msg_list) == 1:
                        if slotname == 'header': ### Timestamp
                            time_val = getattr(msg_list[-1][1], slotname).stamp  ### [-1] extracts latest message
                            time_in_float = time_val.to_sec()
                            dict[slotname] = [time_in_float]

                        else: ### Sensor value or context
                            dict[slotname] = [getattr(msg_list[-1][1], slotname)] ### [-1] extracts latest message
                    else:
                        if slotname == 'header': ### Timestamp
                            time_val = getattr(msg_list[-1][1], slotname).stamp ### [-1] extracts latest message
                            time_in_float = time_val.to_sec()
                            dict[slotname].append(time_in_float)

                        else: ### Sensor value or context
                            dict[slotname].append(getattr(msg_list[-1][1], slotname)) ### [-1] extracts latest message

        ### Convert dictionary to dataframe
        # print('Generating DataFrame...')
        df = pd.DataFrame(data=dict)

        # print('Saving DataFrame in dictionary')
        combDat_dict[msg.topic] = df

    print('Included Topic: ', target_topics_list)
    return combDat_dict

def read2var2(bagfilename, topics_include):

    '''
    rosbagDir: directory that includes your .bag files
    ex)rosbagDir = 'C:/Users/hkim910/Documents/MATLAB/OSL_testData/'

    topics_include: list of topics that you want to include in your data file
    ex)topics_include = ['/SensorData', '/fsm/State', '/matlab/ground_truth',
                      '/ml/estimation_filtered', '/ml/estimation_unfiltered']

    ### Example Available Topic list#### (Subject to change)
    0                     /SensorData ##
    1                     /SensorInfo ##
    2     /decision_making/fsm/events ##
    3                      /fsm/State ##
    4                    /fsm/command ##
    5                  /fsm/commandGT ##
    6                    /fsm/context ##
    7                 /fsm/delay_flag ##
    8            /matlab/ground_truth ##
    9                      /ml/enable ##
    10             /ml/enable_command ##
    11        /ml/estimation_filtered ##
    12      /ml/estimation_unfiltered ##
    13        /ml/features_continuous ##
    14          /ml/features_discrete ##
    ### Example Available Topic list####
    '''

    # ### Test
    # workspace_path = os.getcwd()
    # BagDir = workspace_path + '/Stair_Data_Raw/TF15/'  # This gives us a str with the path and the file we are using
    # datalist = os.listdir(BagDir)  # this gives us a list all the TF's from the Ramp_Data
    # bagfilename = BagDir + datalist[1]
    # topics_include = checkTopics(bagfilename)
    # ### Test

    ### Test2
    # workspace_path = os.getcwd()
    # BagDir = workspace_path + '/Stair_PILOT/ABJM_09_11_23/'  # This gives us a str with the path and the file we are using
    # datalist = os.listdir(BagDir)  # this gives us a list all the TF's from the Ramp_Data
    #
    # target_bag_1 = BagDir + datalist[5]
    # topics_include = checkTopics(target_bag_1)
    ### Test2

    # ### Test3
    # workspace_path = os.getcwd()
    # BagDir = workspace_path + '/UniControl_PILOT/TFHK/'  # This gives us a str with the path and the file we are using
    # datalist = os.listdir(BagDir)  # this gives us a list all the TF's from the Ramp_Data
    #
    # target_bag_1 = BagDir + datalist[0]
    # topics_include = checkTopics(target_bag_1)
    # ### Test3


    print('Bag File Processing: ', bagfilename)
    b = bRead(bagfilename)

    # b = bRead(target_bag_1)

    if os.path.exists(bagfilename[:-4]): ### This will remove redundant folder created by bRead function
        shutil.rmtree(bagfilename[:-4])

    ### Extract entire topics
    topics = b.topic_table.Topics
    target_topics_list = []

    ### Check if every topic of interest is included in raw data
    for t in range(0, len(topics_include)):
        topic_candidate = topics_include[t]
        # print(topic_candidate)

        if topic_candidate in topics.tolist():
            target_topics_list.append(topic_candidate)

    ##### Extract Raw message list from .bag file
    ### Save messages
    combDat_dict = {}

    for target_topic in target_topics_list:
        msg_list = []

        for msg in b.reader.read_messages(topics=target_topic):

            msg_list.append(msg)

            if len(msg_list) == 1:

                slotlist = msg_list[0][1].__slots__  ### slotlist provides 'key' strings for dictionaries
                dict = {}

            if target_topic == '/SensorInfo':

                if len(msg_list) == 1:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message
                    time_in_float = time_val.to_sec()
                    dict['header'] = [time_in_float]

                    knee_setpoint_orgmsg = [getattr(msg_list[-1][1], 'knee_setpoint')]  ### [-1] extracts latest message
                    ankle_setpoint_orgmsg = [getattr(msg_list[-1][1], 'ankle_setpoint')]  ### [-1] extracts latest message

                    # Crop torque values from msg
                    knee_current_setpoint = float(str(knee_setpoint_orgmsg).rsplit('current_setpoint: ')[1].rsplit('\nt')[0])
                    knee_torque_setpoint = float(str(knee_setpoint_orgmsg).rsplit('torque_setpoint: ')[1].rsplit('\nk')[0])
                    knee_current_applied = getattr(msg_list[-1][1], 'knee_current_applied')
                    knee_torque_applied = getattr(msg_list[-1][1], 'knee_torque_applied')
                    knee_k = float(str(knee_setpoint_orgmsg).rsplit('k: ')[1].rsplit('\nb')[0])
                    knee_b = float(str(knee_setpoint_orgmsg).rsplit('b: ')[1].rsplit('\nt')[0])
                    knee_theta_eq = float(str(knee_setpoint_orgmsg).rsplit('theta_eq: ')[1].rsplit('\nzk')[0])
                    knee_zk = float(str(knee_setpoint_orgmsg).rsplit('zk: ')[1].rsplit('\nzb')[0])
                    knee_zb = float(str(knee_setpoint_orgmsg).rsplit('zb: ')[1].rsplit('\nztheta_eq')[0])
                    knee_ztheta_eq = float(str(knee_setpoint_orgmsg).rsplit('ztheta_eq: ')[1].rsplit('\nankle')[0])
                    # knee_ztheta_eq = float(str(knee_setpoint_orgmsg).rsplit('ztheta_eq: ')[1].rsplit(']')[0])
                    knee_theta_mot = float(str(knee_setpoint_orgmsg).rsplit('knee_theta_mot: ')[1].rsplit('\nknee')[0])
                    knee_thetadot_mot = float(str(knee_setpoint_orgmsg).rsplit('knee_thetadot_mot: ')[1].rsplit(']')[0])

                    ankle_current_setpoint = float(str(ankle_setpoint_orgmsg).rsplit('current_setpoint: ')[1].rsplit('\nt')[0])
                    ankle_torque_setpoint = float(str(ankle_setpoint_orgmsg).rsplit('torque_setpoint: ')[1].rsplit('\nk')[0])
                    ankle_current_applied = getattr(msg_list[-1][1], 'ankle_current_applied')
                    ankle_torque_applied = getattr(msg_list[-1][1], 'ankle_torque_applied')
                    ankle_k = float(str(ankle_setpoint_orgmsg).rsplit('k: ')[1].rsplit('\nb')[0])
                    ankle_b = float(str(ankle_setpoint_orgmsg).rsplit('b: ')[1].rsplit('\nt')[0])
                    ankle_theta_eq = float(str(ankle_setpoint_orgmsg).rsplit('theta_eq: ')[1].rsplit('\nzk')[0])
                    ankle_zk = float(str(ankle_setpoint_orgmsg).rsplit('zk: ')[1].rsplit('\nzb')[0])
                    ankle_zb = float(str(ankle_setpoint_orgmsg).rsplit('zb: ')[1].rsplit('\nztheta_eq')[0])
                    ankle_ztheta_eq = float(str(ankle_setpoint_orgmsg).rsplit('ztheta_eq: ')[1].rsplit('\nankle')[0])
                    # ankle_ztheta_eq = float(str(ankle_setpoint_orgmsg).rsplit('ztheta_eq: ')[1].rsplit(']')[0])
                    ankle_theta_mot = float(str(ankle_setpoint_orgmsg).rsplit('ankle_theta_mot: ')[1].rsplit('\nankle')[0])
                    ankle_thetadot_mot = float(str(ankle_setpoint_orgmsg).rsplit('ankle_thetadot_mot: ')[1].rsplit('\nknee')[0])

                    dict['knee_current_setpoint'] = [knee_current_setpoint]  ### [-1] extracts latest message
                    dict['knee_current_applied'] = [knee_current_applied]  ### [-1] extracts latest message
                    dict['knee_torque_setpoint'] = [knee_torque_setpoint] ### [-1] extracts latest message
                    dict['knee_torque_applied'] = [knee_torque_applied]  ### [-1] extracts latest message
                    dict['knee_k'] = [knee_k]  ### [-1] extracts latest message
                    dict['knee_b'] = [knee_b]  ### [-1] extracts latest message
                    dict['knee_theta_eq'] = [knee_theta_eq]  ### [-1] extracts latest message
                    dict['knee_zk'] = [knee_zk]  ### [-1] extracts latest message
                    dict['knee_zb'] = [knee_zb]  ### [-1] extracts latest message
                    dict['knee_ztheta_eq'] = [knee_ztheta_eq]  ### [-1] extracts latest message
                    dict['knee_theta_mot'] = [knee_theta_mot]  ### [-1] extracts latest message
                    dict['knee_thetadot_mot'] = [knee_thetadot_mot]  ### [-1] extracts latest message

                    dict['ankle_current_setpoint'] = [ankle_current_setpoint]  ### [-1] extracts latest message
                    dict['ankle_current_applied'] = [ankle_current_applied]  ### [-1] extracts latest message
                    dict['ankle_torque_setpoint'] = [ankle_torque_setpoint]  ### [-1] extracts latest message
                    dict['ankle_torque_applied'] = [ankle_torque_applied]  ### [-1] extracts latest message
                    dict['ankle_k'] = [ankle_k]  ### [-1] extracts latest message
                    dict['ankle_b'] = [ankle_b]  ### [-1] extracts latest message
                    dict['ankle_theta_eq'] = [ankle_theta_eq]  ### [-1] extracts latest message
                    dict['ankle_zk'] = [ankle_zk]  ### [-1] extracts latest message
                    dict['ankle_zb'] = [ankle_zb]  ### [-1] extracts latest message
                    dict['ankle_ztheta_eq'] = [ankle_ztheta_eq]  ### [-1] extracts latest message
                    dict['ankle_theta_mot'] = [ankle_theta_mot]  ### [-1] extracts latest message
                    dict['ankle_thetadot_mot'] = [ankle_thetadot_mot]  ### [-1] extracts latest message


                else:
                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message
                    time_in_float = time_val.to_sec()
                    dict['header'].append(time_in_float)

                    knee_setpoint_orgmsg = [getattr(msg_list[-1][1], 'knee_setpoint')]  ### [-1] extracts latest message
                    ankle_setpoint_orgmsg = [getattr(msg_list[-1][1], 'ankle_setpoint')]  ### [-1] extracts latest message

                    # Crop torque values from msg
                    knee_current_setpoint = float(str(knee_setpoint_orgmsg).rsplit('current_setpoint: ')[1].rsplit('\nt')[0])
                    knee_torque_setpoint = float(str(knee_setpoint_orgmsg).rsplit('torque_setpoint: ')[1].rsplit('\nk')[0])
                    knee_current_applied = getattr(msg_list[-1][1], 'knee_current_applied')
                    knee_torque_applied = getattr(msg_list[-1][1], 'knee_torque_applied')
                    knee_k = float(str(knee_setpoint_orgmsg).rsplit('k: ')[1].rsplit('\nb')[0])
                    knee_b = float(str(knee_setpoint_orgmsg).rsplit('b: ')[1].rsplit('\nt')[0])
                    knee_theta_eq = float(str(knee_setpoint_orgmsg).rsplit('theta_eq: ')[1].rsplit('\nzk')[0])
                    knee_zk = float(str(knee_setpoint_orgmsg).rsplit('zk: ')[1].rsplit('\nzb')[0])
                    knee_zb = float(str(knee_setpoint_orgmsg).rsplit('zb: ')[1].rsplit('\nztheta_eq')[0])
                    knee_ztheta_eq = float(str(knee_setpoint_orgmsg).rsplit('ztheta_eq: ')[1].rsplit('\nankle')[0])
                    # knee_ztheta_eq = float(str(knee_setpoint_orgmsg).rsplit('ztheta_eq: ')[1].rsplit(']')[0])
                    knee_theta_mot = float(str(knee_setpoint_orgmsg).rsplit('knee_theta_mot: ')[1].rsplit('\nknee')[0])
                    knee_thetadot_mot = float(str(knee_setpoint_orgmsg).rsplit('knee_thetadot_mot: ')[1].rsplit(']')[0])

                    ankle_current_setpoint = float(str(ankle_setpoint_orgmsg).rsplit('current_setpoint: ')[1].rsplit('\nt')[0])
                    ankle_torque_setpoint = float(str(ankle_setpoint_orgmsg).rsplit('torque_setpoint: ')[1].rsplit('\nk')[0])
                    ankle_current_applied = getattr(msg_list[-1][1], 'ankle_current_applied')
                    ankle_torque_applied = getattr(msg_list[-1][1], 'ankle_torque_applied')
                    ankle_k = float(str(ankle_setpoint_orgmsg).rsplit('k: ')[1].rsplit('\nb')[0])
                    ankle_b = float(str(ankle_setpoint_orgmsg).rsplit('b: ')[1].rsplit('\nt')[0])
                    ankle_theta_eq = float(str(ankle_setpoint_orgmsg).rsplit('theta_eq: ')[1].rsplit('\nzk')[0])
                    ankle_zk = float(str(ankle_setpoint_orgmsg).rsplit('zk: ')[1].rsplit('\nzb')[0])
                    ankle_zb = float(str(ankle_setpoint_orgmsg).rsplit('zb: ')[1].rsplit('\nztheta_eq')[0])
                    ankle_ztheta_eq = float(str(ankle_setpoint_orgmsg).rsplit('ztheta_eq: ')[1].rsplit('\nankle')[0])
                    # ankle_ztheta_eq = float(str(ankle_setpoint_orgmsg).rsplit('ztheta_eq: ')[1].rsplit(']')[0])
                    ankle_theta_mot = float(str(ankle_setpoint_orgmsg).rsplit('ankle_theta_mot: ')[1].rsplit('\nankle')[0])
                    ankle_thetadot_mot = float(str(ankle_setpoint_orgmsg).rsplit('ankle_thetadot_mot: ')[1].rsplit('\nknee')[0])

                    dict['knee_current_setpoint'].append(knee_current_setpoint)  ### [-1] extracts latest message
                    dict['knee_current_applied'].append(knee_current_applied)  ### [-1] extracts latest message
                    dict['knee_torque_setpoint'].append(knee_torque_setpoint) ### [-1] extracts latest message
                    dict['knee_torque_applied'].append(knee_torque_applied)  ### [-1] extracts latest message
                    dict['knee_k'].append(knee_k)  ### [-1] extracts latest message
                    dict['knee_b'].append(knee_b)  ### [-1] extracts latest message
                    dict['knee_theta_eq'].append(knee_theta_eq)  ### [-1] extracts latest message
                    dict['knee_zk'].append(knee_zk)  ### [-1] extracts latest message
                    dict['knee_zb'].append(knee_zb)  ### [-1] extracts latest message
                    dict['knee_ztheta_eq'].append(knee_ztheta_eq)  ### [-1] extracts latest message
                    dict['knee_theta_mot'].append(knee_theta_mot)  ### [-1] extracts latest message
                    dict['knee_thetadot_mot'].append(knee_thetadot_mot)  ### [-1] extracts latest message

                    dict['ankle_current_setpoint'].append(ankle_current_setpoint)  ### [-1] extracts latest message
                    dict['ankle_current_applied'].append(ankle_current_applied)  ### [-1] extracts latest message
                    dict['ankle_torque_setpoint'].append(ankle_torque_setpoint)  ### [-1] extracts latest message
                    dict['ankle_torque_applied'].append(ankle_torque_applied)  ### [-1] extracts latest message
                    dict['ankle_k'].append(ankle_k)  ### [-1] extracts latest message
                    dict['ankle_b'].append(ankle_b)  ### [-1] extracts latest message
                    dict['ankle_theta_eq'].append(ankle_theta_eq)  ### [-1] extracts latest message
                    dict['ankle_zk'].append(ankle_zk)  ### [-1] extracts latest message
                    dict['ankle_zb'].append(ankle_zb)  ### [-1] extracts latest message
                    dict['ankle_ztheta_eq'].append(ankle_ztheta_eq)  ### [-1] extracts latest message
                    dict['ankle_theta_mot'].append(ankle_theta_mot)  ### [-1] extracts latest message
                    dict['ankle_thetadot_mot'].append(ankle_thetadot_mot)  ### [-1] extracts latest message

            elif target_topic == '/foot2/imu/data':

                if len(msg_list) == 1:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message
                    time_in_float = time_val.to_sec()
                    dict['header'] = [time_in_float]

                    gyro_orgmsg = [getattr(msg_list[-1][1], 'angular_velocity')]  ### [-1] extracts latest message
                    accel_orgmsg = [getattr(msg_list[-1][1], 'linear_acceleration')]  ### [-1] extracts latest message

                    # Crop torque values from msg
                    gyro_x = float(str(gyro_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])
                    gyro_y = float(str(gyro_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])
                    gyro_z = float(str(gyro_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    acc_x = float(str(accel_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])
                    acc_y = float(str(accel_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])
                    acc_z = float(str(accel_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    dict['foot_gyro_x'] = [gyro_x]
                    dict['foot_gyro_y'] = [gyro_y]
                    dict['foot_gyro_z'] = [gyro_z]

                    dict['foot_acc_x'] = [acc_x]
                    dict['foot_acc_y'] = [acc_y]
                    dict['foot_acc_z'] = [acc_z]

                else:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message
                    time_in_float = time_val.to_sec()
                    dict['header'].append(time_in_float)

                    gyro_orgmsg = [getattr(msg_list[-1][1], 'angular_velocity')]  ### [-1] extracts latest message
                    accel_orgmsg = [
                        getattr(msg_list[-1][1], 'linear_acceleration')]  ### [-1] extracts latest message

                    # Crop torque values from msg
                    gyro_x = float(str(gyro_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])
                    gyro_y = float(str(gyro_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])
                    gyro_z = float(str(gyro_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    acc_x = float(str(accel_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])
                    acc_y = float(str(accel_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])
                    acc_z = float(str(accel_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    dict['foot_gyro_x'].append(gyro_x)
                    dict['foot_gyro_y'].append(gyro_y)
                    dict['foot_gyro_z'].append(gyro_z)

                    dict['foot_acc_x'].append(acc_x)
                    dict['foot_acc_y'].append(acc_y)
                    dict['foot_acc_z'].append(acc_z)

            elif target_topic == '/foot2/nav/filtered_imu/data':

                if len(msg_list) == 1:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message
                    time_in_float = time_val.to_sec()
                    dict['header'] = [time_in_float]

                    foot_orientation_orgmsg = [getattr(msg_list[-1][1], 'orientation')]  ### [-1] extracts latest message

                    # Crop torque values from msg
                    orien_x = float(str(foot_orientation_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])
                    orien_y = float(str(foot_orientation_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])
                    orien_z = float(str(foot_orientation_orgmsg).rsplit('z: ')[1].rsplit('\nw')[0])
                    orien_w = float(str(foot_orientation_orgmsg).rsplit('w: ')[1].rsplit(']')[0])

                    dict['orientation_x'] = [orien_x]
                    dict['orientation_y'] = [orien_y]
                    dict['orientation_z'] = [orien_z]
                    dict['orientation_w'] = [orien_w]

                else:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message
                    time_in_float = time_val.to_sec()
                    dict['header'].append(time_in_float)

                    foot_orientation_orgmsg = [getattr(msg_list[-1][1], 'orientation')]  ### [-1] extracts latest message

                    # Crop torque values from msg
                    orien_x = float(str(foot_orientation_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])
                    orien_y = float(str(foot_orientation_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])
                    orien_z = float(str(foot_orientation_orgmsg).rsplit('z: ')[1].rsplit('\nw')[0])
                    orien_w = float(str(foot_orientation_orgmsg).rsplit('w: ')[1].rsplit(']')[0])

                    dict['orientation_x'].append(orien_x)
                    dict['orientation_y'].append(orien_y)
                    dict['orientation_z'].append(orien_z)
                    dict['orientation_w'].append(orien_w)

            elif target_topic == '/thigh/imu/data':

                if len(msg_list) == 1:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message
                    time_in_float = time_val.to_sec()
                    dict['header'] = [time_in_float]

                    gyro_orgmsg = [getattr(msg_list[-1][1], 'angular_velocity')]  ### [-1] extracts latest message
                    accel_orgmsg = [
                        getattr(msg_list[-1][1], 'linear_acceleration')]  ### [-1] extracts latest message

                    # Crop torque values from msg
                    gyro_x = float(str(gyro_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])
                    gyro_y = float(str(gyro_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])
                    gyro_z = float(str(gyro_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    acc_x = float(str(accel_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])
                    acc_y = float(str(accel_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])
                    acc_z = float(str(accel_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    dict['thigh_gyro_x'] = [gyro_x]
                    dict['thigh_gyro_y'] = [gyro_y]
                    dict['thigh_gyro_z'] = [gyro_z]

                    dict['thigh_acc_x'] = [acc_x]
                    dict['thigh_acc_y'] = [acc_y]
                    dict['thigh_acc_z'] = [acc_z]

                else:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message
                    time_in_float = time_val.to_sec()
                    dict['header'].append(time_in_float)

                    gyro_orgmsg = [getattr(msg_list[-1][1], 'angular_velocity')]  ### [-1] extracts latest message
                    accel_orgmsg = [
                        getattr(msg_list[-1][1], 'linear_acceleration')]  ### [-1] extracts latest message

                    # Crop torque values from msg
                    gyro_x = float(str(gyro_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])
                    gyro_y = float(str(gyro_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])
                    gyro_z = float(str(gyro_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    acc_x = float(str(accel_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])
                    acc_y = float(str(accel_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])
                    acc_z = float(str(accel_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    dict['thigh_gyro_x'].append(gyro_x)
                    dict['thigh_gyro_y'].append(gyro_y)
                    dict['thigh_gyro_z'].append(gyro_z)

                    dict['thigh_acc_x'].append(acc_x)
                    dict['thigh_acc_y'].append(acc_y)
                    dict['thigh_acc_z'].append(acc_z)

            elif target_topic == '/thigh/nav/filtered_imu/data':

                if len(msg_list) == 1:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message
                    time_in_float = time_val.to_sec()
                    dict['header'] = [time_in_float]

                    thigh_orientation_orgmsg = [
                        getattr(msg_list[-1][1], 'orientation')]  ### [-1] extracts latest message

                    # Crop torque values from msg
                    orien_x = float(str(thigh_orientation_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])
                    orien_y = float(str(thigh_orientation_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])
                    orien_z = float(str(thigh_orientation_orgmsg).rsplit('z: ')[1].rsplit('\nw')[0])
                    orien_w = float(str(thigh_orientation_orgmsg).rsplit('w: ')[1].rsplit(']')[0])

                    dict['orientation_x'] = [orien_x]
                    dict['orientation_y'] = [orien_y]
                    dict['orientation_z'] = [orien_z]
                    dict['orientation_w'] = [orien_w]

                else:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message
                    time_in_float = time_val.to_sec()
                    dict['header'].append(time_in_float)

                    thigh_orientation_orgmsg = [
                        getattr(msg_list[-1][1], 'orientation')]  ### [-1] extracts latest message

                    # Crop torque values from msg
                    orien_x = float(str(thigh_orientation_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])
                    orien_y = float(str(thigh_orientation_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])
                    orien_z = float(str(thigh_orientation_orgmsg).rsplit('z: ')[1].rsplit('\nw')[0])
                    orien_w = float(str(thigh_orientation_orgmsg).rsplit('w: ')[1].rsplit(']')[0])

                    dict['orientation_x'].append(orien_x)
                    dict['orientation_y'].append(orien_y)
                    dict['orientation_z'].append(orien_z)
                    dict['orientation_w'].append(orien_w)

            elif target_topic == '/intact_thigh/imu/data':

                if len(msg_list) == 1:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message
                    time_in_float = time_val.to_sec()
                    dict['header'] = [time_in_float]

                    gyro_orgmsg = [getattr(msg_list[-1][1], 'angular_velocity')]  ### [-1] extracts latest message
                    accel_orgmsg = [
                        getattr(msg_list[-1][1], 'linear_acceleration')]  ### [-1] extracts latest message

                    # Crop torque values from msg
                    gyro_x = float(str(gyro_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])
                    gyro_y = float(str(gyro_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])
                    gyro_z = float(str(gyro_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    acc_x = float(str(accel_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])
                    acc_y = float(str(accel_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])
                    acc_z = float(str(accel_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    dict['thigh_gyro_x'] = [gyro_x]
                    dict['thigh_gyro_y'] = [gyro_y]
                    dict['thigh_gyro_z'] = [gyro_z]

                    dict['thigh_acc_x'] = [acc_x]
                    dict['thigh_acc_y'] = [acc_y]
                    dict['thigh_acc_z'] = [acc_z]

                else:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message
                    time_in_float = time_val.to_sec()
                    dict['header'].append(time_in_float)

                    gyro_orgmsg = [getattr(msg_list[-1][1], 'angular_velocity')]  ### [-1] extracts latest message
                    accel_orgmsg = [
                        getattr(msg_list[-1][1], 'linear_acceleration')]  ### [-1] extracts latest message

                    # Crop torque values from msg
                    gyro_x = float(str(gyro_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])
                    gyro_y = float(str(gyro_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])
                    gyro_z = float(str(gyro_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    acc_x = float(str(accel_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])
                    acc_y = float(str(accel_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])
                    acc_z = float(str(accel_orgmsg).rsplit('z: ')[1].rsplit(']')[0])

                    dict['thigh_gyro_x'].append(gyro_x)
                    dict['thigh_gyro_y'].append(gyro_y)
                    dict['thigh_gyro_z'].append(gyro_z)

                    dict['thigh_acc_x'].append(acc_x)
                    dict['thigh_acc_y'].append(acc_y)
                    dict['thigh_acc_z'].append(acc_z)

            elif target_topic == '/intact_thigh/nav/filtered_imu/data':

                if len(msg_list) == 1:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message
                    time_in_float = time_val.to_sec()
                    dict['header'] = [time_in_float]

                    thigh_orientation_orgmsg = [
                        getattr(msg_list[-1][1], 'orientation')]  ### [-1] extracts latest message

                    # Crop torque values from msg
                    orien_x = float(str(thigh_orientation_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])
                    orien_y = float(str(thigh_orientation_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])
                    orien_z = float(str(thigh_orientation_orgmsg).rsplit('z: ')[1].rsplit('\nw')[0])
                    orien_w = float(str(thigh_orientation_orgmsg).rsplit('w: ')[1].rsplit(']')[0])

                    dict['orientation_x'] = [orien_x]
                    dict['orientation_y'] = [orien_y]
                    dict['orientation_z'] = [orien_z]
                    dict['orientation_w'] = [orien_w]

                else:

                    time_val = getattr(msg_list[-1][1], 'header').stamp  ### [-1] extracts latest message
                    time_in_float = time_val.to_sec()
                    dict['header'].append(time_in_float)

                    thigh_orientation_orgmsg = [
                        getattr(msg_list[-1][1], 'orientation')]  ### [-1] extracts latest message

                    # Crop torque values from msg
                    orien_x = float(str(thigh_orientation_orgmsg).rsplit('x: ')[1].rsplit('\ny')[0])
                    orien_y = float(str(thigh_orientation_orgmsg).rsplit('y: ')[1].rsplit('\nz')[0])
                    orien_z = float(str(thigh_orientation_orgmsg).rsplit('z: ')[1].rsplit('\nw')[0])
                    orien_w = float(str(thigh_orientation_orgmsg).rsplit('w: ')[1].rsplit(']')[0])

                    dict['orientation_x'].append(orien_x)
                    dict['orientation_y'].append(orien_y)
                    dict['orientation_z'].append(orien_z)
                    dict['orientation_w'].append(orien_w)


            else:

                for slotname in slotlist:
                    if len(msg_list) == 1:
                        if slotname == 'header': ### Timestamp
                            time_val = getattr(msg_list[-1][1], slotname).stamp  ### [-1] extracts latest message
                            time_in_float = time_val.to_sec()
                            dict[slotname] = [time_in_float]

                        else: ### Sensor value or context
                            dict[slotname] = [getattr(msg_list[-1][1], slotname)] ### [-1] extracts latest message
                    else:
                        if slotname == 'header': ### Timestamp
                            time_val = getattr(msg_list[-1][1], slotname).stamp ### [-1] extracts latest message
                            time_in_float = time_val.to_sec()
                            dict[slotname].append(time_in_float)

                        else: ### Sensor value or context
                            dict[slotname].append(getattr(msg_list[-1][1], slotname)) ### [-1] extracts latest message

        ### Convert dictionary to dataframe
        # print('Generating DataFrame...')
        df = pd.DataFrame(data=dict)

        # print('Saving DataFrame in dictionary')
        combDat_dict[msg.topic] = df

    print('Included Topic: ', target_topics_list)
    return combDat_dict

def align2SInfo(bagfilename, oldIMU = False):

    '''
    This Funcion aligns SensorData, Foot IMU & Thigh IMU orientation, and FSM to SensorInfo timestamp
    side: prosthetic side (default: r), if left, run swapleft for some sensordata channels
    '''

    # ### Test
    # workspace_path = os.getcwd()
    # BagDir = workspace_path + '/Stair_Data_Raw/TF25/'  # This gives us a str with the path and the file we are using
    # datalist = os.listdir(BagDir)
    # bagfilename = BagDir + 'OSL_Stair_Preset_1.bag'
    # oldIMU = False
    # ### Test

    TF_name = 'TF' + os.path.dirname(bagfilename).rsplit('TF')[1]
    TF_name = TF_name[:4]
    infoPath = os.path.dirname(os.path.dirname(bagfilename)) + '/TF_Info.csv'
    TFInfo = pd.read_csv(infoPath)
    TFInfo_idx = np.where(TFInfo.Tfnumber == TF_name)[0]
    TF_Weight = TFInfo.Weight[TFInfo_idx].__array__()[0]
    side = TFInfo.Side[TFInfo_idx].__array__()[0]
    topics = checkTopics(bagfilename)

    wNorm = [
        'forceX', 'forceY', 'forceZ',
        'momentX', 'momentY', 'momentZ'
    ]

    swapLeft = [
        'forceY', 'momentX', 'momentZ',
        'thigh_accelY', 'thigh_gyroX', 'thigh_gyroZ',
        'shank_accelZ', 'shank_gyroX', 'shank_gyroY',
        'foot_accelY', 'foot_gyroX', 'foot_gyroZ'
    ]

    swapOldSetting = [
        'shank_accelX', 'shank_accelY',
        'shank_gyroX', 'shank_gyroY'
    ]

    try:
        rawdat = read2var2(bagfilename, topics_include=topics)

    except:
        rawdat = read2var(bagfilename, topics_include=topics)

    target_SensorData = rawdat['/SensorData']
    target_SensorInfo = rawdat['/SensorInfo']
    target_sync = rawdat['/sync']

    sync_true = np.where(target_sync['data'] == True)[0][0]
    sync_first_header = target_sync['header'][sync_true] ## Find the first time header of sync

    try:
        target_FSM = rawdat['/fsm/StateGT']

    except:
        target_FSM = rawdat['/fsm/State']

    # Set start/end point and crop
    sync_start_SInfo = closest_point(target_SensorInfo['header'], sync_first_header)
    sync_SInfo = []
    for p in range(0, len(target_SensorInfo)):
        if p < sync_start_SInfo:
            sync_SInfo.append(-1)
        else:
            sync_SInfo.append(p-sync_start_SInfo)




    start_time = np.max([target_SensorData['header'].array[0], target_SensorInfo['header'].array[0]])
    end_time = np.min([target_SensorData['header'].array[-1], target_SensorInfo['header'].array[-1]])

    start_SDat_idx = closest_point(target_SensorData['header'], start_time)
    end_SDat_idx = closest_point(target_SensorData['header'], end_time)

    start_SInfo_idx = closest_point(target_SensorInfo['header'], start_time)
    end_SInfo_idx = closest_point(target_SensorInfo['header'], end_time)

    try:
        target_footIMU = rawdat['/foot2/nav/filtered_imu/data']
        target_thighIMU = rawdat['/thigh/nav/filtered_imu/data']

        start_footIMU_idx = closest_point(target_footIMU['header'], start_time)
        end_footIMU_idx = closest_point(target_footIMU['header'], end_time)

        start_thighIMU_idx = closest_point(target_thighIMU['header'], start_time)
        end_thighIMU_idx = closest_point(target_thighIMU['header'], end_time)

        cropped_footIMU = target_footIMU[start_footIMU_idx:end_footIMU_idx]
        cropped_thighIMU = target_thighIMU[start_thighIMU_idx:end_thighIMU_idx]

    except:
        pass
        # print('No IMU orientation')

    cropped_SDat = target_SensorData[start_SDat_idx:end_SDat_idx]
    cropped_SInfo = target_SensorInfo[start_SInfo_idx:end_SInfo_idx]
    cropped_SInfo_sync = sync_SInfo[start_SInfo_idx:end_SInfo_idx]

    cropped_SInfo['sync'] = cropped_SInfo_sync

    x_new = cropped_SInfo['header'].array

    # Align SensorData to SensorInfo
    aligned_SDat = {}
    for SDatKeys in cropped_SDat.keys():

        if SDatKeys == 'header':
            aligned_SDat[SDatKeys] = cropped_SInfo['header']

            continue

        target_to_align = cropped_SDat[SDatKeys].array

        if SDatKeys in wNorm:
            target_to_align /= (TF_Weight*9.8)
            # print('Weight Normalizing: ' + SDatKeys)

        if oldIMU == True:
            if SDatKeys in swapOldSetting:
                if 'accelX' in SDatKeys:
                    target_to_align = cropped_SDat['shank_accelY']
                    # print(SDatKeys)
                elif 'accelY' in SDatKeys:
                    target_to_align = cropped_SDat['shank_accelX']
                    # print(SDatKeys)
                elif 'gyroX' in SDatKeys:
                    target_to_align = cropped_SDat['shank_gyroY']
                    # print(SDatKeys)
                elif 'gyroY' in SDatKeys:
                    target_to_align = -cropped_SDat['shank_gyroX']
                    # print(SDatKeys)

        if side == 'L':
            # print('Prosthesis Side: Left')

            if SDatKeys in swapLeft:
                target_to_align = target_to_align * (-1)
                # print('Swapping to the left: ' + SDatKeys)

        # else:
            # print('Prosthesis Side: Right')
            # print('No Swap to left: ')

        # x_original = np.linspace(start = x_new[0], stop = x_new[-1], num = len(cropped_SDat))
        x_original = cropped_SDat['header'].array

        f_target = itpd(x_original, target_to_align)
        target_aligned = f_target(x_new)

        aligned_SDat[SDatKeys] = target_aligned

    aligned_SDat = pd.DataFrame.from_dict(aligned_SDat)

    try:
        # Align footIMU to SensorInfo
        aligned_footIMU = {}
        for footIMUKeys in cropped_footIMU.keys():

            if footIMUKeys == 'header':
                aligned_footIMU[footIMUKeys] = cropped_SInfo['header']

                continue

            target_to_align = cropped_footIMU[footIMUKeys].array
            x_original = np.linspace(start = x_new[0], stop = x_new[-1], num = len(cropped_footIMU))
            # x_original = cropped_footIMU['header'].array


            f_target = itpd(x_original, target_to_align)
            target_aligned = f_target(x_new)

            aligned_footIMU[footIMUKeys] = target_aligned

        aligned_footIMU = pd.DataFrame.from_dict(aligned_footIMU)

        # Align thighIMU to SensorInfo
        aligned_thighIMU = {}
        for thighIMUKeys in cropped_thighIMU.keys():

            if thighIMUKeys == 'header':
                aligned_thighIMU[thighIMUKeys] = cropped_SInfo['header']

                continue

            target_to_align = cropped_thighIMU[thighIMUKeys].array
            x_original = np.linspace(start = x_new[0], stop = x_new[-1], num = len(cropped_thighIMU))
            # x_original = cropped_thighIMU['header'].array

            f_target = itpd(x_original, target_to_align)
            target_aligned = f_target(x_new)

            aligned_thighIMU[thighIMUKeys] = target_aligned

        aligned_thighIMU = pd.DataFrame.from_dict(aligned_thighIMU)

    except:
        print('No IMU orientation')

    # Align FSM to SensorInfo
    aligned_FSM = target_FSM.copy()
    for i in range(0, len(aligned_FSM)):
        aligned_FSM_newheader = closest_point(x_new, aligned_FSM['header'][i])
        aligned_FSM_newheader = x_new[aligned_FSM_newheader]

        aligned_FSM['header'][i] = aligned_FSM_newheader

    if len(aligned_footIMU) > 0:

        return aligned_SDat, aligned_footIMU, aligned_thighIMU, aligned_FSM, cropped_SInfo

    else:

        return aligned_SDat, aligned_FSM, cropped_SInfo

def df_rearrange_header(target_df):
    # target_df = dict_to_df

    dict_to_numpy = target_df.to_numpy()
    new_dict = {}
    i = 0

    for col in target_df.keys():
        new_dict[col] = dict_to_numpy[:,i]
        i += 1

    rearranged_df = pd.DataFrame.from_dict(new_dict)
    return rearranged_df

class FSMinfo:
    # FSM_df = aligned_FSM

    def __init__(self, FSM_df):

        target_FSM_timeval = FSM_df['header']
        target_FSM_state = FSM_df['state']

        # Force the first step to be LWRA transition step
        ii = 0
        for phase in target_FSM_state:
            if 'EarlyStance' in phase:
                print(ii, phase)

                break
                # continue

            else:
                ii += 1

        # print(ii)

        first_ES_idx = ii
        target_FSM_state[first_ES_idx] = 'LW_EarlyStance'
        target_FSM_state[first_ES_idx + 1] = 'LW_LateStance'
        target_FSM_state[first_ES_idx + 2] = 'LW_SwingFlexion'
        target_FSM_state[first_ES_idx + 3] = 'LW_SwingExtension'
        target_FSM_state[first_ES_idx + 4] = 'LW_EarlyStance'

        try: # If ML statement is included in the FSM state
            MLidx = np.where(target_FSM_state == 'ML')[0]

            target_FSM_timeval_MLdropped = target_FSM_timeval.drop(MLidx)
            target_FSM_state_MLdropped = target_FSM_state.drop(MLidx)

            # Below is required to remove empty idx (dropped idx still remains)
            target_FSM_timeval_MLdropped = np.array(target_FSM_timeval_MLdropped)
            target_FSM_state_MLdropped = np.array(target_FSM_state_MLdropped)

            FSM_total_MLdropped = {'header': target_FSM_timeval_MLdropped, 'state': target_FSM_state_MLdropped}
            FSM_total = pd.DataFrame(data = FSM_total_MLdropped)

            target_FSM_timeval = FSM_total['header']
            target_FSM_state = FSM_total['state']

        except:

            FSM_total = FSM_df

        # Remove any state before Home
        try:
            HomeIdx = np.where(target_FSM_state == 'Home')[0][0]
            if HomeIdx != 0 and HomeIdx != len(target_FSM_state) - 1:
                for rm in range(0, HomeIdx):
                    FSM_total_beforeHomeDropped = FSM_total.drop(rm)

                # Below is required to remove empty idx (dropped idx still remains)
                target_FSM_timeval_beforeHomeDropped = np.array(FSM_total_beforeHomeDropped['header'])
                target_FSM_state_beforeHomeDropped = np.array(FSM_total_beforeHomeDropped['state'])

                FSM_total_beforeHomeDropped = {'header': target_FSM_timeval_beforeHomeDropped,
                                               'state': target_FSM_state_beforeHomeDropped}

                FSM_total = pd.DataFrame(data = FSM_total_beforeHomeDropped)

                target_FSM_timeval = FSM_total['header']
                target_FSM_state = FSM_total['state']

        except:
            pass

        # Remove repeated states
        for s in range(0, len(target_FSM_state)-1):
            if target_FSM_state[s] == target_FSM_state[s+1]:
                FSM_total = FSM_total.drop(s+1)
                # print(s+1)

        # Below is required to remove empty idx (dropped idx still remains)
        target_FSM_timeval_rpdropped = np.array(FSM_total['header'])
        target_FSM_state_rpdropped = np.array(FSM_total['state'])

        FSM_total_MLdropped = {'header': target_FSM_timeval_rpdropped, 'state': target_FSM_state_rpdropped}
        FSM_total = pd.DataFrame(data=FSM_total_MLdropped)

        target_FSM_timeval = FSM_total['header']
        target_FSM_state = FSM_total['state']

        time_idx_ES = []
        time_idx_LS = []
        time_idx_SF = []
        time_idx_SE = []

        i = 0
        for phase in target_FSM_state:
            if 'EarlyStance' in phase:
                time_idx_ES.append(i)
                i += 1
            elif 'LateStance' in phase:
                time_idx_LS.append(i)
                i += 1
            elif 'SwingExtension' in phase:
                time_idx_SE.append(i)
                i += 1
            elif 'SwingFlexion' in phase:
                time_idx_SF.append(i)
                i += 1
            else:
                i += 1

        ##### Ramps
        ### Heel Strike (Early Stance) Timings
        time_idx_RAES = []  ### Ramp Ascend Heel Strike
        time_idx_RDES = []  ### Ramp Descend Heel Strike
        time_idx_LWES = []  ### Level Walking Heel Strike

        for j in range(0, len(time_idx_ES)):
            if 'RA' in target_FSM_state[time_idx_ES[j]]:
                time_idx_RAES.append(time_idx_ES[j])

            if 'RD' in target_FSM_state[time_idx_ES[j]]:
                time_idx_RDES.append(time_idx_ES[j])

            if 'LW' in target_FSM_state[time_idx_ES[j]]:
                time_idx_LWES.append(time_idx_ES[j])

        time_val_RAES = target_FSM_timeval[time_idx_RAES].array
        time_val_RDES = target_FSM_timeval[time_idx_RDES].array
        time_val_LWES = target_FSM_timeval[time_idx_LWES].array

        ### Late Stance Timings
        time_idx_RALS = []  ### Ramp Ascend Toe Off
        time_idx_RDLS = []  ### Ramp Descend Toe Off
        time_idx_LWLS = []  ### Level Walking Toe Off

        for j in range(0, len(time_idx_LS)):
            if 'RA' in target_FSM_state[time_idx_LS[j]]:
                time_idx_RALS.append(time_idx_LS[j])

            if 'RD' in target_FSM_state[time_idx_LS[j]]:
                time_idx_RDLS.append(time_idx_LS[j])

            if 'LW' in target_FSM_state[time_idx_LS[j]]:
                time_idx_LWLS.append(time_idx_LS[j])

        time_val_RALS = target_FSM_timeval[time_idx_RALS].array
        time_val_RDLS = target_FSM_timeval[time_idx_RDLS].array
        time_val_LWLS = target_FSM_timeval[time_idx_LWLS].array

        ### Swing Flexion Timings
        time_idx_RASF = []  ### Ramp Ascend Toe Off
        time_idx_RDSF = []  ### Ramp Descend Toe Off
        time_idx_LWSF = []  ### Level Walking Toe Off

        for j in range(0, len(time_idx_SF)):
            if 'RA' in target_FSM_state[time_idx_SF[j]]:
                time_idx_RASF.append(time_idx_SF[j])

            if 'RD' in target_FSM_state[time_idx_SF[j]]:
                time_idx_RDSF.append(time_idx_SF[j])

            if 'LW' in target_FSM_state[time_idx_SF[j]]:
                time_idx_LWSF.append(time_idx_SF[j])

        time_val_RASF = target_FSM_timeval[time_idx_RASF].array
        time_val_RDSF = target_FSM_timeval[time_idx_RDSF].array
        time_val_LWSF = target_FSM_timeval[time_idx_LWSF].array

        ### Swing Extension Timings
        time_idx_RASE = []  ### Ramp Ascend Swing Extension
        time_idx_RDSE = []  ### Ramp Descend Swing Extension
        time_idx_LWSE = []  ### Level Walking Swing Extension

        for j in range(0, len(time_idx_SE)):
            if 'RA' in target_FSM_state[time_idx_SE[j]]:
                time_idx_RASE.append(time_idx_SE[j])

            if 'RD' in target_FSM_state[time_idx_SE[j]]:
                time_idx_RDSE.append(time_idx_SE[j])

            if 'LW' in target_FSM_state[time_idx_SE[j]]:
                time_idx_LWSE.append(time_idx_SE[j])

        time_val_RASE = target_FSM_timeval[time_idx_RASE].array
        time_val_RDSE = target_FSM_timeval[time_idx_RDSE].array
        time_val_LWSE = target_FSM_timeval[time_idx_LWSE].array

        ##### Stairs
        ### Heel Strike (Early Stance) Timings
        time_idx_SAES = []  ### Stair Ascend Heel Strike
        time_idx_SDES = []  ### Stair Descend Heel Strike

        for j in range(0, len(time_idx_ES)):
            if 'SA' in target_FSM_state[time_idx_ES[j]]:
                time_idx_SAES.append(time_idx_ES[j])

            if 'SD' in target_FSM_state[time_idx_ES[j]]:
                time_idx_SDES.append(time_idx_ES[j])

        time_val_SAES = target_FSM_timeval[time_idx_SAES].array
        time_val_SDES = target_FSM_timeval[time_idx_SDES].array

        ### Late Stance Timings
        time_idx_SALS = []  ### Stair Ascend Toe Off
        time_idx_SDLS = []  ### Stair Descend Toe Off

        for j in range(0, len(time_idx_LS)):
            if 'SA' in target_FSM_state[time_idx_LS[j]]:
                time_idx_SALS.append(time_idx_LS[j])

            if 'SD' in target_FSM_state[time_idx_LS[j]]:
                time_idx_SDLS.append(time_idx_LS[j])

        time_val_SALS = target_FSM_timeval[time_idx_SALS].array
        time_val_SDLS = target_FSM_timeval[time_idx_SDLS].array

        ### Swing Flexion Timings
        time_idx_SASF = []  ### Stair Ascend Toe Off
        time_idx_SDSF = []  ### Stair Descend Toe Off

        for j in range(0, len(time_idx_SF)):
            if 'SA' in target_FSM_state[time_idx_SF[j]]:
                time_idx_SASF.append(time_idx_SF[j])

            if 'SD' in target_FSM_state[time_idx_SF[j]]:
                time_idx_SDSF.append(time_idx_SF[j])

        time_val_SASF = target_FSM_timeval[time_idx_SASF].array
        time_val_SDSF = target_FSM_timeval[time_idx_SDSF].array

        ### Swing Extension Timings
        time_idx_SASE = []  ### Stair Ascend Swing Extension
        time_idx_SDSE = []  ### Stair Descend Swing Extension

        for j in range(0, len(time_idx_SE)):
            if 'SA' in target_FSM_state[time_idx_SE[j]]:
                time_idx_SASE.append(time_idx_SE[j])

            if 'SD' in target_FSM_state[time_idx_SE[j]]:
                time_idx_SDSE.append(time_idx_SE[j])

            if 'LW' in target_FSM_state[time_idx_SE[j]]:
                time_idx_LWSE.append(time_idx_SE[j])

        time_val_SASE = target_FSM_timeval[time_idx_SASE].array
        time_val_SDSE = target_FSM_timeval[time_idx_SDSE].array

        time_val_ES = target_FSM_timeval[time_idx_ES].array
        time_val_LS = target_FSM_timeval[time_idx_LS].array
        time_val_SF = target_FSM_timeval[time_idx_SF].array
        time_val_SE = target_FSM_timeval[time_idx_SE].array

        self.FSM_total = FSM_total

        self.time_val_ES = time_val_ES
        self.time_val_LS = time_val_LS
        self.time_val_SF = time_val_SF
        self.time_val_SE = time_val_SE
        self.time_idx_ES = time_idx_ES
        self.time_idx_LS = time_idx_LS
        self.time_idx_SF = time_idx_SF
        self.time_idx_SE = time_idx_SE

        self.time_val_LWES = time_val_LWES
        self.time_val_LWLS = time_val_LWLS
        self.time_val_LWSF = time_val_LWSF
        self.time_val_LWSE = time_val_LWSE
        self.time_idx_LWES = time_idx_LWES
        self.time_idx_LWLS = time_idx_LWLS
        self.time_idx_LWSF = time_idx_LWSF
        self.time_idx_LWSE = time_idx_LWSE

        self.time_val_RAES = time_val_RAES
        self.time_val_RALS = time_val_RALS
        self.time_val_RASF = time_val_RASF
        self.time_val_RASE = time_val_RASE
        self.time_idx_RAES = time_idx_RAES
        self.time_idx_RALS = time_idx_RALS
        self.time_idx_RASF = time_idx_RASF
        self.time_idx_RASE = time_idx_RASE

        self.time_val_RDES = time_val_RDES
        self.time_val_RDLS = time_val_RDLS
        self.time_val_RDSF = time_val_RDSF
        self.time_val_RDSE = time_val_RDSE
        self.time_idx_RDES = time_idx_RDES
        self.time_idx_RDLS = time_idx_RDLS
        self.time_idx_RDSF = time_idx_RDSF
        self.time_idx_RDSE = time_idx_RDSE

        self.time_val_SAES = time_val_SAES
        self.time_val_SALS = time_val_SALS
        self.time_val_SASF = time_val_SASF
        self.time_val_SASE = time_val_SASE
        self.time_idx_SAES = time_idx_SAES
        self.time_idx_SALS = time_idx_SALS
        self.time_idx_SASF = time_idx_SASF
        self.time_idx_SASE = time_idx_SASE

        self.time_val_SDES = time_val_SDES
        self.time_val_SDLS = time_val_SDLS
        self.time_val_SDSF = time_val_SDSF
        self.time_val_SDSE = time_val_SDSE
        self.time_idx_SDES = time_idx_SDES
        self.time_idx_SDLS = time_idx_SDLS
        self.time_idx_SDSF = time_idx_SDSF
        self.time_idx_SDSE = time_idx_SDSE

class alignedBag_Ramp_OldSetting:

    ### OldSetting means transition LW->RA takes place in EarlyStance
    ### Later settings LW->RA takes place in SwingFlexion

    # def __init__(self, bagfilename, slope, startMSP = 0, endMSP = 1, plot = True):
    def __init__(self, bagfilename, plot=True):

        # ### Test
        # workspace_path = os.getcwd()
        # BagDir = workspace_path + '/Ramp_Data_Raw_Additional/Slope_Online/TF06v2/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(BagDir)
        #
        # bagfilename = BagDir + datalist[12]
        # ### Test

        try:
            aligned_SDat, aligned_footIMU, aligned_thighIMU, aligned_FSM, cropped_SInfo = align2SInfo(bagfilename, oldIMU = True)

        except:
            aligned_SDat, aligned_FSM, cropped_SInfo = align2SInfo(bagfilename, oldIMU = True)

        ##### Align time segment to State
        FSM = FSMinfo(aligned_FSM)
        target_sensor_timeval = aligned_SDat['header']
        transition_idx_total = np.zeros(1)  # For plotting

        transition_idx_total_Sensor = {}  # For later SS/TR separation
        transition_idx_total_Sensor['LW2AS'] = []
        transition_idx_total_Sensor['AS2LW'] = []
        transition_idx_total_Sensor['LW2DS'] = []
        transition_idx_total_Sensor['DS2LW'] = []

        transitionMS_idx_total_Sensor = {}  # For later SS/TR separation
        transitionMS_idx_total_Sensor['LW2AS'] = []
        transitionMS_idx_total_Sensor['AS2LW'] = []
        transitionMS_idx_total_Sensor['LW2DS'] = []
        transitionMS_idx_total_Sensor['DS2LW'] = []

        input_starttime = FSM.time_val_ES[0]
        input_startidx = closest_point(target_sensor_timeval, input_starttime)
        input_endtime = FSM.time_val_ES[-1]
        input_endidx = closest_point(target_sensor_timeval, input_endtime)
        ### Data before the first ES and after the last ES will be excluded in input shaping

        ##### Find Transition time indices (RA<->LW, RD<->LW)
        '''
        Two different version needed...
        Our experiments make subjects to begin w/ sound side, end w/ prosthesis side.
        1) The last ES->ES
        2) One before the last SASF->SAES    
        '''
        # This is version 1)
        transition_idx_total = np.where(np.diff(FSM.time_idx_ES) == 1)[0]  # Where ES->ES happens (LWES<->RA/RDES)

        FSM_idx_LW2RA_end = []
        FSM_idx_RA2LW_end = []
        FSM_idx_LW2RD_end = []
        FSM_idx_RD2LW_end = []

        try:
            # There are some trials that do not have LWES in the first step
            # For those cases, the first RAES is considered as transition step
            if FSM.time_idx_ES[0] == FSM.time_idx_RAES[0]:
                FSM_idx_LW2RA_end.append(FSM.time_idx_RAES[1])

        except:
            pass

        for i in range(0, len(transition_idx_total)):

            trid = transition_idx_total[i]  # index in time_idx_ES matrix
            trid_in_FSM_total = FSM.time_idx_ES[trid]  # index in FSM_total

            if 'RA' in FSM.FSM_total['state'][trid_in_FSM_total + 1]:
                # print(trid_in_FSM_total, ': LW2RA transition')
                FSM_idx_LW2RA_end.append(trid_in_FSM_total)

            elif 'RD' in FSM.FSM_total['state'][trid_in_FSM_total + 1]:
                # print(trid_in_FSM_total, ': LW2RD transition')
                FSM_idx_LW2RD_end.append(trid_in_FSM_total)

            else:
                if 'RA' in FSM.FSM_total['state'][trid_in_FSM_total]:
                    # print(trid_in_FSM_total, ': RA2LW transition')
                    FSM_idx_RA2LW_end.append(trid_in_FSM_total)

                elif 'RD' in FSM.FSM_total['state'][trid_in_FSM_total]:
                    # print(trid_in_FSM_total, ': RD2LW transition')
                    FSM_idx_RD2LW_end.append(trid_in_FSM_total)

        try:
            # There are some trials that do not have LWES in the last step
            # For those cases, the last RDES is considered as transition step
            if FSM.time_idx_ES[-1] == FSM.time_idx_RDES[-1]:
                FSM_idx_RD2LW_end.append(FSM.time_idx_RDES[-1])

        except:
            pass

        # Find time header value that corresponds to transition timing
        for n in range(0, len(FSM_idx_LW2RA_end)):
            transition_LW2RA_end_timeval = FSM.FSM_total['header'][FSM_idx_LW2RA_end[n]]
            transition_LW2RA_start_timeval = FSM.FSM_total['header'][FSM_idx_LW2RA_end[n] - 2]
            transition_RA2LW_end_timeval = FSM.FSM_total['header'][FSM_idx_RA2LW_end[n]]
            transition_RA2LW_start_timeval = FSM.FSM_total['header'][FSM_idx_RA2LW_end[n] - 2]
            transition_LW2RD_end_timeval = FSM.FSM_total['header'][FSM_idx_LW2RD_end[n]]
            transition_LW2RD_start_timeval = FSM.FSM_total['header'][FSM_idx_LW2RD_end[n] - 2]
            transition_RD2LW_end_timeval = FSM.FSM_total['header'][FSM_idx_RD2LW_end[n]]
            transition_RD2LW_start_timeval = FSM.FSM_total['header'][FSM_idx_RD2LW_end[n] - 2]

            # Find FSM transition index in sensor stream
            transition_LW2RA_start_idx_Sensor = closest_point(target_sensor_timeval, transition_LW2RA_start_timeval)
            transition_LW2RA_end_idx_Sensor = closest_point(target_sensor_timeval, transition_LW2RA_end_timeval)
            transition_RA2LW_start_idx_Sensor = closest_point(target_sensor_timeval, transition_RA2LW_start_timeval)
            transition_RA2LW_end_idx_Sensor = closest_point(target_sensor_timeval, transition_RA2LW_end_timeval)
            transition_LW2RD_start_idx_Sensor = closest_point(target_sensor_timeval, transition_LW2RD_start_timeval)
            transition_LW2RD_end_idx_Sensor = closest_point(target_sensor_timeval, transition_LW2RD_end_timeval)
            transition_RD2LW_start_idx_Sensor = closest_point(target_sensor_timeval, transition_RD2LW_start_timeval)
            transition_RD2LW_end_idx_Sensor = closest_point(target_sensor_timeval, transition_RD2LW_end_timeval)

            # Compensate start time offset sensor vs FSM
            transition_idx_total_Sensor['LW2AS'].append(transition_LW2RA_start_idx_Sensor - input_startidx)
            transition_idx_total_Sensor['LW2AS'].append(transition_LW2RA_end_idx_Sensor - input_startidx)
            transition_idx_total_Sensor['AS2LW'].append(transition_RA2LW_start_idx_Sensor - input_startidx)
            transition_idx_total_Sensor['AS2LW'].append(transition_RA2LW_end_idx_Sensor - input_startidx)
            transition_idx_total_Sensor['LW2DS'].append(transition_LW2RD_start_idx_Sensor - input_startidx)
            transition_idx_total_Sensor['LW2DS'].append(transition_LW2RD_end_idx_Sensor - input_startidx)
            transition_idx_total_Sensor['DS2LW'].append(transition_RD2LW_start_idx_Sensor - input_startidx)
            transition_idx_total_Sensor['DS2LW'].append(transition_RD2LW_end_idx_Sensor - input_startidx)

        # Arrange Sensor Data (Include Thigh/Foot IMU orientation in SensorData)
        try:
            foot_qx = aligned_footIMU['orientation_x']
            foot_qy = aligned_footIMU['orientation_y']
            foot_qz = aligned_footIMU['orientation_z']
            foot_qw = aligned_footIMU['orientation_w']

            thigh_qx = aligned_thighIMU['orientation_x']
            thigh_qy = aligned_thighIMU['orientation_y']
            thigh_qz = aligned_thighIMU['orientation_z']
            thigh_qw = aligned_thighIMU['orientation_w']

            ### Compute orientations
            Zvector = np.array([0, 0, 1])

            foot_qmat = np.array([foot_qw, foot_qx, foot_qy, foot_qz]).T
            thigh_qmat = np.array([thigh_qw, thigh_qx, thigh_qy, thigh_qz]).T

            foot_Rmat = np.zeros([len(foot_qmat), 3, 3])
            thigh_Rmat = np.zeros([len(thigh_qmat), 3, 3])

            foot_rotZmat = np.zeros([len(foot_qmat), 3])
            thigh_rotZmat = np.zeros([len(thigh_qmat), 3])

            foot_anglemat = np.zeros([len(foot_qmat)])
            thigh_anglemat = np.zeros([len(thigh_qmat)])

            for i in range(0, len(foot_qmat)):
                foot_Rmat[i, :] = AHRS.q2Rot(foot_qmat[i])
                foot_rotZmat[i, :] = np.matmul(foot_Rmat[i, :], Zvector)

                vlen = np.sign(foot_rotZmat[i, 0]) * np.sqrt(foot_rotZmat[i, 0] ** 2 + foot_rotZmat[i, 1] ** 2)
                hlen = foot_rotZmat[i, 2]

                foot_anglemat[i] = 90 - math.atan2(hlen, vlen) * 180 / np.pi

                # Denoise for realtime
                if i > 0:
                    if foot_anglemat[i] - foot_anglemat[i - 1] > 25:
                        foot_anglemat[i] = foot_anglemat[i - 1]

            for i in range(0, len(thigh_qmat)):
                thigh_Rmat[i, :] = AHRS.q2Rot(thigh_qmat[i])
                thigh_rotZmat[i, :] = np.matmul(thigh_Rmat[i, :], Zvector)

                hlen = np.sign(thigh_rotZmat[i, 0]) * np.sqrt(thigh_rotZmat[i, 0] ** 2 + thigh_rotZmat[i, 1] ** 2)
                vlen = thigh_rotZmat[i, 2]

                thigh_anglemat[i] = 90 - math.atan2(hlen, vlen) * 180 / np.pi

                # Denoise for realtime
                if i > 0:
                    if thigh_anglemat[i] - thigh_anglemat[i - 1] < -15:
                        thigh_anglemat[i] = thigh_anglemat[i - 1]

            offset_end = closest_point(target_sensor_timeval, input_starttime)
            foot_angle_offset = np.mean(foot_anglemat[:offset_end])
            thigh_angle_offset = np.mean(thigh_anglemat[:offset_end])

            foot_anglemat -= foot_angle_offset
            thigh_anglemat -= thigh_angle_offset

            aligned_SDat['thigh_orientation'] = thigh_anglemat
            aligned_SDat['foot_orientation'] = foot_anglemat

        except:
            pass

        # ground_truth = ground_truth[input_startidx:input_endidx]
        aligned_SDat = aligned_SDat[input_startidx:input_endidx]
        cropped_SInfo = cropped_SInfo[input_startidx:input_endidx]

        transition_counts = int(len(transition_idx_total_Sensor['LW2AS']) / 2)

        # Store to this class
        # self.ground_truth = ground_truth
        self.sensor_data = aligned_SDat
        self.sensor_info = cropped_SInfo
        self.FSM = FSM
        self.transition_idx_Sensor = transition_idx_total_Sensor
        self.transition_counts = transition_counts
        # self.transition_idx = transition_idx_total
        # self.transition_MSidx = transition_MSidx_total

class Ramp_GT_byMSP_OldSetting:

    ### OldSetting means transition LW->RA takes place in EarlyStance
    ### Later settings LW->RA takes place in SwingFlexion

    def __init__(self, bagfilename, slope, startMSP_list = [], endMSP_list = []):

        # ### Test
        # workspace_path = os.getcwd()
        # BagDir = workspace_path + '/Ramp_Data_Raw_Additional/Slope_Online/TF15v2/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(BagDir)
        #
        # bagfilename = BagDir + datalist[0]
        # slope = 19.6
        # startMSP_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        # endMSP_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        #
        # startMSP_list = [0]
        # endMSP_list = [0.4, 1.0]
        # ### Test

        aligned_bag = alignedBag_Ramp_OldSetting(bagfilename, plot = False)
        transition_idx_Sensor = aligned_bag.transition_idx_Sensor
        sensor_data_array = aligned_bag.sensor_data.to_numpy()
        transition_counts = int(len(transition_idx_Sensor['LW2AS'])/2)

        # Run slope GT variation
        GT_slope_dict = {}
        for start in range(0, len(startMSP_list)):
            for end in range(0, len(endMSP_list)):

                startMSP = startMSP_list[start]
                endMSP = endMSP_list[end]

                if startMSP > endMSP:
                    continue

                keyName = 'MSP' + str(startMSP) + 'to' + str(endMSP)
                ground_truth = np.zeros([len(sensor_data_array)])
                # print(keyName)

                for n in range(0, transition_counts):
                    # Vary transition timing during Mid-Swing (MS)
                    transitionMS_LW2RA_start_idx_Sensor = int((1 - startMSP) * transition_idx_Sensor['LW2AS'][2 * n] + \
                                                              startMSP * transition_idx_Sensor['LW2AS'][2 * n + 1])

                    transitionMS_LW2RA_end_idx_Sensor = int((1 - endMSP) * transition_idx_Sensor['LW2AS'][2 * n] + \
                                                            endMSP * transition_idx_Sensor['LW2AS'][2 * n + 1])

                    transitionMS_RA2LW_start_idx_Sensor = int((1 - startMSP) * transition_idx_Sensor['AS2LW'][2 * n] + \
                                                              startMSP * transition_idx_Sensor['AS2LW'][2 * n + 1])

                    transitionMS_RA2LW_end_idx_Sensor = int((1 - endMSP) * transition_idx_Sensor['AS2LW'][2 * n] + \
                                                            endMSP * transition_idx_Sensor['AS2LW'][2 * n + 1])

                    transitionMS_LW2RD_start_idx_Sensor = int((1 - startMSP) * transition_idx_Sensor['LW2DS'][2 * n] + \
                                                              startMSP * transition_idx_Sensor['LW2DS'][2 * n + 1])

                    transitionMS_LW2RD_end_idx_Sensor = int((1 - endMSP) * transition_idx_Sensor['LW2DS'][2 * n] + \
                                                            endMSP * transition_idx_Sensor['LW2DS'][2 * n + 1])

                    transitionMS_RD2LW_start_idx_Sensor = int((1 - startMSP) * transition_idx_Sensor['DS2LW'][2 * n] + \
                                                              startMSP * transition_idx_Sensor['DS2LW'][2 * n + 1])

                    transitionMS_RD2LW_end_idx_Sensor = int((1 - endMSP) * transition_idx_Sensor['DS2LW'][2 * n] + \
                                                            endMSP * transition_idx_Sensor['DS2LW'][2 * n + 1])

                    # Fill Steady-State walk
                    ground_truth[transitionMS_LW2RA_end_idx_Sensor:transitionMS_RA2LW_start_idx_Sensor] = 1
                    ground_truth[transitionMS_LW2RD_end_idx_Sensor:transitionMS_RD2LW_start_idx_Sensor] = -1
                    # GT_slope_dict[keyName] = ground_truth

                    if not startMSP == endMSP:
                        # print('MSPs are different')

                        ### Generate linear transition
                        n_transition_LW2RA = transitionMS_LW2RA_end_idx_Sensor - transitionMS_LW2RA_start_idx_Sensor
                        n_transition_RA2LW = transitionMS_RA2LW_end_idx_Sensor - transitionMS_RA2LW_start_idx_Sensor
                        n_transition_LW2RD = transitionMS_LW2RD_end_idx_Sensor - transitionMS_LW2RD_start_idx_Sensor
                        n_transition_RD2LW = transitionMS_RD2LW_end_idx_Sensor - transitionMS_RD2LW_start_idx_Sensor

                        lt_LW2RA = np.linspace(0, 1, n_transition_LW2RA)
                        lt_RA2LW = np.linspace(1, 0, n_transition_RA2LW)
                        lt_LW2RD = np.linspace(0, -1, n_transition_LW2RD)
                        lt_RD2LW = np.linspace(-1, 0, n_transition_RD2LW)

                        ground_truth[transitionMS_LW2RA_start_idx_Sensor:transitionMS_LW2RA_end_idx_Sensor] = lt_LW2RA
                        ground_truth[transitionMS_RA2LW_start_idx_Sensor:transitionMS_RA2LW_end_idx_Sensor] = lt_RA2LW
                        ground_truth[transitionMS_LW2RD_start_idx_Sensor:transitionMS_LW2RD_end_idx_Sensor] = lt_LW2RD
                        ground_truth[transitionMS_RD2LW_start_idx_Sensor:transitionMS_RD2LW_end_idx_Sensor] = lt_RD2LW

                # ground_truth *= slope
                GT_slope_dict[keyName] = slope * ground_truth

        self.GT_slope_dict = GT_slope_dict
        # self.transition_counts = transition_counts

class separation_SSTR_Ramp_OldSetting:
    def __init__(self, bagfilename, plot = False, slope = 0, startMSP_list = [], endMSP_list = [], FWwinlen = 125):
        # super().__init__(bagfilename, plot = plot)

        # ### Test
        # workspace_path = os.getcwd()
        # BagDir = workspace_path + '/Ramp_Data_Raw_Additional/Slope_Online/TF03v2/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(BagDir)
        #
        # bagfilename = BagDir + datalist[3]
        # slope = 0
        # startMSP_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        # endMSP_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        # FWwinlen = 125
        # testload = alignedBag_Ramp_OldSetting(bagfilename, plot = False)
        # testload = testload.__dict__
        # ### Test

        testload = alignedBag_Ramp_OldSetting(bagfilename, plot = plot)
        testload = testload.__dict__

        GT_slope_dict = Ramp_GT_byMSP_OldSetting(bagfilename, slope, startMSP_list = startMSP_list, endMSP_list = endMSP_list).GT_slope_dict
        GT_slope_df = pd.DataFrame.from_dict(GT_slope_dict)

        transition_counts = testload['transition_counts']
        SInfo_noHeader = testload['sensor_info']
        SInfo_noHeader = SInfo_noHeader.drop(['header'], axis = 1)
        dict_to_df = pd.merge(testload['sensor_data'], SInfo_noHeader, left_index= True, right_index=True)
        dict_to_df = df_rearrange_header(dict_to_df)

        State_and_Slope_header = dict_to_df['header'].__array__()
        FSM_header = testload['FSM'].FSM_total['header'].__array__()
        state_mat = np.zeros(len(dict_to_df))

        # state_ESmat = np.zeros(len(dict_to_df))

        for t in range(0, len(State_and_Slope_header)):

            if State_and_Slope_header[t] in FSM_header:

                matchIdx = np.where(FSM_header == State_and_Slope_header[t])[0][0]
                state = testload['FSM'].FSM_total['state'][matchIdx]

                if 'EarlyStance' in state:
                    state_num = 1
                    # state_ESnum = 1
                elif 'LateStance' in state:
                    state_num = 1
                    # state_ESnum = 0
                elif 'SwingFlexion' in state:
                    state_num = 2
                    # state_ESnum = 0
                elif 'SwingExtension' in state:
                    state_num = 3
                    # state_ESnum = 0
                else:
                    state_num = 0
                    # state_ESnum = 0

                # print(state)

            state_mat[t] = state_num

        transition_ref = testload['transition_idx_Sensor']

        # for TRtype in transition_ref.keys():
        #     SWstart = transition_ref[TRtype][0]
        #     SWend = transition_ref[TRtype][1]
        #
        #     state_mat[SWstart:SWend] = 2

        state_mat[0] = 3
        state_mat[-1] = 1

        dict_to_df['state'] = state_mat

        dict_to_df = pd.merge(dict_to_df, GT_slope_df, left_index = True, right_index=True)

        stride_ref = np.diff(state_mat)
        stride_edge = np.where(stride_ref == -2)[0]+1

        # plt.plot(state_mat)
        # plt.plot(ground_truth)
        # plt.plot(stride_ref)
        # plt.plot(stride_edge)

        dict_sep_total = {}

        for s in range(0, len(stride_edge)-1):
            # s = 0
            stride_start = stride_edge[s]
            stride_end = stride_edge[s + 1]

            # Determine stride type, if transition exists since it's not LW trial
            if transition_counts > 0:
                for c in range(0, transition_counts):

                    # If Transition
                    if stride_start <= transition_ref['LW2AS'][2 * c] and stride_end + 1 >= transition_ref['LW2AS'][2 * c]:
                        stride_type = 'TR_LW2AS'
                        # print(s, stride_type)
                    elif stride_start <= transition_ref['AS2LW'][2 * c] and stride_end + 1 >= transition_ref['AS2LW'][2 * c]:
                        stride_type = 'TR_AS2LW'
                        # print(s, stride_type)
                    elif stride_start <= transition_ref['LW2DS'][2 * c] and stride_end + 1 >= transition_ref['LW2DS'][2 * c]:
                        stride_type = 'TR_LW2DS'
                        # print(s, stride_type)
                    elif stride_start <= transition_ref['DS2LW'][2 * c] and stride_end + 1 >= transition_ref['DS2LW'][2 * c]:
                        stride_type = 'TR_DS2LW'
                        # print(s, stride_type)

                    else: # If Steady-State
                        if stride_start + 1 >= transition_ref['LW2AS'][(2 * c) + 1] and \
                                stride_end <= transition_ref['AS2LW'][2 * c]:
                            stride_type = 'SS_AS'
                            # print(s, stride_type)
                        elif stride_start + 1 >= transition_ref['AS2LW'][(2 * c) + 1] and \
                                stride_end <= transition_ref['LW2DS'][2 * c]:
                            stride_type = 'SS_LW'
                            # print(s, stride_type)
                        elif stride_start + 1 >= transition_ref['LW2DS'][(2 * c) + 1] and \
                                stride_end <= transition_ref['DS2LW'][2 * c]:
                            stride_type = 'SS_DS'
                            # print(s, stride_type)
                        else:
                            stride_type = 'SS_LW'
                            # print(s, stride_type)
            else: # LW trial
                stride_type = 'SS_LW'

            if s==0:
                df_crop = dict_to_df[stride_edge[s]:stride_edge[s + 1]]
            else:
                df_crop = dict_to_df[stride_edge[s] - FWwinlen:stride_edge[s+1]]

            dict_sep_total[s] = {'data': df_crop, 'stride_type': stride_type}

        self.dict_sep_total = dict_sep_total

class alignedBag_Stair:

    ### OldSetting means transition LW->RA takes place in EarlyStance
    ### Later settings LW->RA takes place in SwingFlexion

    # def __init__(self, bagfilename, slope, startMSP = 0, endMSP = 1, plot = True):
    def __init__(self, bagfilename, plot=True):

        # ### Test
        # workspace_path = os.getcwd()
        # BagDir = workspace_path + '/Stair_Data_Raw/TF19/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(BagDir)
        # bagfilename = BagDir + 'OSL_Stair_Preset_1.bag'
        # ### Test

        try:
            aligned_SDat, aligned_footIMU, aligned_thighIMU, aligned_FSM, cropped_SInfo = align2SInfo(bagfilename, oldIMU = False)

        except:
            aligned_SDat, aligned_FSM, cropped_SInfo = align2SInfo(bagfilename, oldIMU = False)

        ##### Align time segment to State
        if 'TF19/OSL_Stair_Preset_1.bag' in bagfilename:
            # Special case: manual arrangement
            aligned_FSM['state'][83] = 'SA_SwingFlexion'
            aligned_FSM['state'][84] = 'LW_SwingFlexion'

        if 'TF19/OSL_Stair_Preset_2.bag' in bagfilename:
            # Special case: manual arrangement
            aligned_FSM['state'][139] = 'SA_SwingFlexion'
            aligned_FSM['state'][140] = 'LW_SwingFlexion'
            aligned_FSM = aligned_FSM.drop([244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254])

        FSM = FSMinfo(aligned_FSM)
        target_sensor_timeval = aligned_SDat['header']
        transition_idx_total = np.zeros(1)  # For plotting

        transition_idx_total_Sensor = {}  # For later SS/TR separation
        transition_idx_total_Sensor['LW2AS'] = []
        transition_idx_total_Sensor['AS2LW'] = []
        transition_idx_total_Sensor['LW2DS'] = []
        transition_idx_total_Sensor['DS2LW'] = []

        transitionMS_idx_total_Sensor = {}  # For later SS/TR separation
        transitionMS_idx_total_Sensor['LW2AS'] = []
        transitionMS_idx_total_Sensor['AS2LW'] = []
        transitionMS_idx_total_Sensor['LW2DS'] = []
        transitionMS_idx_total_Sensor['DS2LW'] = []

        input_starttime = FSM.time_val_ES[0]
        input_startidx = closest_point(target_sensor_timeval, input_starttime)
        input_endtime = FSM.time_val_ES[-1]
        input_endidx = closest_point(target_sensor_timeval, input_endtime)
        ### Data before the first ES and after the last ES will be excluded in input shaping

        ##### Find Transition time indices (SA<->LW, SD<->LW)
        '''
        Two different version needed...
        Our experiments make subjects to begin w/ sound side, end w/ prosthesis side.
        1) The last ES->ES
        2) One before the last SASF->SASE    
        '''
        # This is version 2)
        transition_idx_total = []
        for t in range(0, len(FSM.FSM_total) - 1):
            state_current = FSM.FSM_total['state'][t]
            state_next = FSM.FSM_total['state'][t+1]

            mode_current = state_current[:2]
            mode_next = state_next[:2]

            if mode_current == mode_next:
                continue
            else:
                if ('EarlyStance' in state_next) or ('SwingFlexion' in state_next):
                    # print(state_current, state_next)
                    transition_idx_total.append(t+1)

        FSM_idx_LW2SA_end = []
        FSM_idx_SA2LW_end = []
        FSM_idx_LW2SD_end = []
        FSM_idx_SD2LW_end = []

        try:
            # There are some trials that do not have LWES in the first step
            # For those cases, the first RAES is considered as transition step
            if FSM.time_idx_ES[0] == FSM.time_idx_SAES[0]:
                FSM_idx_LW2SA_end.append(FSM.time_idx_SAES[1])

        except:
            pass

        for i in range(0, len(transition_idx_total)):

            trid_in_FSM_total = transition_idx_total[i]  # index in time_idx_ES matrix
            # trid_in_FSM_total = FSM.time_idx_ES[trid]  # index in FSM_total
            # print(FSM.FSM_total['state'][trid_in_FSM_total-1], FSM.FSM_total['state'][trid_in_FSM_total])

            if 'SA' in FSM.FSM_total['state'][trid_in_FSM_total]:
                # print(trid_in_FSM_total + 2, ': LW2SA transition, ', FSM.FSM_total['state'][trid_in_FSM_total + 2])
                FSM_idx_LW2SA_end.append(trid_in_FSM_total + 2) # SA<->LW occurs in SF FSM, so +2 is required

            elif 'SD' in FSM.FSM_total['state'][trid_in_FSM_total]:
                # print(trid_in_FSM_total, ': LW2SD transition, ', FSM.FSM_total['state'][trid_in_FSM_total])
                FSM_idx_LW2SD_end.append(trid_in_FSM_total) # SD<->LW occurs in ES FSM

            else:
                if 'SA' in FSM.FSM_total['state'][trid_in_FSM_total-1]:
                    # print(trid_in_FSM_total + 2, ': SA2LW transition, ', FSM.FSM_total['state'][trid_in_FSM_total + 2])
                    FSM_idx_SA2LW_end.append(trid_in_FSM_total + 2)

                elif 'SD' in FSM.FSM_total['state'][trid_in_FSM_total-1]:
                    # print(trid_in_FSM_total, ': SD2LW transition, ', FSM.FSM_total['state'][trid_in_FSM_total])
                    FSM_idx_SD2LW_end.append(trid_in_FSM_total + 4)

        try:
            # There are some trials that do not have LWES in the last step
            # For those cases, the last RDES is considered as transition step
            if FSM.time_idx_ES[-1] == FSM.time_idx_SDES[-1]:
                FSM_idx_SD2LW_end.append(FSM.time_idx_SDES[-1])

        except:
            pass

        # Find time header value that corresponds to transition timing
        for n in range(0, len(FSM_idx_LW2SA_end)):
            transition_LW2SA_end_timeval = FSM.FSM_total['header'][FSM_idx_LW2SA_end[n]]
            transition_LW2SA_start_timeval = FSM.FSM_total['header'][FSM_idx_LW2SA_end[n] - 3]
            transition_SA2LW_end_timeval = FSM.FSM_total['header'][FSM_idx_SA2LW_end[n]]
            transition_SA2LW_start_timeval = FSM.FSM_total['header'][FSM_idx_SA2LW_end[n] - 3]
            transition_LW2SD_end_timeval = FSM.FSM_total['header'][FSM_idx_LW2SD_end[n]]
            transition_LW2SD_start_timeval = FSM.FSM_total['header'][FSM_idx_LW2SD_end[n] - 3]
            transition_SD2LW_end_timeval = FSM.FSM_total['header'][FSM_idx_SD2LW_end[n]]
            transition_SD2LW_start_timeval = FSM.FSM_total['header'][FSM_idx_SD2LW_end[n] - 4]

            # print(FSM.FSM_total['state'][FSM_idx_LW2SA_end[n]], FSM.FSM_total['state'][FSM_idx_LW2SA_end[n] - 3])
            # print(FSM.FSM_total['state'][FSM_idx_SA2LW_end[n]], FSM.FSM_total['state'][FSM_idx_SA2LW_end[n] - 3])
            # print(FSM.FSM_total['state'][FSM_idx_LW2SD_end[n]], FSM.FSM_total['state'][FSM_idx_LW2SD_end[n] - 3])
            # print(FSM.FSM_total['state'][FSM_idx_SD2LW_end[n]], FSM.FSM_total['state'][FSM_idx_SD2LW_end[n] - 3])

            # Find FSM transition index in sensor stream
            transition_LW2SA_start_idx_Sensor = closest_point(target_sensor_timeval, transition_LW2SA_start_timeval)
            transition_LW2SA_end_idx_Sensor = closest_point(target_sensor_timeval, transition_LW2SA_end_timeval)
            transition_SA2LW_start_idx_Sensor = closest_point(target_sensor_timeval, transition_SA2LW_start_timeval)
            transition_SA2LW_end_idx_Sensor = closest_point(target_sensor_timeval, transition_SA2LW_end_timeval)
            transition_LW2SD_start_idx_Sensor = closest_point(target_sensor_timeval, transition_LW2SD_start_timeval)
            transition_LW2SD_end_idx_Sensor = closest_point(target_sensor_timeval, transition_LW2SD_end_timeval)
            transition_SD2LW_start_idx_Sensor = closest_point(target_sensor_timeval, transition_SD2LW_start_timeval)
            transition_SD2LW_end_idx_Sensor = closest_point(target_sensor_timeval, transition_SD2LW_end_timeval)

            # Compensate start time offset sensor vs FSM
            transition_idx_total_Sensor['LW2AS'].append(transition_LW2SA_start_idx_Sensor - input_startidx)
            transition_idx_total_Sensor['LW2AS'].append(transition_LW2SA_end_idx_Sensor - input_startidx)
            transition_idx_total_Sensor['AS2LW'].append(transition_SA2LW_start_idx_Sensor - input_startidx)
            transition_idx_total_Sensor['AS2LW'].append(transition_SA2LW_end_idx_Sensor - input_startidx)
            transition_idx_total_Sensor['LW2DS'].append(transition_LW2SD_start_idx_Sensor - input_startidx)
            transition_idx_total_Sensor['LW2DS'].append(transition_LW2SD_end_idx_Sensor - input_startidx)
            transition_idx_total_Sensor['DS2LW'].append(transition_SD2LW_start_idx_Sensor - input_startidx)
            transition_idx_total_Sensor['DS2LW'].append(transition_SD2LW_end_idx_Sensor - input_startidx)

        # Arrange Sensor Data (Include Thigh/Foot IMU orientation in SensorData)
        try:
            foot_qx = aligned_footIMU['orientation_x']
            foot_qy = aligned_footIMU['orientation_y']
            foot_qz = aligned_footIMU['orientation_z']
            foot_qw = aligned_footIMU['orientation_w']

            thigh_qx = aligned_thighIMU['orientation_x']
            thigh_qy = aligned_thighIMU['orientation_y']
            thigh_qz = aligned_thighIMU['orientation_z']
            thigh_qw = aligned_thighIMU['orientation_w']

            ### Compute orientations
            Zvector = np.array([0, 0, 1])

            foot_qmat = np.array([foot_qw, foot_qx, foot_qy, foot_qz]).T
            thigh_qmat = np.array([thigh_qw, thigh_qx, thigh_qy, thigh_qz]).T

            foot_Rmat = np.zeros([len(foot_qmat), 3, 3])
            thigh_Rmat = np.zeros([len(thigh_qmat), 3, 3])

            foot_rotZmat = np.zeros([len(foot_qmat), 3])
            thigh_rotZmat = np.zeros([len(thigh_qmat), 3])

            foot_anglemat = np.zeros([len(foot_qmat)])
            thigh_anglemat = np.zeros([len(thigh_qmat)])

            for i in range(0, len(foot_qmat)):
                foot_Rmat[i, :] = AHRS.q2Rot(foot_qmat[i])
                foot_rotZmat[i, :] = np.matmul(foot_Rmat[i, :], Zvector)

                vlen = np.sign(foot_rotZmat[i, 0]) * np.sqrt(foot_rotZmat[i, 0] ** 2 + foot_rotZmat[i, 1] ** 2)
                hlen = foot_rotZmat[i, 2]

                foot_anglemat[i] = 90 - math.atan2(hlen, vlen) * 180 / np.pi

                # Denoise for realtime
                if i > 0:
                    if foot_anglemat[i] - foot_anglemat[i - 1] > 25:
                        foot_anglemat[i] = foot_anglemat[i - 1]

            for i in range(0, len(thigh_qmat)):
                thigh_Rmat[i, :] = AHRS.q2Rot(thigh_qmat[i])
                thigh_rotZmat[i, :] = np.matmul(thigh_Rmat[i, :], Zvector)

                hlen = np.sign(thigh_rotZmat[i, 0]) * np.sqrt(thigh_rotZmat[i, 0] ** 2 + thigh_rotZmat[i, 1] ** 2)
                vlen = thigh_rotZmat[i, 2]

                thigh_anglemat[i] = 90 - math.atan2(hlen, vlen) * 180 / np.pi

                # Denoise for realtime
                if i > 0:
                    if thigh_anglemat[i] - thigh_anglemat[i - 1] < -15:
                        thigh_anglemat[i] = thigh_anglemat[i - 1]

            offset_end = closest_point(target_sensor_timeval, input_starttime)
            foot_angle_offset = np.mean(foot_anglemat[:offset_end])
            thigh_angle_offset = np.mean(thigh_anglemat[:offset_end])

            foot_anglemat -= foot_angle_offset
            thigh_anglemat -= thigh_angle_offset

            aligned_SDat['thigh_orientation'] = thigh_anglemat
            aligned_SDat['foot_orientation'] = foot_anglemat

        except:
            pass

        # ground_truth = ground_truth[input_startidx:input_endidx]
        aligned_SDat = df_rearrange_header(aligned_SDat[input_startidx:input_endidx])
        cropped_SInfo = df_rearrange_header(cropped_SInfo[input_startidx:input_endidx])

        transition_counts = int(len(transition_idx_total_Sensor['LW2AS']) / 2)

        # Store to this class
        # self.ground_truth = ground_truth
        self.sensor_data = aligned_SDat
        self.sensor_info = cropped_SInfo
        self.FSM = FSM
        self.transition_idx_Sensor = transition_idx_total_Sensor
        self.transition_counts = transition_counts
        # self.transition_idx = transition_idx_total
        # self.transition_MSidx = transition_MSidx_total

class alignedBag_Ramp_OldSetting_v2:

    ### OldSetting means transition LW->RA takes place in EarlyStance
    ### Later settings LW->RA takes place in SwingFlexion

    # def __init__(self, bagfilename, slope, startMSP = 0, endMSP = 1, plot = True):
    def __init__(self, bagfilename, FWwinlen, plot=True):

        # ### Test
        # workspace_path = os.getcwd()
        # BagDir = workspace_path + '/Ramp_Data_Raw_Additional/Slope_Online/TF02v2/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(BagDir)
        #
        # bagfilename = BagDir + datalist[7]
        # ### Test

        try:
            aligned_SDat, aligned_footIMU, aligned_thighIMU, aligned_FSM, cropped_SInfo = align2SInfo(bagfilename, oldIMU = True)

        except:
            aligned_SDat, aligned_FSM, cropped_SInfo = align2SInfo(bagfilename, oldIMU = True)

        FSM = FSMinfo(aligned_FSM)
        target_sensor_timeval = aligned_SDat['header']

        input_starttime = FSM.time_val_ES[0]
        input_startidx = closest_point(target_sensor_timeval, input_starttime)
        input_endtime = FSM.time_val_ES[-1]
        input_endidx = closest_point(target_sensor_timeval, input_endtime)
        ### Data before the first ES and after the last ES will be excluded in input shaping

        try:
            foot_qx = aligned_footIMU['orientation_x']
            foot_qy = aligned_footIMU['orientation_y']
            foot_qz = aligned_footIMU['orientation_z']
            foot_qw = aligned_footIMU['orientation_w']

            thigh_qx = aligned_thighIMU['orientation_x']
            thigh_qy = aligned_thighIMU['orientation_y']
            thigh_qz = aligned_thighIMU['orientation_z']
            thigh_qw = aligned_thighIMU['orientation_w']

            ### Compute orientations
            Zvector = np.array([0, 0, 1])

            foot_qmat = np.array([foot_qw, foot_qx, foot_qy, foot_qz]).T
            thigh_qmat = np.array([thigh_qw, thigh_qx, thigh_qy, thigh_qz]).T

            foot_Rmat = np.zeros([len(foot_qmat), 3, 3])
            thigh_Rmat = np.zeros([len(thigh_qmat), 3, 3])

            foot_rotZmat = np.zeros([len(foot_qmat), 3])
            thigh_rotZmat = np.zeros([len(thigh_qmat), 3])

            foot_anglemat = np.zeros([len(foot_qmat)])
            thigh_anglemat = np.zeros([len(thigh_qmat)])

            for i in range(0, len(foot_qmat)):
                foot_Rmat[i, :] = AHRS.q2Rot(foot_qmat[i])
                foot_rotZmat[i, :] = np.matmul(foot_Rmat[i, :], Zvector)

                vlen = np.sign(foot_rotZmat[i, 0]) * np.sqrt(foot_rotZmat[i, 0] ** 2 + foot_rotZmat[i, 1] ** 2)
                hlen = foot_rotZmat[i, 2]

                foot_anglemat[i] = 90 - math.atan2(hlen, vlen) * 180 / np.pi

                # Denoise for realtime
                if i > 0:
                    if foot_anglemat[i] - foot_anglemat[i - 1] > 25:
                        foot_anglemat[i] = foot_anglemat[i - 1]

            for i in range(0, len(thigh_qmat)):
                thigh_Rmat[i, :] = AHRS.q2Rot(thigh_qmat[i])
                thigh_rotZmat[i, :] = np.matmul(thigh_Rmat[i, :], Zvector)

                hlen = np.sign(thigh_rotZmat[i, 0]) * np.sqrt(thigh_rotZmat[i, 0] ** 2 + thigh_rotZmat[i, 1] ** 2)
                vlen = thigh_rotZmat[i, 2]

                thigh_anglemat[i] = 90 - math.atan2(hlen, vlen) * 180 / np.pi

                # Denoise for realtime
                if i > 0:
                    if thigh_anglemat[i] - thigh_anglemat[i - 1] < -15:
                        thigh_anglemat[i] = thigh_anglemat[i - 1]

            offset_end = closest_point(target_sensor_timeval, input_starttime)
            foot_angle_offset = np.mean(foot_anglemat[:offset_end])
            thigh_angle_offset = np.mean(thigh_anglemat[:offset_end])

            foot_anglemat -= foot_angle_offset
            thigh_anglemat -= thigh_angle_offset

            aligned_SDat['thigh_orientation'] = thigh_anglemat
            aligned_SDat['foot_orientation'] = foot_anglemat

        except:
            pass

        aligned_SDat = df_rearrange_header(aligned_SDat[input_startidx:input_endidx + 1])
        cropped_SInfo = df_rearrange_header(cropped_SInfo[input_startidx:input_endidx + 1])

        state_dict = {}
        state_dict['header'] = aligned_SDat['header']
        state_dict['state'] = []
        state_dict['state'].append(FSM.FSM_total['state'][0])

        i = 1
        while i <= len(aligned_SDat) - 1:
            # for i in range(1, len(aligned_SDat) - 1):
            if state_dict['header'][i] in FSM.FSM_total['header'].__array__():
                state_idx_FSM = np.where(FSM.FSM_total['header'].__array__() == state_dict['header'][i])[0]

                if len(state_idx_FSM) == 1:
                    state_dict['state'].append(FSM.FSM_total['state'][state_idx_FSM[0]])
                    i += 1

                else:  ## Sometimes consecutive time header values are exactly same
                    if FSM.FSM_total['state'][state_idx_FSM[1]] != 'Home':
                        print(i)
                        state_dict['state'].append(FSM.FSM_total['state'][state_idx_FSM[0]])
                        state_dict['state'].append(FSM.FSM_total['state'][state_idx_FSM[1]])
                        i += 2

                    else:
                        state_dict['state'].append(FSM.FSM_total['state'][state_idx_FSM[0]])
                        i += 1

            else:
                state_dict['state'].append(state_dict['state'][i - 1])
                i += 1

        state_dict_df = pd.DataFrame.from_dict(state_dict)

        ### state_mat: state as numbers
        state_mat = np.zeros(len(state_dict_df))

        for t in range(0, len(state_mat)):

            state = state_dict_df['state'][t]

            if 'EarlyStance' in state:
                state_num = 1
                # state_ESnum = 1
            elif 'LateStance' in state:
                state_num = 1
                # state_ESnum = 0
            elif 'SwingFlexion' in state:
                state_num = 2
                # state_ESnum = 0
            elif 'SwingExtension' in state:
                state_num = 3
                # state_ESnum = 0
            else:
                state_num = 0
                # state_ESnum = 0

                # print(state)

            state_mat[t] = state_num

        state_dict_df['state_num'] = state_mat
        state_dict_df_noHeader = state_dict_df.drop(['header'], axis=1)
        SInfo_noHeader = cropped_SInfo.drop(['header'], axis=1)

        dict_to_df = pd.merge(aligned_SDat, SInfo_noHeader, left_index=True, right_index=True)
        dict_to_df = pd.merge(dict_to_df, state_dict_df_noHeader, left_index=True, right_index=True)
        dict_to_df = df_rearrange_header(dict_to_df)

        # Store to this class
        self.sensor_data = aligned_SDat
        self.sensor_info = cropped_SInfo
        self.FSM = FSM
        self.dict_arranged = dict_to_df

class alignedBag_Stair_v2:

    ### Save raw .bag file to CSV... before separating into stride files

    # def __init__(self, bagfilename, slope, startMSP = 0, endMSP = 1, plot = True):
    def __init__(self, bagfilename, FWwinlen, plot=False):

        # ### Test
        # workspace_path = os.getcwd()
        # BagDir = workspace_path + '/Stair_Data_Raw/TF21/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(BagDir)
        # bagfilename = BagDir + 'OSL_Stair_Preset_1.bag'
        # FWwinlen = 125
        # ### Test

        try:
            aligned_SDat, aligned_footIMU, aligned_thighIMU, aligned_FSM, cropped_SInfo = align2SInfo(bagfilename, oldIMU = False)

        except:
            aligned_SDat, aligned_FSM, cropped_SInfo = align2SInfo(bagfilename, oldIMU = False)

        FSM = FSMinfo(aligned_FSM)
        target_sensor_timeval = aligned_SDat['header']

        input_starttime = FSM.time_val_ES[0]
        input_startidx = closest_point(target_sensor_timeval, input_starttime)
        input_endtime = FSM.time_val_ES[-1]
        input_endidx = closest_point(target_sensor_timeval, input_endtime)
        ### Data before the first ES and after the last ES will be excluded in input shaping

        try:
            foot_qx = aligned_footIMU['orientation_x']
            foot_qy = aligned_footIMU['orientation_y']
            foot_qz = aligned_footIMU['orientation_z']
            foot_qw = aligned_footIMU['orientation_w']

            thigh_qx = aligned_thighIMU['orientation_x']
            thigh_qy = aligned_thighIMU['orientation_y']
            thigh_qz = aligned_thighIMU['orientation_z']
            thigh_qw = aligned_thighIMU['orientation_w']

            ### Compute orientations
            Zvector = np.array([0, 0, 1])

            foot_qmat = np.array([foot_qw, foot_qx, foot_qy, foot_qz]).T
            thigh_qmat = np.array([thigh_qw, thigh_qx, thigh_qy, thigh_qz]).T

            foot_Rmat = np.zeros([len(foot_qmat), 3, 3])
            thigh_Rmat = np.zeros([len(thigh_qmat), 3, 3])

            foot_rotZmat = np.zeros([len(foot_qmat), 3])
            thigh_rotZmat = np.zeros([len(thigh_qmat), 3])

            foot_anglemat = np.zeros([len(foot_qmat)])
            thigh_anglemat = np.zeros([len(thigh_qmat)])

            for i in range(0, len(foot_qmat)):
                foot_Rmat[i, :] = AHRS.q2Rot(foot_qmat[i])
                foot_rotZmat[i, :] = np.matmul(foot_Rmat[i, :], Zvector)

                vlen = np.sign(foot_rotZmat[i, 0]) * np.sqrt(foot_rotZmat[i, 0] ** 2 + foot_rotZmat[i, 1] ** 2)
                hlen = foot_rotZmat[i, 2]

                foot_anglemat[i] = 90 - math.atan2(hlen, vlen) * 180 / np.pi

                # Denoise for realtime
                if i > 0:
                    if foot_anglemat[i] - foot_anglemat[i - 1] > 25:
                        foot_anglemat[i] = foot_anglemat[i - 1]

            for i in range(0, len(thigh_qmat)):
                thigh_Rmat[i, :] = AHRS.q2Rot(thigh_qmat[i])
                thigh_rotZmat[i, :] = np.matmul(thigh_Rmat[i, :], Zvector)

                hlen = np.sign(thigh_rotZmat[i, 0]) * np.sqrt(thigh_rotZmat[i, 0] ** 2 + thigh_rotZmat[i, 1] ** 2)
                vlen = thigh_rotZmat[i, 2]

                thigh_anglemat[i] = 90 - math.atan2(hlen, vlen) * 180 / np.pi

                # Denoise for realtime
                if i > 0:
                    if thigh_anglemat[i] - thigh_anglemat[i - 1] < -15:
                        thigh_anglemat[i] = thigh_anglemat[i - 1]

            offset_end = closest_point(target_sensor_timeval, input_starttime)
            foot_angle_offset = np.mean(foot_anglemat[:offset_end])
            thigh_angle_offset = np.mean(thigh_anglemat[:offset_end])

            foot_anglemat -= foot_angle_offset
            thigh_anglemat -= thigh_angle_offset

            aligned_SDat['thigh_orientation'] = thigh_anglemat
            aligned_SDat['foot_orientation'] = foot_anglemat

        except:
            pass

        aligned_SDat = df_rearrange_header(aligned_SDat[input_startidx:input_endidx + 1])
        cropped_SInfo = df_rearrange_header(cropped_SInfo[input_startidx:input_endidx + 1])

        # state_dict: state as string
        state_dict = {}
        state_dict['header'] = aligned_SDat['header']
        state_dict['state'] = []
        state_dict['state'].append(FSM.FSM_total['state'][0])

        i = 1
        while i <= len(aligned_SDat) - 1:
        # for i in range(1, len(aligned_SDat) - 1):
            if state_dict['header'][i] in FSM.FSM_total['header'].__array__():
                state_idx_FSM = np.where(FSM.FSM_total['header'].__array__() == state_dict['header'][i])[0]

                if len(state_idx_FSM) == 1:
                    state_dict['state'].append(FSM.FSM_total['state'][state_idx_FSM[0]])
                    i += 1

                else: ## Sometimes consecutive time header values are exactly same
                    state_dict['state'].append(FSM.FSM_total['state'][state_idx_FSM[0]])
                    state_dict['state'].append(FSM.FSM_total['state'][state_idx_FSM[1]])
                    i += 2

            else:
                state_dict['state'].append(state_dict['state'][i-1])
                i += 1

        state_dict_df = pd.DataFrame.from_dict(state_dict)

        ### state_mat: state as numbers
        state_mat = np.zeros(len(state_dict_df))

        for t in range(0, len(state_mat)):

            state = state_dict_df['state'][t]

            if 'EarlyStance' in state:
                state_num = 1
                # state_ESnum = 1
            elif 'LateStance' in state:
                state_num = 1
                # state_ESnum = 0
            elif 'SwingFlexion' in state:
                state_num = 2
                # state_ESnum = 0
            elif 'SwingExtension' in state:
                state_num = 3
                # state_ESnum = 0
            else:
                state_num = 0
                # state_ESnum = 0

                # print(state)

            state_mat[t] = state_num

        state_dict_df['state_num'] = state_mat
        state_dict_df_noHeader = state_dict_df.drop(['header'], axis = 1)
        SInfo_noHeader = cropped_SInfo.drop(['header'], axis = 1)

        dict_to_df = pd.merge(aligned_SDat, SInfo_noHeader, left_index= True, right_index=True)
        dict_to_df = pd.merge(dict_to_df, state_dict_df_noHeader, left_index=True, right_index=True)
        dict_to_df = df_rearrange_header(dict_to_df)

        # Store to this class
        self.sensor_data = aligned_SDat
        self.sensor_info = cropped_SInfo
        self.FSM = FSM
        self.dict_arranged = dict_to_df

class Stair_GT_byMSP:

    ### OldSetting means transition LW->RA takes place in EarlyStance
    ### Later settings LW->RA takes place in SwingFlexion

    def __init__(self, bagfilename, slope, startMSP_list = [], endMSP_list = []):

        # ### Test
        # workspace_path = os.getcwd()
        # BagDir = workspace_path + '/Stair_Data_Raw/TF19/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(BagDir)
        # bagfilename = BagDir + datalist[2]
        # slope = 19.6
        # startMSP_list = [0, 0.25, 0.5, 0.75, 1.0]
        # endMSP_list = [0, 0.25, 0.5, 0.75, 1.0]
        # ### Test

        aligned_bag = alignedBag_Stair(bagfilename, plot = False)
        transition_idx_Sensor = aligned_bag.transition_idx_Sensor
        sensor_data_array = aligned_bag.sensor_data.to_numpy()
        transition_counts = int(len(transition_idx_Sensor['LW2AS'])/2)

        # Run slope GT variation
        GT_slope_dict = {}
        for start in range(0, len(startMSP_list)):
            for end in range(0, len(endMSP_list)):

                startMSP = startMSP_list[start]
                endMSP = endMSP_list[end]

                if startMSP > endMSP:
                    continue

                keyName = 'MSP' + str(startMSP) + 'to' + str(endMSP)
                ground_truth = np.zeros([len(sensor_data_array)])
                # print(keyName)

                for n in range(0, transition_counts):
                    # Vary transition timing during Mid-Swing (MS)
                    transitionMS_LW2SA_start_idx_Sensor = int((1 - startMSP) * transition_idx_Sensor['LW2AS'][2 * n] + \
                                                              startMSP * transition_idx_Sensor['LW2AS'][2 * n + 1])

                    transitionMS_LW2SA_end_idx_Sensor = int((1 - endMSP) * transition_idx_Sensor['LW2AS'][2 * n] + \
                                                            endMSP * transition_idx_Sensor['LW2AS'][2 * n + 1])

                    transitionMS_SA2LW_start_idx_Sensor = int((1 - startMSP) * transition_idx_Sensor['AS2LW'][2 * n] + \
                                                              startMSP * transition_idx_Sensor['AS2LW'][2 * n + 1])

                    transitionMS_SA2LW_end_idx_Sensor = int((1 - endMSP) * transition_idx_Sensor['AS2LW'][2 * n] + \
                                                            endMSP * transition_idx_Sensor['AS2LW'][2 * n + 1])

                    transitionMS_LW2SD_start_idx_Sensor = int((1 - startMSP) * transition_idx_Sensor['LW2DS'][2 * n] + \
                                                              startMSP * transition_idx_Sensor['LW2DS'][2 * n + 1])

                    transitionMS_LW2SD_end_idx_Sensor = int((1 - endMSP) * transition_idx_Sensor['LW2DS'][2 * n] + \
                                                            endMSP * transition_idx_Sensor['LW2DS'][2 * n + 1])

                    transitionMS_SD2LW_start_idx_Sensor = int((1 - startMSP) * transition_idx_Sensor['DS2LW'][2 * n] + \
                                                              startMSP * transition_idx_Sensor['DS2LW'][2 * n + 1])

                    transitionMS_SD2LW_end_idx_Sensor = int((1 - endMSP) * transition_idx_Sensor['DS2LW'][2 * n] + \
                                                            endMSP * transition_idx_Sensor['DS2LW'][2 * n + 1])

                    # Fill Steady-State walk
                    ground_truth[transitionMS_LW2SA_end_idx_Sensor:transitionMS_SA2LW_start_idx_Sensor] = 1
                    ground_truth[transitionMS_LW2SD_end_idx_Sensor:transitionMS_SD2LW_start_idx_Sensor] = -1
                    # GT_slope_dict[keyName] = ground_truth

                    if not startMSP == endMSP:
                        # print('MSPs are different')

                        ### Generate linear transition
                        n_transition_LW2SA = transitionMS_LW2SA_end_idx_Sensor - transitionMS_LW2SA_start_idx_Sensor
                        n_transition_SA2LW = transitionMS_SA2LW_end_idx_Sensor - transitionMS_SA2LW_start_idx_Sensor
                        n_transition_LW2SD = transitionMS_LW2SD_end_idx_Sensor - transitionMS_LW2SD_start_idx_Sensor
                        n_transition_SD2LW = transitionMS_SD2LW_end_idx_Sensor - transitionMS_SD2LW_start_idx_Sensor

                        lt_LW2SA = np.linspace(0, 1, n_transition_LW2SA)
                        lt_SA2LW = np.linspace(1, 0, n_transition_SA2LW)
                        lt_LW2SD = np.linspace(0, -1, n_transition_LW2SD)
                        lt_SD2LW = np.linspace(-1, 0, n_transition_SD2LW)

                        ground_truth[transitionMS_LW2SA_start_idx_Sensor:transitionMS_LW2SA_end_idx_Sensor] = lt_LW2SA
                        ground_truth[transitionMS_SA2LW_start_idx_Sensor:transitionMS_SA2LW_end_idx_Sensor] = lt_SA2LW
                        ground_truth[transitionMS_LW2SD_start_idx_Sensor:transitionMS_LW2SD_end_idx_Sensor] = lt_LW2SD
                        ground_truth[transitionMS_SD2LW_start_idx_Sensor:transitionMS_SD2LW_end_idx_Sensor] = lt_SD2LW

                # ground_truth *= slope
                GT_slope_dict[keyName] = slope * ground_truth

        self.GT_slope_dict = GT_slope_dict
        # self.transition_counts = transition_counts

class separation_SSTR_Stair:
    def __init__(self, bagfilename, plot = False, slope = 0, startMSP_list = [], endMSP_list = [], FWwinlen = 125):

        # ### Test
        # workspace_path = os.getcwd()
        # BagDir = workspace_path + '/Stair_Data_Raw/TF19/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(BagDir)
        # bagfilename = BagDir + datalist[3]
        # slope = 19.6
        # startMSP_list = [0, 0.25, 0.5, 0.75, 1.0]
        # endMSP_list = [0, 0.25, 0.5, 0.75, 1.0]
        #
        # FWwinlen = 125
        # plot = False
        # ### Test

        testload = alignedBag_Stair(bagfilename, plot = plot)
        testload = testload.__dict__

        GT_slope_dict = Stair_GT_byMSP(bagfilename, slope, startMSP_list = startMSP_list, endMSP_list = endMSP_list).GT_slope_dict
        GT_slope_df = pd.DataFrame.from_dict(GT_slope_dict)

        transition_counts = testload['transition_counts']
        SInfo_noHeader = testload['sensor_info']
        SInfo_noHeader = SInfo_noHeader.drop(['header'], axis = 1)
        dict_to_df = pd.merge(testload['sensor_data'], SInfo_noHeader, left_index= True, right_index=True)
        dict_to_df = df_rearrange_header(dict_to_df)

        State_and_Slope_header = dict_to_df['header'].__array__()
        FSM_header = testload['FSM'].FSM_total['header'].__array__()
        state_mat = np.zeros(len(dict_to_df))

        for t in range(0, len(State_and_Slope_header)):

            if State_and_Slope_header[t] in FSM_header:

                matchIdx = np.where(FSM_header == State_and_Slope_header[t])[0][0]
                state = testload['FSM'].FSM_total['state'][matchIdx]

                if 'EarlyStance' in state:
                    state_num = 1
                    # state_ESnum = 1
                elif 'LateStance' in state:
                    state_num = 1
                    # state_ESnum = 0
                elif 'SwingFlexion' in state:
                    state_num = 2
                    # state_ESnum = 0
                elif 'SwingExtension' in state:
                    state_num = 3
                    # state_ESnum = 0
                else:
                    state_num = 0
                    # state_ESnum = 0

                # print(state)

            state_mat[t] = state_num

        transition_ref = testload['transition_idx_Sensor']

        state_mat[0] = 3
        state_mat[-1] = 1

        dict_to_df['state'] = state_mat

        dict_to_df = pd.merge(dict_to_df, GT_slope_df, left_index = True, right_index=True)

        stride_ref = np.diff(state_mat)
        stride_edge = np.where(stride_ref == -2)[0]+1

        # plt.plot(state_mat)
        # plt.plot(ground_truth)
        # plt.plot(stride_ref)
        # plt.plot(stride_edge)

        dict_sep_total = {}

        for s in range(0, len(stride_edge)-1):
            # s = 9
            stride_start = stride_edge[s]
            stride_end = stride_edge[s + 1]

            print(stride_start)
            print(stride_end)

            # Determine stride type, if transition exists since it's not LW trial
            if transition_counts > 0:
                for c in range(0, transition_counts):

                    # c = 0
                    # c = 1
                    # c = 2
                    # c = 3

                    # If Transition
                    if stride_start <= transition_ref['LW2AS'][2 * c] and stride_end + 1 >= transition_ref['LW2AS'][2 * c]:
                        stride_type = 'TR_LW2AS'
                        print(s, c, stride_type)
                    elif stride_start <= transition_ref['AS2LW'][2 * c] and stride_end + 1 >= transition_ref['AS2LW'][2 * c]:
                        stride_type = 'TR_AS2LW'
                        print(s, c, stride_type)
                    elif stride_start <= transition_ref['LW2DS'][2 * c] and stride_end + 1 >= transition_ref['LW2DS'][2 * c]:
                        stride_type = 'TR_LW2DS'
                        # print(s, c, stride_type)
                    elif stride_start <= transition_ref['DS2LW'][2 * c] and stride_end > transition_ref['DS2LW'][2 * c]:
                        stride_type = 'TR_DS2LW'
                        # print(s, c, stride_type)

                    else: # If Steady-State
                        # if (s == 0) or (s == len(stride_edge)-2):
                        #     stride_type = 'SS_LW'
                        #     # print(s, c, stride_type)

                        if stride_end <= transition_ref['LW2AS'][0]:
                            stride_type = 'SS_LW'
                            # print(s, c, stride_type)

                        elif stride_start >= transition_ref['DS2LW'][-1]:
                            stride_type = 'SS_LW'
                            # print(s, c, stride_type)

                        elif stride_start + 1 >= transition_ref['LW2AS'][(2 * c) + 1] and \
                                stride_end <= transition_ref['AS2LW'][2 * c]:
                            stride_type = 'SS_AS'
                            # print(s, c, stride_type)
                        elif stride_start + 1 >= transition_ref['AS2LW'][(2 * c) + 1] and \
                                stride_end <= transition_ref['LW2DS'][2 * c]:
                            stride_type = 'SS_LW'
                            # print(s, c, stride_type)
                        elif stride_start + 1 >= transition_ref['LW2DS'][(2 * c) + 1] and \
                                stride_end <= transition_ref['DS2LW'][2 * c]:
                            stride_type = 'SS_DS'
                            # print(s, c, stride_type)
                        else:
                            try:
                                if stride_start + 1 >= transition_ref['DS2LW'][(2 * c) + 1] and \
                                        stride_end <= transition_ref['LW2AS'][2 * (c + 1)]:
                                    stride_type = 'SS_LW'
                                    # print(s, c, stride_type)
                            except:
                                if stride_start + 1 >= transition_ref['DS2LW'][(2 * c) + 1]:
                                    stride_type = 'SS_LW'
                                    # print(s, c, stride_type)
                                # pass
            print(s, c, stride_type)
            #
            # else: # LW trial
            #     stride_type = 'SS_LW'

            if s==0:
                df_crop = dict_to_df[stride_edge[s]:stride_edge[s + 1]]
            else:
                df_crop = dict_to_df[stride_edge[s] - FWwinlen:stride_edge[s+1]]

            dict_sep_total[s] = {'data': df_crop, 'stride_type': stride_type}

        self.dict_sep_total = dict_sep_total

class separation_stride_Stair:
    def __init__(self, bagfilename, plot = False, slope = 0, startMSP_list = [], endMSP_list = [], FWwinlen = 125):

        # ### Test
        # workspace_path = os.getcwd()
        # BagDir = workspace_path + '/Stair_Data_Raw/TF19/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(BagDir)
        # bagfilename = BagDir + datalist[3]
        # slope = 19.6
        # startMSP_list = [0, 0.25, 0.5, 0.75, 1.0]
        # endMSP_list = [0, 0.25, 0.5, 0.75, 1.0]
        #
        # FWwinlen = 125
        # plot = False
        # ### Test

        testload = alignedBag_Stair(bagfilename, plot = plot)
        testload = testload.__dict__

        GT_slope_dict = Stair_GT_byMSP(bagfilename, slope, startMSP_list = startMSP_list, endMSP_list = endMSP_list).GT_slope_dict
        GT_slope_df = pd.DataFrame.from_dict(GT_slope_dict)

        transition_counts = testload['transition_counts']
        SInfo_noHeader = testload['sensor_info']
        SInfo_noHeader = SInfo_noHeader.drop(['header'], axis = 1)
        dict_to_df = pd.merge(testload['sensor_data'], SInfo_noHeader, left_index= True, right_index=True)
        dict_to_df = df_rearrange_header(dict_to_df)

        State_and_Slope_header = dict_to_df['header'].__array__()
        FSM_header = testload['FSM'].FSM_total['header'].__array__()
        state_mat = np.zeros(len(dict_to_df))

        for t in range(0, len(State_and_Slope_header)):

            if State_and_Slope_header[t] in FSM_header:

                matchIdx = np.where(FSM_header == State_and_Slope_header[t])[0][0]
                state = testload['FSM'].FSM_total['state'][matchIdx]

                if 'EarlyStance' in state:
                    state_num = 1
                    # state_ESnum = 1
                elif 'LateStance' in state:
                    state_num = 1
                    # state_ESnum = 0
                elif 'SwingFlexion' in state:
                    state_num = 2
                    # state_ESnum = 0
                elif 'SwingExtension' in state:
                    state_num = 3
                    # state_ESnum = 0
                else:
                    state_num = 0
                    # state_ESnum = 0

                # print(state)

            state_mat[t] = state_num

        transition_ref = testload['transition_idx_Sensor']

        state_mat[0] = 3
        state_mat[-1] = 1

        dict_to_df['state'] = state_mat

        dict_to_df = pd.merge(dict_to_df, GT_slope_df, left_index = True, right_index=True)

        stride_ref = np.diff(state_mat)
        stride_edge = np.where(stride_ref == -2)[0]+1

        # plt.plot(state_mat)
        # plt.plot(ground_truth)
        # plt.plot(stride_ref)
        # plt.plot(stride_edge)

        dict_sep_total = {}

        for s in range(0, len(stride_edge)-1):
            # s = 9
            stride_start = stride_edge[s]
            stride_end = stride_edge[s + 1]

            print(stride_start)
            print(stride_end)

            # Determine stride type, if transition exists since it's not LW trial
            if transition_counts > 0:
                for c in range(0, transition_counts):

                    # c = 0
                    # c = 1
                    # c = 2
                    # c = 3

                    # If Transition
                    if stride_start <= transition_ref['LW2AS'][2 * c] and stride_end + 1 >= transition_ref['LW2AS'][2 * c]:
                        stride_type = 'TR_LW2AS'
                        print(s, c, stride_type)
                    elif stride_start <= transition_ref['AS2LW'][2 * c] and stride_end + 1 >= transition_ref['AS2LW'][2 * c]:
                        stride_type = 'TR_AS2LW'
                        print(s, c, stride_type)
                    elif stride_start <= transition_ref['LW2DS'][2 * c] and stride_end + 1 >= transition_ref['LW2DS'][2 * c]:
                        stride_type = 'TR_LW2DS'
                        # print(s, c, stride_type)
                    elif stride_start <= transition_ref['DS2LW'][2 * c] and stride_end > transition_ref['DS2LW'][2 * c]:
                        stride_type = 'TR_DS2LW'
                        # print(s, c, stride_type)

                    else: # If Steady-State
                        # if (s == 0) or (s == len(stride_edge)-2):
                        #     stride_type = 'SS_LW'
                        #     # print(s, c, stride_type)

                        if stride_end <= transition_ref['LW2AS'][0]:
                            stride_type = 'SS_LW'
                            # print(s, c, stride_type)

                        elif stride_start >= transition_ref['DS2LW'][-1]:
                            stride_type = 'SS_LW'
                            # print(s, c, stride_type)

                        elif stride_start + 1 >= transition_ref['LW2AS'][(2 * c) + 1] and \
                                stride_end <= transition_ref['AS2LW'][2 * c]:
                            stride_type = 'SS_AS'
                            # print(s, c, stride_type)
                        elif stride_start + 1 >= transition_ref['AS2LW'][(2 * c) + 1] and \
                                stride_end <= transition_ref['LW2DS'][2 * c]:
                            stride_type = 'SS_LW'
                            # print(s, c, stride_type)
                        elif stride_start + 1 >= transition_ref['LW2DS'][(2 * c) + 1] and \
                                stride_end <= transition_ref['DS2LW'][2 * c]:
                            stride_type = 'SS_DS'
                            # print(s, c, stride_type)
                        else:
                            try:
                                if stride_start + 1 >= transition_ref['DS2LW'][(2 * c) + 1] and \
                                        stride_end <= transition_ref['LW2AS'][2 * (c + 1)]:
                                    stride_type = 'SS_LW'
                                    # print(s, c, stride_type)
                            except:
                                if stride_start + 1 >= transition_ref['DS2LW'][(2 * c) + 1]:
                                    stride_type = 'SS_LW'
                                    # print(s, c, stride_type)
                                # pass
            print(s, c, stride_type)
            #
            # else: # LW trial
            #     stride_type = 'SS_LW'

            if s==0:
                df_crop = dict_to_df[stride_edge[s]:stride_edge[s + 1]]
            else:
                df_crop = dict_to_df[stride_edge[s] - FWwinlen:stride_edge[s+1]]

            dict_sep_total[s] = {'data': df_crop, 'stride_type': stride_type}

        self.dict_sep_total = dict_sep_total

class separation_stride_Stair_v2:
    def __init__(self, csvfilename, plot = False, slope = 0, FWwinlen = 125, delay = 15):

        # Reads CSV file format

        # ### Test
        # workspace_path = os.getcwd()
        # csvDir = workspace_path + '/processedCSV_StairDat_byBagfile/Offline/TF08/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(csvDir)
        # csvfilename = csvDir + datalist[6]
        # slope = 24
        #
        # FWwinlen = 125
        # plot = False
        # delay = 30
        # ### Test

        raw_csv = pd.read_csv(csvfilename)

        # Check data quality
        quality_array = raw_csv['quality'].__array__()
        seg_starts = []
        seg_ends = []
        for q in range(0, len(quality_array)):
            if quality_array[q] == 'G':
                if q == 0: # When the first element is good
                    seg_starts.append(q)
                elif quality_array[q - 1] != 'G':
                    seg_starts.append(q)
                elif q == len(quality_array) - 1:
                    seg_ends.append(q)

            else:
                if q > 0:
                    if quality_array[q - 1] == 'G':
                        seg_ends.append(q - 1)

        segment_dict = {}

        for i in range(0, len(seg_starts)):
            segment_dict[i] = raw_csv[seg_starts[i] : seg_ends[i] + 1]

        # Collect stride start/end index
        stride_intervals = []
        for i in range(0, len(segment_dict)):
            # i = 3

            series_offset = segment_dict[i]['Unnamed: 0'].__array__()[0]
            target_segment_dict = df_rearrange_header(segment_dict[i])

            swing_end_idx = np.where(np.diff(target_segment_dict['state_num']) == -2)[0]
            stance_end_or_swing_partition_idx = np.where(np.diff(target_segment_dict['state_num']) == 1)[0]
            stance_end_idx = stance_end_or_swing_partition_idx[target_segment_dict['state_num'][stance_end_or_swing_partition_idx] == 1]

            stance_start_raw = 0 # Initialize
            if target_segment_dict['state_num'][0] == 1: # If selected segment begin with stance
                stance_start_raw = np.append(np.array([0]), swing_end_idx + 1)
            else:
                stance_start_raw = swing_end_idx + 1

            # crop stride range: check if selected stride includes transition
            for s in range(0, len(stance_start_raw) - 1):
                # s = 6

                fsm_states_stance = target_segment_dict['state'][stance_start_raw[s] : stance_end_idx[s] + 1]
                stance_unique = np.unique(fsm_states_stance)

                # print(s, stance_unique)

                escount = 0
                for ss in range(0, len(stance_unique)):
                    if 'EarlyStance' in stance_unique[ss]:
                        escount += 1

                if escount > 1: # 1 step after LW<->SD transition
                    if 'LW_LateStance' in stance_unique:
                        stride_start_idx = stance_start_raw[s] + np.where(fsm_states_stance == 'LW_EarlyStance')[0][0] #SD->LW stride: stride begin from LWES
                        # print(s, stance_unique, 'SD2LW is included in this stance')
                    if 'SD_LateStance' in stance_unique:
                        stride_start_idx = stance_start_raw[s] + np.where(fsm_states_stance == 'SD_EarlyStance')[0][0] #LW->SD stride: stride begin from SDES
                        # print(s, stance_unique, 'LW2SD is included in this stance')

                else: # No LW<->SD transition
                    # print(s, stance_unique, 'Transition is not included in this stance')
                    stride_start_idx = stance_start_raw[s]

                # stride_end_idx = swing_end_idx[s] + delay
                stride_end_idx = stance_start_raw[s + 1] - 1 + delay
                if stride_end_idx > len(target_segment_dict):
                    stride_end_idx -= delay - 1

                stride_intervals.append([stride_start_idx + series_offset, stride_end_idx + series_offset])

        sep_dict = []
        for ints in range(0, len(stride_intervals)):

            if ints == 0:
                sep_df = raw_csv[stride_intervals[ints][0]: stride_intervals[ints][1] + 1]
            else:
                sep_df = raw_csv[stride_intervals[ints][0] - FWwinlen: stride_intervals[ints][1] + 1]

            sep_dict.append(sep_df)

        self.dict_result = sep_dict

class separation_stride_Stair_v2_forTRtest:
    def __init__(self, csvfilename, plot = False, slope = 0, FWwinlen = 125, delay = 15):

        # Reads CSV file format

        # ### Test
        # workspace_path = os.getcwd()
        # csvDir = workspace_path + '/processedCSV_StairDat_byBagfile/Offline/TF08/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(csvDir)
        # csvfilename = csvDir + datalist[6]
        # slope = 24
        #
        # FWwinlen = 125
        # plot = False
        # delay = 30
        # ### Test

        raw_csv = pd.read_csv(csvfilename)

        # Check data quality
        quality_array = raw_csv['quality'].__array__()
        seg_starts = []
        seg_ends = []
        for q in range(0, len(quality_array)):
            if quality_array[q] == 'G':
                if q == 0: # When the first element is good
                    seg_starts.append(q)
                elif quality_array[q - 1] != 'G':
                    seg_starts.append(q)
                elif q == len(quality_array) - 1:
                    seg_ends.append(q)

            else:
                if q > 0:
                    if quality_array[q - 1] == 'G':
                        seg_ends.append(q - 1)

        segment_dict = {}

        for i in range(0, len(seg_starts)):
            segment_dict[i] = raw_csv[seg_starts[i] : seg_ends[i] + 1]

        # Collect stride start/end index
        stride_intervals = []
        for i in range(0, len(segment_dict)):
            # i = 1

            series_offset = segment_dict[i]['Unnamed: 0'].__array__()[0]
            target_segment_dict = df_rearrange_header(segment_dict[i])

            stance_end_or_swing_partition_idx = np.where(np.diff(target_segment_dict['state_num']) == 1)[0]
            stance_end_idx = stance_end_or_swing_partition_idx[target_segment_dict['state_num'][stance_end_or_swing_partition_idx] == 1]
            swing_end_idx = np.where(np.diff(target_segment_dict['state_num']) == -2)[0]
            swing_start_idx = stance_end_idx + 1

            stance_start_raw = 0 # Initialize
            if target_segment_dict['state_num'][0] == 1: # If selected segment begin with stance
                stance_start_raw = np.append(np.array([0]), swing_end_idx + 1)
            else:
                stance_start_raw = swing_end_idx + 1

            # crop stride range: check if selected stride includes transition
            for s in range(0, len(stance_start_raw) - 1):
                # s = 6

                fsm_states_stance = target_segment_dict['state'][stance_start_raw[s] : stance_end_idx[s] + 1]
                fsm_states_swing = target_segment_dict['state'][swing_start_idx[s]: swing_end_idx[s] + 1]
                stance_unique = np.unique(fsm_states_stance)
                swing_unique = np.unique(fsm_states_swing)

                # print(s, stance_unique)

                escount = 0
                for ss in range(0, len(stance_unique)):
                    if 'EarlyStance' in stance_unique[ss]:
                        escount += 1

                if escount > 1: # 1 step after LW<->SD transition
                    if 'LW_LateStance' in stance_unique:
                        stride_start_idx = stance_start_raw[s] + np.where(fsm_states_stance == 'LW_EarlyStance')[0][0] #SD->LW stride: stride begin from LWES
                        # print(s, stance_unique, 'SD2LW is included in this stance')
                    if 'SD_LateStance' in stance_unique:
                        stride_start_idx = stance_start_raw[s] + np.where(fsm_states_stance == 'SD_EarlyStance')[0][0] #LW->SD stride: stride begin from SDES
                        # print(s, stance_unique, 'LW2SD is included in this stance')

                else: # No LW<->SD transition
                    # print(s, stance_unique, 'Transition is not included in this stance')
                    stride_start_idx = stance_start_raw[s]

                sfcount = 0
                for ss in range(0, len(swing_unique)):
                    if 'SwingFlexion' in swing_unique[ss]:
                        sfcount += 1

                if sfcount > 1: # 1 step after LW<->SA transition
                    if 'LW_SwingExtension' in swing_unique:
                        stride_start_idx = swing_start_idx[s] + np.where(fsm_states_swing == 'LW_SwingFlexion')[0][0] #SA->LW stride: stride begin from LWSF
                        # print(s, swing_unique, 'SA2LW is included in this swing')
                    if 'SA_SwingExtension' in swing_unique:
                        stride_start_idx = swing_start_idx[s] + np.where(fsm_states_swing == 'SA_SwingFlexion')[0][0] #LW->SA stride: stride begin from SASF
                        # print(s, swing_unique, 'LW2SA is included in this swing')

                else: # No LW<->SA transition
                    # print(s, stance_unique, 'Transition is not included in this stance')
                    stride_start_idx = stance_start_raw[s]

                # stride_end_idx = swing_end_idx[s] + delay
                stride_end_idx = stance_start_raw[s + 1] - 1 + delay
                if stride_end_idx > len(target_segment_dict):
                    stride_end_idx -= delay - 1

                stride_intervals.append([stride_start_idx + series_offset, stride_end_idx + series_offset])

        sep_dict = []
        for ints in range(0, len(stride_intervals)):

            if ints == 0:
                sep_df = raw_csv[stride_intervals[ints][0]: stride_intervals[ints][1] + 1]
            else:
                sep_df = raw_csv[stride_intervals[ints][0] - FWwinlen: stride_intervals[ints][1] + 1]

            sep_dict.append(sep_df)

        self.dict_result = sep_dict

class separation_stride_Ramp_OldSetting_v2:
    def __init__(self, csvfilename, plot = False, slope = 0, FWwinlen = 125, delay = 15):

        # Reads CSV file format

        # ### Test
        # workspace_path = os.getcwd()
        # csvDir = workspace_path + '/processedCSV_RampDat_byBagfile/RT/TF02/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(csvDir)
        # csvfilename = csvDir + datalist[0]
        # slope = 9.6
        #
        # FWwinlen = 125
        # plot = False
        # delay = 30
        # ### Test

        raw_csv = pd.read_csv(csvfilename)
        segment_dict = {}
        segment_dict[0] = raw_csv

        # Collect stride start/end index
        stride_intervals = []
        for i in range(0, len(segment_dict)):

            series_offset = segment_dict[i]['Unnamed: 0'].__array__()[0]
            target_segment_dict = df_rearrange_header(segment_dict[i])

            swing_end_idx = np.where(np.diff(target_segment_dict['state_num']) == -2)[0]
            stance_end_or_swing_partition_idx = np.where(np.diff(target_segment_dict['state_num']) == 1)[0]
            stance_end_idx = stance_end_or_swing_partition_idx[target_segment_dict['state_num'][stance_end_or_swing_partition_idx] == 1]

            stance_start_raw = 0 # Initialize
            if target_segment_dict['state_num'][0] == 1: # If selected segment begin with stance
                stance_start_raw = np.append(np.array([0]), swing_end_idx + 1)
            else:
                stance_start_raw = swing_end_idx + 1

            # crop stride range: check if selected stride includes transition
            for s in range(0, len(stance_start_raw) - 1):
                # s = 6

                fsm_states_stance = target_segment_dict['state'][stance_start_raw[s] : stance_end_idx[s] + 1]
                stance_unique = np.unique(fsm_states_stance)

                # print(s, stance_unique)

                escount = 0
                for ss in range(0, len(stance_unique)):
                    if 'EarlyStance' in stance_unique[ss]:
                        escount += 1

                if escount > 1: # 1 step after RA<->LW<->RD transition
                    if 'LW_LateStance' in stance_unique:
                        stride_start_idx = stance_start_raw[s] + np.where(fsm_states_stance == 'LW_EarlyStance')[0][0] #RA/RD->LW stride: stride begin from LWES
                        # print(s, stance_unique, 'SD2LW is included in this stance')
                    if 'RA_LateStance' in stance_unique:
                        stride_start_idx = stance_start_raw[s] + np.where(fsm_states_stance == 'RA_EarlyStance')[0][0] #LW->RA stride: stride begin from RAES
                        # print(s, stance_unique, 'LW2SD is included in this stance')
                    if 'RD_LateStance' in stance_unique:
                        stride_start_idx = stance_start_raw[s] + np.where(fsm_states_stance == 'RD_EarlyStance')[0][0] #LW->RD stride: stride begin from RDES
                        # print(s, stance_unique, 'LW2SD is included in this stance')

                else: # No LW<->SD transition
                    # print(s, stance_unique, 'Transition is not included in this stance')
                    stride_start_idx = stance_start_raw[s]

                # stride_end_idx = swing_end_idx[s] + delay
                stride_end_idx = stance_start_raw[s + 1] - 1 + delay
                if stride_end_idx > len(target_segment_dict):
                    stride_end_idx -= delay - 1

                stride_intervals.append([stride_start_idx + series_offset, stride_end_idx + series_offset])

        sep_dict = []
        for ints in range(0, len(stride_intervals)):

            if ints == 0:
                sep_df = raw_csv[stride_intervals[ints][0]: stride_intervals[ints][1] + 1]
            else:
                sep_df = raw_csv[stride_intervals[ints][0] - FWwinlen: stride_intervals[ints][1] + 1]

            sep_dict.append(sep_df)

        self.dict_result = sep_dict

class separation_stride_Ramp_OldSetting_v2_forTRtest:
    def __init__(self, csvfilename, plot = False, slope = 0, FWwinlen = 125, delay = 15):

        # Reads CSV file format

        # ### Test
        # workspace_path = os.getcwd()
        # csvDir = workspace_path + '/processedCSV_RampDat_byBagfile/RT/TF02/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(csvDir)
        # csvfilename = csvDir + datalist[0]
        # slope = 9.6
        #
        # FWwinlen = 125
        # plot = False
        # delay = 30
        # ### Test

        raw_csv = pd.read_csv(csvfilename)
        segment_dict = {}
        segment_dict[0] = raw_csv

        # Collect stride start/end index
        stride_intervals = []
        for i in range(0, len(segment_dict)):

            series_offset = segment_dict[i]['Unnamed: 0'].__array__()[0]
            target_segment_dict = df_rearrange_header(segment_dict[i])

            swing_end_idx = np.where(np.diff(target_segment_dict['state_num']) == -2)[0]
            stance_end_or_swing_partition_idx = np.where(np.diff(target_segment_dict['state_num']) == 1)[0]
            stance_end_idx = stance_end_or_swing_partition_idx[target_segment_dict['state_num'][stance_end_or_swing_partition_idx] == 1]

            stance_start_raw = 0 # Initialize
            if target_segment_dict['state_num'][0] == 1: # If selected segment begin with stance
                stance_start_raw = np.append(np.array([0]), swing_end_idx + 1)
            else:
                stance_start_raw = swing_end_idx + 1

            # crop stride range: check if selected stride includes transition
            for s in range(0, len(stance_start_raw) - 1):
                # s = 6

                fsm_states_stance = target_segment_dict['state'][stance_start_raw[s] : stance_end_idx[s] + 1]
                stance_unique = np.unique(fsm_states_stance)

                # print(s, stance_unique)

                escount = 0
                for ss in range(0, len(stance_unique)):
                    if 'EarlyStance' in stance_unique[ss]:
                        escount += 1

                if escount > 1: # 1 step after RA<->LW<->RD transition
                    if 'LW_LateStance' in stance_unique:
                        stride_start_idx = stance_start_raw[s] + np.where(fsm_states_stance == 'LW_EarlyStance')[0][0] #RA/RD->LW stride: stride begin from LWES
                        # print(s, stance_unique, 'SD2LW is included in this stance')
                    if 'RA_LateStance' in stance_unique:
                        stride_start_idx = stance_start_raw[s] + np.where(fsm_states_stance == 'RA_EarlyStance')[0][0] #LW->RA stride: stride begin from RAES
                        # print(s, stance_unique, 'LW2SD is included in this stance')
                    if 'RD_LateStance' in stance_unique:
                        stride_start_idx = stance_start_raw[s] + np.where(fsm_states_stance == 'RD_EarlyStance')[0][0] #LW->RD stride: stride begin from RDES
                        # print(s, stance_unique, 'LW2SD is included in this stance')

                else: # No LW<->SD transition
                    # print(s, stance_unique, 'Transition is not included in this stance')
                    stride_start_idx = stance_start_raw[s]

                # stride_end_idx = swing_end_idx[s] + delay
                stride_end_idx = stance_start_raw[s + 1] - 1 + delay
                if stride_end_idx > len(target_segment_dict):
                    stride_end_idx -= delay - 1

                stride_intervals.append([stride_start_idx + series_offset, stride_end_idx + series_offset])

        sep_dict_TR = []
        for ints in range(0, len(stride_intervals)):

            if ints == 0:
                sep_df = raw_csv[stride_intervals[ints][0]: stride_intervals[ints][1] + 1]
            else:
                sep_df = raw_csv[stride_intervals[ints][0] - FWwinlen: stride_intervals[ints][1] + 1]

            sep_df_aligned = df_rearrange_header(sep_df)

            if sep_df_aligned['state'][FWwinlen][:2] != sep_df_aligned['state'][FWwinlen - 1][:2]:
                sep_dict_TR.append(sep_df_aligned)

        self.dict_result = sep_dict_TR

class alignedBag_PILOT_viz_ASonly:

    ### Right now it only has Ascent, but later include transitions and descent
    ### And this function now is only for interpolation + plotting (vizualization)

    def __init__(self, bagfilename):

        # ### Test
        # workspace_path = os.getcwd()
        # BagDir = workspace_path + '/UniControl_PILOT/TFA2/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(BagDir)
        #
        # bagfilename = BagDir + datalist[1]
        # ### Test

        aligned_SDat, aligned_footIMU, aligned_thighIMU, aligned_FSM, cropped_SInfo = align2SInfo(bagfilename, oldIMU = True)

        ##### Align time segment to State
        FSM = FSMinfo(aligned_FSM)
        target_sensor_timeval = aligned_SDat['header']
        ground_truth = np.zeros([len(aligned_SDat)])
        transition_idx_total = np.zeros(1)  # For plotting

        ### Target to plot: knee theta, knee torque applied, forceZ, momentY, thighAngle, knee_k
        # Stance + Swing interpolation

        time_idx_RAES = FSM.time_idx_RAES  # HS (end of swing)
        time_val_RAES = FSM.time_val_RAES  # HS (end of swing)

        FSM_timeval = FSM.FSM_total['header']
        sensor_timeval = aligned_SDat['header']

        input_starttime = FSM.time_val_ES[0]
        input_startidx = closest_point(target_sensor_timeval, input_starttime)
        input_endtime = FSM.time_val_ES[-1]
        input_endidx = closest_point(target_sensor_timeval, input_endtime)
        ### Data before the first ES and after the last ES will be excluded in input shaping

        # Arrange Sensor Data (Include Thigh/Foot IMU orientation in SensorData)
        try:
            foot_qx = aligned_footIMU['orientation_x']
            foot_qy = aligned_footIMU['orientation_y']
            foot_qz = aligned_footIMU['orientation_z']
            foot_qw = aligned_footIMU['orientation_w']

            thigh_qx = aligned_thighIMU['orientation_x']
            thigh_qy = aligned_thighIMU['orientation_y']
            thigh_qz = aligned_thighIMU['orientation_z']
            thigh_qw = aligned_thighIMU['orientation_w']

            ### Compute orientations
            Zvector = np.array([0, 0, 1])

            foot_qmat = np.array([foot_qw, foot_qx, foot_qy, foot_qz]).T
            thigh_qmat = np.array([thigh_qw, thigh_qx, thigh_qy, thigh_qz]).T

            foot_Rmat = np.zeros([len(foot_qmat), 3, 3])
            thigh_Rmat = np.zeros([len(thigh_qmat), 3, 3])

            foot_rotZmat = np.zeros([len(foot_qmat), 3])
            thigh_rotZmat = np.zeros([len(thigh_qmat), 3])

            foot_anglemat = np.zeros([len(foot_qmat)])
            thigh_anglemat = np.zeros([len(thigh_qmat)])

            for i in range(0, len(foot_qmat)):
                foot_Rmat[i, :] = AHRS.q2Rot(foot_qmat[i])
                foot_rotZmat[i, :] = np.matmul(foot_Rmat[i, :], Zvector)

                vlen = np.sign(foot_rotZmat[i, 0]) * np.sqrt(foot_rotZmat[i, 0] ** 2 + foot_rotZmat[i, 1] ** 2)
                hlen = foot_rotZmat[i, 2]

                foot_anglemat[i] = 90 - math.atan2(hlen, vlen) * 180 / np.pi

                # Denoise for realtime
                if i > 0:
                    if foot_anglemat[i] - foot_anglemat[i-1] > 25:
                        foot_anglemat[i] = foot_anglemat[i-1]

            for i in range(0, len(thigh_qmat)):
                thigh_Rmat[i, :] = AHRS.q2Rot(thigh_qmat[i])
                thigh_rotZmat[i, :] = np.matmul(thigh_Rmat[i, :], Zvector)

                hlen = np.sign(thigh_rotZmat[i, 0]) * np.sqrt(thigh_rotZmat[i, 0] ** 2 + thigh_rotZmat[i, 1] ** 2)
                vlen = thigh_rotZmat[i, 2]

                thigh_anglemat[i] = 90 - math.atan2(hlen, vlen) * 180 / np.pi

                # Denoise for realtime
                if i > 0:
                    if thigh_anglemat[i] - thigh_anglemat[i-1] < -15:
                        thigh_anglemat[i] = thigh_anglemat[i-1]

            offset_end = closest_point(target_sensor_timeval, input_starttime)
            foot_angle_offset = np.mean(foot_anglemat[:offset_end])
            thigh_angle_offset = np.mean(thigh_anglemat[:offset_end])

            foot_anglemat -= foot_angle_offset
            thigh_anglemat -= thigh_angle_offset

            aligned_SDat['thigh_orientation'] = thigh_anglemat
            aligned_SDat['foot_orientation'] = foot_anglemat

        except:
            pass

        # print(np.isnan(np.sum(thigh_anglemat)))

        x_new_stance = np.linspace(start = 0, stop = 299, num = 300)
        x_new_Estance = np.linspace(start=0, stop=149, num=150)
        x_new_Lstance = np.linspace(start=0, stop=149, num=150)
        x_new_swing = np.linspace(start=0, stop=199, num=200)

        knee_theta_intp = []
        knee_torque_applied_intp = []
        ankle_theta_intp = []
        ankle_torque_applied_intp = []
        knee_k_intp = []
        knee_b_intp = []
        ankle_k_intp = []
        ankle_b_intp = []
        forceZ_intp = []
        momentY_intp = []
        thigh_orientation_intp = []

        for i in range(0, len(time_idx_RAES) - 1):
            # i = 1
            time_val_RAES = FSM_timeval[time_idx_RAES[i]]
            time_val_RALS = FSM_timeval[time_idx_RAES[i] + 1]
            time_val_RASF = FSM_timeval[time_idx_RAES[i] + 2]
            time_val_RAES_next = FSM_timeval[time_idx_RAES[i + 1]]

            sensor_RAES_idx = np.where(sensor_timeval == time_val_RAES)[0][0]
            sensor_RALS_idx = np.where(sensor_timeval == time_val_RALS)[0][0]
            sensor_RASF_idx = np.where(sensor_timeval == time_val_RASF)[0][0]
            sensor_RAES_next_idx = np.where(sensor_timeval == time_val_RAES_next)[0][0]

            SDat_stance = aligned_SDat[sensor_RAES_idx:sensor_RASF_idx]
            SDat_Estance = aligned_SDat[sensor_RAES_idx:sensor_RALS_idx]
            SDat_Lstance = aligned_SDat[sensor_RALS_idx:sensor_RASF_idx]
            SDat_swing = aligned_SDat[sensor_RASF_idx:sensor_RAES_next_idx]

            SInfo_stance = cropped_SInfo[sensor_RAES_idx:sensor_RASF_idx]
            SInfo_Estance = cropped_SInfo[sensor_RAES_idx:sensor_RALS_idx]
            SInfo_Lstance = cropped_SInfo[sensor_RALS_idx:sensor_RASF_idx]
            SInfo_swing = cropped_SInfo[sensor_RASF_idx:sensor_RAES_next_idx]

            # Interpolation: 150 Estance + 150 Lstance + 200 swing
            knee_theta_Estance_original = SDat_Estance['knee_theta'].__array__()
            ankle_theta_Estance_original = SDat_Estance['ankle_theta'].__array__()
            forceZ_Estance_original = SDat_Estance['forceZ'].__array__()
            momentY_Estance_original = SDat_Estance['momentY'].__array__()
            thighorient_Estance_original = SDat_Estance['thigh_orientation'].__array__()
            knee_torque_applied_Estance_original = SInfo_Estance['knee_torque_applied'].__array__()
            ankle_torque_applied_Estance_original = SInfo_Estance['ankle_torque_applied'].__array__()
            knee_k_Estance_original = SInfo_Estance['knee_k'].__array__()
            ankle_k_Estance_original = SInfo_Estance['ankle_k'].__array__()
            knee_b_Estance_original = SInfo_Estance['knee_b'].__array__()
            ankle_b_Estance_original = SInfo_Estance['ankle_b'].__array__()

            Estance_total_original = np.array([knee_theta_Estance_original,
                                              ankle_theta_Estance_original,
                                              forceZ_Estance_original,
                                              momentY_Estance_original,
                                              thighorient_Estance_original,
                                              knee_torque_applied_Estance_original,
                                              ankle_torque_applied_Estance_original,
                                              knee_k_Estance_original,
                                              ankle_k_Estance_original,
                                              knee_b_Estance_original,
                                              ankle_b_Estance_original])

            knee_theta_Lstance_original = SDat_Lstance['knee_theta'].__array__()
            ankle_theta_Lstance_original = SDat_Lstance['ankle_theta'].__array__()
            forceZ_Lstance_original = SDat_Lstance['forceZ'].__array__()
            momentY_Lstance_original = SDat_Lstance['momentY'].__array__()
            thighorient_Lstance_original = SDat_Lstance['thigh_orientation'].__array__()
            knee_torque_applied_Lstance_original = SInfo_Lstance['knee_torque_applied'].__array__()
            ankle_torque_applied_Lstance_original = SInfo_Lstance['ankle_torque_applied'].__array__()
            knee_k_Lstance_original = SInfo_Lstance['knee_k'].__array__()
            ankle_k_Lstance_original = SInfo_Lstance['ankle_k'].__array__()
            knee_b_Lstance_original = SInfo_Lstance['knee_b'].__array__()
            ankle_b_Lstance_original = SInfo_Lstance['ankle_b'].__array__()

            Lstance_total_original = np.array([knee_theta_Lstance_original,
                                              ankle_theta_Lstance_original,
                                              forceZ_Lstance_original,
                                              momentY_Lstance_original,
                                              thighorient_Lstance_original,
                                              knee_torque_applied_Lstance_original,
                                              ankle_torque_applied_Lstance_original,
                                              knee_k_Lstance_original,
                                              ankle_k_Lstance_original,
                                              knee_b_Lstance_original,
                                              ankle_b_Lstance_original])



            knee_theta_stance_original = SDat_stance['knee_theta'].__array__()
            ankle_theta_stance_original = SDat_stance['ankle_theta'].__array__()
            forceZ_stance_original = SDat_stance['forceZ'].__array__()
            momentY_stance_original = SDat_stance['momentY'].__array__()
            thighorient_stance_original = SDat_stance['thigh_orientation'].__array__()
            knee_torque_applied_stance_original = SInfo_stance['knee_torque_applied'].__array__()
            ankle_torque_applied_stance_original = SInfo_stance['ankle_torque_applied'].__array__()
            knee_k_stance_original = SInfo_stance['knee_k'].__array__()
            ankle_k_stance_original = SInfo_stance['ankle_k'].__array__()
            knee_b_stance_original = SInfo_stance['knee_b'].__array__()
            ankle_b_stance_original = SInfo_stance['ankle_b'].__array__()

            stance_total_original = np.array([knee_theta_stance_original,
                                              ankle_theta_stance_original,
                                              forceZ_stance_original,
                                              momentY_stance_original,
                                              thighorient_stance_original,
                                              knee_torque_applied_stance_original,
                                              ankle_torque_applied_stance_original,
                                              knee_k_stance_original,
                                              ankle_k_stance_original,
                                              knee_b_stance_original,
                                              ankle_b_stance_original])

            knee_theta_swing_original = SDat_swing['knee_theta'].__array__()
            ankle_theta_swing_original = SDat_swing['ankle_theta'].__array__()
            forceZ_swing_original = SDat_swing['forceZ'].__array__()
            momentY_swing_original = SDat_swing['momentY'].__array__()
            thighorient_swing_original = SDat_swing['thigh_orientation'].__array__()
            knee_torque_applied_swing_original = SInfo_swing['knee_torque_applied'].__array__()
            ankle_torque_applied_swing_original = SInfo_swing['ankle_torque_applied'].__array__()
            knee_k_swing_original = SInfo_swing['knee_k'].__array__()
            ankle_k_swing_original = SInfo_swing['ankle_k'].__array__()
            knee_b_swing_original = SInfo_swing['knee_b'].__array__()
            ankle_b_swing_original = SInfo_swing['ankle_b'].__array__()

            swing_total_original = np.array([knee_theta_swing_original,
                                              ankle_theta_swing_original,
                                              forceZ_swing_original,
                                              momentY_swing_original,
                                              thighorient_swing_original,
                                              knee_torque_applied_swing_original,
                                              ankle_torque_applied_swing_original,
                                              knee_k_swing_original,
                                              ankle_k_swing_original,
                                              knee_b_swing_original,
                                              ankle_b_swing_original])


            x_original_stance = np.linspace(start=x_new_stance[0],
                                            stop=x_new_stance[-1],
                                            num=stance_total_original.shape[1])

            x_original_Estance = np.linspace(start=x_new_Estance[0],
                                            stop=x_new_Estance[-1],
                                            num=Estance_total_original.shape[1])

            x_original_Lstance = np.linspace(start=x_new_Lstance[0],
                                            stop=x_new_Lstance[-1],
                                            num=Lstance_total_original.shape[1])

            x_original_swing = np.linspace(start=x_new_swing[0],
                                            stop=x_new_swing[-1],
                                            num=swing_total_original.shape[1])

            target_aligned_entireGait_total = []
            for ii in range(0, len(Estance_total_original)):
                # ii=0

                target_to_align_Estance = Estance_total_original[ii,:]
                f_target_Estance = itpd(x_original_Estance, target_to_align_Estance)
                target_aligned_Estance = f_target_Estance(x_new_Estance)

                target_to_align_Lstance = Lstance_total_original[ii,:]
                f_target_Lstance = itpd(x_original_Lstance, target_to_align_Lstance)
                target_aligned_Lstance = f_target_Lstance(x_new_Lstance)

                target_to_align_swing = swing_total_original[ii, :]
                f_target_swing = itpd(x_original_swing, target_to_align_swing)
                target_aligned_swing = f_target_swing(x_new_swing)

                target_aligned_entireGait = np.concatenate([target_aligned_Estance,
                                                            target_aligned_Lstance,
                                                            target_aligned_swing], axis = 0)

                target_aligned_entireGait_total.append(target_aligned_entireGait)

            knee_theta_intp.append(target_aligned_entireGait_total[0])
            ankle_theta_intp.append(target_aligned_entireGait_total[1])
            forceZ_intp.append(target_aligned_entireGait_total[2])
            momentY_intp.append(target_aligned_entireGait_total[3])
            thigh_orientation_intp.append(target_aligned_entireGait_total[4])
            knee_torque_applied_intp.append(target_aligned_entireGait_total[5])
            ankle_torque_applied_intp.append(target_aligned_entireGait_total[6])
            knee_k_intp.append(target_aligned_entireGait_total[7])
            ankle_k_intp.append(target_aligned_entireGait_total[8])
            knee_b_intp.append(target_aligned_entireGait_total[9])
            ankle_b_intp.append(target_aligned_entireGait_total[10])

        self.knee_theta = knee_theta_intp
        self.ankle_theta = ankle_theta_intp
        self.forceZ = forceZ_intp
        self.momentY = momentY_intp
        self.thigh_orientation = thigh_orientation_intp
        self.knee_torque_applied = knee_torque_applied_intp
        self.ankle_torque_applied = ankle_torque_applied_intp
        self.knee_k = knee_k_intp
        self.ankle_k = ankle_k_intp
        self.knee_b = knee_b_intp
        self.ankle_b = ankle_b_intp

class UniCont_PILOT_viz_ASonly:

    ### Right now it only has Ascent, but later include transitions and descent
    ### And this function now is only for interpolation + plotting (vizualization)

    def __init__(self, BagDir):

        # ### Test
        # workspace_path = os.getcwd()
        # BagDir = workspace_path + '/UniControl_PILOT/TFHK/'  # This gives us a str with the path and the file we are using
        # ### Test

        datalist = os.listdir(BagDir)
        KeyInfoDir = BagDir + 'KEY.csv'
        KeyInfo = pd.read_csv(KeyInfoDir)
        data_dict_total = {}

        for datFile in datalist:

            # datFile = '11.bag'

            if '.bag' not in datFile:
                continue

            datFileNo = datFile.rsplit('.bag')[0]
            File_idx = np.where(KeyInfo.fileName == int(datFileNo))[0][0]
            slope = KeyInfo.slope[File_idx]

            bagfilename = BagDir + datFile
            aligned_data = alignedBag_PILOT_viz_ASonly(bagfilename)

            if slope not in data_dict_total.keys():
                data_dict_total[slope] = {}
                data_dict_total[slope]['knee_theta'] = []
                data_dict_total[slope]['knee_torque_applied'] = []
                data_dict_total[slope]['ankle_theta'] = []
                data_dict_total[slope]['ankle_torque_applied'] = []
                data_dict_total[slope]['forceZ'] = []
                data_dict_total[slope]['momentY'] = []
                data_dict_total[slope]['thigh_orientation'] = []
                data_dict_total[slope]['knee_k'] = []
                data_dict_total[slope]['knee_b'] = []
                data_dict_total[slope]['ankle_k'] = []
                data_dict_total[slope]['ankle_b'] = []

            for j in range(0, len(aligned_data.forceZ)):
                data_dict_total[slope]['knee_theta'].append(aligned_data.knee_theta[j])
                data_dict_total[slope]['knee_torque_applied'].append(aligned_data.knee_torque_applied[j])
                data_dict_total[slope]['ankle_theta'].append(aligned_data.ankle_theta[j])
                data_dict_total[slope]['ankle_torque_applied'].append(aligned_data.ankle_torque_applied[j])
                data_dict_total[slope]['forceZ'].append(aligned_data.forceZ[j])
                data_dict_total[slope]['momentY'].append(aligned_data.momentY[j])
                data_dict_total[slope]['thigh_orientation'].append(aligned_data.thigh_orientation[j])
                data_dict_total[slope]['knee_k'].append(aligned_data.knee_k[j])
                data_dict_total[slope]['knee_b'].append(aligned_data.knee_b[j])
                data_dict_total[slope]['ankle_k'].append(aligned_data.ankle_k[j])
                data_dict_total[slope]['ankle_b'].append(aligned_data.ankle_b[j])

        data_dict_average = {}
        data_dict_average['knee_theta'] = {}
        data_dict_average['knee_torque_applied'] = {}
        data_dict_average['ankle_theta'] = {}
        data_dict_average['ankle_torque_applied'] = {}
        data_dict_average['forceZ'] = {}
        data_dict_average['momentY'] = {}
        data_dict_average['thigh_orientation'] = {}
        data_dict_average['knee_k'] = {}
        data_dict_average['knee_b'] = {}
        data_dict_average['ankle_k'] = {}
        data_dict_average['ankle_b'] = {}

        for dtype in data_dict_average.keys():
            # dtype = 'knee_theta'

            for slope in data_dict_total.keys():
                # slope = 5.2

                target_data = np.array(data_dict_total[slope][dtype])

                average_curve = np.zeros([target_data.shape[1]])

                for l in range(0, target_data.shape[1]):
                    average_curve[l] = np.mean(target_data[:,l])

                data_dict_average[dtype][slope] = average_curve

        self.average_curve = data_dict_average

class inputProcess_FWNN_noDelay(separation_SSTR_Stair):
    def __init__(self, bagfilename, slope, startMSP=0, endMSP=1, plot=False, FWwinlen = 125, winStride = 1, sensors = None):
        super().__init__(bagfilename, slope, startMSP, endMSP, plot, FWwinlen)

        testload = self.__dict__

        # ### Test
        # workspace_path = os.getcwd()
        # BagDir = workspace_path + '/Stair_Data_Raw/TF23/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(BagDir)
        #
        # bagfilename = BagDir + datalist[5]
        # slope = 19.6
        # startMSP = 0.3
        # endMSP = 0.5
        # FWwinlen = 125
        # winStride = 1
        #
        # sensors = ['forceX',
        #            'forceY',
        #            'forceZ',
        #            'momentX',
        #            'momentY',
        #            'momentZ',
        #            'thigh_orientation',
        #            'thigh_accelX',
        #            'thigh_accelY',
        #            'thigh_accelZ',
        #            'thigh_gyroX',
        #            'thigh_gyroY',
        #            'thigh_gyroZ',
        #            'foot_accelX',
        #            'foot_accelY',
        #            'foot_accelZ',
        #            'foot_gyroX',
        #            'foot_gyroY',
        #            'foot_gyroZ',
        #            'shank_accelX',
        #            'shank_accelY',
        #            'shank_accelZ',
        #            'shank_gyroX',
        #            'shank_gyroY',
        #            'shank_gyroZ',
        #            'knee_theta',
        #            'ankle_theta',
        #            'knee_thetadot',
        #            'ankle_thetadot',
        #            ]
        #
        # testload = separation_SSTR(bagfilename, slope, startMSP=startMSP, endMSP=endMSP, plot=False)
        # testload = testload.__dict__
        # ### Test

        sensor_data = testload['separated_dict_MS']['sensor_data']
        ground_truth = testload['separated_dict_MS']['ground_truth']

        sensor_data_FSM = testload['separated_dict_FSM']['sensor_data']
        ground_truth_FSM = testload['separated_dict_FSM']['ground_truth']

        X_train = {}
        Y_train = {}
        X_test = {}
        Y_test = {}

        X_train_FSM = {}
        Y_train_FSM = {}
        X_test_FSM = {}
        Y_test_FSM = {}

        for strideType in sensor_data.keys():

            # Initiate
            if strideType not in X_train.keys():
                X_train[strideType] = {}
                Y_train[strideType] = {}
                X_test[strideType] = {}
                Y_test[strideType] = {}

            target_stride_ground_truth = ground_truth[strideType]
            target_stride_sensor_data = sensor_data[strideType]

            for mode in target_stride_ground_truth.keys():

                target_mode_ground_truth = target_stride_ground_truth[mode]
                target_mode_sensor_data = target_stride_sensor_data[mode]

                test_n = int(np.round(len(target_mode_ground_truth) * 0.2))
                train_n = int(len(target_mode_ground_truth) - test_n)

                for l in range(0, train_n):
                    target_ground_truth = target_mode_ground_truth[l]
                    target_sensor_data = target_mode_sensor_data[l]

                    ground_truth_windowed = fncs.rawsignal_windowed(input=target_ground_truth,
                                                                    window_length=FWwinlen,
                                                                    overlap_length=FWwinlen - winStride)

                    ### Selecting the most forward one (Forward estimator)
                    ground_truth_windowed_forward = ground_truth_windowed[:,-1]

                    ### Sensor selection
                    if sensors == None:
                        selected_sensor_data = np.array(target_sensor_data)
                        selected_sensor_data = selected_sensor_data[:, 1:]  ### For sensor channel selection

                    else:
                        sensordata_arranged = {}
                        for sensor in sensors:
                            sensordata_arranged[sensor] = target_sensor_data[sensor]

                        sensordata_arranged = pd.DataFrame.from_dict(sensordata_arranged)
                        selected_sensor_data = sensordata_arranged.copy()
                        selected_sensor_data = np.array(selected_sensor_data)


                    firstchannel_windowed = fncs.rawsignal_windowed(input=selected_sensor_data[:, 0],
                                                                    window_length=FWwinlen,
                                                                    overlap_length=FWwinlen - winStride)

                    sensordata_windowed = np.zeros([firstchannel_windowed.shape[0],
                                                      firstchannel_windowed.shape[1],
                                                      selected_sensor_data.shape[1]])

                    sensordata_windowed[:, :, 0] = firstchannel_windowed  ### train_X or test_X

                    for s in range(1, selected_sensor_data.shape[1]):  ### First one is already done
                        target_channel_to_window = selected_sensor_data[:, s]

                        windowed_target = fncs.rawsignal_windowed(input=target_channel_to_window,
                                                                  window_length=FWwinlen,
                                                                  overlap_length=FWwinlen - winStride)

                        sensordata_windowed[:, :, s] = windowed_target

                    # if (mode == 'LW'):
                    #     print(mode, ' / ', sensordata_windowed.shape[0])

                    ### Initiate
                    if mode not in X_train[strideType].keys():
                        X_train[strideType][mode] = sensordata_windowed
                        Y_train[strideType][mode] = ground_truth_windowed_forward

                    else:
                        X_train[strideType][mode] = np.append(X_train[strideType][mode],
                                                              sensordata_windowed, axis = 0)
                        Y_train[strideType][mode] = np.append(Y_train[strideType][mode],
                                                              ground_truth_windowed_forward, axis = 0)

                for l in range(train_n, train_n + test_n):
                    target_ground_truth = target_mode_ground_truth[l]
                    target_sensor_data = target_mode_sensor_data[l]

                    ground_truth_windowed = fncs.rawsignal_windowed(input=target_ground_truth,
                                                                    window_length=FWwinlen,
                                                                    overlap_length=FWwinlen - winStride)

                    ### Selecting the most forward one (Forward estimator)
                    ground_truth_windowed_forward = ground_truth_windowed[:, -1]

                    ### Sensor selection
                    if sensors == None:
                        selected_sensor_data = np.array(target_sensor_data)
                        selected_sensor_data = selected_sensor_data[:, 1:]  ### For sensor channel selection

                    else:
                        sensordata_arranged = {}
                        for sensor in sensors:
                            sensordata_arranged[sensor] = target_sensor_data[sensor]

                        sensordata_arranged = pd.DataFrame.from_dict(sensordata_arranged)
                        selected_sensor_data = sensordata_arranged.copy()
                        selected_sensor_data = np.array(selected_sensor_data)

                    firstchannel_windowed = fncs.rawsignal_windowed(input=selected_sensor_data[:, 0],
                                                                    window_length=FWwinlen,
                                                                    overlap_length=FWwinlen - winStride)

                    sensordata_windowed = np.zeros([firstchannel_windowed.shape[0],
                                                    firstchannel_windowed.shape[1],
                                                    selected_sensor_data.shape[1]])

                    sensordata_windowed[:, :, 0] = firstchannel_windowed  ### train_X or test_X

                    for s in range(1, selected_sensor_data.shape[1]):  ### First one is already done
                        target_channel_to_window = selected_sensor_data[:, s]

                        windowed_target = fncs.rawsignal_windowed(input=target_channel_to_window,
                                                                  window_length=FWwinlen,
                                                                  overlap_length=FWwinlen - winStride)

                        sensordata_windowed[:, :, s] = windowed_target

                    ### Initiate
                    if mode not in X_test[strideType].keys():
                        X_test[strideType][mode] = sensordata_windowed
                        Y_test[strideType][mode] = ground_truth_windowed_forward

                    else:
                        X_test[strideType][mode] = np.append(X_test[strideType][mode],
                                                             sensordata_windowed, axis=0)
                        Y_test[strideType][mode] = np.append(Y_test[strideType][mode],
                                                             ground_truth_windowed_forward, axis=0)


        for strideType in sensor_data_FSM.keys():

            # Initiate
            if strideType not in X_train_FSM.keys():
                X_train_FSM[strideType] = {}
                Y_train_FSM[strideType] = {}
                X_test_FSM[strideType] = {}
                Y_test_FSM[strideType] = {}

            target_stride_ground_truth = ground_truth_FSM[strideType]
            target_stride_sensor_data = sensor_data_FSM[strideType]

            if strideType == 'TR':

                for mode in target_stride_ground_truth.keys():

                    # Initiate
                    if mode not in X_train_FSM[strideType].keys():
                        X_train_FSM[strideType][mode] = []
                        Y_train_FSM[strideType][mode] = []
                        X_test_FSM[strideType][mode] = []
                        Y_test_FSM[strideType][mode] = []

                    target_mode_ground_truth = target_stride_ground_truth[mode]
                    target_mode_sensor_data = target_stride_sensor_data[mode]

                    test_n = int(np.round(len(target_mode_ground_truth) * 0.2))
                    train_n = int(len(target_mode_ground_truth) - test_n)

                    ## Training Data
                    for l in range(0, train_n):
                        target_ground_truth = target_mode_ground_truth[l]
                        target_sensor_data = target_mode_sensor_data[l]

                        ground_truth_windowed = fncs.rawsignal_windowed(input=target_ground_truth,
                                                                        window_length=FWwinlen,
                                                                        overlap_length=FWwinlen - winStride)

                        ### Selecting the most forward one (Forward estimator)
                        ground_truth_windowed_forward = ground_truth_windowed[:,-1]

                        ### Sensor selection
                        if sensors == None:
                            selected_sensor_data = np.array(target_sensor_data)
                            selected_sensor_data = selected_sensor_data[:, 1:]  ### For sensor channel selection

                        else:
                            sensordata_arranged = {}
                            for sensor in sensors:
                                sensordata_arranged[sensor] = target_sensor_data[sensor]

                            sensordata_arranged = pd.DataFrame.from_dict(sensordata_arranged)
                            selected_sensor_data = sensordata_arranged.copy()
                            selected_sensor_data = np.array(selected_sensor_data)

                        firstchannel_windowed = fncs.rawsignal_windowed(input=selected_sensor_data[:, 0],
                                                                        window_length=FWwinlen,
                                                                        overlap_length=FWwinlen - winStride)

                        sensordata_windowed = np.zeros([firstchannel_windowed.shape[0],
                                                          firstchannel_windowed.shape[1],
                                                          selected_sensor_data.shape[1]])

                        sensordata_windowed[:, :, 0] = firstchannel_windowed  ### train_X or test_X

                        for s in range(1, selected_sensor_data.shape[1]):  ### First one is already done
                            target_channel_to_window = selected_sensor_data[:, s]

                            windowed_target = fncs.rawsignal_windowed(input=target_channel_to_window,
                                                                      window_length=FWwinlen,
                                                                      overlap_length=FWwinlen - winStride)

                            sensordata_windowed[:, :, s] = windowed_target


                        X_train_FSM[strideType][mode].append(sensordata_windowed)
                        Y_train_FSM[strideType][mode].append(ground_truth_windowed_forward)


                    ## Test Data
                    for l in range(train_n, train_n + test_n):
                        target_ground_truth = target_mode_ground_truth[l]
                        target_sensor_data = target_mode_sensor_data[l]

                        ground_truth_windowed = fncs.rawsignal_windowed(input=target_ground_truth,
                                                                        window_length=FWwinlen,
                                                                        overlap_length=FWwinlen - winStride)

                        ### Selecting the most forward one (Forward estimator)
                        ground_truth_windowed_forward = ground_truth_windowed[:, -1]

                        ### Sensor selection
                        if sensors == None:
                            selected_sensor_data = np.array(target_sensor_data)
                            selected_sensor_data = selected_sensor_data[:, 1:]  ### For sensor channel selection

                        else:
                            sensordata_arranged = {}
                            for sensor in sensors:
                                sensordata_arranged[sensor] = target_sensor_data[sensor]

                            sensordata_arranged = pd.DataFrame.from_dict(sensordata_arranged)
                            selected_sensor_data = sensordata_arranged.copy()
                            selected_sensor_data = np.array(selected_sensor_data)

                        firstchannel_windowed = fncs.rawsignal_windowed(input=selected_sensor_data[:, 0],
                                                                        window_length=FWwinlen,
                                                                        overlap_length=FWwinlen - winStride)

                        sensordata_windowed = np.zeros([firstchannel_windowed.shape[0],
                                                        firstchannel_windowed.shape[1],
                                                        selected_sensor_data.shape[1]])

                        sensordata_windowed[:, :, 0] = firstchannel_windowed  ### train_X or test_X

                        for s in range(1, selected_sensor_data.shape[1]):  ### First one is already done
                            target_channel_to_window = selected_sensor_data[:, s]

                            windowed_target = fncs.rawsignal_windowed(input=target_channel_to_window,
                                                                      window_length=FWwinlen,
                                                                      overlap_length=FWwinlen - winStride)

                            sensordata_windowed[:, :, s] = windowed_target


                        X_test_FSM[strideType][mode].append(sensordata_windowed)
                        Y_test_FSM[strideType][mode].append(ground_truth_windowed_forward)

            else:
                for mode in target_stride_ground_truth.keys():
                    target_mode_ground_truth = target_stride_ground_truth[mode]
                    target_mode_sensor_data = target_stride_sensor_data[mode]

                    X_train_FSM[strideType][mode] = {}
                    Y_train_FSM[strideType][mode] = {}
                    X_test_FSM[strideType][mode] = {}
                    Y_test_FSM[strideType][mode] = {}

                    for datType in target_mode_ground_truth.keys():

                        if datType not in X_train_FSM[strideType][mode].keys():
                            X_train_FSM[strideType][mode][datType] = []
                            Y_train_FSM[strideType][mode][datType] = []
                            X_test_FSM[strideType][mode][datType] = []
                            Y_test_FSM[strideType][mode][datType] = []

                        target_datType_ground_truth = target_mode_ground_truth[datType]
                        target_datType_sensor_data = target_mode_sensor_data[datType]

                        test_n = int(np.round(len(target_datType_ground_truth) * 0.2))
                        train_n = int(len(target_datType_ground_truth) - test_n)

                        ## Training Data
                        for l in range(0, train_n):
                            target_ground_truth = target_datType_ground_truth[l]
                            target_sensor_data = target_datType_sensor_data[l]

                            ground_truth_windowed = fncs.rawsignal_windowed(input=target_ground_truth,
                                                                            window_length=FWwinlen,
                                                                            overlap_length=FWwinlen - winStride)

                            ### Selecting the most forward one (Forward estimator)
                            ground_truth_windowed_forward = ground_truth_windowed[:, -1]

                            ### Sensor selection
                            if sensors == None:
                                selected_sensor_data = np.array(target_sensor_data)
                                selected_sensor_data = selected_sensor_data[:, 1:]  ### For sensor channel selection

                            else:
                                sensordata_arranged = {}
                                for sensor in sensors:
                                    sensordata_arranged[sensor] = target_sensor_data[sensor]

                                sensordata_arranged = pd.DataFrame.from_dict(sensordata_arranged)
                                selected_sensor_data = sensordata_arranged.copy()
                                selected_sensor_data = np.array(selected_sensor_data)

                            firstchannel_windowed = fncs.rawsignal_windowed(input=selected_sensor_data[:, 0],
                                                                            window_length=FWwinlen,
                                                                            overlap_length=FWwinlen - winStride)

                            sensordata_windowed = np.zeros([firstchannel_windowed.shape[0],
                                                            firstchannel_windowed.shape[1],
                                                            selected_sensor_data.shape[1]])

                            sensordata_windowed[:, :, 0] = firstchannel_windowed  ### train_X or test_X

                            for s in range(1, selected_sensor_data.shape[1]):  ### First one is already done
                                target_channel_to_window = selected_sensor_data[:, s]

                                windowed_target = fncs.rawsignal_windowed(input=target_channel_to_window,
                                                                          window_length=FWwinlen,
                                                                          overlap_length=FWwinlen - winStride)

                                sensordata_windowed[:, :, s] = windowed_target

                            # if (mode == 'LW') and (datType == 'Total'):
                            #     print(mode, datType, ' / ', sensordata_windowed.shape[0])

                            X_train_FSM[strideType][mode][datType].append(sensordata_windowed)
                            Y_train_FSM[strideType][mode][datType].append(ground_truth_windowed_forward)

                        ## Test Data
                        for l in range(train_n, train_n + test_n):
                            target_ground_truth = target_datType_ground_truth[l]
                            target_sensor_data = target_datType_sensor_data[l]

                            ground_truth_windowed = fncs.rawsignal_windowed(input=target_ground_truth,
                                                                            window_length=FWwinlen,
                                                                            overlap_length=FWwinlen - winStride)

                            ### Selecting the most forward one (Forward estimator)
                            ground_truth_windowed_forward = ground_truth_windowed[:, -1]

                            ### Sensor selection
                            if sensors == None:
                                selected_sensor_data = np.array(target_sensor_data)
                                selected_sensor_data = selected_sensor_data[:, 1:]  ### For sensor channel selection

                            else:
                                sensordata_arranged = {}
                                for sensor in sensors:
                                    sensordata_arranged[sensor] = target_sensor_data[sensor]

                                sensordata_arranged = pd.DataFrame.from_dict(sensordata_arranged)
                                selected_sensor_data = sensordata_arranged.copy()
                                selected_sensor_data = np.array(selected_sensor_data)

                            firstchannel_windowed = fncs.rawsignal_windowed(input=selected_sensor_data[:, 0],
                                                                            window_length=FWwinlen,
                                                                            overlap_length=FWwinlen - winStride)

                            sensordata_windowed = np.zeros([firstchannel_windowed.shape[0],
                                                            firstchannel_windowed.shape[1],
                                                            selected_sensor_data.shape[1]])

                            sensordata_windowed[:, :, 0] = firstchannel_windowed  ### train_X or test_X

                            for s in range(1, selected_sensor_data.shape[1]):  ### First one is already done
                                target_channel_to_window = selected_sensor_data[:, s]

                                windowed_target = fncs.rawsignal_windowed(input=target_channel_to_window,
                                                                          window_length=FWwinlen,
                                                                          overlap_length=FWwinlen - winStride)

                                sensordata_windowed[:, :, s] = windowed_target

                            X_test_FSM[strideType][mode][datType].append(sensordata_windowed)
                            Y_test_FSM[strideType][mode][datType].append(ground_truth_windowed_forward)



        # Combined Input/Output
        firstidx = 0
        for strideType in X_train.keys():

            target_stride_X_train = X_train[strideType]
            target_stride_Y_train = Y_train[strideType]
            target_stride_X_test = X_test[strideType]
            target_stride_Y_test = Y_test[strideType]

            for mode in target_stride_X_train.keys():
                target_mode_X_train = target_stride_X_train[mode]
                target_mode_Y_train = target_stride_Y_train[mode]
                target_mode_X_test = target_stride_X_test[mode]
                target_mode_Y_test = target_stride_Y_test[mode]

                if firstidx == 0:
                    X_train_total = target_mode_X_train
                    Y_train_total = target_mode_Y_train
                    X_test_total = target_mode_X_test
                    Y_test_total = target_mode_Y_test

                    firstidx += 1

                else:
                    X_train_total = np.append(X_train_total, target_mode_X_train, axis = 0)
                    Y_train_total = np.append(Y_train_total, target_mode_Y_train, axis = 0)
                    X_test_total = np.append(X_test_total, target_mode_X_test, axis = 0)
                    Y_test_total = np.append(Y_test_total, target_mode_Y_test, axis = 0)

        X_train['Total'] = X_train_total
        Y_train['Total'] = Y_train_total
        X_test['Total'] = X_test_total
        Y_test['Total'] = Y_test_total

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_train_FSM = X_train_FSM
        self.Y_train_FSM = Y_train_FSM
        self.X_test_FSM = X_test_FSM
        self.Y_test_FSM = Y_test_FSM