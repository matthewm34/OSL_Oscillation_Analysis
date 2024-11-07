import sys
import os

workspace_path = os.getcwd()
sys.path.insert(1, workspace_path)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import useful_functions
from useful_functions import Functions_SP as fncs
from useful_functions.OSL_Slope_InputProcessing_fromCSV import CSV_FWNNinput_varTransition as inputProc
from useful_functions.OSL_Slope_Offline_bagProc_Functions import checkTopics
from useful_functions import align_sensor2ES_v2 as al
import scipy.interpolate as itpd
import joblib as jb

matplotlib.use('TkAgg')

loadfileDir = workspace_path + '/Unified_Controller_Analysis/PILOT/AB01_2024.Oct.23(HK)/'
bagfilelist = os.listdir(loadfileDir)
jb_savefilename = loadfileDir + 'Unified_AB01_AS.dict'
W = 75
data_load = True

# Test
# target_bag = loadfileDir + bagfilelist[20]
# target_bag_prev_TF = loadfileDir + 'OSL_Stair_Preset_2.bag'
#
# topic_list = checkTopics(target_bag)
# topic_list_prev_TF = checkTopics(target_bag_prev_TF)
#
# loaded_bag = useful_functions.OSL_Slope_Offline_bagProc_Functions.read2var2(target_bag, topic_list)
# loaded_bag_prev_TF = useful_functions.OSL_Slope_Offline_bagProc_Functions.read2var2(target_bag_prev_TF, topic_list_prev_TF)
# # Test

#0. Classify LW and RA and store data
data_dict = {}
data_dict['knee_torque_applied'] = {}
data_dict['ankle_torque_applied'] = {}
data_dict['knee_k_stance'] = {}
data_dict['knee_theta_eq_stance'] = {}
data_dict['knee_b_stance'] = {}
data_dict['ankle_k_stance'] = {}
data_dict['ankle_b_stance'] = {}

data_dict['forceZ_stance'] = {}
data_dict['knee_theta_stance'] = {}
data_dict['ankle_theta_stance'] = {}
data_dict['thigh_orientation_stance'] = {}
data_dict['intact_thigh_orientation_stance'] = {}
data_dict['knee_theta_swing'] = {}
data_dict['ankle_theta_swing'] = {}
data_dict['thigh_orientation_swing'] = {}
data_dict['intact_thigh_orientation_swing'] = {}
gait_phase_new = np.linspace(0, 100, 101)

#1
for b in range(0, len(bagfilelist)):

    if data_load:
        break

    if '.bag' not in bagfilelist[b]:
        continue

    if 'OSL' in bagfilelist[b]:
        continue

    target_bag = loadfileDir + bagfilelist[b]
    topic_list = checkTopics(target_bag)
    loaded_bag = useful_functions.OSL_Slope_Offline_bagProc_Functions.read2var2(target_bag, topic_list)
    print(bagfilelist[b])

    #2. Split data into gait cycles
    FSM = loaded_bag["/fsm/State"]
    FSM_states = FSM['state']
    FSM_headers = FSM['header']

    #3. Collect Early Stances
    FSM_ES_idx = []
    for f in range(0, len(FSM_states)):
        if 'EarlyStance' in FSM_states[f]:
            FSM_ES_idx.append(f)

    for ff in range(0, len(FSM_ES_idx) - 1):

        # For Test
        # ff = 1
        # For Test

        start_ES_mode = FSM_states[FSM_ES_idx[ff]]
        end_ES_mode = FSM_states[FSM_ES_idx[ff + 1]]

        # If ESs are different: transition stride: skip for now
        if start_ES_mode != end_ES_mode:
            # print('transition step', ff)
            continue

        # If ESs are same: steady-state stride
        start_ES_time = FSM_headers[FSM_ES_idx[ff]]
        SF_time = FSM_headers[FSM_ES_idx[ff] + 2]
        end_ES_time = FSM_headers[FSM_ES_idx[ff + 1]]

        start_ES_idx_SensorData = al.closest_point(loaded_bag['/SensorData']['header'], start_ES_time)
        SF_idx_SensorData = al.closest_point(loaded_bag['/SensorData']['header'], SF_time)
        end_ES_idx_SensorData = al.closest_point(loaded_bag['/SensorData']['header'], end_ES_time)

        start_ES_idx_SensorInfo = al.closest_point(loaded_bag['/SensorInfo']['header'], start_ES_time)
        SF_idx_SensorInfo = al.closest_point(loaded_bag['/SensorInfo']['header'], SF_time)
        end_ES_idx_SensorInfo = al.closest_point(loaded_bag['/SensorInfo']['header'], end_ES_time)

        # Target data to analyze

        ## SensorInfo: Torques, IPs
        knee_torque_applied_stance_crop = loaded_bag['/SensorInfo']['knee_torque_applied'][
                                          start_ES_idx_SensorInfo:SF_idx_SensorInfo]
        ankle_torque_applied_stance_crop = loaded_bag['/SensorInfo']['ankle_torque_applied'][
                                          start_ES_idx_SensorInfo:SF_idx_SensorInfo]

        knee_k_stance_crop = loaded_bag['/SensorInfo']['knee_k'][
                                          start_ES_idx_SensorInfo:SF_idx_SensorInfo]
        ankle_k_stance_crop = loaded_bag['/SensorInfo']['ankle_k'][
                                           start_ES_idx_SensorInfo:SF_idx_SensorInfo]

        knee_b_stance_crop = loaded_bag['/SensorInfo']['knee_b'][
                                          start_ES_idx_SensorInfo:SF_idx_SensorInfo]
        ankle_b_stance_crop = loaded_bag['/SensorInfo']['ankle_b'][
                                           start_ES_idx_SensorInfo:SF_idx_SensorInfo]

        knee_theta_eq_stance_crop = loaded_bag['/SensorInfo']['knee_theta_eq'][
                                    start_ES_idx_SensorInfo:SF_idx_SensorInfo]

        ## SensorData: force, kinematics
        forceZ_stance_crop = loaded_bag['/SensorData']['forceZ'][
                             start_ES_idx_SensorData:SF_idx_SensorData]
        knee_theta_stance_crop = loaded_bag['/SensorData']['knee_theta'][
                                          start_ES_idx_SensorData:SF_idx_SensorData]
        ankle_theta_stance_crop = loaded_bag['/SensorData']['ankle_theta'][
                                          start_ES_idx_SensorData:SF_idx_SensorData]
        thigh_orientation_stance_crop = loaded_bag['/SensorData']['thigh_orientation'][
                                          start_ES_idx_SensorData:SF_idx_SensorData]
        intact_thigh_orientation_stance_crop = 180 - loaded_bag['/SensorData']['intact_thigh_orientation'][
                                                 start_ES_idx_SensorData:SF_idx_SensorData]

        knee_theta_swing_crop = loaded_bag['/SensorData']['knee_theta'][
                                 SF_idx_SensorData:end_ES_idx_SensorData]
        ankle_theta_swing_crop = loaded_bag['/SensorData']['ankle_theta'][
                                  SF_idx_SensorData:end_ES_idx_SensorData]
        thigh_orientation_swing_crop = loaded_bag['/SensorData']['thigh_orientation'][
                                        SF_idx_SensorData:end_ES_idx_SensorData]
        intact_thigh_orientation_swing_crop = 180 - loaded_bag['/SensorData']['intact_thigh_orientation'][
                                                     SF_idx_SensorData:end_ES_idx_SensorData]

        gait_phase_old_stance_SI = np.linspace(0, 100, len(knee_torque_applied_stance_crop))
        gait_phase_old_stance_SD = np.linspace(0, 100, len(knee_theta_stance_crop))
        gait_phase_old_swing_SD = np.linspace(0, 100, len(knee_theta_swing_crop))

        # Interpolate

        ## SensorInfo
        f_knee_torque_applied_stance = itpd.interp1d(gait_phase_old_stance_SI, knee_torque_applied_stance_crop)
        knee_torque_applied_stance_itpd = f_knee_torque_applied_stance(gait_phase_new)

        f_ankle_torque_applied_stance = itpd.interp1d(gait_phase_old_stance_SI, ankle_torque_applied_stance_crop)
        ankle_torque_applied_stance_itpd = f_ankle_torque_applied_stance(gait_phase_new)

        f_knee_k_stance = itpd.interp1d(gait_phase_old_stance_SI, knee_k_stance_crop)
        knee_k_stance_itpd = f_knee_k_stance(gait_phase_new)

        f_ankle_k_stance = itpd.interp1d(gait_phase_old_stance_SI, ankle_k_stance_crop)
        ankle_k_stance_itpd = f_ankle_k_stance(gait_phase_new)

        f_knee_b_stance = itpd.interp1d(gait_phase_old_stance_SI, knee_b_stance_crop)
        knee_b_stance_itpd = f_knee_b_stance(gait_phase_new)

        f_ankle_b_stance = itpd.interp1d(gait_phase_old_stance_SI, ankle_b_stance_crop)
        ankle_b_stance_itpd = f_ankle_b_stance(gait_phase_new)

        f_knee_theta_eq_stance = itpd.interp1d(gait_phase_old_stance_SI, knee_theta_eq_stance_crop)
        knee_theta_eq_stance_itpd = f_knee_theta_eq_stance(gait_phase_new)

        ## SensorData
        ### Stance
        f_forceZ_stance = itpd.interp1d(gait_phase_old_stance_SD, forceZ_stance_crop)
        forceZ_stance_itpd = f_forceZ_stance(gait_phase_new)

        f_knee_theta_stance = itpd.interp1d(gait_phase_old_stance_SD, knee_theta_stance_crop)
        knee_theta_stance_itpd = f_knee_theta_stance(gait_phase_new)

        f_ankle_theta_stance = itpd.interp1d(gait_phase_old_stance_SD, ankle_theta_stance_crop)
        ankle_theta_stance_itpd = f_ankle_theta_stance(gait_phase_new)

        f_thigh_orientation_stance = itpd.interp1d(gait_phase_old_stance_SD, thigh_orientation_stance_crop)
        thigh_orientation_stance_itpd = f_thigh_orientation_stance(gait_phase_new)

        f_intact_thigh_orientation_stance = itpd.interp1d(gait_phase_old_stance_SD, intact_thigh_orientation_stance_crop)
        intact_thigh_orientation_stance_itpd = f_intact_thigh_orientation_stance(gait_phase_new)

        ### Swing
        f_knee_theta_swing = itpd.interp1d(gait_phase_old_swing_SD, knee_theta_swing_crop)
        knee_theta_swing_itpd = f_knee_theta_swing(gait_phase_new)

        f_ankle_theta_swing = itpd.interp1d(gait_phase_old_swing_SD, ankle_theta_swing_crop)
        ankle_theta_swing_itpd = f_ankle_theta_swing(gait_phase_new)

        f_thigh_orientation_swing = itpd.interp1d(gait_phase_old_swing_SD, thigh_orientation_swing_crop)
        thigh_orientation_swing_itpd = f_thigh_orientation_swing(gait_phase_new)

        f_intact_thigh_orientation_swing = itpd.interp1d(gait_phase_old_swing_SD, intact_thigh_orientation_swing_crop)
        intact_thigh_orientation_swing_itpd = f_intact_thigh_orientation_swing(gait_phase_new)

        if 'LW' in start_ES_mode:
            print(ff, 'LW stride')

            if 0 not in data_dict['knee_torque_applied'].keys():
                data_dict['knee_torque_applied'][0] = []
                data_dict['ankle_torque_applied'][0] = []
                data_dict['knee_k_stance'][0] = []
                data_dict['knee_b_stance'][0] = []
                data_dict['knee_theta_eq_stance'][0] = []
                data_dict['ankle_k_stance'][0] = []
                data_dict['ankle_b_stance'][0] = []

                data_dict['forceZ_stance'][0] = []
                data_dict['knee_theta_stance'][0] = []
                data_dict['ankle_theta_stance'][0] = []
                data_dict['thigh_orientation_stance'][0] = []
                data_dict['intact_thigh_orientation_stance'][0] = []
                data_dict['knee_theta_swing'][0] = []
                data_dict['ankle_theta_swing'][0] = []
                data_dict['thigh_orientation_swing'][0] = []
                data_dict['intact_thigh_orientation_swing'][0] = []

            data_dict['knee_torque_applied'][0].append(knee_torque_applied_stance_itpd)
            data_dict['ankle_torque_applied'][0].append(ankle_torque_applied_stance_itpd)
            data_dict['knee_k_stance'][0].append(knee_k_stance_itpd)
            data_dict['knee_b_stance'][0].append(knee_b_stance_itpd)
            data_dict['knee_theta_eq_stance'][0].append(knee_theta_eq_stance_itpd)
            data_dict['ankle_k_stance'][0].append(ankle_k_stance_itpd)
            data_dict['ankle_b_stance'][0].append(ankle_b_stance_itpd)

            data_dict['forceZ_stance'][0].append(forceZ_stance_itpd)
            data_dict['knee_theta_stance'][0].append(knee_theta_stance_itpd)
            data_dict['ankle_theta_stance'][0].append(ankle_theta_stance_itpd)
            data_dict['thigh_orientation_stance'][0].append(thigh_orientation_stance_itpd)
            data_dict['intact_thigh_orientation_stance'][0].append(intact_thigh_orientation_stance_itpd)
            data_dict['knee_theta_swing'][0].append(knee_theta_swing_itpd)
            data_dict['ankle_theta_swing'][0].append(ankle_theta_swing_itpd)
            data_dict['thigh_orientation_swing'][0].append(thigh_orientation_swing_itpd)
            data_dict['intact_thigh_orientation_swing'][0].append(intact_thigh_orientation_swing_itpd)

        if 'RA' in start_ES_mode:
            print(ff, 'Ascent stride')
            slope = loaded_bag['/fsm/context']['value'][0]

            if slope not in data_dict['knee_torque_applied'].keys():
                data_dict['knee_torque_applied'][slope] = []
                data_dict['ankle_torque_applied'][slope] = []
                data_dict['knee_k_stance'][slope] = []
                data_dict['knee_b_stance'][slope] = []
                data_dict['knee_theta_eq_stance'][slope] = []
                data_dict['ankle_k_stance'][slope] = []
                data_dict['ankle_b_stance'][slope] = []

                data_dict['forceZ_stance'][slope] = []
                data_dict['knee_theta_stance'][slope] = []
                data_dict['ankle_theta_stance'][slope] = []
                data_dict['thigh_orientation_stance'][slope] = []
                data_dict['intact_thigh_orientation_stance'][slope] = []
                data_dict['knee_theta_swing'][slope] = []
                data_dict['ankle_theta_swing'][slope] = []
                data_dict['thigh_orientation_swing'][slope] = []
                data_dict['intact_thigh_orientation_swing'][slope] = []

            data_dict['knee_torque_applied'][slope].append(knee_torque_applied_stance_itpd)
            data_dict['ankle_torque_applied'][slope].append(ankle_torque_applied_stance_itpd)
            data_dict['knee_k_stance'][slope].append(knee_k_stance_itpd)
            data_dict['knee_b_stance'][slope].append(knee_b_stance_itpd)
            data_dict['knee_theta_eq_stance'][slope].append(knee_theta_eq_stance_itpd)
            data_dict['ankle_k_stance'][slope].append(ankle_k_stance_itpd)
            data_dict['ankle_b_stance'][slope].append(ankle_b_stance_itpd)

            data_dict['forceZ_stance'][slope].append(forceZ_stance_itpd)
            data_dict['knee_theta_stance'][slope].append(knee_theta_stance_itpd)
            data_dict['ankle_theta_stance'][slope].append(ankle_theta_stance_itpd)
            data_dict['thigh_orientation_stance'][slope].append(thigh_orientation_stance_itpd)
            data_dict['intact_thigh_orientation_stance'][slope].append(intact_thigh_orientation_stance_itpd)
            data_dict['knee_theta_swing'][slope].append(knee_theta_swing_itpd)
            data_dict['ankle_theta_swing'][slope].append(ankle_theta_swing_itpd)
            data_dict['thigh_orientation_swing'][slope].append(thigh_orientation_swing_itpd)
            data_dict['intact_thigh_orientation_swing'][slope].append(intact_thigh_orientation_swing_itpd)

if data_load == False:
    # Save processed PILOT data
    jb.dump(data_dict, jb_savefilename)

data_loaded = jb.load(jb_savefilename)


# Plot
avg_dict = {}
for channel in data_loaded.keys():

    if channel not in avg_dict.keys():
        avg_dict[channel] = {}

    for slope in data_loaded[channel].keys():
        if slope == 30.0 or np.round(slope,2) == 9.2:
            continue

        avg_dict[channel][slope] = np.zeros(101)

        target_slope_channel_data = np.array(data_loaded[channel][slope])

        for p in range(0, len(gait_phase_new)):
            avg_dict[channel][slope][p] = np.mean(target_slope_channel_data[:,p])

# Plot
# colors = ['#96EAB6', '#29C564', '#156935', '#0C3A1E', '#010703']
rgb_start = np.array(fncs.hex2rgb('96EAB6'))/255
rgb_end = np.array(fncs.hex2rgb('010703'))/255
sorted_slope = sorted(avg_dict['knee_torque_applied'].keys())
t = len(sorted_slope)

for channel in avg_dict.keys():
    plt.figure(channel)
    c = 0

    for slope in sorted_slope:

        target_slope_channel_data_avg = np.array(avg_dict[channel][slope])

        if 'torque' in channel:
            target_slope_channel_data_avg = target_slope_channel_data_avg / W
        if 'force' in channel:
            target_slope_channel_data_avg = target_slope_channel_data_avg / W / 9.8
        if 'intact' in channel:
            target_slope_channel_data_avg = 180 - target_slope_channel_data_avg

        plot_color = rgb_start + (rgb_end-rgb_start) * c/t
        plot_color = tuple(plot_color)
        # print(plot_color)
        plt.plot(target_slope_channel_data_avg, color = plot_color, linewidth = 3)
        c +=1

    plt.legend(np.round(sorted_slope,2))