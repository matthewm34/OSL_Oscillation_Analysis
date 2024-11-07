'''
This script processes sensor data dictionary to input data for Neural Network process.
'''

import pandas as pd
# from useful_functions import Functions_SP as fncs
# import joblib as jb
import numpy as np
# from useful_functions import align_sensor2ES_v2 as al
import os
import matplotlib.pyplot as plt
import math

def windowsignal_1D(input, window_length, overlap_length):
    '''

    '''
    num_increment = math.floor( len(input) / (window_length-overlap_length) )
    signal_windowed = np.zeros([num_increment, window_length])

    for i in range(0, num_increment):
        if len(input) - ((window_length) + i * (window_length-overlap_length)) <0 :
            break

        signal_windowed[i,:] = np.array(input[i * (window_length - overlap_length): window_length + i * (window_length - overlap_length)])

    signal_windowed = signal_windowed[:i,:]
    return signal_windowed

def windowsignal_2D(target_sensordata, window_length, overlap_length):

    ### Windowing sensordata and GT
    firstchannel_windowed = windowsignal_1D(input=target_sensordata[:, 0],
                                            window_length=window_length,
                                            overlap_length=overlap_length)

    windowed_signal_total = np.zeros([firstchannel_windowed.shape[0],
                                      firstchannel_windowed.shape[1],
                                      target_sensordata.shape[1]])

    windowed_signal_total[:, :, 0] = firstchannel_windowed  ### train_X or test_X

    for s in range(1, target_sensordata.shape[1]):  ### First one is already done
        target_to_window = target_sensordata[:, s]
        windowed_target = windowsignal_1D(input=target_to_window,
                                          window_length=window_length,
                                          overlap_length=overlap_length)

        windowed_signal_total[:, :, s] = windowed_target

    return windowed_signal_total

def CSV_SortCount(TFDir, test_ratio=0.2):
    '''
    In each designated TF data folder, split scv files into specified ratio
    in each preset(0,1,2, ... ) - stride type(SS/TR) - mode(LW, AS/DS, LW2AS, AS2LW, ...)
    '''

    # ### Function test - start
    # workspace_path = os.getcwd()
    # TFDir = workspace_path + '/processedCSV_byStride/FW125/Offline/Ramp/TF02/'
    # test_ratio = 0.2
    # ### Function test - end

    csv_list = os.listdir(TFDir)

    count_dict = {}

    ##for loop below
    for csv_name in csv_list:

        ### Test
        # csv_name = csv_list[0]
        ### Test

        # Initiate dictionary
        preset = csv_name.rsplit('_')[3]
        if preset not in count_dict.keys():
            count_dict[preset] = {}

        stride_type = csv_name.rsplit('_')[4]
        if stride_type not in count_dict[preset].keys():
            count_dict[preset][stride_type] = {}

        mode = csv_name.rsplit('_')[5]
        if mode not in count_dict[preset][stride_type].keys():
            count_dict[preset][stride_type][mode] = []

        # Fully counted dictionary
        count_dict[preset][stride_type][mode].append(csv_name)

    # Initiate split dictionary
    train_list = []
    test_list = []

    for preset in count_dict.keys():
        for stride_type in count_dict[preset].keys():
            for mode in count_dict[preset][stride_type].keys():
                n_stride = len(count_dict[preset][stride_type][mode])

                test_count = int(np.ceil(n_stride * test_ratio)) # Include at least 1 stride in test/valid data
                train_count = int(n_stride - test_count)

                for i1 in range(0, train_count):
                    train_list.append(count_dict[preset][stride_type][mode][i1])

                for i2 in range(train_count, n_stride):
                    test_list.append(count_dict[preset][stride_type][mode][i2])

    return train_list, test_list

def CSV_FWNNinput(TFDir, test_ratio=0.2, sensor_list = [], FW_winlen = 125, stride_len = 2, delay = 25, ramp_off = False):
    '''
    Using CSV_SortCount Function, converts CSV files into practical FW NN input
    '''

    # ### Function test - start
    # workspace_path = os.getcwd()
    # TFDir = workspace_path + '/processedCSV_byStride/FW125/Offline/Ramp/TF02/'
    # test_ratio = 0.2
    # FW_winlen = 125
    # stride_len = 2
    # delay = 25
    #
    # # Full Sensor
    # sensor_list = ['ankle_theta', 'ankle_thetadot', 'knee_theta', 'knee_thetadot',
    #               'forceX', 'forceY', 'forceZ', 'momentX', 'momentY', 'momentZ',
    #               'shank_accelX', 'shank_accelY', 'shank_accelZ', 'shank_gyroX', 'shank_gyroY', 'shank_gyroZ',
    #               'foot_accelX', 'foot_accelY', 'foot_accelZ', 'foot_gyroX', 'foot_gyroY', 'foot_gyroZ',
    #               'thigh_accelX', 'thigh_accelY', 'thigh_accelZ', 'thigh_gyroX', 'thigh_gyroY', 'thigh_gyroZ']
    #
    # # Only with biarticular movements
    # sensor_list =  ['ankle_theta', 'ankle_thetadot', 'knee_theta', 'knee_thetadot',
    #               'forceX', 'forceZ', 'momentY',
    #               'shank_accelX', 'shank_accelY', 'shank_gyroZ',
    #               'foot_accelX', 'foot_accelZ', 'foot_gyroY',
    #               'thigh_accelX', 'thigh_accelZ', 'thigh_gyroY']
    #
    # ramp_off = True
    # ### Function test - end

    train_list, test_list = CSV_SortCount(TFDir, test_ratio)
    overlap_len = FW_winlen - stride_len

    # Form train dataset
    train_total_count = 0
    for train_CSV in train_list:
        eachFileDir = TFDir + train_CSV
        target_CSV = pd.read_csv(eachFileDir) # Load each file
        # target_CSV = target_CSV[:-delay]

        # Convert to sensor data array
        sensordata_raw = []
        for sChannel in sensor_list:
            sensordata_raw.append(target_CSV[sChannel].__array__())

        sensordata_raw = np.array(sensordata_raw).T
        sensordata_FWNNinput = windowsignal_2D(sensordata_raw, FW_winlen, overlap_len)

        # GT slope labels
        slopelabel_raw = []
        for MSPs in target_CSV.keys():
            if 'MSP' not in MSPs:
                continue

            slopelabel_raw.append(target_CSV[MSPs].__array__())

        slopelabel_raw = np.array(slopelabel_raw).T

        if ramp_off == True: ## Applying 150ms delay
            slopelabel_raw[-delay:-delay+15] = slopelabel_raw[FW_winlen]

        slopelabel_FWNNinput = windowsignal_2D(slopelabel_raw, FW_winlen, overlap_len)

        if train_total_count == 0:
            train_X = sensordata_FWNNinput
            train_Y = slopelabel_FWNNinput
            train_total_count += 1
            print('Train Data:' + str(train_total_count) + '/' + str(len(train_list)), end = '\r')

        else:
            train_X = np.concatenate([train_X, sensordata_FWNNinput], axis = 0)
            train_Y = np.concatenate([train_Y, slopelabel_FWNNinput], axis = 0)
            train_total_count += 1
            print('Train Data:' + str(train_total_count) + '/' + str(len(train_list)), end = '\r')

    # Form test dataset
    test_total_count = 0
    for test_CSV in test_list:
        eachFileDir = TFDir + test_CSV
        target_CSV = pd.read_csv(eachFileDir) # Load each file
        # target_CSV = target_CSV[:-delay]

        # Convert to sensor data array
        sensordata_raw = []
        for sChannel in sensor_list:
            sensordata_raw.append(target_CSV[sChannel].__array__())

        sensordata_raw = np.array(sensordata_raw).T

        if ramp_off == True: ## Applying 150ms delay
            slopelabel_raw[-delay:-delay+15] = slopelabel_raw[FW_winlen]

        sensordata_FWNNinput = windowsignal_2D(sensordata_raw, FW_winlen, overlap_len)

        # GT slope labels
        slopelabel_raw = []
        for MSPs in target_CSV.keys():
            if 'MSP' not in MSPs:
                continue

            slopelabel_raw.append(target_CSV[MSPs].__array__())

        slopelabel_raw = np.array(slopelabel_raw).T
        slopelabel_FWNNinput = windowsignal_2D(slopelabel_raw, FW_winlen, overlap_len)

        if test_total_count == 0:
            test_X = sensordata_FWNNinput
            test_Y = slopelabel_FWNNinput
            test_total_count += 1
            print('Test Data:' + str(test_total_count) + '/' + str(len(test_list)), end = '\r')
        else:
            test_X = np.concatenate([test_X, sensordata_FWNNinput], axis = 0)
            test_Y = np.concatenate([test_Y, slopelabel_FWNNinput], axis = 0)
            test_total_count += 1
            print('Test Data:' + str(test_total_count) + '/' + str(len(test_list)), end = '\r')

    return train_X, train_Y, test_X, test_Y

def CSV_varTransition_save2CSV(csvName, MSP_list):
    '''
    Using CSV_SortCount Function, converts CSV files into practical FW NN input
    '''

    # # ### Function test - start
    # workspace_path = os.getcwd()
    # # # csvName = workspace_path + '/processedCSV_byStride/FW125/RT/Ramp/TF02/TF02_RampRT_preset_2_TR_LW2AS_2_bag16preset_2_slope8.7.csv'
    # # csvName = workspace_path + '\processedCSV_byStride\FW125\Offline\Stair\TF03\TF03_StairOffline_preset_1_TR_LW2AS_1_bagStair_Preset_1.csv'
    # csvName = workspace_path + '\processedCSV_byStride\FW125\Offline\Stair\TF08\TF08_StairOffline_preset_3_TR_LW2DS_6_bagStair_Preset_3_1.csv'
    # MSP_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # # ### Function test - end

    eachFileDir = csvName
    target_CSV = pd.read_csv(eachFileDir) # Load each file

    # GT slope labels
    slopelabel_raw = []
    for MSPs in target_CSV.keys():
        if 'MSP' not in MSPs:
            continue

        slopelabel_raw.append(target_CSV[MSPs].__array__())

    slopelabel_raw = np.array(slopelabel_raw)[0].T

    if 'TR' not in csvName:
        target_CSV['slopelabel_raw'] = slopelabel_raw

    if 'TR' in csvName: # Check Transition CSVs

        '''
        Transition types
        1. Ramp
         - LW<->RA/RD: In HCs (Type 1) 
        2. Stair
         - LW<->SA: In SFs(TOs) (Type 2)
         - LW<->SD: In HCs (Type 1)
        '''

        if 'Ramp' in csvName: # Ramp: All Type 1

            '''
            In Ramp Data
            1. Offline: 150ms transition delay is not applied
            2. RT: 150ms transition delay is already applied
            '''

            # Find Toe off and Transition HC
            target_state_num = target_CSV['state_num'].__array__()

            # Last frame of stance
            for st_HC_crop in range(0, len(target_state_num) - 1):
                st_HC_crop += 1
                st_HC_crop *= -1

                if target_state_num[st_HC_crop] == 1: # If the state is Stance
                    break

            target_CSV = target_CSV[:st_HC_crop + 1 + len(slopelabel_raw)] # the last state should be stance
            target_state_num = target_CSV['state_num'].__array__() # Update with cropped data

            # GT slope labels
            slopelabel_raw = []
            for MSPs in target_CSV.keys():
                if 'MSP' not in MSPs:
                    continue

                slopelabel_raw.append(target_CSV[MSPs].__array__())

            slopelabel_raw = np.array(slopelabel_raw)[0].T

            # First frame of stance
            for st_HC in range(0, len(target_state_num) - 1):
                st_HC += 1
                st_HC *= -1
                st_HC_prev = st_HC - 1

                if target_state_num[st_HC] - target_state_num[st_HC_prev] == -2:
                    break

            # print(st_HC)
            if st_HC == -1: # Last strides of each trial
                delayed_HC_idx = st_HC
            else:
                delayed_HC_idx = st_HC + 15

            slopelabel_delayed = np.copy(slopelabel_raw)
            slopelabel_delayed[st_HC:delayed_HC_idx] = slopelabel_raw[st_HC-1]
            slopelabel_delayed[delayed_HC_idx:] = slopelabel_raw[-1]

            # Find TO
            for st_TO in range(0, len(target_state_num) - 1):
                st_TO += 1
                st_TO *= -1
                st_TO_prev = st_TO - 1

                if (target_state_num[st_TO] == 2) and (target_state_num[st_TO_prev] == 1):
                    break

            # print(st_TO)
            TO_idx = st_TO

            transition_len = st_HC - TO_idx

            target_CSV['slopelabel_raw'] = slopelabel_raw
            target_CSV['slopelabel_delayed'] = slopelabel_delayed

            for MSP in MSP_list:
                transition_start_idx = TO_idx + int(MSP * transition_len)
                transition_end_idx = delayed_HC_idx
                transition_start_slope = slopelabel_raw[TO_idx - 1]
                transition_end_slope = slopelabel_raw[-1]

                transition_GT = np.linspace(start=transition_start_slope, stop=transition_end_slope,
                                            num=transition_end_idx - transition_start_idx + 2)

                ### Check
                slopelabel_updated = np.copy(slopelabel_delayed)
                slopelabel_updated[transition_start_idx-1:transition_end_idx+1] = transition_GT
                ### Check

                slope_keyname = 'slopelabel_smoothed' + str(MSP)
                target_CSV[slope_keyname] = slopelabel_updated

        if 'Stair' in csvName: # Stair: LW<->SD Type 1 / LW<->SA Type 2

            '''
            In Stair Data - Offline: 150ms transition delay is not applied
            '''

            # Find Toe off and Transition HC
            target_state_num = target_CSV['state_num'].__array__()

            # Last frame of stance
            for st_HC_crop in range(0, len(target_state_num) - 1):
                st_HC_crop += 1
                st_HC_crop *= -1

                if target_state_num[st_HC_crop] == 1:  # If the state is Stance
                    break

            target_CSV = target_CSV[:st_HC_crop + 1 + len(slopelabel_raw)]  # the last state should be stance
            target_state_num = target_CSV['state_num'].__array__()  # Update with cropped data

            # GT slope labels
            slopelabel_raw = []
            for MSPs in target_CSV.keys():
                if 'MSP' not in MSPs:
                    continue

                slopelabel_raw.append(target_CSV[MSPs].__array__())

            slopelabel_raw = np.array(slopelabel_raw)[0].T

            # First frame of stance
            for st_HC in range(0, len(target_state_num) - 1):
                st_HC += 1
                st_HC *= -1
                st_HC_prev = st_HC - 1

                if target_state_num[st_HC] - target_state_num[st_HC_prev] == -2:
                    break

            # Find TO
            for st_TO in range(0, len(target_state_num) - 1):
                st_TO += 1
                st_TO *= -1
                st_TO_prev = st_TO - 1

                if (target_state_num[st_TO] == 2) and (target_state_num[st_TO_prev] == 1):
                    break

            # print(st_TO)
            TO_idx = st_TO

            # Case 1: LW<->SA - Transitions take place at TO
            if 'LW2AS' in csvName or 'AS2LW' in csvName:

                delayed_TO_idx = TO_idx + 15

                slopelabel_delayed = np.copy(slopelabel_raw)
                slopelabel_delayed[st_TO:delayed_TO_idx] = slopelabel_raw[st_TO - 1]
                slopelabel_delayed[delayed_TO_idx:] = slopelabel_raw[-1]

                # First stance frame of the stride
                for st_HC2 in range(-st_HC, len(target_state_num) - 1):
                    st_HC2 += 1
                    st_HC2 *= -1
                    st_HC2_prev = st_HC2 - 1

                    if target_state_num[st_HC2] - target_state_num[st_HC2_prev] == -2:
                        break

                transition_len = st_TO - st_HC2

                target_CSV['slopelabel_raw'] = slopelabel_raw
                target_CSV['slopelabel_delayed'] = slopelabel_delayed

                for MSP in MSP_list:
                    transition_start_idx = st_HC2 + int(MSP * transition_len)
                    transition_end_idx = delayed_TO_idx
                    transition_start_slope = slopelabel_raw[TO_idx - 1]
                    transition_end_slope = slopelabel_raw[-1]

                    transition_GT = np.linspace(start=transition_start_slope, stop=transition_end_slope,
                                                num=transition_end_idx - transition_start_idx + 2)

                    ### Check
                    slopelabel_updated = np.copy(slopelabel_delayed)
                    slopelabel_updated[transition_start_idx - 1:transition_end_idx + 1] = transition_GT
                    ### Check

                    slope_keyname = 'slopelabel_smoothed' + str(MSP)
                    target_CSV[slope_keyname] = slopelabel_updated

            # Case 2: LW<->SD - Transitions take place at HC
            if 'LW2DS' in csvName or 'DS2LW' in csvName:

                # print(st_HC)
                if st_HC == -1:  # Last strides of each trial
                    delayed_HC_idx = st_HC
                else:
                    delayed_HC_idx = st_HC + 15

                slopelabel_delayed = np.copy(slopelabel_raw)
                slopelabel_delayed[st_HC:delayed_HC_idx] = slopelabel_raw[st_HC - 1]
                slopelabel_delayed[delayed_HC_idx:] = slopelabel_raw[-1]

                transition_len = st_HC - TO_idx

                target_CSV['slopelabel_raw'] = slopelabel_raw
                target_CSV['slopelabel_delayed'] = slopelabel_delayed

                for MSP in MSP_list:
                    transition_start_idx = TO_idx + int(MSP * transition_len)
                    transition_end_idx = delayed_HC_idx
                    transition_start_slope = slopelabel_raw[TO_idx - 1]
                    transition_end_slope = slopelabel_raw[-1]

                    transition_GT = np.linspace(start=transition_start_slope, stop=transition_end_slope,
                                                num=transition_end_idx - transition_start_idx + 2)

                    ### Check
                    slopelabel_updated = np.copy(slopelabel_delayed)
                    slopelabel_updated[transition_start_idx - 1:transition_end_idx + 1] = transition_GT
                    ### Check

                    slope_keyname = 'slopelabel_smoothed' + str(MSP)
                    target_CSV[slope_keyname] = slopelabel_updated

    return target_CSV

def CSV_varTransition_save2CSV_temp(csvName, MSP_list):
    '''
    Using CSV_SortCount Function, converts CSV files into practical FW NN input
    Temporarily test 100ms delay
    '''

    # # ### Function test - start
    # workspace_path = os.getcwd()
    # # # csvName = workspace_path + '/processedCSV_byStride/FW125/RT/Ramp/TF02/TF02_RampRT_preset_2_TR_LW2AS_2_bag16preset_2_slope8.7.csv'
    # # csvName = workspace_path + '\processedCSV_byStride\FW125\Offline\Stair\TF03\TF03_StairOffline_preset_1_TR_LW2AS_1_bagStair_Preset_1.csv'
    # csvName = workspace_path + '\processedCSV_byStride\FW125\Offline\Stair\TF08\TF08_StairOffline_preset_3_TR_LW2DS_6_bagStair_Preset_3_1.csv'
    # MSP_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # # ### Function test - end

    eachFileDir = csvName
    target_CSV = pd.read_csv(eachFileDir) # Load each file

    # GT slope labels
    slopelabel_raw = []
    for MSPs in target_CSV.keys():
        if 'MSP' not in MSPs:
            continue

        slopelabel_raw.append(target_CSV[MSPs].__array__())

    slopelabel_raw = np.array(slopelabel_raw)[0].T

    if 'TR' not in csvName:
        target_CSV['slopelabel_raw'] = slopelabel_raw

    if 'TR' in csvName: # Check Transition CSVs

        '''
        Transition types
        1. Ramp
         - LW<->RA/RD: In HCs (Type 1) 
        2. Stair
         - LW<->SA: In SFs(TOs) (Type 2)
         - LW<->SD: In HCs (Type 1)
        '''

        if 'Ramp' in csvName: # Ramp: All Type 1

            '''
            In Ramp Data
            1. Offline: 150ms transition delay is not applied
            2. RT: 150ms transition delay is already applied
            '''

            # Find Toe off and Transition HC
            target_state_num = target_CSV['state_num'].__array__()

            # Last frame of stance
            for st_HC_crop in range(0, len(target_state_num) - 1):
                st_HC_crop += 1
                st_HC_crop *= -1

                if target_state_num[st_HC_crop] == 1: # If the state is Stance
                    break

            target_CSV = target_CSV[:st_HC_crop + 1 + len(slopelabel_raw)] # the last state should be stance
            target_state_num = target_CSV['state_num'].__array__() # Update with cropped data

            # GT slope labels
            slopelabel_raw = []
            for MSPs in target_CSV.keys():
                if 'MSP' not in MSPs:
                    continue

                slopelabel_raw.append(target_CSV[MSPs].__array__())

            slopelabel_raw = np.array(slopelabel_raw)[0].T

            # First frame of stance
            for st_HC in range(0, len(target_state_num) - 1):
                st_HC += 1
                st_HC *= -1
                st_HC_prev = st_HC - 1

                if target_state_num[st_HC] - target_state_num[st_HC_prev] == -2:
                    break

            # print(st_HC)
            if st_HC == -1: # Last strides of each trial
                delayed_HC_idx = st_HC
            else:
                delayed_HC_idx = st_HC + 10

            slopelabel_delayed = np.copy(slopelabel_raw)
            slopelabel_delayed[st_HC:delayed_HC_idx] = slopelabel_raw[st_HC-1]
            slopelabel_delayed[delayed_HC_idx:] = slopelabel_raw[-1]

            # Find TO
            for st_TO in range(0, len(target_state_num) - 1):
                st_TO += 1
                st_TO *= -1
                st_TO_prev = st_TO - 1

                if (target_state_num[st_TO] == 2) and (target_state_num[st_TO_prev] == 1):
                    break

            # print(st_TO)
            TO_idx = st_TO

            transition_len = st_HC - TO_idx

            target_CSV['slopelabel_raw'] = slopelabel_raw
            target_CSV['slopelabel_delayed'] = slopelabel_delayed

            for MSP in MSP_list:
                transition_start_idx = TO_idx + int(MSP * transition_len)
                transition_end_idx = delayed_HC_idx
                transition_start_slope = slopelabel_raw[TO_idx - 1]
                transition_end_slope = slopelabel_raw[-1]

                transition_GT = np.linspace(start=transition_start_slope, stop=transition_end_slope,
                                            num=transition_end_idx - transition_start_idx + 2)

                ### Check
                slopelabel_updated = np.copy(slopelabel_delayed)
                slopelabel_updated[transition_start_idx-1:transition_end_idx+1] = transition_GT
                ### Check

                slope_keyname = 'slopelabel_smoothed' + str(MSP)
                target_CSV[slope_keyname] = slopelabel_updated

        if 'Stair' in csvName: # Stair: LW<->SD Type 1 / LW<->SA Type 2

            '''
            In Stair Data - Offline: 150ms transition delay is not applied
            '''

            # Find Toe off and Transition HC
            target_state_num = target_CSV['state_num'].__array__()

            # Last frame of stance
            for st_HC_crop in range(0, len(target_state_num) - 1):
                st_HC_crop += 1
                st_HC_crop *= -1

                if target_state_num[st_HC_crop] == 1:  # If the state is Stance
                    break

            target_CSV = target_CSV[:st_HC_crop + 1 + len(slopelabel_raw)]  # the last state should be stance
            target_state_num = target_CSV['state_num'].__array__()  # Update with cropped data

            # GT slope labels
            slopelabel_raw = []
            for MSPs in target_CSV.keys():
                if 'MSP' not in MSPs:
                    continue

                slopelabel_raw.append(target_CSV[MSPs].__array__())

            slopelabel_raw = np.array(slopelabel_raw)[0].T

            # First frame of stance
            for st_HC in range(0, len(target_state_num) - 1):
                st_HC += 1
                st_HC *= -1
                st_HC_prev = st_HC - 1

                if target_state_num[st_HC] - target_state_num[st_HC_prev] == -2:
                    break

            # Find TO
            for st_TO in range(0, len(target_state_num) - 1):
                st_TO += 1
                st_TO *= -1
                st_TO_prev = st_TO - 1

                if (target_state_num[st_TO] == 2) and (target_state_num[st_TO_prev] == 1):
                    break

            # print(st_TO)
            TO_idx = st_TO

            # Case 1: LW<->SA - Transitions take place at TO
            if 'LW2AS' in csvName or 'AS2LW' in csvName:

                delayed_TO_idx = TO_idx + 10

                slopelabel_delayed = np.copy(slopelabel_raw)
                slopelabel_delayed[st_TO:delayed_TO_idx] = slopelabel_raw[st_TO - 1]
                slopelabel_delayed[delayed_TO_idx:] = slopelabel_raw[-1]

                # First stance frame of the stride
                for st_HC2 in range(-st_HC, len(target_state_num) - 1):
                    st_HC2 += 1
                    st_HC2 *= -1
                    st_HC2_prev = st_HC2 - 1

                    if target_state_num[st_HC2] - target_state_num[st_HC2_prev] == -2:
                        break

                transition_len = st_TO - st_HC2

                target_CSV['slopelabel_raw'] = slopelabel_raw
                target_CSV['slopelabel_delayed'] = slopelabel_delayed

                for MSP in MSP_list:
                    transition_start_idx = st_HC2 + int(MSP * transition_len)
                    transition_end_idx = delayed_TO_idx
                    transition_start_slope = slopelabel_raw[TO_idx - 1]
                    transition_end_slope = slopelabel_raw[-1]

                    transition_GT = np.linspace(start=transition_start_slope, stop=transition_end_slope,
                                                num=transition_end_idx - transition_start_idx + 2)

                    ### Check
                    slopelabel_updated = np.copy(slopelabel_delayed)
                    slopelabel_updated[transition_start_idx - 1:transition_end_idx + 1] = transition_GT
                    ### Check

                    slope_keyname = 'slopelabel_smoothed' + str(MSP)
                    target_CSV[slope_keyname] = slopelabel_updated

            # Case 2: LW<->SD - Transitions take place at HC
            if 'LW2DS' in csvName or 'DS2LW' in csvName:

                # print(st_HC)
                if st_HC == -1:  # Last strides of each trial
                    delayed_HC_idx = st_HC
                else:
                    delayed_HC_idx = st_HC + 10

                slopelabel_delayed = np.copy(slopelabel_raw)
                slopelabel_delayed[st_HC:delayed_HC_idx] = slopelabel_raw[st_HC - 1]
                slopelabel_delayed[delayed_HC_idx:] = slopelabel_raw[-1]

                transition_len = st_HC - TO_idx

                target_CSV['slopelabel_raw'] = slopelabel_raw
                target_CSV['slopelabel_delayed'] = slopelabel_delayed

                for MSP in MSP_list:
                    transition_start_idx = TO_idx + int(MSP * transition_len)
                    transition_end_idx = delayed_HC_idx
                    transition_start_slope = slopelabel_raw[TO_idx - 1]
                    transition_end_slope = slopelabel_raw[-1]

                    transition_GT = np.linspace(start=transition_start_slope, stop=transition_end_slope,
                                                num=transition_end_idx - transition_start_idx + 2)

                    ### Check
                    slopelabel_updated = np.copy(slopelabel_delayed)
                    slopelabel_updated[transition_start_idx - 1:transition_end_idx + 1] = transition_GT
                    ### Check

                    slope_keyname = 'slopelabel_smoothed' + str(MSP)
                    target_CSV[slope_keyname] = slopelabel_updated

    return target_CSV

def CSV_FWNNinput_varTransition(TFDir, test_ratio=0.2, sensor_list = [], FW_winlen = 125, stride_len = 2, MSP = 'baseline'):
    '''
    Using CSV_SortCount Function, converts CSV files into practical FW NN input
    '''

    # ### Function test - start
    # workspace_path = os.getcwd()
    # TFDir = workspace_path + '/processedCSV_byStride_MSPapplied/FW125/Offline/Stair/TF03/'
    # # TFDir = workspace_path + '/processedCSV_byStride_MSPapplied/FW125/Offline/Ramp/TF03/'
    # test_ratio = 0.2
    # FW_winlen = 125
    # stride_len = 2
    # MSP = 'baseline'
    # # MSP = 0.0
    #
    # # Full Sensor
    # sensor_list = ['ankle_theta', 'ankle_thetadot', 'knee_theta', 'knee_thetadot',
    #               'forceX', 'forceY', 'forceZ', 'momentX', 'momentY', 'momentZ',
    #               'shank_accelX', 'shank_accelY', 'shank_accelZ', 'shank_gyroX', 'shank_gyroY', 'shank_gyroZ',
    #               'foot_accelX', 'foot_accelY', 'foot_accelZ', 'foot_gyroX', 'foot_gyroY', 'foot_gyroZ',
    #               'thigh_accelX', 'thigh_accelY', 'thigh_accelZ', 'thigh_gyroX', 'thigh_gyroY', 'thigh_gyroZ']
    #
    # # # Only with biarticular movements
    # # sensor_list =  ['ankle_theta', 'ankle_thetadot', 'knee_theta', 'knee_thetadot',
    # #               'forceX', 'forceZ', 'momentY',
    # #               'shank_accelX', 'shank_accelY', 'shank_gyroZ',
    # #               'foot_accelX', 'foot_accelZ', 'foot_gyroY',
    # #               'thigh_accelX', 'thigh_accelZ', 'thigh_gyroY']
    # ### Function test - end

    train_list, test_list = CSV_SortCount(TFDir, test_ratio)
    overlap_len = FW_winlen - stride_len

    # Form train dataset
    train_total_count = 0
    for train_CSV in train_list:

        # ### Test
        # train_CSV = 'TF02_RampOffline_preset_4_SS_AS_1_bag28preset_4_slope12.4.csv'
        # train_CSV = 'TF02_RampOffline_preset_1_TR_LW2AS_1_bag38preset_1_slope7.8.csv'
        # ### Test

        eachFileDir = TFDir + train_CSV
        target_CSV = pd.read_csv(eachFileDir)  # Load each file

        if 'EarlyStance' not in target_CSV['state'][FW_winlen]:
            print(train_CSV, 'skip this trial')

        # Convert to sensor data array
        sensordata_raw = []
        for sChannel in sensor_list:
            sensordata_raw.append(target_CSV[sChannel].__array__())

        sensordata_raw = np.array(sensordata_raw).T
        sensordata_FWNNinput = windowsignal_2D(sensordata_raw, FW_winlen, overlap_len)

        # GT slope labels
        if 'SS' in train_CSV:
            slopelabel_raw = target_CSV['slopelabel_raw'].__array__()
            slopelabel_raw = np.array(slopelabel_raw).T

            slopelabel_FWNNinput = windowsignal_1D(slopelabel_raw, FW_winlen, overlap_len)
            slopelabel_FWNNinput = slopelabel_FWNNinput[:, -1]

            windowed_delay = int(np.ceil(15/stride_len))

            '''
            Offline Ramp: Some SS strides include transition part from previous mode ---> exclude first 150ms
            Offline Stair, RT Ramp: No need to change SS strides
            '''

            if 'Ramp' in train_CSV and 'Off' in train_CSV:

                if 'SwingExtension' in target_CSV['state'][FW_winlen - 1]:

                    prev_mode = target_CSV['state'][FW_winlen - 1][:2]
                    next_mode = target_CSV['state'][FW_winlen][:2]

                    if prev_mode != next_mode: # target SS strides that include transition
                        sensordata_FWNNinput = sensordata_FWNNinput[windowed_delay:]
                        slopelabel_FWNNinput = slopelabel_FWNNinput[windowed_delay:]

        if 'TR' in train_CSV:

            if MSP == 'baseline':
                slopelabel_target = target_CSV['slopelabel_delayed'].__array__()
                slopelabel_target = np.array(slopelabel_target).T
            else:
                MSPkey = 'slopelabel_smoothed' + str(MSP)
                slopelabel_target = target_CSV[MSPkey].__array__()
                slopelabel_target = np.array(slopelabel_target).T

            slopelabel_FWNNinput = windowsignal_1D(slopelabel_target, FW_winlen, overlap_len)
            slopelabel_FWNNinput = slopelabel_FWNNinput[:, -1]

        if train_total_count == 0:
            train_X = sensordata_FWNNinput
            train_Y = slopelabel_FWNNinput
            train_total_count += 1
            print('Train Data:' + str(train_total_count) + '/' + str(len(train_list)), end='\r')

        else:
            train_X = np.concatenate([train_X, sensordata_FWNNinput], axis=0)
            train_Y = np.concatenate([train_Y, slopelabel_FWNNinput], axis=0)
            train_total_count += 1
            print('Train Data:' + str(train_total_count) + '/' + str(len(train_list)), end='\r')

    # Form test dataset
    test_total_count = 0
    for test_CSV in test_list:

        eachFileDir = TFDir + test_CSV
        target_CSV = pd.read_csv(eachFileDir)  # Load each file

        if 'EarlyStance' not in target_CSV['state'][FW_winlen]:
            print(test_CSV, 'skip this trial')

        # Convert to sensor data array
        sensordata_raw = []
        for sChannel in sensor_list:
            sensordata_raw.append(target_CSV[sChannel].__array__())

        sensordata_raw = np.array(sensordata_raw).T
        sensordata_FWNNinput = windowsignal_2D(sensordata_raw, FW_winlen, overlap_len)

        # GT slope labels
        if 'SS' in test_CSV:
            slopelabel_raw = target_CSV['slopelabel_raw'].__array__()
            slopelabel_raw = np.array(slopelabel_raw).T

            slopelabel_FWNNinput = windowsignal_1D(slopelabel_raw, FW_winlen, overlap_len)
            slopelabel_FWNNinput = slopelabel_FWNNinput[:, -1]

            windowed_delay = int(np.ceil(15 / stride_len))

            '''
            Offline Ramp: Some SS strides include transition part from previous mode ---> exclude first 150ms
            Offline Stair, RT Ramp: No need to change SS strides
            '''

            if 'Ramp' in test_CSV and 'Off' in test_CSV:

                if 'SwingExtension' in target_CSV['state'][FW_winlen - 1]:

                    prev_mode = target_CSV['state'][FW_winlen - 1][:2]
                    next_mode = target_CSV['state'][FW_winlen][:2]

                    if prev_mode != next_mode:  # target SS strides that include transition
                        sensordata_FWNNinput = sensordata_FWNNinput[windowed_delay:]
                        slopelabel_FWNNinput = slopelabel_FWNNinput[windowed_delay:]

        if 'TR' in test_CSV:

            if MSP == 'baseline':
                slopelabel_target = target_CSV['slopelabel_delayed'].__array__()
                slopelabel_target = np.array(slopelabel_target).T
            else:
                MSPkey = 'slopelabel_smoothed' + str(MSP)
                slopelabel_target = target_CSV[MSPkey].__array__()
                slopelabel_target = np.array(slopelabel_target).T

            slopelabel_FWNNinput = windowsignal_1D(slopelabel_target, FW_winlen, overlap_len)
            slopelabel_FWNNinput = slopelabel_FWNNinput[:, -1]

        if test_total_count == 0:
            test_X = sensordata_FWNNinput
            test_Y = slopelabel_FWNNinput
            test_total_count += 1
            print('Test Data:' + str(test_total_count) + '/' + str(len(test_list)), end='\r')

        else:
            test_X = np.concatenate([test_X, sensordata_FWNNinput], axis=0)
            test_Y = np.concatenate([test_Y, slopelabel_FWNNinput], axis=0)
            test_total_count += 1
            print('Test Data:' + str(test_total_count) + '/' + str(len(test_list)), end='\r')

    return train_X, train_Y, test_X, test_Y

def CSV_FWNNinput_varTransition_temp(TFDir, test_ratio=0.2, sensor_list = [], FW_winlen = 125, stride_len = 2, MSP = 'baseline'):
    '''
    Using CSV_SortCount Function, converts CSV files into practical FW NN input
    '''

    # ### Function test - start
    # workspace_path = os.getcwd()
    # TFDir = workspace_path + '/processedCSV_byStride_MSPapplied/FW125/Offline/Stair/TF03/'
    # # TFDir = workspace_path + '/processedCSV_byStride_MSPapplied/FW125/Offline/Ramp/TF03/'
    # test_ratio = 0.2
    # FW_winlen = 125
    # stride_len = 2
    # MSP = 'baseline'
    # # MSP = 0.0
    #
    # # Full Sensor
    # sensor_list = ['ankle_theta', 'ankle_thetadot', 'knee_theta', 'knee_thetadot',
    #               'forceX', 'forceY', 'forceZ', 'momentX', 'momentY', 'momentZ',
    #               'shank_accelX', 'shank_accelY', 'shank_accelZ', 'shank_gyroX', 'shank_gyroY', 'shank_gyroZ',
    #               'foot_accelX', 'foot_accelY', 'foot_accelZ', 'foot_gyroX', 'foot_gyroY', 'foot_gyroZ',
    #               'thigh_accelX', 'thigh_accelY', 'thigh_accelZ', 'thigh_gyroX', 'thigh_gyroY', 'thigh_gyroZ']
    #
    # # # Only with biarticular movements
    # # sensor_list =  ['ankle_theta', 'ankle_thetadot', 'knee_theta', 'knee_thetadot',
    # #               'forceX', 'forceZ', 'momentY',
    # #               'shank_accelX', 'shank_accelY', 'shank_gyroZ',
    # #               'foot_accelX', 'foot_accelZ', 'foot_gyroY',
    # #               'thigh_accelX', 'thigh_accelZ', 'thigh_gyroY']
    # ### Function test - end

    train_list, test_list = CSV_SortCount(TFDir, test_ratio)
    overlap_len = FW_winlen - stride_len

    # Form train dataset
    train_total_count = 0
    for train_CSV in train_list:

        # ### Test
        # train_CSV = 'TF02_RampOffline_preset_4_SS_AS_1_bag28preset_4_slope12.4.csv'
        # train_CSV = 'TF02_RampOffline_preset_1_TR_LW2AS_1_bag38preset_1_slope7.8.csv'
        # ### Test

        eachFileDir = TFDir + train_CSV
        target_CSV = pd.read_csv(eachFileDir)  # Load each file

        if 'EarlyStance' not in target_CSV['state'][FW_winlen]:
            print(train_CSV, 'skip this trial')

        # Convert to sensor data array
        sensordata_raw = []
        for sChannel in sensor_list:
            sensordata_raw.append(target_CSV[sChannel].__array__())

        sensordata_raw = np.array(sensordata_raw).T
        sensordata_FWNNinput = windowsignal_2D(sensordata_raw, FW_winlen, overlap_len)

        # GT slope labels
        if 'SS' in train_CSV:
            slopelabel_raw = target_CSV['slopelabel_raw'].__array__()
            slopelabel_raw = np.array(slopelabel_raw).T

            slopelabel_FWNNinput = windowsignal_1D(slopelabel_raw, FW_winlen, overlap_len)
            slopelabel_FWNNinput = slopelabel_FWNNinput[:, -1]

            windowed_delay = int(np.ceil(10/stride_len))

            '''
            Offline Ramp: Some SS strides include transition part from previous mode ---> exclude first 150ms
            Offline Stair, RT Ramp: No need to change SS strides
            '''

            if 'Ramp' in train_CSV and 'Off' in train_CSV:

                if 'SwingExtension' in target_CSV['state'][FW_winlen - 1]:

                    prev_mode = target_CSV['state'][FW_winlen - 1][:2]
                    next_mode = target_CSV['state'][FW_winlen][:2]

                    if prev_mode != next_mode: # target SS strides that include transition
                        sensordata_FWNNinput = sensordata_FWNNinput[windowed_delay:]
                        slopelabel_FWNNinput = slopelabel_FWNNinput[windowed_delay:]

        if 'TR' in train_CSV:

            if MSP == 'baseline':
                slopelabel_target = target_CSV['slopelabel_delayed'].__array__()
                slopelabel_target = np.array(slopelabel_target).T
            else:
                MSPkey = 'slopelabel_smoothed' + str(MSP)
                slopelabel_target = target_CSV[MSPkey].__array__()
                slopelabel_target = np.array(slopelabel_target).T

            slopelabel_FWNNinput = windowsignal_1D(slopelabel_target, FW_winlen, overlap_len)
            slopelabel_FWNNinput = slopelabel_FWNNinput[:, -1]

        if train_total_count == 0:
            train_X = sensordata_FWNNinput
            train_Y = slopelabel_FWNNinput
            train_total_count += 1
            print('Train Data:' + str(train_total_count) + '/' + str(len(train_list)), end='\r')

        else:
            train_X = np.concatenate([train_X, sensordata_FWNNinput], axis=0)
            train_Y = np.concatenate([train_Y, slopelabel_FWNNinput], axis=0)
            train_total_count += 1
            print('Train Data:' + str(train_total_count) + '/' + str(len(train_list)), end='\r')

    # Form test dataset
    test_total_count = 0
    for test_CSV in test_list:

        eachFileDir = TFDir + test_CSV
        target_CSV = pd.read_csv(eachFileDir)  # Load each file

        if 'EarlyStance' not in target_CSV['state'][FW_winlen]:
            print(test_CSV, 'skip this trial')

        # Convert to sensor data array
        sensordata_raw = []
        for sChannel in sensor_list:
            sensordata_raw.append(target_CSV[sChannel].__array__())

        sensordata_raw = np.array(sensordata_raw).T
        sensordata_FWNNinput = windowsignal_2D(sensordata_raw, FW_winlen, overlap_len)

        # GT slope labels
        if 'SS' in test_CSV:
            slopelabel_raw = target_CSV['slopelabel_raw'].__array__()
            slopelabel_raw = np.array(slopelabel_raw).T

            slopelabel_FWNNinput = windowsignal_1D(slopelabel_raw, FW_winlen, overlap_len)
            slopelabel_FWNNinput = slopelabel_FWNNinput[:, -1]

            windowed_delay = int(np.ceil(10 / stride_len))

            '''
            Offline Ramp: Some SS strides include transition part from previous mode ---> exclude first 150ms
            Offline Stair, RT Ramp: No need to change SS strides
            '''

            if 'Ramp' in test_CSV and 'Off' in test_CSV:

                if 'SwingExtension' in target_CSV['state'][FW_winlen - 1]:

                    prev_mode = target_CSV['state'][FW_winlen - 1][:2]
                    next_mode = target_CSV['state'][FW_winlen][:2]

                    if prev_mode != next_mode:  # target SS strides that include transition
                        sensordata_FWNNinput = sensordata_FWNNinput[windowed_delay:]
                        slopelabel_FWNNinput = slopelabel_FWNNinput[windowed_delay:]

        if 'TR' in test_CSV:

            if MSP == 'baseline':
                slopelabel_target = target_CSV['slopelabel_delayed'].__array__()
                slopelabel_target = np.array(slopelabel_target).T
            else:
                MSPkey = 'slopelabel_smoothed' + str(MSP)
                slopelabel_target = target_CSV[MSPkey].__array__()
                slopelabel_target = np.array(slopelabel_target).T

            slopelabel_FWNNinput = windowsignal_1D(slopelabel_target, FW_winlen, overlap_len)
            slopelabel_FWNNinput = slopelabel_FWNNinput[:, -1]

        if test_total_count == 0:
            test_X = sensordata_FWNNinput
            test_Y = slopelabel_FWNNinput
            test_total_count += 1
            print('Test Data:' + str(test_total_count) + '/' + str(len(test_list)), end='\r')

        else:
            test_X = np.concatenate([test_X, sensordata_FWNNinput], axis=0)
            test_Y = np.concatenate([test_Y, slopelabel_FWNNinput], axis=0)
            test_total_count += 1
            print('Test Data:' + str(test_total_count) + '/' + str(len(test_list)), end='\r')

    return train_X, train_Y, test_X, test_Y











def CSV_FWNNinput_NoLW(TFDir, test_ratio=0.2, sensor_list = [], FW_winlen = 125, stride_len = 2):
    '''
    Using CSV_SortCount Function, converts CSV files into practical FW NN input
    '''

    # ### Function test - start
    # workspace_path = os.getcwd()
    # TFDir = workspace_path + '/processedCSV/FW125/Offline/Ramp/TF02/'
    # test_ratio = 0.2
    # FW_winlen = 125
    # stride_len = 2
    #
    # # Full Sensor
    # sensor_list = ['ankle_theta', 'ankle_thetadot', 'knee_theta', 'knee_thetadot',
    #               'forceX', 'forceY', 'forceZ', 'momentX', 'momentY', 'momentZ',
    #               'shank_accelX', 'shank_accelY', 'shank_accelZ', 'shank_gyroX', 'shank_gyroY', 'shank_gyroZ',
    #               'foot_accelX', 'foot_accelY', 'foot_accelZ', 'foot_gyroX', 'foot_gyroY', 'foot_gyroZ',
    #               'thigh_accelX', 'thigh_accelY', 'thigh_accelZ', 'thigh_gyroX', 'thigh_gyroY', 'thigh_gyroZ']
    #
    # # Only with biarticular movements
    # sensor_list =  ['ankle_theta', 'ankle_thetadot', 'knee_theta', 'knee_thetadot',
    #               'forceX', 'forceZ', 'momentY',
    #               'shank_accelX', 'shank_accelY', 'shank_gyroZ',
    #               'foot_accelX', 'foot_accelZ', 'foot_gyroY',
    #               'thigh_accelX', 'thigh_accelZ', 'thigh_gyroY']
    # ### Function test - end

    train_list, test_list = CSV_SortCount(TFDir, test_ratio)
    overlap_len = FW_winlen - stride_len

    # Form train dataset
    train_total_count = 0
    for train_CSV in train_list:

        if 'SS_LW' in train_CSV:
            continue

        if 'TR_LW' in train_CSV:
            continue

        eachFileDir = TFDir + train_CSV
        target_CSV = pd.read_csv(eachFileDir) # Load each file

        # Convert to sensor data array
        sensordata_raw = []
        for sChannel in sensor_list:
            sensordata_raw.append(target_CSV[sChannel].__array__())

        sensordata_raw = np.array(sensordata_raw).T
        sensordata_FWNNinput = windowsignal_2D(sensordata_raw, FW_winlen, overlap_len)

        # GT slope labels
        slopelabel_raw = []
        for MSPs in target_CSV.keys():
            if 'MSP' not in MSPs:
                continue

            slopelabel_raw.append(target_CSV[MSPs].__array__())

        slopelabel_raw = np.array(slopelabel_raw).T
        slopelabel_FWNNinput = windowsignal_2D(slopelabel_raw, FW_winlen, overlap_len)

        if train_total_count == 0:
            train_X = sensordata_FWNNinput
            train_Y = slopelabel_FWNNinput
            train_total_count += 1
            print('Train Data:' + str(train_total_count) + '/' + str(len(train_list)), end = '\r')

        else:
            train_X = np.concatenate([train_X, sensordata_FWNNinput], axis = 0)
            train_Y = np.concatenate([train_Y, slopelabel_FWNNinput], axis = 0)
            train_total_count += 1
            print('Train Data:' + str(train_total_count) + '/' + str(len(train_list)), end = '\r')

    # Form test dataset
    test_total_count = 0
    for test_CSV in test_list:

        if 'SS_LW' in test_CSV:
            continue

        if 'TR_LW' in test_CSV:
            continue

        eachFileDir = TFDir + test_CSV
        target_CSV = pd.read_csv(eachFileDir) # Load each file

        # Convert to sensor data array
        sensordata_raw = []
        for sChannel in sensor_list:
            sensordata_raw.append(target_CSV[sChannel].__array__())

        sensordata_raw = np.array(sensordata_raw).T
        sensordata_FWNNinput = windowsignal_2D(sensordata_raw, FW_winlen, overlap_len)

        # GT slope labels
        slopelabel_raw = []
        for MSPs in target_CSV.keys():
            if 'MSP' not in MSPs:
                continue

            slopelabel_raw.append(target_CSV[MSPs].__array__())

        slopelabel_raw = np.array(slopelabel_raw).T
        slopelabel_FWNNinput = windowsignal_2D(slopelabel_raw, FW_winlen, overlap_len)

        if test_total_count == 0:
            test_X = sensordata_FWNNinput
            test_Y = slopelabel_FWNNinput
            test_total_count += 1
            print('Test Data:' + str(test_total_count) + '/' + str(len(test_list)), end = '\r')
        else:
            test_X = np.concatenate([test_X, sensordata_FWNNinput], axis = 0)
            test_Y = np.concatenate([test_Y, slopelabel_FWNNinput], axis = 0)
            test_total_count += 1
            print('Test Data:' + str(test_total_count) + '/' + str(len(test_list)), end = '\r')

    return train_X, train_Y, test_X, test_Y

def CSV_FWNNinput_LW_and_ASonly(TFDir, test_ratio=0.2, sensor_list = [], FW_winlen = 125, stride_len = 2):
    '''
    Using CSV_SortCount Function, converts CSV files into practical FW NN input
    '''

    # ### Function test - start
    # workspace_path = os.getcwd()
    # TFDir = workspace_path + '/processedCSV/FW125/Offline/Ramp/TF02/'
    # test_ratio = 0.2
    # FW_winlen = 125
    # stride_len = 2
    #
    # # Full Sensor
    # sensor_list = ['ankle_theta', 'ankle_thetadot', 'knee_theta', 'knee_thetadot',
    #               'forceX', 'forceY', 'forceZ', 'momentX', 'momentY', 'momentZ',
    #               'shank_accelX', 'shank_accelY', 'shank_accelZ', 'shank_gyroX', 'shank_gyroY', 'shank_gyroZ',
    #               'foot_accelX', 'foot_accelY', 'foot_accelZ', 'foot_gyroX', 'foot_gyroY', 'foot_gyroZ',
    #               'thigh_accelX', 'thigh_accelY', 'thigh_accelZ', 'thigh_gyroX', 'thigh_gyroY', 'thigh_gyroZ']
    #
    # # Only with biarticular movements
    # sensor_list =  ['ankle_theta', 'ankle_thetadot', 'knee_theta', 'knee_thetadot',
    #               'forceX', 'forceZ', 'momentY',
    #               'shank_accelX', 'shank_accelY', 'shank_gyroZ',
    #               'foot_accelX', 'foot_accelZ', 'foot_gyroY',
    #               'thigh_accelX', 'thigh_accelZ', 'thigh_gyroY']
    # ### Function test - end

    train_list, test_list = CSV_SortCount(TFDir, test_ratio)
    overlap_len = FW_winlen - stride_len

    # Form train dataset
    train_total_count = 0
    for train_CSV in train_list:

        if 'DS' in train_CSV:
            continue

        eachFileDir = TFDir + train_CSV
        target_CSV = pd.read_csv(eachFileDir) # Load each file

        # Convert to sensor data array
        sensordata_raw = []
        for sChannel in sensor_list:
            sensordata_raw.append(target_CSV[sChannel].__array__())

        sensordata_raw = np.array(sensordata_raw).T
        sensordata_FWNNinput = windowsignal_2D(sensordata_raw, FW_winlen, overlap_len)

        # GT slope labels
        slopelabel_raw = []
        for MSPs in target_CSV.keys():
            if 'MSP' not in MSPs:
                continue

            slopelabel_raw.append(target_CSV[MSPs].__array__())

        slopelabel_raw = np.array(slopelabel_raw).T
        slopelabel_FWNNinput = windowsignal_2D(slopelabel_raw, FW_winlen, overlap_len)

        if train_total_count == 0:
            train_X = sensordata_FWNNinput
            train_Y = slopelabel_FWNNinput
            train_total_count += 1
            print('Train Data:' + str(train_total_count) + '/' + str(len(train_list)), end = '\r')

        else:
            train_X = np.concatenate([train_X, sensordata_FWNNinput], axis = 0)
            train_Y = np.concatenate([train_Y, slopelabel_FWNNinput], axis = 0)
            train_total_count += 1
            print('Train Data:' + str(train_total_count) + '/' + str(len(train_list)), end = '\r')

    # Form test dataset
    test_total_count = 0
    for test_CSV in test_list:

        if 'DS' in test_CSV:
            continue

        eachFileDir = TFDir + test_CSV
        target_CSV = pd.read_csv(eachFileDir) # Load each file

        # Convert to sensor data array
        sensordata_raw = []
        for sChannel in sensor_list:
            sensordata_raw.append(target_CSV[sChannel].__array__())

        sensordata_raw = np.array(sensordata_raw).T
        sensordata_FWNNinput = windowsignal_2D(sensordata_raw, FW_winlen, overlap_len)

        # GT slope labels
        slopelabel_raw = []
        for MSPs in target_CSV.keys():
            if 'MSP' not in MSPs:
                continue

            slopelabel_raw.append(target_CSV[MSPs].__array__())

        slopelabel_raw = np.array(slopelabel_raw).T
        slopelabel_FWNNinput = windowsignal_2D(slopelabel_raw, FW_winlen, overlap_len)

        if test_total_count == 0:
            test_X = sensordata_FWNNinput
            test_Y = slopelabel_FWNNinput
            test_total_count += 1
            print('Test Data:' + str(test_total_count) + '/' + str(len(test_list)), end = '\r')
        else:
            test_X = np.concatenate([test_X, sensordata_FWNNinput], axis = 0)
            test_Y = np.concatenate([test_Y, slopelabel_FWNNinput], axis = 0)
            test_total_count += 1
            print('Test Data:' + str(test_total_count) + '/' + str(len(test_list)), end = '\r')

    return train_X, train_Y, test_X, test_Y

def CSV_FWNNinput_NoLW_ASonly(TFDir, test_ratio=0.2, sensor_list = [], FW_winlen = 125, stride_len = 2):
    '''
    Using CSV_SortCount Function, converts CSV files into practical FW NN input
    '''

    # ### Function test - start
    # workspace_path = os.getcwd()
    # TFDir = workspace_path + '/processedCSV/FW125/Offline/Ramp/TF02/'
    # test_ratio = 0.2
    # FW_winlen = 125
    # stride_len = 2
    #
    # # Full Sensor
    # sensor_list = ['ankle_theta', 'ankle_thetadot', 'knee_theta', 'knee_thetadot',
    #               'forceX', 'forceY', 'forceZ', 'momentX', 'momentY', 'momentZ',
    #               'shank_accelX', 'shank_accelY', 'shank_accelZ', 'shank_gyroX', 'shank_gyroY', 'shank_gyroZ',
    #               'foot_accelX', 'foot_accelY', 'foot_accelZ', 'foot_gyroX', 'foot_gyroY', 'foot_gyroZ',
    #               'thigh_accelX', 'thigh_accelY', 'thigh_accelZ', 'thigh_gyroX', 'thigh_gyroY', 'thigh_gyroZ']
    #
    # # Only with biarticular movements
    # sensor_list =  ['ankle_theta', 'ankle_thetadot', 'knee_theta', 'knee_thetadot',
    #               'forceX', 'forceZ', 'momentY',
    #               'shank_accelX', 'shank_accelY', 'shank_gyroZ',
    #               'foot_accelX', 'foot_accelZ', 'foot_gyroY',
    #               'thigh_accelX', 'thigh_accelZ', 'thigh_gyroY']
    # ### Function test - end

    train_list, test_list = CSV_SortCount(TFDir, test_ratio)
    overlap_len = FW_winlen - stride_len

    # Form train dataset
    train_total_count = 0
    for train_CSV in train_list:

        if 'SS_LW' in train_CSV:
            continue

        if 'TR_LW' in train_CSV:
            continue

        if 'DS' in train_CSV:
            continue

        eachFileDir = TFDir + train_CSV
        target_CSV = pd.read_csv(eachFileDir) # Load each file

        # Convert to sensor data array
        sensordata_raw = []
        for sChannel in sensor_list:
            sensordata_raw.append(target_CSV[sChannel].__array__())

        sensordata_raw = np.array(sensordata_raw).T
        sensordata_FWNNinput = windowsignal_2D(sensordata_raw, FW_winlen, overlap_len)

        # GT slope labels
        slopelabel_raw = []
        for MSPs in target_CSV.keys():
            if 'MSP' not in MSPs:
                continue

            slopelabel_raw.append(target_CSV[MSPs].__array__())

        slopelabel_raw = np.array(slopelabel_raw).T
        slopelabel_FWNNinput = windowsignal_2D(slopelabel_raw, FW_winlen, overlap_len)

        if train_total_count == 0:
            train_X = sensordata_FWNNinput
            train_Y = slopelabel_FWNNinput
            train_total_count += 1
            print('Train Data:' + str(train_total_count) + '/' + str(len(train_list)), end = '\r')

        else:
            train_X = np.concatenate([train_X, sensordata_FWNNinput], axis = 0)
            train_Y = np.concatenate([train_Y, slopelabel_FWNNinput], axis = 0)
            train_total_count += 1
            print('Train Data:' + str(train_total_count) + '/' + str(len(train_list)), end = '\r')

    # Form test dataset
    test_total_count = 0
    for test_CSV in test_list:

        if 'SS_LW' in test_CSV:
            continue

        if 'TR_LW' in test_CSV:
            continue

        if 'DS' in test_CSV:
            continue

        eachFileDir = TFDir + test_CSV
        target_CSV = pd.read_csv(eachFileDir) # Load each file

        # Convert to sensor data array
        sensordata_raw = []
        for sChannel in sensor_list:
            sensordata_raw.append(target_CSV[sChannel].__array__())

        sensordata_raw = np.array(sensordata_raw).T
        sensordata_FWNNinput = windowsignal_2D(sensordata_raw, FW_winlen, overlap_len)

        # GT slope labels
        slopelabel_raw = []
        for MSPs in target_CSV.keys():
            if 'MSP' not in MSPs:
                continue

            slopelabel_raw.append(target_CSV[MSPs].__array__())

        slopelabel_raw = np.array(slopelabel_raw).T
        slopelabel_FWNNinput = windowsignal_2D(slopelabel_raw, FW_winlen, overlap_len)

        if test_total_count == 0:
            test_X = sensordata_FWNNinput
            test_Y = slopelabel_FWNNinput
            test_total_count += 1
            print('Test Data:' + str(test_total_count) + '/' + str(len(test_list)), end = '\r')
        else:
            test_X = np.concatenate([test_X, sensordata_FWNNinput], axis = 0)
            test_Y = np.concatenate([test_Y, slopelabel_FWNNinput], axis = 0)
            test_total_count += 1
            print('Test Data:' + str(test_total_count) + '/' + str(len(test_list)), end = '\r')

    return train_X, train_Y, test_X, test_Y

def CSV_FWNNinput_NoLW_DSonly(TFDir, test_ratio=0.2, sensor_list=[], FW_winlen=125, stride_len=2):
    '''
    Using CSV_SortCount Function, converts CSV files into practical FW NN input
    '''

    # ### Function test - start
    # workspace_path = os.getcwd()
    # TFDir = workspace_path + '/processedCSV/FW125/Offline/Ramp/TF02/'
    # test_ratio = 0.2
    # FW_winlen = 125
    # stride_len = 2
    #
    # # Full Sensor
    # sensor_list = ['ankle_theta', 'ankle_thetadot', 'knee_theta', 'knee_thetadot',
    #               'forceX', 'forceY', 'forceZ', 'momentX', 'momentY', 'momentZ',
    #               'shank_accelX', 'shank_accelY', 'shank_accelZ', 'shank_gyroX', 'shank_gyroY', 'shank_gyroZ',
    #               'foot_accelX', 'foot_accelY', 'foot_accelZ', 'foot_gyroX', 'foot_gyroY', 'foot_gyroZ',
    #               'thigh_accelX', 'thigh_accelY', 'thigh_accelZ', 'thigh_gyroX', 'thigh_gyroY', 'thigh_gyroZ']
    #
    # # Only with biarticular movements
    # sensor_list =  ['ankle_theta', 'ankle_thetadot', 'knee_theta', 'knee_thetadot',
    #               'forceX', 'forceZ', 'momentY',
    #               'shank_accelX', 'shank_accelY', 'shank_gyroZ',
    #               'foot_accelX', 'foot_accelZ', 'foot_gyroY',
    #               'thigh_accelX', 'thigh_accelZ', 'thigh_gyroY']
    # ### Function test - end

    train_list, test_list = CSV_SortCount(TFDir, test_ratio)
    overlap_len = FW_winlen - stride_len

    # Form train dataset
    train_total_count = 0
    for train_CSV in train_list:

        if 'SS_LW' in train_CSV:
            continue

        if 'TR_LW' in train_CSV:
            continue

        if 'AS' in train_CSV:
            continue

        eachFileDir = TFDir + train_CSV
        target_CSV = pd.read_csv(eachFileDir)  # Load each file

        # Convert to sensor data array
        sensordata_raw = []
        for sChannel in sensor_list:
            sensordata_raw.append(target_CSV[sChannel].__array__())

        sensordata_raw = np.array(sensordata_raw).T
        sensordata_FWNNinput = windowsignal_2D(sensordata_raw, FW_winlen, overlap_len)

        # GT slope labels
        slopelabel_raw = []
        for MSPs in target_CSV.keys():
            if 'MSP' not in MSPs:
                continue

            slopelabel_raw.append(target_CSV[MSPs].__array__())

        slopelabel_raw = np.array(slopelabel_raw).T
        slopelabel_FWNNinput = windowsignal_2D(slopelabel_raw, FW_winlen, overlap_len)

        if train_total_count == 0:
            train_X = sensordata_FWNNinput
            train_Y = slopelabel_FWNNinput
            train_total_count += 1
            print('Train Data:' + str(train_total_count) + '/' + str(len(train_list)), end='\r')

        else:
            train_X = np.concatenate([train_X, sensordata_FWNNinput], axis=0)
            train_Y = np.concatenate([train_Y, slopelabel_FWNNinput], axis=0)
            train_total_count += 1
            print('Train Data:' + str(train_total_count) + '/' + str(len(train_list)), end='\r')

    # Form test dataset
    test_total_count = 0
    for test_CSV in test_list:

        if 'SS_LW' in test_CSV:
            continue

        if 'TR_LW' in test_CSV:
            continue

        if 'AS' in test_CSV:
            continue

        eachFileDir = TFDir + test_CSV
        target_CSV = pd.read_csv(eachFileDir)  # Load each file

        # Convert to sensor data array
        sensordata_raw = []
        for sChannel in sensor_list:
            sensordata_raw.append(target_CSV[sChannel].__array__())

        sensordata_raw = np.array(sensordata_raw).T
        sensordata_FWNNinput = windowsignal_2D(sensordata_raw, FW_winlen, overlap_len)

        # GT slope labels
        slopelabel_raw = []
        for MSPs in target_CSV.keys():
            if 'MSP' not in MSPs:
                continue

            slopelabel_raw.append(target_CSV[MSPs].__array__())

        slopelabel_raw = np.array(slopelabel_raw).T
        slopelabel_FWNNinput = windowsignal_2D(slopelabel_raw, FW_winlen, overlap_len)

        if test_total_count == 0:
            test_X = sensordata_FWNNinput
            test_Y = slopelabel_FWNNinput
            test_total_count += 1
            print('Test Data:' + str(test_total_count) + '/' + str(len(test_list)), end='\r')
        else:
            test_X = np.concatenate([test_X, sensordata_FWNNinput], axis=0)
            test_Y = np.concatenate([test_Y, slopelabel_FWNNinput], axis=0)
            test_total_count += 1
            print('Test Data:' + str(test_total_count) + '/' + str(len(test_list)), end='\r')

    return train_X, train_Y, test_X, test_Y