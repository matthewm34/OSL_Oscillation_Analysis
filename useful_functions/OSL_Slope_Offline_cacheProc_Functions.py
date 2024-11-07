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
import joblib as jb

def align2SDat(cachefilename):

    '''
    This Funcion aligns SensorData, Foot IMU & Thigh IMU orientation, and FSM to SensorInfo timestamp
    side: prosthetic side (default: r), if left, run swapleft for some sensordata channels
    '''

    # ### Test
    # workspace_path = os.getcwd()
    # cacheDir = workspace_path + '/Ramp_Data/TF02v2/'  # This gives us a str with the path and the file we are using
    # datalist = os.listdir(cacheDir)  # this gives us a list all the TF's from the Ramp_Data
    # cachefilename = cacheDir + datalist[1]
    # ###Test

    TF_name = 'TF' + os.path.dirname(cachefilename).rsplit('TF')[1]
    TF_name = TF_name[:4]
    infoPath = os.path.dirname(os.path.dirname(cachefilename)) + '/TF_Info.csv'
    TFInfo = pd.read_csv(infoPath)
    TFInfo_idx = np.where(TFInfo.Tfnumber == TF_name)[0]
    TF_Weight = TFInfo.Weight[TFInfo_idx].__array__()[0]
    side = TFInfo.Side[TFInfo_idx].__array__()[0]

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

    rawdat = jb.load(cachefilename)

    target_SensorData = rawdat['/SensorData']
    # target_SensorInfo = rawdat['/SensorInfo'] # Skipping SensorInfo for processing cache files

    try:
        target_FSM = rawdat['/fsm/State'] # No StateGT for offline data

    except:
        pass

    # Set start/end point and crop
    start_time = target_SensorData['header'].array[0]
    end_time = target_SensorData['header'].array[-1]

    x_new = target_SensorData['header'].array

    # Align SensorData
    aligned_SDat = {}

    for SDatKeys in target_SensorData.keys():

        if SDatKeys == 'header':
            aligned_SDat[SDatKeys] = target_SensorData['header']

            continue

        target_to_align = target_SensorData[SDatKeys].array

        # Weight normalization
        if SDatKeys in wNorm:
            target_to_align /= (TF_Weight*9.8)
            # print('Weight Normalizing: ' + SDatKeys)

        if SDatKeys in swapOldSetting:
            if 'accelX' in SDatKeys:
                target_to_align = target_SensorData['shank_accelY']
                # print(SDatKeys)
            elif 'accelY' in SDatKeys:
                target_to_align = target_SensorData['shank_accelX']
                # print(SDatKeys)
            elif 'gyroX' in SDatKeys:
                target_to_align = target_SensorData['shank_gyroY']
                # print(SDatKeys)
            elif 'gyroY' in SDatKeys:
                target_to_align = -target_SensorData['shank_gyroX']
                # print(SDatKeys)

        # SwapLeft
        if side == 'L':
            # print('Prosthesis Side: Left')

            if SDatKeys in swapLeft:
                target_to_align = target_to_align * (-1)
                # print('Swapping to the left: ' + SDatKeys)

        aligned_SDat[SDatKeys] = target_to_align

    aligned_SDat = pd.DataFrame.from_dict(aligned_SDat)

    # Align FSM to SensorInfo
    aligned_FSM = target_FSM.copy()
    for i in range(0, len(aligned_FSM)):
        aligned_FSM_newheader = closest_point(x_new, aligned_FSM['header'][i])
        aligned_FSM_newheader = x_new[aligned_FSM_newheader]

        aligned_FSM['header'][i] = aligned_FSM_newheader

    return aligned_SDat, aligned_FSM

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
                # print(ii, phase)

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
        # target_FSM_state[first_ES_idx + 4] = 'LW_EarlyStance'

        try:  # If ML statement is included in the FSM state
            MLidx = np.where(target_FSM_state == 'ML')[0]

            target_FSM_timeval_MLdropped = target_FSM_timeval.drop(MLidx)
            target_FSM_state_MLdropped = target_FSM_state.drop(MLidx)

            # Below is required to remove empty idx (dropped idx still remains)
            target_FSM_timeval_MLdropped = np.array(target_FSM_timeval_MLdropped)
            target_FSM_state_MLdropped = np.array(target_FSM_state_MLdropped)

            FSM_total_MLdropped = {'header': target_FSM_timeval_MLdropped, 'state': target_FSM_state_MLdropped}
            FSM_total = pd.DataFrame(data=FSM_total_MLdropped)

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

                FSM_total = pd.DataFrame(data=FSM_total_beforeHomeDropped)

                target_FSM_timeval = FSM_total['header']
                target_FSM_state = FSM_total['state']

        except:
            pass

        # Remove repeated states
        for s in range(0, len(target_FSM_state) - 1):
            if target_FSM_state[s] == target_FSM_state[s + 1]:
                FSM_total = FSM_total.drop(s + 1)
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

class alignedCache_Ramp_OldSetting:

    ### OldSetting means transition LW->RA takes place in EarlyStance
    ### Later settings LW->RA takes place in SwingFlexion

    # def __init__(self, bagfilename, slope, startMSP = 0, endMSP = 1, plot = True):
    def __init__(self, cachefilename, plot=True):

        ### Test
        workspace_path = os.getcwd()
        cacheDir = workspace_path + '/Ramp_Data/TF02v2/'  # This gives us a str with the path and the file we are using
        datalist = os.listdir(cacheDir)  # this gives us a list all the TF's from the Ramp_Data
        cachefilename = cacheDir + datalist[0]
        ### Test

        aligned_SDat, aligned_FSM = align2SDat(cachefilename)

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
        3) LWSE->RAES   
        '''
        # This is version 3)
        # transition_idx_total = np.where(np.diff(FSM.time_idx_ES) == 1)[0]  # Where ES->ES happens (LWES<->RA/RDES)
        transition_idx_total = []
        for t in range(0, len(FSM.FSM_total) - 1):
            state_current = FSM.FSM_total['state'][t]
            state_next = FSM.FSM_total['state'][t+1]

            mode_current = state_current[:2]
            mode_next = state_next[:2]

            if mode_current == mode_next:
                continue
            else:
                if 'EarlyStance' in state_next:
                    # print(state_next)
                    transition_idx_total.append(t+1)

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

            # trid = transition_idx_total[i]  # index in time_idx_ES matrix
            trid_in_FSM_total = transition_idx_total[i]  # index in FSM_total

            if 'RA' in FSM.FSM_total['state'][trid_in_FSM_total]:
                # print(trid_in_FSM_total, ': LW2RA transition')
                FSM_idx_LW2RA_end.append(trid_in_FSM_total)

            elif 'RD' in FSM.FSM_total['state'][trid_in_FSM_total]:
                # print(trid_in_FSM_total, ': LW2RD transition')
                FSM_idx_LW2RD_end.append(trid_in_FSM_total)

            else:
                if 'RA' in FSM.FSM_total['state'][trid_in_FSM_total-1]:
                    # print(trid_in_FSM_total, ': RA2LW transition')
                    FSM_idx_RA2LW_end.append(trid_in_FSM_total)

                elif 'RD' in FSM.FSM_total['state'][trid_in_FSM_total-1]:
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

        aligned_SDat = aligned_SDat[input_startidx:input_endidx]

        transition_counts = int(len(transition_idx_total_Sensor['LW2AS']) / 2)

        # Store to this class
        self.sensor_data = aligned_SDat
        self.FSM = FSM
        self.transition_idx_Sensor = transition_idx_total_Sensor
        self.transition_counts = transition_counts

class Ramp_GT_byMSP_OldSetting:

    ### OldSetting means transition LW->RA takes place in EarlyStance
    ### Later settings LW->RA takes place in SwingFlexion

    def __init__(self, cachefilename, slope, startMSP_list = [], endMSP_list = []):

        # ### Test
        # workspace_path = os.getcwd()
        # cacheDir = workspace_path + '/Ramp_Data/TF16v2/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(cacheDir)  # this gives us a list all the TF's from the Ramp_Data
        # cachefilename = cacheDir + datalist[1]
        #
        # slope = 19.6
        # startMSP_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        # endMSP_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        # ### Test

        aligned_bag = alignedCache_Ramp_OldSetting(cachefilename, plot = False)
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

class separation_SSTR_Ramp_OldSetting:
    def __init__(self, cachefilename, plot = False, slope = 0, startMSP_list = [], endMSP_list = [], FWwinlen = 125):

        # ### Test
        # workspace_path = os.getcwd()
        # cacheDir = workspace_path + '/Ramp_Data/TF02v2/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(cacheDir)  # this gives us a list all the TF's from the Ramp_Data
        # cachefilename = cacheDir + datalist[1]
        #
        # slope = 19.6
        # startMSP_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        # endMSP_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        # FWwinlen = 125
        # plot = False
        # ### Test

        testload = alignedCache_Ramp_OldSetting(cachefilename, plot = False)
        testload = testload.__dict__

        GT_slope_dict = Ramp_GT_byMSP_OldSetting(cachefilename, slope, startMSP_list = startMSP_list, endMSP_list = endMSP_list).GT_slope_dict
        GT_slope_df = pd.DataFrame.from_dict(GT_slope_dict)

        transition_counts = testload['transition_counts']
        dict_to_df = df_rearrange_header(testload['sensor_data'])

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

class alignedCache_Ramp_OldSetting_v2:

    ### OldSetting means transition LW->RA takes place in EarlyStance
    ### Later settings LW->RA takes place in SwingFlexion

    # def __init__(self, bagfilename, slope, startMSP = 0, endMSP = 1, plot = True):
    def __init__(self, cachefilename, FWwinlen, plot=True):

        # ### Test
        # workspace_path = os.getcwd()
        # cacheDir = workspace_path + '/Ramp_Data/TF02v2/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(cacheDir)  # this gives us a list all the TF's from the Ramp_Data
        # cachefilename = cacheDir + datalist[0]
        # ### Test

        aligned_SDat, aligned_FSM = align2SDat(cachefilename)

        ##### Align time segment to State
        FSM = FSMinfo(aligned_FSM)
        target_sensor_timeval = aligned_SDat['header']

        input_starttime = FSM.time_val_ES[0]
        input_startidx = closest_point(target_sensor_timeval, input_starttime)
        input_endtime = FSM.time_val_ES[-1]
        input_endidx = closest_point(target_sensor_timeval, input_endtime)

        aligned_SDat = df_rearrange_header(aligned_SDat[input_startidx:input_endidx + 1])

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

        dict_to_df = pd.merge(aligned_SDat, state_dict_df_noHeader, left_index=True, right_index=True)
        dict_to_df = df_rearrange_header(dict_to_df)

        # Store to this class
        self.sensor_data = aligned_SDat
        self.FSM = FSM
        self.dict_arranged = dict_to_df

class separation_stride_Ramp_OldSetting_v2:
    def __init__(self, csvfilename, plot = False, slope = 0, FWwinlen = 125, delay = 15):

        # ### Test
        # workspace_path = os.getcwd()
        # csvDir = workspace_path + '/processedCSV_RampDat_byBagfile/Offline/TF02/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(csvDir)
        # csvfilename = csvDir + datalist[0]
        # slope = 10.8
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
            stance_end_idx = stance_end_or_swing_partition_idx[
                target_segment_dict['state_num'][stance_end_or_swing_partition_idx] == 1]

            stance_start_raw = 0  # Initialize
            if target_segment_dict['state_num'][0] == 1:  # If selected segment begin with stance
                stance_start_raw = np.append(np.array([0]), swing_end_idx + 1)
            else:
                stance_start_raw = swing_end_idx + 1

            # crop stride range: check if selected stride includes transition
            for s in range(0, len(stance_start_raw) - 1):
                # s = 0

                stride_start_idx = stance_start_raw[s]
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

        # ### Test
        # workspace_path = os.getcwd()
        # csvDir = workspace_path + '/processedCSV_RampDat_byBagfile/Offline/TF02/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(csvDir)
        # csvfilename = csvDir + datalist[0]
        # slope = 10.8
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
            stance_end_idx = stance_end_or_swing_partition_idx[
                target_segment_dict['state_num'][stance_end_or_swing_partition_idx] == 1]

            stance_start_raw = 0  # Initialize
            if target_segment_dict['state_num'][0] == 1:  # If selected segment begin with stance
                stance_start_raw = np.append(np.array([0]), swing_end_idx + 1)
            else:
                stance_start_raw = swing_end_idx + 1

            # crop stride range: check if selected stride includes transition
            for s in range(0, len(stance_start_raw) - 1):
                # s = 0

                stride_start_idx = stance_start_raw[s]
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

            if sep_df_aligned['state'][FWwinlen][:2] != sep_df_aligned['state'][FWwinlen-1][:2]:
                sep_dict_TR.append(sep_df_aligned)

        self.dict_result = sep_dict_TR

class separation_stride_Ramp_OldSetting_v2_forTRtest_temp:
    def __init__(self, csvfilename, plot = False, slope = 0, FWwinlen = 125, delay = 10):

        # ### Test
        # workspace_path = os.getcwd()
        # csvDir = workspace_path + '/processedCSV_RampDat_byBagfile/Offline/TF02/'  # This gives us a str with the path and the file we are using
        # datalist = os.listdir(csvDir)
        # csvfilename = csvDir + datalist[0]
        # slope = 10.8
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
            stance_end_idx = stance_end_or_swing_partition_idx[
                target_segment_dict['state_num'][stance_end_or_swing_partition_idx] == 1]

            stance_start_raw = 0  # Initialize
            if target_segment_dict['state_num'][0] == 1:  # If selected segment begin with stance
                stance_start_raw = np.append(np.array([0]), swing_end_idx + 1)
            else:
                stance_start_raw = swing_end_idx + 1

            # crop stride range: check if selected stride includes transition
            for s in range(0, len(stance_start_raw) - 1):
                # s = 0

                stride_start_idx = stance_start_raw[s]
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

            if sep_df_aligned['state'][FWwinlen][:2] != sep_df_aligned['state'][FWwinlen-1][:2]:
                sep_dict_TR.append(sep_df_aligned)

        self.dict_result = sep_dict_TR
