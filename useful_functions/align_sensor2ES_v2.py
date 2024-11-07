import numpy as np
import joblib as jb
import matplotlib.pyplot as plt ### For check
import pandas as pd
import math
from useful_functions import AHRS_Functions as AHRS
from scipy.signal import butter as bt
import scipy.signal as sig
from scipy.interpolate import interp1d as intp
import os

def closest_point(target_array, lookup_value):
    '''
    Useful function to point out the closest point in timestamp to specific timing
    '''

    mindiff = np.min(np.abs(target_array - lookup_value))
    lookup_idx = np.where(np.abs(target_array - lookup_value) == mindiff)
    lookup_idx = lookup_idx[0][0]

    return lookup_idx

def timealign(startstamp, endstamp, target_data):
    '''
    Aligns 'target_data' timestamp to designated start and end timestamp
    Use this to align sensor data to specific phases
    target_data should include 'header' as time stamp history
    '''

    time_val = target_data['header'].array

    time_idx_start = closest_point(time_val, startstamp)
    time_idx_end = closest_point(time_val, endstamp)

    # Crop data
    cropped_data = target_data[time_idx_start:time_idx_end + 1]

    return cropped_data

def sensor2phase(loadfileDir):
    '''

    '''
    rawdatdict = jb.load(loadfileDir)

    sensor_phase_dict = {}
    for TF in rawdatdict.keys():
        target_TFdata = rawdatdict[TF]

        sensor_phase_dict[TF] = {}

        for slope in target_TFdata.keys():
            target_slopedata = target_TFdata[slope]

            sensor_phase_dict[TF][slope] = {}

            ### Break down cases
            sensor_phase_dict[TF][slope]['LW2LW'] = []
            sensor_phase_dict[TF][slope]['LW2RA'] = []
            sensor_phase_dict[TF][slope]['LW2RD'] = []

            sensor_phase_dict[TF][slope]['RA2RA'] = []
            sensor_phase_dict[TF][slope]['RA2LW'] = []

            sensor_phase_dict[TF][slope]['RD2RD'] = []
            sensor_phase_dict[TF][slope]['RD2LW'] = []

            for bagnum in target_slopedata.keys():
                target_data = target_slopedata[bagnum]

                ##### Align time segment to State
                sensordata = target_data['sensordata']
                time_val_sensor = sensordata['header']
                state = target_data['state']
                time_val_state = state['header']

                time_idx_ES = []
                time_idx_LS = []
                time_idx_SE = []
                time_idx_SF = []

                i = 0
                for phase in state['state']:
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

                time_val_ES = time_val_state[time_idx_ES].array
                time_val_LS = time_val_state[time_idx_LS].array
                time_val_SE = time_val_state[time_idx_SE].array
                time_val_SF = time_val_state[time_idx_SF].array

                time_idx_RAHS = []  ### Ramp Ascend Heel Strike
                time_idx_RDHS = []  ### Ramp Descend Heel Strike
                time_idx_LWHS = []  ### Level Walking Heel Strike

                for j in range(0, len(time_idx_ES)):
                    if 'RA' in state['state'][time_idx_ES[j]]:
                        time_idx_RAHS.append(time_idx_ES[j])

                    if 'RD' in state['state'][time_idx_ES[j]]:
                        time_idx_RDHS.append(time_idx_ES[j])

                    if 'LW' in state['state'][time_idx_ES[j]]:
                        time_idx_LWHS.append(time_idx_ES[j])

                time_val_RAHS = time_val_state[time_idx_RAHS].array
                time_val_RDHS = time_val_state[time_idx_RDHS].array
                time_val_LWHS = time_val_state[time_idx_LWHS].array

                for k in range(0, len(time_idx_ES)-1):
                    start_idx = time_idx_ES[k]

                    if start_idx in time_idx_LWHS:
                        start_mode = 'LW'
                    elif start_idx in time_idx_RAHS:
                        start_mode = 'RA'
                    elif start_idx in time_idx_RDHS:
                        start_mode = 'RD'

                    end_idx = time_idx_ES[k+1]

                    if end_idx in time_idx_LWHS:
                        end_mode = 'LW'
                    elif end_idx in time_idx_RAHS:
                        end_mode = 'RA'
                    elif end_idx in time_idx_RDHS:
                        end_mode = 'RD'

                    label = start_mode + '2' + end_mode

                    time_val_state_start = time_val_state[start_idx]
                    time_val_state_end = time_val_state[end_idx]
                    time_idx_sensor_start = closest_point(time_val_sensor, time_val_state_start)
                    time_idx_sensor_end = closest_point(time_val_sensor, time_val_state_end)
                    interval = time_idx_sensor_end-time_idx_sensor_start

                    original_window = np.arange(0, interval+1)/interval
                    gaitphase = np.arange(0,101)/100

                    signal2phase = {}
                    for sensor in sensordata.keys():
                        if sensor == 'header':
                            continue

                        target_signal_to_crop = sensordata[sensor]
                        cropped_signal = target_signal_to_crop[time_idx_sensor_start:time_idx_sensor_end+1]

                        f = intp(original_window, cropped_signal)

                        signal2phase[sensor] = f(gaitphase)

                    sensor_phase_dict[TF][slope][label].append(signal2phase)

    return sensor_phase_dict

def sensor2ES_LW(target_data_dict):
    '''
    You can either put .dict data directory, or dataframe dictionary
    This function returns not only aligned sensor data, but also aligned Stance Phase time stamps
    * Stance:1, Swing:0
    '''

    if target_data_dict.__class__ == str:
        target_rawdat = jb.load(target_data_dict)
    elif target_data_dict.__class__ == dict:
        target_rawdat = target_data_dict
    else:
        print('Error! Check the input type (Directory or Dictionary)')

    target_sensordat = target_rawdat['/SensorData']

    ##### Align time segment to State
    time_val = target_rawdat['/fsm/State']['header']
    state = target_rawdat['/fsm/State']['state']

    time_idx_ES = []
    time_idx_LS = []
    time_idx_SE = []
    time_idx_SF = []

    i = 0
    for phase in state:
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

    time_val_ES = time_val[time_idx_ES].array
    time_val_LS = time_val[time_idx_LS].array
    time_val_SE = time_val[time_idx_SE].array
    time_val_SF = time_val[time_idx_SF].array

    if len(time_val_ES) > len(time_val_LS):
        print('ES and LS have different length: Erasing ' ,len(time_val_ES) - len(time_val_LS), 'Element...')
        time_val_ES = time_val_ES[:-1] ### Erase last element

    target_sensor_timeval = target_sensordat['header']

    stationary_foot = {}
    stationary_foot['header'] = target_sensor_timeval
    stationary_history = np.zeros(len(target_sensor_timeval)) + 1

    ### Find ES and LS (Stance Phases) time indices in sensor timestamps
    for i in range(0,len(time_val_ES)):
        sensortime_idx_ES = closest_point(target_sensor_timeval, time_val_ES[i])
        sensortime_idx_SF = closest_point(target_sensor_timeval, time_val_SF[i])

        stationary_history[sensortime_idx_ES:sensortime_idx_SF] = 0

    stationary_foot['classifier'] = stationary_history

    df = pd.DataFrame(data=stationary_foot)

    time_val_start = time_val_ES[1] ### Removing First step, can change to 0 if you want include the first step
    time_val_end = time_val_ES[-1]

    sensordat_aligned = timealign(time_val_start, time_val_end, target_sensordat)
    stationary_history_aligned = timealign(time_val_start, time_val_end, df)

    return sensordat_aligned, stationary_history_aligned

def phasefilter_Stance(state_dataframe, time_vector, ES_offset = 20):
    '''
    state_dataframe: Time stamp value vs FSM state provided by cache file
    time_vector: Target time frame that you want to align FSM state with (Usually Sensordata timestamp)
    ES_offset: Since the foot is not probably flat at the moment of ES, flat foot stance starts from ES + ES_offset
       (Default: 20 points = 200 ms)
    '''

    ### Function Check
    # workspace_path = os.getcwd()
    # loadfileDir = workspace_path + '/OSL_Slope_SensorDat_FootThighIMU_v2'
    # ES_offset = 20
    # rawdatdict = jb.load(loadfileDir)
    #
    # ### Test
    # state_dataframe = rawdatdict['TF02v2'][7.8]['38']['state']
    # time_vector = np.array(rawdatdict['TF02v2'][7.8]['38']['sensordata']['header'])
    #
    # del rawdatdict


    state = state_dataframe['state']
    statetime = state_dataframe['header']

    time_idx_ES = []
    time_idx_LS = []
    time_idx_SE = []
    time_idx_SF = []

    i = 0
    for phase in state:
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

    time_val_ES = statetime[time_idx_ES].array
    time_val_LS = statetime[time_idx_LS].array
    # time_val_SE = statetime[time_idx_SE].array # for later use
    # time_val_SF = statetime[time_idx_SF].array

    phasefilter_stance = {}
    phasefilter_stance['header'] = time_vector
    fullstance = np.zeros(len(time_vector))
    flatfoot = np.zeros(len(time_vector))

    ### Find ES and LS (Stance Phases) time indices in sensor timestamps
    for i in range(0,len(time_val_ES)):
        if i < len(time_val_ES) - 1:
            sensortime_idx_ES = closest_point(time_vector, time_val_ES[i])
            sensortime_idx_LS = closest_point(time_vector, time_val_LS[i])

            if i == 0:
                fullstance[:sensortime_idx_LS] = 1
                flatfoot[:sensortime_idx_LS] = 1

                start_time_idx = sensortime_idx_ES ### First ES timing w.r.t sensordata timeframe

            fullstance[sensortime_idx_ES:sensortime_idx_LS] = 1
            flatfoot[sensortime_idx_ES + ES_offset:sensortime_idx_LS] = 1

        else:
            sensortime_idx_ES = closest_point(time_vector, time_val_ES[i])

            end_time_idx = sensortime_idx_ES  ### Last ES timing w.r.t sensordata timeframe

            fullstance[sensortime_idx_ES:] = 1
            flatfoot[sensortime_idx_ES + ES_offset:] = 1

    phasefilter_stance['flatfoot'] = flatfoot
    phasefilter_stance['fullstance'] = fullstance
    phasefilter_stance['start_time_idx'] = start_time_idx
    phasefilter_stance['end_time_idx'] = end_time_idx

    df = pd.DataFrame(data=phasefilter_stance)

    return df

def Ramp_GT(target_data_dict, ref_signal_topic):
    '''
    Interpolate Grount Truth (GT) slope values aligned with sensor timestamp
    * Ascent:1, Descent:-1, LW:0

    Transition point: ES

    ref_signal_topic: enter the topic that you will use for reference
    ex: '/SensorData'
    '''

    if target_data_dict.__class__ == str:
        target_rawdata = jb.load(target_data_dict)
    elif target_data_dict.__class__ == dict:
        target_rawdata = target_data_dict
    else:
        print('Error! Check the input type (Directory or Dictionary)')

    ### Function test
    # workspace_path = os.getcwd()
    # cacheDir = workspace_path + '/Ramp_Data/'
    #
    # loadDir = cacheDir + 'TF02v2/'  ### Put your Cache file directory here
    # loadfileDir = loadDir + '23'
    # target_rawdata = jb.load(loadfileDir)
    # ref_signal_topic = str('/SensorData')
    ### Function test

    target_sensordata = target_rawdata[ref_signal_topic]
    target_sensor_timeval = target_sensordata['header']

    ##### Align time segment to State
    time_val = target_rawdata['/fsm/State']['header']
    state = target_rawdata['/fsm/State']['state']

    time_idx_ES = []
    time_idx_LS = []
    time_idx_SE = []
    time_idx_SF = []

    i = 0
    for phase in state:
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

    time_idx_RAHS = []  ### Ramp Ascend Heel Strike
    time_idx_RDHS = []  ### Ramp Descend Heel Strike
    time_idx_LWHS = []  ### Level Walking Heel Strike

    for j in range(0, len(time_idx_ES)):
        if 'RA' in state[time_idx_ES[j]]:
            time_idx_RAHS.append(time_idx_ES[j])

        if 'RD' in state[time_idx_ES[j]]:
            time_idx_RDHS.append(time_idx_ES[j])

        if 'LW' in state[time_idx_ES[j]]:
            time_idx_LWHS.append(time_idx_ES[j])

    time_val_RAHS = time_val[time_idx_RAHS].array
    time_val_RDHS = time_val[time_idx_RDHS].array
    time_val_LWHS = time_val[time_idx_LWHS].array

    ### Late Stance Timings
    time_idx_RALS = []  ### Ramp Ascend Late Stance
    time_idx_RDLS = []  ### Ramp Descend Late Stance
    time_idx_LWLS = []  ### Level Walking Late Stance

    for j in range(0, len(time_idx_LS)):
        if 'RA' in state[time_idx_LS[j]]:
            time_idx_RALS.append(time_idx_LS[j])

        if 'RD' in state[time_idx_LS[j]]:
            time_idx_RDLS.append(time_idx_LS[j])

        if 'LW' in state[time_idx_LS[j]]:
            time_idx_LWLS.append(time_idx_LS[j])

    time_val_RALS = time_val[time_idx_RALS].array
    time_val_RDLS = time_val[time_idx_RDLS].array
    time_val_LWLS = time_val[time_idx_LWLS].array

    if len(time_val_RDHS) < 1: ### If there is no ramp descent
        return

    ground_truth = np.zeros([len(target_sensordata)])

    ### Find Transition time indices (RA->LW, RD->LW)
    transitionIdx_RA_to_LW = closest_point(time_val_LWHS, time_val_RAHS[-1])
    transitionIdx_RD_to_LW = closest_point(time_val_LWHS, time_val_RDHS[-1])
    ### LW->RA and LW->RD transition timings are time_val_RAHS[0] and time_val_RDHS[0] respectively

    ### To compare estimation performance during transition, LS(LW)~LS(Ramp) and LS(Ramp)~LS(LW) is required
    transitionLS_startIdx_LW_to_RA = closest_point(time_val_LWLS, time_val_RAHS[0])
    transitionLS_startIdx_LW_to_RD = closest_point(time_val_LWLS, time_val_RDHS[0])
    transitionLS_endIdx_RA_to_LW = closest_point(time_val_LWLS, time_val_RAHS[-1])
    transitionLS_endIdx_RD_to_LW = closest_point(time_val_LWLS, time_val_RDHS[-1])

    ### Transition timings (RA->LW, RD->LW)
    transition_time_RA_to_LW = time_val_LWHS[transitionIdx_RA_to_LW]
    transition_time_RD_to_LW = time_val_LWHS[transitionIdx_RD_to_LW]
    ### LW->RA and LW->RD transition timings are time_val_RAHS[0] and time_val_RDHS[0] respectively

    sensortime_transition_timeIdx_LW_to_RA = closest_point(target_sensor_timeval,
                                                           time_val_RAHS[0])
    sensortime_transition_timeIdx_RA_to_LW = closest_point(target_sensor_timeval,
                                                           transition_time_RA_to_LW)
    sensortime_transition_timeIdx_LW_to_RD = closest_point(target_sensor_timeval,
                                                           time_val_RDHS[0])
    sensortime_transition_timeIdx_RD_to_LW = closest_point(target_sensor_timeval,
                                                           transition_time_RD_to_LW)

    transitionLS_starttime_LW_to_RA = time_val_LWLS[transitionLS_startIdx_LW_to_RA]
    transitionLS_starttime_RA_to_LW = time_val_RALS[-1]
    transitionLS_starttime_LW_to_RD = time_val_LWLS[transitionLS_startIdx_LW_to_RD]
    transitionLS_starttime_RD_to_LW = time_val_RDLS[-1]

    transitionLS_endtime_LW_to_RA = time_val_RALS[0]
    transitionLS_endtime_RA_to_LW = time_val_LWLS[transitionLS_endIdx_RA_to_LW]
    transitionLS_endtime_LW_to_RD = time_val_RDLS[0]
    transitionLS_endtime_RD_to_LW = time_val_LWLS[transitionLS_endIdx_RD_to_LW]

    ### LS~LS intervals
    sensortimeLS_transition_startIdx_LW_to_RA = closest_point(target_sensor_timeval,
                                                              transitionLS_starttime_LW_to_RA)
    sensortimeLS_transition_endIdx_LW_to_RA = closest_point(target_sensor_timeval,
                                                            transitionLS_endtime_LW_to_RA)

    sensortimeLS_transition_startIdx_RA_to_LW = closest_point(target_sensor_timeval,
                                                              transitionLS_starttime_RA_to_LW)
    sensortimeLS_transition_endIdx_RA_to_LW = closest_point(target_sensor_timeval,
                                                            transitionLS_endtime_RA_to_LW)

    sensortimeLS_transition_startIdx_LW_to_RD = closest_point(target_sensor_timeval,
                                                              transitionLS_starttime_LW_to_RD)
    sensortimeLS_transition_endIdx_LW_to_RD = closest_point(target_sensor_timeval,
                                                            transitionLS_endtime_LW_to_RD)

    sensortimeLS_transition_startIdx_RD_to_LW = closest_point(target_sensor_timeval,
                                                              transitionLS_starttime_RD_to_LW)
    sensortimeLS_transition_endIdx_RD_to_LW = closest_point(target_sensor_timeval,
                                                            transitionLS_endtime_RD_to_LW)

    transitionLS_idx = [sensortimeLS_transition_startIdx_LW_to_RA,
                        sensortimeLS_transition_endIdx_LW_to_RA,
                        sensortimeLS_transition_startIdx_RA_to_LW,
                        sensortimeLS_transition_endIdx_RA_to_LW,
                        sensortimeLS_transition_startIdx_LW_to_RD,
                        sensortimeLS_transition_endIdx_LW_to_RD,
                        sensortimeLS_transition_startIdx_RD_to_LW,
                        sensortimeLS_transition_endIdx_RD_to_LW]

    ground_truth[sensortime_transition_timeIdx_LW_to_RA:sensortime_transition_timeIdx_RA_to_LW] = 1
    ground_truth[sensortime_transition_timeIdx_LW_to_RD:sensortime_transition_timeIdx_RD_to_LW] = -1

    ## Check
    # plt.plot(ground_truth[sensortimeLS_transition_startIdx_LW_to_RA:sensortimeLS_transition_endIdx_LW_to_RA])

    return ground_truth, transitionLS_idx

def Ramp_GT2(target_data_dict, ref_signal_topic): ### Previous version
    '''
    Interpolate Grount Truth (GT) slope values aligned with sensor timestamp
    * Ascent:1, Descent:-1, LW:0

    ref_signal_topic: enter the topic that you will use for reference
    ex: '/SensorData'

    Transition: linearly smoothed between SF-HS in each transition
    '''

    if target_data_dict.__class__ == str:
        target_rawdata = jb.load(target_data_dict)
    elif target_data_dict.__class__ == dict:
        target_rawdata = target_data_dict
    else:
        print('Error! Check the input type (Directory or Dictionary)')


    ### Function test
    # workspace_path = os.getcwd()
    # cacheDir = workspace_path + '/Ramp_Data/'
    #
    # loadDir = cacheDir + 'TF02v2/'  ### Put your Cache file directory here
    # loadfileDir = loadDir + '41'
    # target_rawdata = jb.load(loadfileDir)
    # ref_signal_topic = str('/SensorData')
    ### Function test


    target_sensordata = target_rawdata[ref_signal_topic]
    target_sensor_timeval = target_sensordata['header']

    ##### Align time segment to State
    time_val = target_rawdata['/fsm/State']['header']
    state = target_rawdata['/fsm/State']['state']

    time_idx_ES = []
    time_idx_LS = []
    time_idx_SE = []
    time_idx_SF = []

    i = 0
    for phase in state:
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

    ### Heel Strike Timings
    time_idx_RAHS = []  ### Ramp Ascend Heel Strike
    time_idx_RDHS = []  ### Ramp Descend Heel Strike
    time_idx_LWHS = []  ### Level Walking Heel Strike

    for j in range(0, len(time_idx_ES)):
        if 'RA' in state[time_idx_ES[j]]:
            time_idx_RAHS.append(time_idx_ES[j])

        if 'RD' in state[time_idx_ES[j]]:
            time_idx_RDHS.append(time_idx_ES[j])

        if 'LW' in state[time_idx_ES[j]]:
            time_idx_LWHS.append(time_idx_ES[j])

    time_val_RAHS = time_val[time_idx_RAHS].array
    time_val_RDHS = time_val[time_idx_RDHS].array
    time_val_LWHS = time_val[time_idx_LWHS].array

    ### Swing Flexion Timings
    time_idx_RASF = []  ### Ramp Ascend Swing Flexion
    time_idx_RDSF = []  ### Ramp Descend Swing Flexion
    time_idx_LWSF = []  ### Level Walking Swing Flexion

    for j in range(0, len(time_idx_SF)):
        if 'RA' in state[time_idx_SF[j]]:
            time_idx_RASF.append(time_idx_SF[j])

        if 'RD' in state[time_idx_SF[j]]:
            time_idx_RDSF.append(time_idx_SF[j])

        if 'LW' in state[time_idx_SF[j]]:
            time_idx_LWSF.append(time_idx_SF[j])

    time_val_RASF = time_val[time_idx_RASF].array
    time_val_RDSF = time_val[time_idx_RDSF].array
    time_val_LWSF = time_val[time_idx_LWSF].array

    ### Late Stance Timings
    time_idx_RALS = []  ### Ramp Ascend Late Stance
    time_idx_RDLS = []  ### Ramp Descend Late Stance
    time_idx_LWLS = []  ### Level Walking Late Stance

    for j in range(0, len(time_idx_LS)):
        if 'RA' in state[time_idx_LS[j]]:
            time_idx_RALS.append(time_idx_LS[j])

        if 'RD' in state[time_idx_LS[j]]:
            time_idx_RDLS.append(time_idx_LS[j])

        if 'LW' in state[time_idx_LS[j]]:
            time_idx_LWLS.append(time_idx_LS[j])

    time_val_RALS = time_val[time_idx_RALS].array
    time_val_RDLS = time_val[time_idx_RDLS].array
    time_val_LWLS = time_val[time_idx_LWLS].array
    time_val_LS = time_val[time_idx_LS].array
    time_val_SF = time_val[time_idx_SF].array

    if len(time_val_RDHS) < 1: ### If there is no ramp descent, It's LW trial
        ground_truth = np.zeros([len(target_sensordata)])

        transitionLS_idx = np.zeros([len(time_val_LWLS)])
        transitionSF_idx = np.zeros([len(time_val_LWSF)])

        for ls in range(0, len(transitionLS_idx)):
            transitionLS_idx[ls] = closest_point(target_sensor_timeval, time_val_LWLS[ls])

        for sf in range(0, len(transitionSF_idx)):
            transitionSF_idx[sf] = closest_point(target_sensor_timeval, time_val_LWSF[sf])

        return ground_truth, transitionLS_idx, transitionSF_idx

    ground_truth = np.zeros([len(target_sensordata)])

    ##### Find Transition time indices (RA->LW, RD->LW)
    #### 1. LW-RA
    ## Indices
    transitionLS_startIdx_LW_to_RA = closest_point(time_val_LWLS, time_val_RAHS[0]) # LWLS before linear transition start LW-RA
    transitionSF_startIdx_LW_to_RA = closest_point(time_val_LWSF, time_val_RAHS[0]) # Linear transition start LW-RA

    ## Time values
    transitionLS_starttime_LW_to_RA = time_val_LWLS[transitionLS_startIdx_LW_to_RA]
    transitionSF_starttime_LW_to_RA = time_val_LWSF[transitionSF_startIdx_LW_to_RA]
    transitionHS_endtime_LW_to_RA = time_val_RAHS[0]
    transitionLS_endtime_LW_to_RA = time_val_RALS[0]
    transitionSF_endtime_LW_to_RA = time_val_RASF[0]

    ## Time index in sensor timeframe
    sensortimeLS_transition_startIdx_LW_to_RA = closest_point(target_sensor_timeval,  transitionLS_starttime_LW_to_RA)
    sensortimeSF_transition_startIdx_LW_to_RA = closest_point(target_sensor_timeval, transitionSF_starttime_LW_to_RA)
    sensortimeHS_transition_endIdx_LW_to_RA = closest_point(target_sensor_timeval, transitionHS_endtime_LW_to_RA)
    sensortimeLS_transition_endIdx_LW_to_RA = closest_point(target_sensor_timeval, transitionLS_endtime_LW_to_RA)
    sensortimeSF_transition_endIdx_LW_to_RA = closest_point(target_sensor_timeval, transitionSF_endtime_LW_to_RA)

    #### 2. RA-LW
    ## Indices
    transitionHS_endIdx_RA_to_LW = closest_point(time_val_LWHS, time_val_RASF[-1]) # Linear transition end RA-LW
    transitionLS_endIdx_RA_to_LW = closest_point(time_val_LWHS, time_val_RASF[-1]) # LWLS after Linear transition end RA-LW
    transitionSF_endIdx_RA_to_LW = closest_point(time_val_LWSF, time_val_RASF[-1]) # LWSF after linear transition end RA-LW

    ## Time values
    transitionLS_starttime_RA_to_LW = time_val_RALS[-1]
    transitionSF_starttime_RA_to_LW = time_val_RASF[-1]
    transitionHS_endtime_RA_to_LW = time_val_LWHS[transitionHS_endIdx_RA_to_LW]
    transitionLS_endtime_RA_to_LW = time_val_LWLS[transitionLS_endIdx_RA_to_LW]
    transitionSF_endtime_RA_to_LW = time_val_LWSF[transitionSF_endIdx_RA_to_LW]

    ## Time index in sensor timeframe
    sensortimeLS_transition_startIdx_RA_to_LW = closest_point(target_sensor_timeval, transitionLS_starttime_RA_to_LW)
    sensortimeSF_transition_startIdx_RA_to_LW = closest_point(target_sensor_timeval, transitionSF_starttime_RA_to_LW)
    sensortimeHS_transition_endIdx_RA_to_LW = closest_point(target_sensor_timeval, transitionHS_endtime_RA_to_LW)
    sensortimeLS_transition_endIdx_RA_to_LW = closest_point(target_sensor_timeval, transitionLS_endtime_RA_to_LW)
    sensortimeSF_transition_endIdx_RA_to_LW = closest_point(target_sensor_timeval, transitionSF_endtime_RA_to_LW)


    #### 3. LW-RD
    ## Indices
    transitionLS_startIdx_LW_to_RD = closest_point(time_val_LWLS, time_val_RDHS[0])  # LWLS before linear transition start LW-RA
    transitionSF_startIdx_LW_to_RD = closest_point(time_val_LWSF, time_val_RDHS[0])  # Linear transition start LW-RA

    ## Time values
    transitionLS_starttime_LW_to_RD = time_val_LWLS[transitionLS_startIdx_LW_to_RD]
    transitionSF_starttime_LW_to_RD = time_val_LWSF[transitionSF_startIdx_LW_to_RD]
    transitionHS_endtime_LW_to_RD = time_val_RDHS[0]
    transitionLS_endtime_LW_to_RD = time_val_RDLS[0]
    transitionSF_endtime_LW_to_RD = time_val_RDSF[0]

    ## Time index in sensor timeframe
    sensortimeLS_transition_startIdx_LW_to_RD = closest_point(target_sensor_timeval, transitionLS_starttime_LW_to_RD)
    sensortimeSF_transition_startIdx_LW_to_RD = closest_point(target_sensor_timeval, transitionSF_starttime_LW_to_RD)
    sensortimeHS_transition_endIdx_LW_to_RD = closest_point(target_sensor_timeval, transitionHS_endtime_LW_to_RD)
    sensortimeLS_transition_endIdx_LW_to_RD = closest_point(target_sensor_timeval, transitionLS_endtime_LW_to_RD)
    sensortimeSF_transition_endIdx_LW_to_RD = closest_point(target_sensor_timeval, transitionSF_endtime_LW_to_RD)


    #### 4. RD-LW
    ## Indices
    transitionHS_endIdx_RD_to_LW = closest_point(time_val_LWHS, time_val_RDSF[-1]) # Linear transition end RA-LW
    transitionLS_endIdx_RD_to_LW = closest_point(time_val_LWHS, time_val_RDSF[-1]) # LWLS after Linear transition end RA-LW
    transitionSF_endIdx_RD_to_LW = closest_point(time_val_LWSF, time_val_RDSF[-1]) # LWSF after linear transition end RA-LW

    ## Time values
    transitionLS_starttime_RD_to_LW = time_val_RDLS[-1]
    transitionSF_starttime_RD_to_LW = time_val_RDSF[-1]
    transitionHS_endtime_RD_to_LW = time_val_LWHS[transitionHS_endIdx_RD_to_LW]
    transitionLS_endtime_RD_to_LW = time_val_LWLS[transitionLS_endIdx_RD_to_LW]
    transitionSF_endtime_RD_to_LW = time_val_LWSF[transitionSF_endIdx_RD_to_LW]

    ## Time index in sensor timeframe
    sensortimeLS_transition_startIdx_RD_to_LW = closest_point(target_sensor_timeval, transitionLS_starttime_RD_to_LW)
    sensortimeSF_transition_startIdx_RD_to_LW = closest_point(target_sensor_timeval, transitionSF_starttime_RD_to_LW)
    sensortimeHS_transition_endIdx_RD_to_LW = closest_point(target_sensor_timeval, transitionHS_endtime_RD_to_LW)
    sensortimeLS_transition_endIdx_RD_to_LW = closest_point(target_sensor_timeval, transitionLS_endtime_RD_to_LW)
    sensortimeSF_transition_endIdx_RD_to_LW = closest_point(target_sensor_timeval, transitionSF_endtime_RD_to_LW)

    ground_truth[sensortimeHS_transition_endIdx_LW_to_RA:sensortimeSF_transition_startIdx_RA_to_LW] = 1
    ground_truth[sensortimeHS_transition_endIdx_LW_to_RD:sensortimeSF_transition_startIdx_RD_to_LW] = -1

    ### Generate linear transition
    n_transition_LW_to_RA = sensortimeHS_transition_endIdx_LW_to_RA - sensortimeSF_transition_startIdx_LW_to_RA
    n_transition_RA_to_LW = sensortimeHS_transition_endIdx_RA_to_LW - sensortimeSF_transition_startIdx_RA_to_LW
    n_transition_LW_to_RD = sensortimeHS_transition_endIdx_LW_to_RD - sensortimeSF_transition_startIdx_LW_to_RD
    n_transition_RD_to_LW = sensortimeHS_transition_endIdx_RD_to_LW - sensortimeSF_transition_startIdx_RD_to_LW

    lt_LW_to_RA = np.linspace(0, 1, n_transition_LW_to_RA)
    lt_RA_to_LW = np.linspace(1, 0, n_transition_RA_to_LW)
    lt_LW_to_RD = np.linspace(0, -1, n_transition_LW_to_RD)
    lt_RD_to_LW = np.linspace(-1, 0, n_transition_RD_to_LW)

    ground_truth[sensortimeSF_transition_startIdx_LW_to_RA:sensortimeHS_transition_endIdx_LW_to_RA] = lt_LW_to_RA
    ground_truth[sensortimeSF_transition_startIdx_RA_to_LW:sensortimeHS_transition_endIdx_RA_to_LW] = lt_RA_to_LW
    ground_truth[sensortimeSF_transition_startIdx_LW_to_RD:sensortimeHS_transition_endIdx_LW_to_RD] = lt_LW_to_RD
    ground_truth[sensortimeSF_transition_startIdx_RD_to_LW:sensortimeHS_transition_endIdx_RD_to_LW] = lt_RD_to_LW

    transitionLS_idx = np.zeros([len(time_val_LS)])
    transitionSF_idx = np.zeros([len(time_val_SF)])

    for ls in range(0, len(transitionLS_idx)):
        transitionLS_idx[ls] = closest_point(target_sensor_timeval, time_val_LS[ls])

    for sf in range(0, len(transitionSF_idx)):
        transitionSF_idx[sf] = closest_point(target_sensor_timeval, time_val_SF[sf])

    ## Check
    # plt.plot(ground_truth[sensortimeLS_transition_startIdx_LW_to_RA:sensortimeLS_transition_endIdx_LW_to_RA])
    # plt.plot(ground_truth[sensortimeSF_transition_startIdx_LW_to_RA:sensortimeSF_transition_endIdx_LW_to_RA])
    # plt.plot(ground_truth)

    return ground_truth, transitionLS_idx, transitionSF_idx

def Ramp_GT3_end(target_data_dict, ref_signal_topic, swing_percentage): ### Previous version
    '''
    Interpolate Grount Truth (GT) slope values aligned with sensor timestamp
    * Ascent:1, Descent:-1, LW:0

    ref_signal_topic: enter the topic that you will use for reference
    ex: '/SensorData'

    Transition: linearly smoothed between SF-Midswing-HS in each transition
    '''

    if target_data_dict.__class__ == str:
        target_rawdata = jb.load(target_data_dict)
    elif target_data_dict.__class__ == dict:
        target_rawdata = target_data_dict
    else:
        print('Error! Check the input type (Directory or Dictionary)')

    ### Function test
    # workspace_path = os.getcwd()
    # cacheDir = workspace_path + '/Ramp_Data/'
    #
    # loadDir = cacheDir + 'TF02v2/'  ### Put your Cache file directory here
    # loadfileDir = loadDir + '23'
    # target_rawdata = jb.load(loadfileDir)
    # ref_signal_topic = str('/SensorData')
    # swing_percentage = 0.5
    ### Function test


    target_sensordata = target_rawdata[ref_signal_topic]
    target_sensor_timeval = target_sensordata['header']

    ##### Align time segment to State
    time_val = target_rawdata['/fsm/State']['header']
    state = target_rawdata['/fsm/State']['state']

    time_idx_ES = []
    time_idx_LS = []
    time_idx_SE = []
    time_idx_SF = []

    i = 0
    for phase in state:
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

    ### Heel Strike Timings
    time_idx_RAHS = []  ### Ramp Ascend Heel Strike
    time_idx_RDHS = []  ### Ramp Descend Heel Strike
    time_idx_LWHS = []  ### Level Walking Heel Strike

    for j in range(0, len(time_idx_ES)):
        if 'RA' in state[time_idx_ES[j]]:
            time_idx_RAHS.append(time_idx_ES[j])

        if 'RD' in state[time_idx_ES[j]]:
            time_idx_RDHS.append(time_idx_ES[j])

        if 'LW' in state[time_idx_ES[j]]:
            time_idx_LWHS.append(time_idx_ES[j])

    time_val_RAHS = time_val[time_idx_RAHS].array
    time_val_RDHS = time_val[time_idx_RDHS].array
    time_val_LWHS = time_val[time_idx_LWHS].array

    ### Late Stance Timings
    time_idx_RALS = []  ### Ramp Ascend Late Stance
    time_idx_RDLS = []  ### Ramp Descend Late Stance
    time_idx_LWLS = []  ### Level Walking Late Stance

    for j in range(0, len(time_idx_LS)):
        if 'RA' in state[time_idx_LS[j]]:
            time_idx_RALS.append(time_idx_LS[j])

        if 'RD' in state[time_idx_LS[j]]:
            time_idx_RDLS.append(time_idx_LS[j])

        if 'LW' in state[time_idx_LS[j]]:
            time_idx_LWLS.append(time_idx_LS[j])

    time_val_RALS = time_val[time_idx_RALS].array
    time_val_RDLS = time_val[time_idx_RDLS].array
    time_val_LWLS = time_val[time_idx_LWLS].array

    ### Swing Flexion Timings
    time_idx_RASF = []  ### Ramp Ascend Swing Flexion
    time_idx_RDSF = []  ### Ramp Descend Swing Flexion
    time_idx_LWSF = []  ### Level Walking Swing Flexion

    for j in range(0, len(time_idx_SF)):
        if 'RA' in state[time_idx_SF[j]]:
            time_idx_RASF.append(time_idx_SF[j])

        if 'RD' in state[time_idx_SF[j]]:
            time_idx_RDSF.append(time_idx_SF[j])

        if 'LW' in state[time_idx_SF[j]]:
            time_idx_LWSF.append(time_idx_SF[j])

    time_val_RASF = time_val[time_idx_RASF].array
    time_val_RDSF = time_val[time_idx_RDSF].array
    time_val_LWSF = time_val[time_idx_LWSF].array

    ### Swing Extension Timings
    time_idx_RASE = []  ### Ramp Ascend Swing Flexion
    time_idx_RDSE = []  ### Ramp Descend Swing Flexion
    time_idx_LWSE = []  ### Level Walking Swing Flexion

    for j in range(0, len(time_idx_SE)):
        if 'RA' in state[time_idx_SE[j]]:
            time_idx_RASE.append(time_idx_SE[j])

        if 'RD' in state[time_idx_SE[j]]:
            time_idx_RDSE.append(time_idx_SE[j])

        if 'LW' in state[time_idx_SE[j]]:
            time_idx_LWSE.append(time_idx_SE[j])

    time_val_RASE = time_val[time_idx_RASE].array
    time_val_RDSE = time_val[time_idx_RDSE].array
    time_val_LWSE = time_val[time_idx_LWSE].array

    time_val_ES = time_val[time_idx_ES].array
    time_val_LS = time_val[time_idx_LS].array
    time_val_SF = time_val[time_idx_SF].array
    time_val_SE = time_val[time_idx_SE].array

    if len(time_val_RDHS) < 1: ### If there is no ramp descent, It's LW trial
        ground_truth = np.zeros([len(target_sensordata)])

        transitionES_idx = np.zeros([len(time_val_ES)])
        transitionLS_idx = np.zeros([len(time_val_LS)])
        transitionSF_idx = np.zeros([len(time_val_SF)])
        transitionSE_idx = np.zeros([len(time_val_SE)])

        for es in range(0, len(transitionES_idx)):
            transitionES_idx[es] = closest_point(target_sensor_timeval, time_val_ES[es])

        for ls in range(0, len(transitionLS_idx)):
            transitionLS_idx[ls] = closest_point(target_sensor_timeval, time_val_LS[ls])

        for sf in range(0, len(transitionSF_idx)):
            transitionSF_idx[sf] = closest_point(target_sensor_timeval, time_val_SF[sf])

        for se in range(0, len(transitionSE_idx)):
            transitionSE_idx[se] = closest_point(target_sensor_timeval, time_val_SE[se])

        return ground_truth, transitionES_idx, transitionLS_idx, transitionSF_idx, transitionSE_idx, 0, 0, 0, 0, 0, 0, 0, 0

    ground_truth = np.zeros([len(target_sensordata)])
    ground_truth2 = np.zeros([len(target_sensordata)])

    ##### Find Transition time indices (RA->LW, RD->LW)
    #### 1. LW-RA
    ## Indices
    transitionSF_startIdx_LW_to_RA = closest_point(time_val_LWSF, time_val_RAHS[0]) # Linear transition start LW-RA

    ## Time values
    transitionSF_starttime_LW_to_RA = time_val_LWSF[transitionSF_startIdx_LW_to_RA]
    transitionHS_endtime_LW_to_RA = time_val_RAHS[0]

    transitionMS_endtime_LW_to_RA = (1 - swing_percentage) * transitionSF_starttime_LW_to_RA\
                                    + swing_percentage * transitionHS_endtime_LW_to_RA

    ## Time index in sensor timeframe
    sensortimeSF_transition_startIdx_LW_to_RA = closest_point(target_sensor_timeval, transitionSF_starttime_LW_to_RA)
    sensortimeMS_transition_endIdx_LW_to_RA = closest_point(target_sensor_timeval, transitionMS_endtime_LW_to_RA) # Midswing

    #### 2. RA-LW
    ## Indices
    transitionHS_endIdx_RA_to_LW = closest_point(time_val_LWHS, time_val_RASF[-1]) # Linear transition end RA-LW

    ## Time values
    transitionSF_starttime_RA_to_LW = time_val_RASF[-1]
    transitionHS_endtime_RA_to_LW = time_val_LWHS[transitionHS_endIdx_RA_to_LW]

    transitionMS_endtime_RA_to_LW = (1 - swing_percentage) * transitionSF_starttime_RA_to_LW\
                                    + swing_percentage * transitionHS_endtime_RA_to_LW

    ## Time index in sensor timeframe
    sensortimeSF_transition_startIdx_RA_to_LW = closest_point(target_sensor_timeval, transitionSF_starttime_RA_to_LW)
    sensortimeMS_transition_endIdx_RA_to_LW = closest_point(target_sensor_timeval, transitionMS_endtime_RA_to_LW) # Midswing

    #### 3. LW-RD
    ## Indices
    transitionSF_startIdx_LW_to_RD = closest_point(time_val_LWSF, time_val_RDHS[0])  # Linear transition start LW-RA

    ## Time values
    transitionSF_starttime_LW_to_RD = time_val_LWSF[transitionSF_startIdx_LW_to_RD]
    transitionHS_endtime_LW_to_RD = time_val_RDHS[0]

    transitionMS_endtime_LW_to_RD = (1 - swing_percentage) * transitionSF_starttime_LW_to_RD \
                                    + swing_percentage * transitionHS_endtime_LW_to_RD

    ## Time index in sensor timeframe
    sensortimeSF_transition_startIdx_LW_to_RD = closest_point(target_sensor_timeval, transitionSF_starttime_LW_to_RD)
    sensortimeMS_transition_endIdx_LW_to_RD = closest_point(target_sensor_timeval, transitionMS_endtime_LW_to_RD) # Midswing

    #### 4. RD-LW
    ## Indices
    transitionHS_endIdx_RD_to_LW = closest_point(time_val_LWHS, time_val_RDSF[-1]) # Linear transition end RA-LW

    ## Time values
    transitionSF_starttime_RD_to_LW = time_val_RDSF[-1]
    transitionHS_endtime_RD_to_LW = time_val_LWHS[transitionHS_endIdx_RD_to_LW]

    transitionMS_endtime_RD_to_LW = (1 - swing_percentage) * transitionSF_starttime_RD_to_LW\
                                    + swing_percentage * transitionHS_endtime_RD_to_LW

    ## Time index in sensor timeframe
    sensortimeSF_transition_startIdx_RD_to_LW = closest_point(target_sensor_timeval, transitionSF_starttime_RD_to_LW)
    sensortimeMS_transition_endIdx_RD_to_LW = closest_point(target_sensor_timeval, transitionMS_endtime_RD_to_LW)  # Midswing

    ground_truth[sensortimeMS_transition_endIdx_LW_to_RA:sensortimeSF_transition_startIdx_RA_to_LW] = 1
    ground_truth[sensortimeMS_transition_endIdx_LW_to_RD:sensortimeSF_transition_startIdx_RD_to_LW] = -1

    ### Generate linear transition
    n_transition_LW_to_RA = sensortimeMS_transition_endIdx_LW_to_RA - sensortimeSF_transition_startIdx_LW_to_RA
    n_transition_RA_to_LW = sensortimeMS_transition_endIdx_RA_to_LW - sensortimeSF_transition_startIdx_RA_to_LW
    n_transition_LW_to_RD = sensortimeMS_transition_endIdx_LW_to_RD - sensortimeSF_transition_startIdx_LW_to_RD
    n_transition_RD_to_LW = sensortimeMS_transition_endIdx_RD_to_LW - sensortimeSF_transition_startIdx_RD_to_LW

    lt_LW_to_RA = np.linspace(0, 1, n_transition_LW_to_RA)
    lt_RA_to_LW = np.linspace(1, 0, n_transition_RA_to_LW)
    lt_LW_to_RD = np.linspace(0, -1, n_transition_LW_to_RD)
    lt_RD_to_LW = np.linspace(-1, 0, n_transition_RD_to_LW)

    ground_truth[sensortimeSF_transition_startIdx_LW_to_RA:sensortimeMS_transition_endIdx_LW_to_RA] = lt_LW_to_RA
    ground_truth[sensortimeSF_transition_startIdx_RA_to_LW:sensortimeMS_transition_endIdx_RA_to_LW] = lt_RA_to_LW
    ground_truth[sensortimeSF_transition_startIdx_LW_to_RD:sensortimeMS_transition_endIdx_LW_to_RD] = lt_LW_to_RD
    ground_truth[sensortimeSF_transition_startIdx_RD_to_LW:sensortimeMS_transition_endIdx_RD_to_LW] = lt_RD_to_LW

    transitionES_idx = np.zeros([len(time_val_ES)])
    transitionLS_idx = np.zeros([len(time_val_LS)])
    transitionSF_idx = np.zeros([len(time_val_SF)])
    transitionSE_idx = np.zeros([len(time_val_SE)])

    transitionRAES_idx = np.zeros([len(time_val_RAHS)])
    transitionRALS_idx = np.zeros([len(time_val_RALS)])
    transitionRASF_idx = np.zeros([len(time_val_RASF)])
    transitionRASE_idx = np.zeros([len(time_val_RASE)])

    transitionRDES_idx = np.zeros([len(time_val_RDHS)])
    transitionRDLS_idx = np.zeros([len(time_val_RDLS)])
    transitionRDSF_idx = np.zeros([len(time_val_RDSF)])
    transitionRDSE_idx = np.zeros([len(time_val_RDSE)])

    for es in range(0, len(transitionES_idx)):
        transitionES_idx[es] = closest_point(target_sensor_timeval, time_val_ES[es])

    for ls in range(0, len(transitionLS_idx)):
        transitionLS_idx[ls] = closest_point(target_sensor_timeval, time_val_LS[ls])

    for sf in range(0, len(transitionSF_idx)):
        transitionSF_idx[sf] = closest_point(target_sensor_timeval, time_val_SF[sf])

    for se in range(0, len(transitionSE_idx)):
        transitionSE_idx[se] = closest_point(target_sensor_timeval, time_val_SE[se])

    for raes in range(0, len(transitionRAES_idx)):
        transitionRAES_idx[raes] = closest_point(target_sensor_timeval, time_val_RAHS[raes])

    for rals in range(0, len(transitionRALS_idx)):
        transitionRALS_idx[rals] = closest_point(target_sensor_timeval, time_val_RALS[rals])

    for rasf in range(0, len(transitionRASF_idx)):
        transitionRASF_idx[rasf] = closest_point(target_sensor_timeval, time_val_RASF[rasf])

    for rase in range(0, len(transitionRASE_idx)):
        transitionRASE_idx[rase] = closest_point(target_sensor_timeval, time_val_RASE[rase])

    for rdes in range(0, len(transitionRDES_idx)):
        transitionRDES_idx[rdes] = closest_point(target_sensor_timeval, time_val_RDHS[rdes])

    for rdls in range(0, len(transitionRDLS_idx)):
        transitionRDLS_idx[rdls] = closest_point(target_sensor_timeval, time_val_RDLS[rdls])

    for rdsf in range(0, len(transitionRDSF_idx)):
        transitionRDSF_idx[rdsf] = closest_point(target_sensor_timeval, time_val_RDSF[rdsf])

    for rdse in range(0, len(transitionRDSE_idx)):
        transitionRDSE_idx[rdse] = closest_point(target_sensor_timeval, time_val_RDSE[rdse])

    ## Check
    # plt.plot(ground_truth)
    # plt.plot(ground_truth2)

    return ground_truth, transitionES_idx, transitionLS_idx, transitionSF_idx, transitionSE_idx, \
           transitionRAES_idx, transitionRALS_idx, transitionRASF_idx, transitionRASE_idx, \
           transitionRDES_idx, transitionRDLS_idx, transitionRDSF_idx, transitionRDSE_idx

def Ramp_GT3_start(target_data_dict, ref_signal_topic, swing_percentage): ### Previous version
    '''
    Interpolate Grount Truth (GT) slope values aligned with sensor timestamp
    * Ascent:1, Descent:-1, LW:0

    ref_signal_topic: enter the topic that you will use for reference
    ex: '/SensorData'

    Transition: linearly smoothed between SF-Midswing-HS in each transition
    '''

    if target_data_dict.__class__ == str:
        target_rawdata = jb.load(target_data_dict)
    elif target_data_dict.__class__ == dict:
        target_rawdata = target_data_dict
    else:
        print('Error! Check the input type (Directory or Dictionary)')

    ### Function test
    # workspace_path = os.getcwd()
    # cacheDir = workspace_path + '/Ramp_Data/'
    #
    # loadDir = cacheDir + 'TF02v2/'  ### Put your Cache file directory here
    # loadfileDir = loadDir + '23'
    # target_rawdata = jb.load(loadfileDir)
    # ref_signal_topic = str('/SensorData')
    # swing_percentage = 0.5
    ### Function test


    target_sensordata = target_rawdata[ref_signal_topic]
    target_sensor_timeval = target_sensordata['header']

    ##### Align time segment to State
    time_val = target_rawdata['/fsm/State']['header']
    state = target_rawdata['/fsm/State']['state']

    time_idx_ES = []
    time_idx_LS = []
    time_idx_SE = []
    time_idx_SF = []

    i = 0
    for phase in state:
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

    ### Heel Strike Timings
    time_idx_RAHS = []  ### Ramp Ascend Heel Strike
    time_idx_RDHS = []  ### Ramp Descend Heel Strike
    time_idx_LWHS = []  ### Level Walking Heel Strike

    for j in range(0, len(time_idx_ES)):
        if 'RA' in state[time_idx_ES[j]]:
            time_idx_RAHS.append(time_idx_ES[j])

        if 'RD' in state[time_idx_ES[j]]:
            time_idx_RDHS.append(time_idx_ES[j])

        if 'LW' in state[time_idx_ES[j]]:
            time_idx_LWHS.append(time_idx_ES[j])

    time_val_RAHS = time_val[time_idx_RAHS].array
    time_val_RDHS = time_val[time_idx_RDHS].array
    time_val_LWHS = time_val[time_idx_LWHS].array

    ### Late Stance Timings
    time_idx_RALS = []  ### Ramp Ascend Late Stance
    time_idx_RDLS = []  ### Ramp Descend Late Stance
    time_idx_LWLS = []  ### Level Walking Late Stance

    for j in range(0, len(time_idx_LS)):
        if 'RA' in state[time_idx_LS[j]]:
            time_idx_RALS.append(time_idx_LS[j])

        if 'RD' in state[time_idx_LS[j]]:
            time_idx_RDLS.append(time_idx_LS[j])

        if 'LW' in state[time_idx_LS[j]]:
            time_idx_LWLS.append(time_idx_LS[j])

    time_val_RALS = time_val[time_idx_RALS].array
    time_val_RDLS = time_val[time_idx_RDLS].array
    time_val_LWLS = time_val[time_idx_LWLS].array

    ### Swing Flexion Timings
    time_idx_RASF = []  ### Ramp Ascend Swing Flexion
    time_idx_RDSF = []  ### Ramp Descend Swing Flexion
    time_idx_LWSF = []  ### Level Walking Swing Flexion

    for j in range(0, len(time_idx_SF)):
        if 'RA' in state[time_idx_SF[j]]:
            time_idx_RASF.append(time_idx_SF[j])

        if 'RD' in state[time_idx_SF[j]]:
            time_idx_RDSF.append(time_idx_SF[j])

        if 'LW' in state[time_idx_SF[j]]:
            time_idx_LWSF.append(time_idx_SF[j])

    time_val_RASF = time_val[time_idx_RASF].array
    time_val_RDSF = time_val[time_idx_RDSF].array
    time_val_LWSF = time_val[time_idx_LWSF].array

    ### Swing Extension Timings
    time_idx_RASE = []  ### Ramp Ascend Swing Flexion
    time_idx_RDSE = []  ### Ramp Descend Swing Flexion
    time_idx_LWSE = []  ### Level Walking Swing Flexion

    for j in range(0, len(time_idx_SE)):
        if 'RA' in state[time_idx_SE[j]]:
            time_idx_RASE.append(time_idx_SE[j])

        if 'RD' in state[time_idx_SE[j]]:
            time_idx_RDSE.append(time_idx_SE[j])

        if 'LW' in state[time_idx_SE[j]]:
            time_idx_LWSE.append(time_idx_SE[j])

    time_val_RASE = time_val[time_idx_RASE].array
    time_val_RDSE = time_val[time_idx_RDSE].array
    time_val_LWSE = time_val[time_idx_LWSE].array

    time_val_ES = time_val[time_idx_ES].array
    time_val_LS = time_val[time_idx_LS].array
    time_val_SF = time_val[time_idx_SF].array
    time_val_SE = time_val[time_idx_SE].array

    if len(time_val_RDHS) < 1: ### If there is no ramp descent, It's LW trial
        ground_truth = np.zeros([len(target_sensordata)])

        transitionES_idx = np.zeros([len(time_val_ES)])
        transitionLS_idx = np.zeros([len(time_val_LS)])
        transitionSF_idx = np.zeros([len(time_val_SF)])
        transitionSE_idx = np.zeros([len(time_val_SE)])

        for es in range(0, len(transitionES_idx)):
            transitionES_idx[es] = closest_point(target_sensor_timeval, time_val_ES[es])

        for ls in range(0, len(transitionLS_idx)):
            transitionLS_idx[ls] = closest_point(target_sensor_timeval, time_val_LS[ls])

        for sf in range(0, len(transitionSF_idx)):
            transitionSF_idx[sf] = closest_point(target_sensor_timeval, time_val_SF[sf])

        for se in range(0, len(transitionSE_idx)):
            transitionSE_idx[se] = closest_point(target_sensor_timeval, time_val_SE[se])

        return ground_truth, transitionES_idx, transitionLS_idx, transitionSF_idx, transitionSE_idx, 0, 0, 0, 0, 0, 0, 0, 0

    ground_truth = np.zeros([len(target_sensordata)])
    ground_truth2 = np.zeros([len(target_sensordata)])

    ##### Find Transition time indices (RA->LW, RD->LW)
    #### 1. LW-RA
    ## Indices
    transitionSF_startIdx_LW_to_RA = closest_point(time_val_LWSF, time_val_RAHS[0]) # Linear transition start LW-RA

    ## Time values
    transitionSF_starttime_LW_to_RA = time_val_LWSF[transitionSF_startIdx_LW_to_RA]
    transitionHS_endtime_LW_to_RA = time_val_RAHS[0]

    transitionMS_starttime_LW_to_RA = (1 - swing_percentage) * transitionSF_starttime_LW_to_RA +\
                                      swing_percentage * transitionHS_endtime_LW_to_RA

    ## Time index in sensor timeframe
    sensortimeMS_transition_startIdx_LW_to_RA = closest_point(target_sensor_timeval,
                                                            transitionMS_starttime_LW_to_RA)  # Midswing
    sensortimeHS_transition_endIdx_LW_to_RA = closest_point(target_sensor_timeval, transitionHS_endtime_LW_to_RA) # HS

    #### 2. RA-LW
    ## Indices
    transitionHS_endIdx_RA_to_LW = closest_point(time_val_LWHS, time_val_RASF[-1]) # Linear transition end RA-LW

    ## Time values
    transitionSF_starttime_RA_to_LW = time_val_RASF[-1]
    transitionHS_endtime_RA_to_LW = time_val_LWHS[transitionHS_endIdx_RA_to_LW]

    transitionMS_starttime_RA_to_LW = (1 - swing_percentage) * transitionSF_starttime_RA_to_LW +\
                                      swing_percentage * transitionHS_endtime_RA_to_LW

    ## Time index in sensor timeframe
    sensortimeMS_transition_startIdx_RA_to_LW = closest_point(target_sensor_timeval,
                                                            transitionMS_starttime_RA_to_LW)  # Midswing
    sensortimeHS_transition_endIdx_RA_to_LW = closest_point(target_sensor_timeval, transitionHS_endtime_RA_to_LW) # HS

    #### 3. LW-RD
    ## Indices
    transitionSF_startIdx_LW_to_RD = closest_point(time_val_LWSF, time_val_RDHS[0])  # Linear transition start LW-RA

    ## Time values
    transitionSF_starttime_LW_to_RD = time_val_LWSF[transitionSF_startIdx_LW_to_RD]
    transitionHS_endtime_LW_to_RD = time_val_RDHS[0]

    transitionMS_starttime_LW_to_RD = (1 - swing_percentage) * transitionSF_starttime_LW_to_RD +\
                                      swing_percentage * transitionHS_endtime_LW_to_RD

    ## Time index in sensor timeframe
    sensortimeMS_transition_startIdx_LW_to_RD = closest_point(target_sensor_timeval, transitionMS_starttime_LW_to_RD)
    sensortimeHS_transition_endIdx_LW_to_RD = closest_point(target_sensor_timeval, transitionHS_endtime_LW_to_RD) # Midswing

    #### 4. RD-LW
    ## Indices
    transitionHS_endIdx_RD_to_LW = closest_point(time_val_LWHS, time_val_RDSF[-1]) # Linear transition end RA-LW

    ## Time values
    transitionSF_starttime_RD_to_LW = time_val_RDSF[-1]
    transitionHS_endtime_RD_to_LW = time_val_LWHS[transitionHS_endIdx_RD_to_LW]

    transitionMS_starttime_RD_to_LW = (1 - swing_percentage) * transitionSF_starttime_RD_to_LW \
                                    + swing_percentage * transitionHS_endtime_RD_to_LW

    ## Time index in sensor timeframe
    sensortimeMS_transition_startIdx_RD_to_LW = closest_point(target_sensor_timeval, transitionMS_starttime_RD_to_LW)
    sensortimeHS_transition_endIdx_RD_to_LW = closest_point(target_sensor_timeval, transitionHS_endtime_RD_to_LW)  # Midswing

    ground_truth[sensortimeHS_transition_endIdx_LW_to_RA:sensortimeMS_transition_startIdx_RA_to_LW] = 1
    ground_truth[sensortimeHS_transition_endIdx_LW_to_RD:sensortimeMS_transition_startIdx_RD_to_LW] = -1

    ### Generate linear transition
    n_transition_LW_to_RA = sensortimeHS_transition_endIdx_LW_to_RA - sensortimeMS_transition_startIdx_LW_to_RA
    n_transition_RA_to_LW = sensortimeHS_transition_endIdx_RA_to_LW - sensortimeMS_transition_startIdx_RA_to_LW
    n_transition_LW_to_RD = sensortimeHS_transition_endIdx_LW_to_RD - sensortimeMS_transition_startIdx_LW_to_RD
    n_transition_RD_to_LW = sensortimeHS_transition_endIdx_RD_to_LW - sensortimeMS_transition_startIdx_RD_to_LW

    lt_LW_to_RA = np.linspace(0, 1, n_transition_LW_to_RA)
    lt_RA_to_LW = np.linspace(1, 0, n_transition_RA_to_LW)
    lt_LW_to_RD = np.linspace(0, -1, n_transition_LW_to_RD)
    lt_RD_to_LW = np.linspace(-1, 0, n_transition_RD_to_LW)

    ground_truth[sensortimeMS_transition_startIdx_LW_to_RA:sensortimeHS_transition_endIdx_LW_to_RA] = lt_LW_to_RA
    ground_truth[sensortimeMS_transition_startIdx_RA_to_LW:sensortimeHS_transition_endIdx_RA_to_LW] = lt_RA_to_LW
    ground_truth[sensortimeMS_transition_startIdx_LW_to_RD:sensortimeHS_transition_endIdx_LW_to_RD] = lt_LW_to_RD
    ground_truth[sensortimeMS_transition_startIdx_RD_to_LW:sensortimeHS_transition_endIdx_RD_to_LW] = lt_RD_to_LW

    transitionES_idx = np.zeros([len(time_val_ES)])
    transitionLS_idx = np.zeros([len(time_val_LS)])
    transitionSF_idx = np.zeros([len(time_val_SF)])
    transitionSE_idx = np.zeros([len(time_val_SE)])

    transitionRAES_idx = np.zeros([len(time_val_RAHS)])
    transitionRALS_idx = np.zeros([len(time_val_RALS)])
    transitionRASF_idx = np.zeros([len(time_val_RASF)])
    transitionRASE_idx = np.zeros([len(time_val_RASE)])

    transitionRDES_idx = np.zeros([len(time_val_RDHS)])
    transitionRDLS_idx = np.zeros([len(time_val_RDLS)])
    transitionRDSF_idx = np.zeros([len(time_val_RDSF)])
    transitionRDSE_idx = np.zeros([len(time_val_RDSE)])

    for es in range(0, len(transitionES_idx)):
        transitionES_idx[es] = closest_point(target_sensor_timeval, time_val_ES[es])

    for ls in range(0, len(transitionLS_idx)):
        transitionLS_idx[ls] = closest_point(target_sensor_timeval, time_val_LS[ls])

    for sf in range(0, len(transitionSF_idx)):
        transitionSF_idx[sf] = closest_point(target_sensor_timeval, time_val_SF[sf])

    for se in range(0, len(transitionSE_idx)):
        transitionSE_idx[se] = closest_point(target_sensor_timeval, time_val_SE[se])

    for raes in range(0, len(transitionRAES_idx)):
        transitionRAES_idx[raes] = closest_point(target_sensor_timeval, time_val_RAHS[raes])

    for rals in range(0, len(transitionRALS_idx)):
        transitionRALS_idx[rals] = closest_point(target_sensor_timeval, time_val_RALS[rals])

    for rasf in range(0, len(transitionRASF_idx)):
        transitionRASF_idx[rasf] = closest_point(target_sensor_timeval, time_val_RASF[rasf])

    for rase in range(0, len(transitionRASE_idx)):
        transitionRASE_idx[rase] = closest_point(target_sensor_timeval, time_val_RASE[rase])

    for rdes in range(0, len(transitionRDES_idx)):
        transitionRDES_idx[rdes] = closest_point(target_sensor_timeval, time_val_RDHS[rdes])

    for rdls in range(0, len(transitionRDLS_idx)):
        transitionRDLS_idx[rdls] = closest_point(target_sensor_timeval, time_val_RDLS[rdls])

    for rdsf in range(0, len(transitionRDSF_idx)):
        transitionRDSF_idx[rdsf] = closest_point(target_sensor_timeval, time_val_RDSF[rdsf])

    for rdse in range(0, len(transitionRDSE_idx)):
        transitionRDSE_idx[rdse] = closest_point(target_sensor_timeval, time_val_RDSE[rdse])

    ## Check
    # plt.plot(ground_truth)
    # plt.plot(ground_truth2)

    return ground_truth, transitionES_idx, transitionLS_idx, transitionSF_idx, transitionSE_idx, \
           transitionRAES_idx, transitionRALS_idx, transitionRASF_idx, transitionRASE_idx, \
           transitionRDES_idx, transitionRDLS_idx, transitionRDSF_idx, transitionRDSE_idx

def Ramp_GT4(target_data_dict, ref_signal_topic): ### Previous version
    '''
    Interpolate Grount Truth (GT) slope values aligned with sensor timestamp
    * Ascent:1, Descent:-1, LW:0

    ref_signal_topic: enter the topic that you will use for reference
    ex: '/SensorData'

    Transition: linearly smoothed between SF-Midswing-HS in each transition
    '''

    if target_data_dict.__class__ == str:
        target_rawdata = jb.load(target_data_dict)
    elif target_data_dict.__class__ == dict:
        target_rawdata = target_data_dict
    else:
        print('Error! Check the input type (Directory or Dictionary)')

    ### Function test
    # workspace_path = os.getcwd()
    # cacheDir = workspace_path + '/Ramp_Data/'
    #
    # loadDir = cacheDir + 'TF02v2/'  ### Put your Cache file directory here
    # loadfileDir = loadDir + '23'
    # target_rawdata = jb.load(loadfileDir)
    # ref_signal_topic = str('/SensorData')
    ### Function test

    target_sensordata = target_rawdata[ref_signal_topic]
    target_sensor_timeval = target_sensordata['header']

    ##### Align time segment to State
    time_val = target_rawdata['/fsm/State']['header']
    state = target_rawdata['/fsm/State']['state']

    time_idx_ES = []
    time_idx_LS = []
    time_idx_SE = []
    time_idx_SF = []

    i = 0
    for phase in state:
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

    ### Heel Strike Timings
    time_idx_RAHS = []  ### Ramp Ascend Heel Strike
    time_idx_RDHS = []  ### Ramp Descend Heel Strike
    time_idx_LWHS = []  ### Level Walking Heel Strike

    for j in range(0, len(time_idx_ES)):
        if 'RA' in state[time_idx_ES[j]]:
            time_idx_RAHS.append(time_idx_ES[j])

        if 'RD' in state[time_idx_ES[j]]:
            time_idx_RDHS.append(time_idx_ES[j])

        if 'LW' in state[time_idx_ES[j]]:
            time_idx_LWHS.append(time_idx_ES[j])

    time_val_RAHS = time_val[time_idx_RAHS].array
    time_val_RDHS = time_val[time_idx_RDHS].array
    time_val_LWHS = time_val[time_idx_LWHS].array

    ### Late Stance Timings
    time_idx_RALS = []  ### Ramp Ascend Late Stance
    time_idx_RDLS = []  ### Ramp Descend Late Stance
    time_idx_LWLS = []  ### Level Walking Late Stance

    for j in range(0, len(time_idx_LS)):
        if 'RA' in state[time_idx_LS[j]]:
            time_idx_RALS.append(time_idx_LS[j])

        if 'RD' in state[time_idx_LS[j]]:
            time_idx_RDLS.append(time_idx_LS[j])

        if 'LW' in state[time_idx_LS[j]]:
            time_idx_LWLS.append(time_idx_LS[j])

    time_val_RALS = time_val[time_idx_RALS].array
    time_val_RDLS = time_val[time_idx_RDLS].array
    time_val_LWLS = time_val[time_idx_LWLS].array

    ### Swing Flexion Timings
    time_idx_RASF = []  ### Ramp Ascend Swing Flexion
    time_idx_RDSF = []  ### Ramp Descend Swing Flexion
    time_idx_LWSF = []  ### Level Walking Swing Flexion

    for j in range(0, len(time_idx_SF)):
        if 'RA' in state[time_idx_SF[j]]:
            time_idx_RASF.append(time_idx_SF[j])

        if 'RD' in state[time_idx_SF[j]]:
            time_idx_RDSF.append(time_idx_SF[j])

        if 'LW' in state[time_idx_SF[j]]:
            time_idx_LWSF.append(time_idx_SF[j])

    time_val_RASF = time_val[time_idx_RASF].array
    time_val_RDSF = time_val[time_idx_RDSF].array
    time_val_LWSF = time_val[time_idx_LWSF].array

    ### Swing Extension Timings
    time_idx_RASE = []  ### Ramp Ascend Swing Flexion
    time_idx_RDSE = []  ### Ramp Descend Swing Flexion
    time_idx_LWSE = []  ### Level Walking Swing Flexion

    for j in range(0, len(time_idx_SE)):
        if 'RA' in state[time_idx_SE[j]]:
            time_idx_RASE.append(time_idx_SE[j])

        if 'RD' in state[time_idx_SE[j]]:
            time_idx_RDSE.append(time_idx_SE[j])

        if 'LW' in state[time_idx_SE[j]]:
            time_idx_LWSE.append(time_idx_SE[j])

    time_val_RASE = time_val[time_idx_RASE].array
    time_val_RDSE = time_val[time_idx_RDSE].array
    time_val_LWSE = time_val[time_idx_LWSE].array

    time_val_ES = time_val[time_idx_ES].array
    time_val_LS = time_val[time_idx_LS].array
    time_val_SF = time_val[time_idx_SF].array
    time_val_SE = time_val[time_idx_SE].array

    if len(time_val_RDHS) < 1: ### If there is no ramp descent, It's LW trial
        ground_truth = np.zeros([len(target_sensordata)])

        transitionES_idx = np.zeros([len(time_val_ES)])
        transitionLS_idx = np.zeros([len(time_val_LS)])
        transitionSF_idx = np.zeros([len(time_val_SF)])
        transitionSE_idx = np.zeros([len(time_val_SE)])

        for es in range(0, len(transitionES_idx)):
            transitionES_idx[es] = closest_point(target_sensor_timeval, time_val_ES[es])

        for ls in range(0, len(transitionLS_idx)):
            transitionLS_idx[ls] = closest_point(target_sensor_timeval, time_val_LS[ls])

        for sf in range(0, len(transitionSF_idx)):
            transitionSF_idx[sf] = closest_point(target_sensor_timeval, time_val_SF[sf])

        for se in range(0, len(transitionSE_idx)):
            transitionSE_idx[se] = closest_point(target_sensor_timeval, time_val_SE[se])

        return ground_truth, transitionES_idx, transitionLS_idx, transitionSF_idx, transitionSE_idx, 0, 0, 0, 0, 0, 0, 0, 0

    ground_truth = np.zeros([len(target_sensordata)])
    # ground_truth2 = np.zeros([len(target_sensordata)])

    ##### Find Transition time indices (RA->LW, RD->LW)
    #### 1. LW-RA
    ## Indices
    transitionSF_startIdx_LW_to_RA = closest_point(time_val_LWSF, time_val_RAHS[0]) # Linear transition start LW-RA

    ## Time values
    transitionSF_starttime_LW_to_RA = time_val_LWSF[transitionSF_startIdx_LW_to_RA]
    # transitionHS_endtime_LW_to_RA = time_val_RAHS[0]
    #
    # transitionMS_endtime_LW_to_RA = (1 - swing_percentage) * transitionSF_starttime_LW_to_RA\
    #                                 + swing_percentage * transitionHS_endtime_LW_to_RA

    ## Time index in sensor timeframe
    sensortimeSF_transition_startIdx_LW_to_RA = closest_point(target_sensor_timeval, transitionSF_starttime_LW_to_RA)
    sensortimeMS_transition_endIdx_LW_to_RA = sensortimeSF_transition_startIdx_LW_to_RA + 1 # Midswing

    #### 2. RA-LW
    ## Indices
    transitionHS_endIdx_RA_to_LW = closest_point(time_val_LWHS, time_val_RASF[-1]) # Linear transition end RA-LW

    ## Time values
    transitionSF_starttime_RA_to_LW = time_val_RASF[-1]
    # transitionHS_endtime_RA_to_LW = time_val_LWHS[transitionHS_endIdx_RA_to_LW]
    #
    # transitionMS_endtime_RA_to_LW = (1 - swing_percentage) * transitionSF_starttime_RA_to_LW\
    #                                 + swing_percentage * transitionHS_endtime_RA_to_LW

    ## Time index in sensor timeframe
    sensortimeSF_transition_startIdx_RA_to_LW = closest_point(target_sensor_timeval, transitionSF_starttime_RA_to_LW)
    sensortimeMS_transition_endIdx_RA_to_LW = sensortimeSF_transition_startIdx_RA_to_LW + 1 # Midswing

    #### 3. LW-RD
    ## Indices
    transitionSF_startIdx_LW_to_RD = closest_point(time_val_LWSF, time_val_RDHS[0])  # Linear transition start LW-RA

    ## Time values
    transitionSF_starttime_LW_to_RD = time_val_LWSF[transitionSF_startIdx_LW_to_RD]
    # transitionHS_endtime_LW_to_RD = time_val_RDHS[0]
    #
    # transitionMS_endtime_LW_to_RD = (1 - swing_percentage) * transitionSF_starttime_LW_to_RD \
    #                                 + swing_percentage * transitionHS_endtime_LW_to_RD

    ## Time index in sensor timeframe
    sensortimeSF_transition_startIdx_LW_to_RD = closest_point(target_sensor_timeval, transitionSF_starttime_LW_to_RD)
    sensortimeMS_transition_endIdx_LW_to_RD = sensortimeSF_transition_startIdx_LW_to_RD + 1 # Midswing

    #### 4. RD-LW
    ## Indices
    transitionHS_endIdx_RD_to_LW = closest_point(time_val_LWHS, time_val_RDSF[-1]) # Linear transition end RA-LW

    ## Time values
    transitionSF_starttime_RD_to_LW = time_val_RDSF[-1]
    # transitionHS_endtime_RD_to_LW = time_val_LWHS[transitionHS_endIdx_RD_to_LW]
    #
    # transitionMS_endtime_RD_to_LW = (1 - swing_percentage) * transitionSF_starttime_RD_to_LW\
    #                                 + swing_percentage * transitionHS_endtime_RD_to_LW

    ## Time index in sensor timeframe
    sensortimeSF_transition_startIdx_RD_to_LW = closest_point(target_sensor_timeval, transitionSF_starttime_RD_to_LW)
    sensortimeMS_transition_endIdx_RD_to_LW = sensortimeSF_transition_startIdx_RD_to_LW + 1  # Midswing

    ground_truth[sensortimeMS_transition_endIdx_LW_to_RA:sensortimeSF_transition_startIdx_RA_to_LW] = 1
    ground_truth[sensortimeMS_transition_endIdx_LW_to_RD:sensortimeSF_transition_startIdx_RD_to_LW] = -1

    ### Generate linear transition
    n_transition_LW_to_RA = sensortimeMS_transition_endIdx_LW_to_RA - sensortimeSF_transition_startIdx_LW_to_RA
    n_transition_RA_to_LW = sensortimeMS_transition_endIdx_RA_to_LW - sensortimeSF_transition_startIdx_RA_to_LW
    n_transition_LW_to_RD = sensortimeMS_transition_endIdx_LW_to_RD - sensortimeSF_transition_startIdx_LW_to_RD
    n_transition_RD_to_LW = sensortimeMS_transition_endIdx_RD_to_LW - sensortimeSF_transition_startIdx_RD_to_LW

    lt_LW_to_RA = np.linspace(0, 1, n_transition_LW_to_RA)
    lt_RA_to_LW = np.linspace(1, 0, n_transition_RA_to_LW)
    lt_LW_to_RD = np.linspace(0, -1, n_transition_LW_to_RD)
    lt_RD_to_LW = np.linspace(-1, 0, n_transition_RD_to_LW)

    ground_truth[sensortimeSF_transition_startIdx_LW_to_RA:sensortimeMS_transition_endIdx_LW_to_RA] = lt_LW_to_RA
    ground_truth[sensortimeSF_transition_startIdx_RA_to_LW:sensortimeMS_transition_endIdx_RA_to_LW] = lt_RA_to_LW
    ground_truth[sensortimeSF_transition_startIdx_LW_to_RD:sensortimeMS_transition_endIdx_LW_to_RD] = lt_LW_to_RD
    ground_truth[sensortimeSF_transition_startIdx_RD_to_LW:sensortimeMS_transition_endIdx_RD_to_LW] = lt_RD_to_LW

    transitionES_idx = np.zeros([len(time_val_ES)])
    transitionLS_idx = np.zeros([len(time_val_LS)])
    transitionSF_idx = np.zeros([len(time_val_SF)])
    transitionSE_idx = np.zeros([len(time_val_SE)])

    transitionRAES_idx = np.zeros([len(time_val_RAHS)])
    transitionRALS_idx = np.zeros([len(time_val_RALS)])
    transitionRASF_idx = np.zeros([len(time_val_RASF)])
    transitionRASE_idx = np.zeros([len(time_val_RASE)])

    transitionRDES_idx = np.zeros([len(time_val_RDHS)])
    transitionRDLS_idx = np.zeros([len(time_val_RDLS)])
    transitionRDSF_idx = np.zeros([len(time_val_RDSF)])
    transitionRDSE_idx = np.zeros([len(time_val_RDSE)])

    for es in range(0, len(transitionES_idx)):
        transitionES_idx[es] = closest_point(target_sensor_timeval, time_val_ES[es])

    for ls in range(0, len(transitionLS_idx)):
        transitionLS_idx[ls] = closest_point(target_sensor_timeval, time_val_LS[ls])

    for sf in range(0, len(transitionSF_idx)):
        transitionSF_idx[sf] = closest_point(target_sensor_timeval, time_val_SF[sf])

    for se in range(0, len(transitionSE_idx)):
        transitionSE_idx[se] = closest_point(target_sensor_timeval, time_val_SE[se])

    for raes in range(0, len(transitionRAES_idx)):
        transitionRAES_idx[raes] = closest_point(target_sensor_timeval, time_val_RAHS[raes])

    for rals in range(0, len(transitionRALS_idx)):
        transitionRALS_idx[rals] = closest_point(target_sensor_timeval, time_val_RALS[rals])

    for rasf in range(0, len(transitionRASF_idx)):
        transitionRASF_idx[rasf] = closest_point(target_sensor_timeval, time_val_RASF[rasf])

    for rase in range(0, len(transitionRASE_idx)):
        transitionRASE_idx[rase] = closest_point(target_sensor_timeval, time_val_RASE[rase])

    for rdes in range(0, len(transitionRDES_idx)):
        transitionRDES_idx[rdes] = closest_point(target_sensor_timeval, time_val_RDHS[rdes])

    for rdls in range(0, len(transitionRDLS_idx)):
        transitionRDLS_idx[rdls] = closest_point(target_sensor_timeval, time_val_RDLS[rdls])

    for rdsf in range(0, len(transitionRDSF_idx)):
        transitionRDSF_idx[rdsf] = closest_point(target_sensor_timeval, time_val_RDSF[rdsf])

    for rdse in range(0, len(transitionRDSE_idx)):
        transitionRDSE_idx[rdse] = closest_point(target_sensor_timeval, time_val_RDSE[rdse])

    ## Check
    # plt.plot(ground_truth)
    # plt.plot(ground_truth2)

    return ground_truth, transitionES_idx, transitionLS_idx, transitionSF_idx, transitionSE_idx, \
           transitionRAES_idx, transitionRALS_idx, transitionRASF_idx, transitionRASE_idx, \
           transitionRDES_idx, transitionRDLS_idx, transitionRDSF_idx, transitionRDSE_idx

def Ramp_GT3_nolin(target_data_dict, ref_signal_topic, swing_percentage): ### Previous version
    '''
    Interpolate Grount Truth (GT) slope values aligned with sensor timestamp
    * Ascent:1, Descent:-1, LW:0

    ref_signal_topic: enter the topic that you will use for reference
    ex: '/SensorData'

    Transition: linearly smoothed between SF-Midswing-HS in each transition
    '''

    if target_data_dict.__class__ == str:
        target_rawdata = jb.load(target_data_dict)
    elif target_data_dict.__class__ == dict:
        target_rawdata = target_data_dict
    else:
        print('Error! Check the input type (Directory or Dictionary)')

    ### Function test
    # workspace_path = os.getcwd()
    # cacheDir = workspace_path + '/Ramp_Data/'
    #
    # loadDir = cacheDir + 'TF02v2/'  ### Put your Cache file directory here
    # loadfileDir = loadDir + '23'
    # target_rawdata = jb.load(loadfileDir)
    # ref_signal_topic = str('/SensorData')
    # swing_percentage = 0.5
    ### Function test


    target_sensordata = target_rawdata[ref_signal_topic]
    target_sensor_timeval = target_sensordata['header']

    ##### Align time segment to State
    time_val = target_rawdata['/fsm/State']['header']
    state = target_rawdata['/fsm/State']['state']

    time_idx_ES = []
    time_idx_LS = []
    time_idx_SE = []
    time_idx_SF = []

    i = 0
    for phase in state:
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

    ### Heel Strike Timings
    time_idx_RAHS = []  ### Ramp Ascend Heel Strike
    time_idx_RDHS = []  ### Ramp Descend Heel Strike
    time_idx_LWHS = []  ### Level Walking Heel Strike

    for j in range(0, len(time_idx_ES)):
        if 'RA' in state[time_idx_ES[j]]:
            time_idx_RAHS.append(time_idx_ES[j])

        if 'RD' in state[time_idx_ES[j]]:
            time_idx_RDHS.append(time_idx_ES[j])

        if 'LW' in state[time_idx_ES[j]]:
            time_idx_LWHS.append(time_idx_ES[j])

    time_val_RAHS = time_val[time_idx_RAHS].array
    time_val_RDHS = time_val[time_idx_RDHS].array
    time_val_LWHS = time_val[time_idx_LWHS].array

    ### Swing Flexion Timings
    time_idx_RASF = []  ### Ramp Ascend Swing Flexion
    time_idx_RDSF = []  ### Ramp Descend Swing Flexion
    time_idx_LWSF = []  ### Level Walking Swing Flexion

    for j in range(0, len(time_idx_SF)):
        if 'RA' in state[time_idx_SF[j]]:
            time_idx_RASF.append(time_idx_SF[j])

        if 'RD' in state[time_idx_SF[j]]:
            time_idx_RDSF.append(time_idx_SF[j])

        if 'LW' in state[time_idx_SF[j]]:
            time_idx_LWSF.append(time_idx_SF[j])

    time_val_RASF = time_val[time_idx_RASF].array
    time_val_RDSF = time_val[time_idx_RDSF].array
    time_val_LWSF = time_val[time_idx_LWSF].array

    ### Late Stance Timings
    time_idx_RALS = []  ### Ramp Ascend Late Stance
    time_idx_RDLS = []  ### Ramp Descend Late Stance
    time_idx_LWLS = []  ### Level Walking Late Stance

    for j in range(0, len(time_idx_LS)):
        if 'RA' in state[time_idx_LS[j]]:
            time_idx_RALS.append(time_idx_LS[j])

        if 'RD' in state[time_idx_LS[j]]:
            time_idx_RDLS.append(time_idx_LS[j])

        if 'LW' in state[time_idx_LS[j]]:
            time_idx_LWLS.append(time_idx_LS[j])

    time_val_RALS = time_val[time_idx_RALS].array
    time_val_RDLS = time_val[time_idx_RDLS].array
    time_val_LWLS = time_val[time_idx_LWLS].array
    time_val_ES = time_val[time_idx_ES].array
    time_val_LS = time_val[time_idx_LS].array
    time_val_SF = time_val[time_idx_SF].array

    if len(time_val_RDHS) < 1: ### If there is no ramp descent, It's LW trial
        ground_truth = np.zeros([len(target_sensordata)])

        transitionES_idx = np.zeros([len(time_val_ES)])
        transitionLS_idx = np.zeros([len(time_val_LS)])
        transitionSF_idx = np.zeros([len(time_val_SF)])

        for es in range(0, len(transitionES_idx)):
            transitionES_idx[es] = closest_point(target_sensor_timeval, time_val_ES[es])

        for ls in range(0, len(transitionLS_idx)):
            transitionLS_idx[ls] = closest_point(target_sensor_timeval, time_val_LS[ls])

        for sf in range(0, len(transitionSF_idx)):
            transitionSF_idx[sf] = closest_point(target_sensor_timeval, time_val_SF[sf])

        return ground_truth, transitionES_idx, transitionLS_idx, transitionSF_idx

    ground_truth = np.zeros([len(target_sensordata)])
    ground_truth2 = np.zeros([len(target_sensordata)])

    ##### Find Transition time indices (RA->LW, RD->LW)
    #### 1. LW-RA
    ## Indices
    transitionSF_startIdx_LW_to_RA = closest_point(time_val_LWSF, time_val_RAHS[0]) # Linear transition start LW-RA

    ## Time values
    transitionSF_starttime_LW_to_RA = time_val_LWSF[transitionSF_startIdx_LW_to_RA]
    transitionHS_endtime_LW_to_RA = time_val_RAHS[0]

    transitionMS_endtime_LW_to_RA = (1 - swing_percentage) * transitionSF_starttime_LW_to_RA\
                                    + swing_percentage * transitionHS_endtime_LW_to_RA

    ## Time index in sensor timeframe
    sensortimeSF_transition_startIdx_LW_to_RA = closest_point(target_sensor_timeval, transitionSF_starttime_LW_to_RA)
    sensortimeMS_transition_endIdx_LW_to_RA = closest_point(target_sensor_timeval, transitionMS_endtime_LW_to_RA) # Midswing

    #### 2. RA-LW
    ## Indices
    transitionHS_endIdx_RA_to_LW = closest_point(time_val_LWHS, time_val_RASF[-1]) # Linear transition end RA-LW

    ## Time values
    transitionSF_starttime_RA_to_LW = time_val_RASF[-1]
    transitionHS_endtime_RA_to_LW = time_val_LWHS[transitionHS_endIdx_RA_to_LW]

    transitionMS_endtime_RA_to_LW = (1 - swing_percentage) * transitionSF_starttime_RA_to_LW\
                                    + swing_percentage * transitionHS_endtime_RA_to_LW

    ## Time index in sensor timeframe
    sensortimeSF_transition_startIdx_RA_to_LW = closest_point(target_sensor_timeval, transitionSF_starttime_RA_to_LW)
    sensortimeMS_transition_endIdx_RA_to_LW = closest_point(target_sensor_timeval, transitionMS_endtime_RA_to_LW) # Midswing

    #### 3. LW-RD
    ## Indices
    transitionSF_startIdx_LW_to_RD = closest_point(time_val_LWSF, time_val_RDHS[0])  # Linear transition start LW-RA

    ## Time values
    transitionSF_starttime_LW_to_RD = time_val_LWSF[transitionSF_startIdx_LW_to_RD]
    transitionHS_endtime_LW_to_RD = time_val_RDHS[0]

    transitionMS_endtime_LW_to_RD = (1 - swing_percentage) * transitionSF_starttime_LW_to_RD \
                                    + swing_percentage * transitionHS_endtime_LW_to_RD

    ## Time index in sensor timeframe
    sensortimeSF_transition_startIdx_LW_to_RD = closest_point(target_sensor_timeval, transitionSF_starttime_LW_to_RD)
    sensortimeMS_transition_endIdx_LW_to_RD = closest_point(target_sensor_timeval, transitionMS_endtime_LW_to_RD) # Midswing

    #### 4. RD-LW
    ## Indices
    transitionHS_endIdx_RD_to_LW = closest_point(time_val_LWHS, time_val_RDSF[-1]) # Linear transition end RA-LW

    ## Time values
    transitionSF_starttime_RD_to_LW = time_val_RDSF[-1]
    transitionHS_endtime_RD_to_LW = time_val_LWHS[transitionHS_endIdx_RD_to_LW]

    transitionMS_endtime_RD_to_LW = (1 - swing_percentage) * transitionSF_starttime_RD_to_LW\
                                    + swing_percentage * transitionHS_endtime_RD_to_LW

    ## Time index in sensor timeframe
    sensortimeSF_transition_startIdx_RD_to_LW = closest_point(target_sensor_timeval, transitionSF_starttime_RD_to_LW)
    sensortimeMS_transition_endIdx_RD_to_LW = closest_point(target_sensor_timeval, transitionMS_endtime_RD_to_LW)  # Midswing

    ground_truth[sensortimeMS_transition_endIdx_LW_to_RA:sensortimeSF_transition_startIdx_RA_to_LW] = 1
    ground_truth[sensortimeMS_transition_endIdx_LW_to_RD:sensortimeSF_transition_startIdx_RD_to_LW] = -1

    ### Generate linear transition
    n_transition_LW_to_RA = sensortimeMS_transition_endIdx_LW_to_RA - sensortimeSF_transition_startIdx_LW_to_RA
    n_transition_RA_to_LW = sensortimeMS_transition_endIdx_RA_to_LW - sensortimeSF_transition_startIdx_RA_to_LW
    n_transition_LW_to_RD = sensortimeMS_transition_endIdx_LW_to_RD - sensortimeSF_transition_startIdx_LW_to_RD
    n_transition_RD_to_LW = sensortimeMS_transition_endIdx_RD_to_LW - sensortimeSF_transition_startIdx_RD_to_LW

    # lt_LW_to_RA = np.linspace(0, 1, n_transition_LW_to_RA)
    # lt_RA_to_LW = np.linspace(1, 0, n_transition_RA_to_LW)
    # lt_LW_to_RD = np.linspace(0, -1, n_transition_LW_to_RD)
    # lt_RD_to_LW = np.linspace(-1, 0, n_transition_RD_to_LW)
    #
    # ground_truth[sensortimeSF_transition_startIdx_LW_to_RA:sensortimeMS_transition_endIdx_LW_to_RA] = lt_LW_to_RA
    # ground_truth[sensortimeSF_transition_startIdx_RA_to_LW:sensortimeMS_transition_endIdx_RA_to_LW] = lt_RA_to_LW
    # ground_truth[sensortimeSF_transition_startIdx_LW_to_RD:sensortimeMS_transition_endIdx_LW_to_RD] = lt_LW_to_RD
    # ground_truth[sensortimeSF_transition_startIdx_RD_to_LW:sensortimeMS_transition_endIdx_RD_to_LW] = lt_RD_to_LW

    transitionES_idx = np.zeros([len(time_val_ES)])
    transitionLS_idx = np.zeros([len(time_val_LS)])
    transitionSF_idx = np.zeros([len(time_val_SF)])

    for es in range(0, len(transitionES_idx)):
        transitionES_idx[es] = closest_point(target_sensor_timeval, time_val_ES[es])

    for ls in range(0, len(transitionLS_idx)):
        transitionLS_idx[ls] = closest_point(target_sensor_timeval, time_val_LS[ls])

    for sf in range(0, len(transitionSF_idx)):
        transitionSF_idx[sf] = closest_point(target_sensor_timeval, time_val_SF[sf])

    ## Check
    # plt.plot(ground_truth)
    # plt.plot(ground_truth2)

    return ground_truth, transitionES_idx, transitionLS_idx, transitionSF_idx

# def Stair_GT_start(loadfileDir, ref_signal_topic, swing_percentage): ### Previous version
#     '''
#     Interpolate Grount Truth (GT) slope values aligned with sensor timestamp
#     * Ascent:1, Descent:-1, LW:0
#
#     ref_signal_topic: enter the topic that you will use for reference
#     ex: '/SensorData'
#
#     Transition: linearly smoothed between SF-Midswing-HS in each transition
#     '''
#
#     ### Function test
#     workspace_path = os.getcwd()
#     cacheDir = workspace_path + '/Stair_Data_Raw/'
#
#     loadDir = cacheDir + 'TF15/'  ### Put your Cache file directory here
#     loadfileDir = loadDir + 'OSL_Stair_Preset_1.bag'
#     topics = rsbg.checkTopics(loadfileDir)
#     try:
#         target_rawdata = rsbg.read2var(loadfileDir, topics)
#     except:
#         target_rawdata = rsbg.read2var2(loadfileDir, topics)
#
#     ref_signal_topic = str('/SensorData')
#     swing_percentage = 1
#     ### Function test
#
#
#     target_sensordata = target_rawdata[ref_signal_topic]
#     target_sensor_timeval = target_sensordata['header']
#
#     ##### Align time segment to State
#     time_val = target_rawdata['/fsm/State']['header']
#     state = target_rawdata['/fsm/State']['state']
#
#     time_idx_ES = []
#     time_idx_LS = []
#     time_idx_SE = []
#     time_idx_SF = []
#
#     i = 0
#     for phase in state:
#         if 'EarlyStance' in phase:
#             time_idx_ES.append(i)
#             i += 1
#         elif 'LateStance' in phase:
#             time_idx_LS.append(i)
#             i += 1
#         elif 'SwingExtension' in phase:
#             time_idx_SE.append(i)
#             i += 1
#         elif 'SwingFlexion' in phase:
#             time_idx_SF.append(i)
#             i += 1
#         else:
#             i += 1
#
#     ### Heel Strike Timings
#     time_idx_SAHS = []  ### Ramp Ascend Heel Strike
#     time_idx_SDHS = []  ### Ramp Descend Heel Strike
#     time_idx_LWHS = []  ### Level Walking Heel Strike
#
#     for j in range(0, len(time_idx_ES)):
#         if 'SA' in state[time_idx_ES[j]]:
#             time_idx_SAHS.append(time_idx_ES[j])
#
#         if 'SD' in state[time_idx_ES[j]]:
#             time_idx_SDHS.append(time_idx_ES[j])
#
#         if 'LW' in state[time_idx_ES[j]]:
#             time_idx_LWHS.append(time_idx_ES[j])
#
#     time_val_SAHS = time_val[time_idx_SAHS].array
#     time_val_SDHS = time_val[time_idx_SDHS].array
#     time_val_LWHS = time_val[time_idx_LWHS].array
#
#     ### Late Stance Timings
#     time_idx_SALS = []  ### Ramp Ascend Late Stance
#     time_idx_SDLS = []  ### Ramp Descend Late Stance
#     time_idx_LWLS = []  ### Level Walking Late Stance
#
#     for j in range(0, len(time_idx_LS)):
#         if 'SA' in state[time_idx_LS[j]]:
#             time_idx_SALS.append(time_idx_LS[j])
#
#         if 'SD' in state[time_idx_LS[j]]:
#             time_idx_SDLS.append(time_idx_LS[j])
#
#         if 'LW' in state[time_idx_LS[j]]:
#             time_idx_LWLS.append(time_idx_LS[j])
#
#     time_val_SALS = time_val[time_idx_SALS].array
#     time_val_SDLS = time_val[time_idx_SDLS].array
#     time_val_LWLS = time_val[time_idx_LWLS].array
#
#     ### Swing Flexion Timings
#     time_idx_SASF = []  ### Ramp Ascend Swing Flexion
#     time_idx_SDSF = []  ### Ramp Descend Swing Flexion
#     time_idx_LWSF = []  ### Level Walking Swing Flexion
#
#     for j in range(0, len(time_idx_SF)):
#         if 'SA' in state[time_idx_SF[j]]:
#             time_idx_SASF.append(time_idx_SF[j])
#
#         if 'SD' in state[time_idx_SF[j]]:
#             time_idx_SDSF.append(time_idx_SF[j])
#
#         if 'LW' in state[time_idx_SF[j]]:
#             time_idx_LWSF.append(time_idx_SF[j])
#
#     time_val_SASF = time_val[time_idx_SASF].array
#     time_val_SDSF = time_val[time_idx_SDSF].array
#     time_val_LWSF = time_val[time_idx_LWSF].array
#
#     ### Swing Extension Timings
#     time_idx_SASE = []  ### Ramp Ascend Swing Flexion
#     time_idx_SDSE = []  ### Ramp Descend Swing Flexion
#     time_idx_LWSE = []  ### Level Walking Swing Flexion
#
#     for j in range(0, len(time_idx_SE)):
#         if 'SA' in state[time_idx_SE[j]]:
#             time_idx_SASE.append(time_idx_SE[j])
#
#         if 'SD' in state[time_idx_SE[j]]:
#             time_idx_SDSE.append(time_idx_SE[j])
#
#         if 'LW' in state[time_idx_SE[j]]:
#             time_idx_LWSE.append(time_idx_SE[j])
#
#     time_val_SASE = time_val[time_idx_SASE].array
#     time_val_SDSE = time_val[time_idx_SDSE].array
#     time_val_LWSE = time_val[time_idx_LWSE].array
#
#     time_val_ES = time_val[time_idx_ES].array
#     time_val_LS = time_val[time_idx_LS].array
#     time_val_SF = time_val[time_idx_SF].array
#     time_val_SE = time_val[time_idx_SE].array
#
#     if len(time_val_SDHS) < 1: ### If there is no ramp descent, It's LW trial
#         ground_truth = np.zeros([len(target_sensordata)])
#
#         transitionES_idx = np.zeros([len(time_val_ES)])
#         transitionLS_idx = np.zeros([len(time_val_LS)])
#         transitionSF_idx = np.zeros([len(time_val_SF)])
#         transitionSE_idx = np.zeros([len(time_val_SE)])
#
#         for es in range(0, len(transitionES_idx)):
#             transitionES_idx[es] = closest_point(target_sensor_timeval, time_val_ES[es])
#
#         for ls in range(0, len(transitionLS_idx)):
#             transitionLS_idx[ls] = closest_point(target_sensor_timeval, time_val_LS[ls])
#
#         for sf in range(0, len(transitionSF_idx)):
#             transitionSF_idx[sf] = closest_point(target_sensor_timeval, time_val_SF[sf])
#
#         for se in range(0, len(transitionSE_idx)):
#             transitionSE_idx[se] = closest_point(target_sensor_timeval, time_val_SE[se])
#
#         return ground_truth, transitionES_idx, transitionLS_idx, transitionSF_idx, transitionSE_idx, 0, 0, 0, 0, 0, 0, 0, 0
#
#     ground_truth = np.zeros([len(target_sensordata)])
#     ground_truth2 = np.zeros([len(target_sensordata)])
#
#     ##### Find Transition time indices (SA->LW, SD->LW)
#     #### 1. LW-SA
#     ## Indices
#     transitionSF_startIdx_LW_to_SA = closest_point(time_val_LWSF, time_val_SAHS[0]) # Linear transition start LW-SA
#
#     ## Time values
#     transitionSF_starttime_LW_to_SA = time_val_LWSF[transitionSF_startIdx_LW_to_SA]
#     transitionHS_endtime_LW_to_SA = time_val_SAHS[0]
#
#     transitionMS_starttime_LW_to_SA = (1 - swing_percentage) * transitionSF_starttime_LW_to_SA +\
#                                       swing_percentage * transitionHS_endtime_LW_to_SA
#
#     ## Time index in sensor timeframe
#     sensortimeMS_transition_startIdx_LW_to_SA = closest_point(target_sensor_timeval,
#                                                             transitionMS_starttime_LW_to_SA)  # Midswing
#     sensortimeHS_transition_endIdx_LW_to_SA = closest_point(target_sensor_timeval, transitionHS_endtime_LW_to_SA) # HS
#
#     #### 2. SA-LW
#     ## Indices
#     transitionHS_endIdx_SA_to_LW = closest_point(time_val_LWHS, time_val_SASF[-1]) # Linear transition end SA-LW
#
#     ## Time values
#     transitionSF_starttime_SA_to_LW = time_val_SASF[-1]
#     transitionHS_endtime_SA_to_LW = time_val_LWHS[transitionHS_endIdx_SA_to_LW]
#
#     transitionMS_starttime_SA_to_LW = (1 - swing_percentage) * transitionSF_starttime_SA_to_LW +\
#                                       swing_percentage * transitionHS_endtime_SA_to_LW
#
#     ## Time index in sensor timeframe
#     sensortimeMS_transition_startIdx_SA_to_LW = closest_point(target_sensor_timeval,
#                                                             transitionMS_starttime_SA_to_LW)  # Midswing
#     sensortimeHS_transition_endIdx_SA_to_LW = closest_point(target_sensor_timeval, transitionHS_endtime_SA_to_LW) # HS
#
#     #### 3. LW-SD
#     ## Indices
#     transitionSF_startIdx_LW_to_SD = closest_point(time_val_LWSF, time_val_SDHS[0])  # Linear transition start LW-SA
#
#     ## Time values
#     transitionSF_starttime_LW_to_SD = time_val_LWSF[transitionSF_startIdx_LW_to_SD]
#     transitionHS_endtime_LW_to_SD = time_val_SDHS[0]
#
#     transitionMS_starttime_LW_to_SD = (1 - swing_percentage) * transitionSF_starttime_LW_to_SD +\
#                                       swing_percentage * transitionHS_endtime_LW_to_SD
#
#     ## Time index in sensor timeframe
#     sensortimeMS_transition_startIdx_LW_to_SD = closest_point(target_sensor_timeval, transitionMS_starttime_LW_to_SD)
#     sensortimeHS_transition_endIdx_LW_to_SD = closest_point(target_sensor_timeval, transitionHS_endtime_LW_to_SD) # Midswing
#
#     #### 4. SD-LW
#     ## Indices
#     transitionHS_endIdx_SD_to_LW = closest_point(time_val_LWHS, time_val_SDSF[-1]) # Linear transition end SA-LW
#
#     ## Time values
#     transitionSF_starttime_SD_to_LW = time_val_SDSF[-1]
#     transitionHS_endtime_SD_to_LW = time_val_LWHS[transitionHS_endIdx_SD_to_LW]
#
#     transitionMS_starttime_SD_to_LW = (1 - swing_percentage) * transitionSF_starttime_SD_to_LW \
#                                     + swing_percentage * transitionHS_endtime_SD_to_LW
#
#     ## Time index in sensor timeframe
#     sensortimeMS_transition_startIdx_SD_to_LW = closest_point(target_sensor_timeval, transitionMS_starttime_SD_to_LW)
#     sensortimeHS_transition_endIdx_SD_to_LW = closest_point(target_sensor_timeval, transitionHS_endtime_SD_to_LW)  # Midswing
#
#     ground_truth[sensortimeHS_transition_endIdx_LW_to_SA:sensortimeMS_transition_startIdx_SA_to_LW] = 1
#     ground_truth[sensortimeHS_transition_endIdx_LW_to_SD:sensortimeMS_transition_startIdx_SD_to_LW] = -1
#
#     ### Generate linear transition
#     n_transition_LW_to_SA = sensortimeHS_transition_endIdx_LW_to_SA - sensortimeMS_transition_startIdx_LW_to_SA
#     n_transition_SA_to_LW = sensortimeHS_transition_endIdx_SA_to_LW - sensortimeMS_transition_startIdx_SA_to_LW
#     n_transition_LW_to_SD = sensortimeHS_transition_endIdx_LW_to_SD - sensortimeMS_transition_startIdx_LW_to_SD
#     n_transition_SD_to_LW = sensortimeHS_transition_endIdx_SD_to_LW - sensortimeMS_transition_startIdx_SD_to_LW
#
#     lt_LW_to_SA = np.linspace(0, 1, n_transition_LW_to_SA)
#     lt_SA_to_LW = np.linspace(1, 0, n_transition_SA_to_LW)
#     lt_LW_to_SD = np.linspace(0, -1, n_transition_LW_to_SD)
#     lt_SD_to_LW = np.linspace(-1, 0, n_transition_SD_to_LW)
#
#     ground_truth[sensortimeMS_transition_startIdx_LW_to_SA:sensortimeHS_transition_endIdx_LW_to_SA] = lt_LW_to_SA
#     ground_truth[sensortimeMS_transition_startIdx_SA_to_LW:sensortimeHS_transition_endIdx_SA_to_LW] = lt_SA_to_LW
#     ground_truth[sensortimeMS_transition_startIdx_LW_to_SD:sensortimeHS_transition_endIdx_LW_to_SD] = lt_LW_to_SD
#     ground_truth[sensortimeMS_transition_startIdx_SD_to_LW:sensortimeHS_transition_endIdx_SD_to_LW] = lt_SD_to_LW
#
#     transitionES_idx = np.zeros([len(time_val_ES)])
#     transitionLS_idx = np.zeros([len(time_val_LS)])
#     transitionSF_idx = np.zeros([len(time_val_SF)])
#     transitionSE_idx = np.zeros([len(time_val_SE)])
#
#     transitionSAES_idx = np.zeros([len(time_val_SAHS)])
#     transitionSALS_idx = np.zeros([len(time_val_SALS)])
#     transitionSASF_idx = np.zeros([len(time_val_SASF)])
#     transitionSASE_idx = np.zeros([len(time_val_SASE)])
#
#     transitionSDES_idx = np.zeros([len(time_val_SDHS)])
#     transitionSDLS_idx = np.zeros([len(time_val_SDLS)])
#     transitionSDSF_idx = np.zeros([len(time_val_SDSF)])
#     transitionSDSE_idx = np.zeros([len(time_val_SDSE)])
#
#     for es in range(0, len(transitionES_idx)):
#         transitionES_idx[es] = closest_point(target_sensor_timeval, time_val_ES[es])
#
#     for ls in range(0, len(transitionLS_idx)):
#         transitionLS_idx[ls] = closest_point(target_sensor_timeval, time_val_LS[ls])
#
#     for sf in range(0, len(transitionSF_idx)):
#         transitionSF_idx[sf] = closest_point(target_sensor_timeval, time_val_SF[sf])
#
#     for se in range(0, len(transitionSE_idx)):
#         transitionSE_idx[se] = closest_point(target_sensor_timeval, time_val_SE[se])
#
#     for raes in range(0, len(transitionSAES_idx)):
#         transitionSAES_idx[raes] = closest_point(target_sensor_timeval, time_val_SAHS[raes])
#
#     for rals in range(0, len(transitionSALS_idx)):
#         transitionSALS_idx[rals] = closest_point(target_sensor_timeval, time_val_SALS[rals])
#
#     for rasf in range(0, len(transitionSASF_idx)):
#         transitionSASF_idx[rasf] = closest_point(target_sensor_timeval, time_val_SASF[rasf])
#
#     for rase in range(0, len(transitionSASE_idx)):
#         transitionSASE_idx[rase] = closest_point(target_sensor_timeval, time_val_SASE[rase])
#
#     for rdes in range(0, len(transitionSDES_idx)):
#         transitionSDES_idx[rdes] = closest_point(target_sensor_timeval, time_val_SDHS[rdes])
#
#     for rdls in range(0, len(transitionSDLS_idx)):
#         transitionSDLS_idx[rdls] = closest_point(target_sensor_timeval, time_val_SDLS[rdls])
#
#     for rdsf in range(0, len(transitionSDSF_idx)):
#         transitionSDSF_idx[rdsf] = closest_point(target_sensor_timeval, time_val_SDSF[rdsf])
#
#     for rdse in range(0, len(transitionSDSE_idx)):
#         transitionSDSE_idx[rdse] = closest_point(target_sensor_timeval, time_val_SDSE[rdse])
#
#     ## Check
#     # plt.plot(ground_truth)
#     # plt.plot(ground_truth2)
#
#     return ground_truth, transitionES_idx, transitionLS_idx, transitionSF_idx, transitionSE_idx, \
#            transitionSAES_idx, transitionSALS_idx, transitionSASF_idx, transitionSASE_idx, \
#            transitionSDES_idx, transitionSDLS_idx, transitionSDSF_idx, transitionSDSE_idx

def Stair_GT(target_data_dict, ref_signal_topic = '/Sensordata'): ### Previous version

    '''
     Interpolate Grount Truth (GT) slope values aligned with sensor timestamp
     * Ascent:1, Descent:-1, LW:0

     ref_signal_topic: enter the topic that you will use for reference
     ex: '/SensorData'

     SA: Begins with Swing Extension
     SD: Begins with Swing Flexion
     '''

    if target_data_dict.__class__ == str:
        target_rawdata = jb.load(target_data_dict)
    elif target_data_dict.__class__ == dict:
        target_rawdata = target_data_dict
    else:
        print('Error! Check the input type (Directory or Dictionary)')

    ### Function test
    workspace_path = os.getcwd()
    cacheDir = workspace_path + '/Stair_Data_Raw/'

    loadDir = cacheDir + 'TF15/'  ### Put your Cache file directory here
    loadfileDir = loadDir + 'OSL_Stair_Preset_1.dict'
    target_rawdata = jb.load(loadfileDir)
    ref_signal_topic = str('/SensorData')
    ### Function test

    target_sensordata = target_rawdata[ref_signal_topic]
    target_sensor_timeval = target_sensordata['header']

    ##### Align time segment to State
    time_val = target_rawdata['/fsm/State']['header']
    state = target_rawdata['/fsm/State']['state']

    ### Find SA start points
    SA_startpoint_index = []
    SA_startpoint_time = []

    for s in range(0, len(state)):

        if 'SA' not in state[s]:
            continue

        if s==0:
            SA_startpoint_index.append(s)
            SA_startpoint_time.append(time_val[s])

        elif 'SA' not in state[s - 1]:
            SA_startpoint_index.append(s)
            SA_startpoint_time.append(time_val[s])

    ### Find SA end points
    SA_endpoint_index = []
    SA_endpoint_time = []

    for s in range(0, len(state)):

        if 'SA' not in state[s]:
            continue

        elif 'SA' not in state[s + 1]:
            SA_endpoint_index.append(s)
            SA_endpoint_time.append(time_val[s])

    ### Find SD start points
    SD_startpoint_index = []
    SD_startpoint_time = []

    for s in range(0, len(state)):

        if 'SD' not in state[s]:
            continue

        if s==0:
            SD_startpoint_index.append(s)
            SD_startpoint_time.append(time_val[s])

        elif 'SD' not in state[s - 1]:
            SD_startpoint_index.append(s)
            SD_startpoint_time.append(time_val[s])

    ### Find SD end points
    SD_endpoint_index = []
    SD_endpoint_time = []

    for s in range(0, len(state)):

        if 'SD' not in state[s]:
            continue

        elif 'SD' not in state[s + 1]:
            SD_endpoint_index.append(s)
            SD_endpoint_time.append(time_val[s])

    ground_truth = np.zeros(len(target_sensordata))

    for sa in range(0, len(SA_endpoint_time)):
        SA_start = closest_point(target_sensor_timeval,SA_startpoint_time[sa])
        SA_end = closest_point(target_sensor_timeval, SA_endpoint_time[sa])

        ground_truth[SA_start:SA_end] = 1

    for sd in range(0, len(SD_endpoint_time)):
        SD_start = closest_point(target_sensor_timeval, SD_startpoint_time[sd])
        SD_end = closest_point(target_sensor_timeval, SD_endpoint_time[sd])

        ground_truth[SD_start:SD_end] = -1

    ## Check
    # plt.plot(ground_truth[sensortimeLS_transition_startIdx_LW_to_RA:sensortimeLS_transition_endIdx_LW_to_RA])
    # plt.plot(ground_truth[sensortimeSF_transition_startIdx_LW_to_RA:sensortimeSF_transition_endIdx_LW_to_RA])
    # plt.plot(ground_truth)

    return ground_truth

def Ramp_GT_LS(target_data_dict, ref_signal_topic):
    '''
    Interpolate Grount Truth (GT) slope values aligned with sensor timestamp
    * Ascent:1, Descent:-1, LW:0

    ref_signal_topic: enter the topic that you will use for reference
    ex: '/SensorData'
    '''


    ### Function test part - start
    # workspace_path = os.getcwd()
    # loadDir = workspace_path + '/Ramp_Data/TF16v2/'  ### Put your Cache file directory here
    # loadfileDir = loadDir + '1'
    # target_rawdata = jb.load(loadfileDir)
    #
    # ref_signal_topic = str('/SensorData')
    ### Function test part - end


    if target_data_dict.__class__ == str:
        target_rawdata = jb.load(target_data_dict)
    elif target_data_dict.__class__ == dict:
        target_rawdata = target_data_dict
    else:
        print('Error! Check the input type (Directory or Dictionary)')

    target_sensordata = target_rawdata[ref_signal_topic]
    target_sensor_timeval = target_sensordata['header']

    ##### Align time segment to State
    time_val = target_rawdata['/fsm/State']['header']
    state = target_rawdata['/fsm/State']['state']

    ### Phase indexing
    time_idx_ES = []
    time_idx_LS = []
    time_idx_SE = []
    time_idx_SF = []

    i = 0
    for phase in state:
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

    ### Mode indexing: HS
    time_idx_RAHS = []  ### Ramp Ascend Heel Strike
    time_idx_RDHS = []  ### Ramp Descend Heel Strike
    time_idx_LWHS = []  ### Level Walking Heel Strike

    for j in range(0, len(time_idx_ES)):
        if 'RA' in state[time_idx_ES[j]]:
            time_idx_RAHS.append(time_idx_ES[j])

        if 'RD' in state[time_idx_ES[j]]:
            time_idx_RDHS.append(time_idx_ES[j])

        if 'LW' in state[time_idx_ES[j]]:
            time_idx_LWHS.append(time_idx_ES[j])

    time_val_RAHS = time_val[time_idx_RAHS].array
    time_val_RDHS = time_val[time_idx_RDHS].array
    time_val_LWHS = time_val[time_idx_LWHS].array

    if len(time_val_RDHS) < 1: ### If there is no ramp descent
        return

    ### Mode indexing: LS
    time_idx_RALS = []  ### Ramp Ascend Heel Strike
    time_idx_RDLS = []  ### Ramp Descend Heel Strike
    time_idx_LWLS = []  ### Level Walking Heel Strike

    for j in range(0, len(time_idx_LS)):
        if 'RA' in state[time_idx_LS[j]]:
            time_idx_RALS.append(time_idx_LS[j])

        if 'RD' in state[time_idx_LS[j]]:
            time_idx_RDLS.append(time_idx_LS[j])

        if 'LW' in state[time_idx_LS[j]]:
            time_idx_LWLS.append(time_idx_LS[j])

    time_val_RALS = time_val[time_idx_RALS].array
    time_val_RDLS = time_val[time_idx_RDLS].array
    time_val_LWLS = time_val[time_idx_LWLS].array

    if len(time_val_RDLS) < 1: ### If there is no ramp descent
        return

    ### Labeling
    ground_truth = np.zeros([len(target_sensordata)])

    ### Find Transition time indices (RA<->LW, RD<->LW)
    transitionIdx_LW_to_RA = closest_point(time_val_LWLS, time_val_RALS[0])
    transitionIdx_LW_to_RD = closest_point(time_val_LWLS, time_val_RDLS[0])

    ### Transition timings (RA->LW, RD->LW)
    transition_time_RA_to_LW = time_val_RALS[-1]
    transition_time_RD_to_LW = time_val_RDLS[-1]
    transition_time_LW_to_RA = time_val_LWLS[transitionIdx_LW_to_RA]
    transition_time_LW_to_RD = time_val_LWLS[transitionIdx_LW_to_RD]

    sensortime_transition_timeIdx_LW_to_RA = closest_point(target_sensor_timeval,
                                                           transition_time_LW_to_RA)
    sensortime_transition_timeIdx_RA_to_LW = closest_point(target_sensor_timeval,
                                                           transition_time_RA_to_LW)
    sensortime_transition_timeIdx_LW_to_RD = closest_point(target_sensor_timeval,
                                                           transition_time_LW_to_RD)
    sensortime_transition_timeIdx_RD_to_LW = closest_point(target_sensor_timeval,
                                                           transition_time_RD_to_LW)

    ground_truth[sensortime_transition_timeIdx_LW_to_RA:sensortime_transition_timeIdx_RA_to_LW] = 1
    ground_truth[sensortime_transition_timeIdx_LW_to_RD:sensortime_transition_timeIdx_RD_to_LW] = -1

    return ground_truth

def Ramp_GT_SF(target_data_dict, ref_signal_topic):
    '''
    Interpolate Grount Truth (GT) slope values aligned with sensor timestamp
    * Ascent:1, Descent:-1, LW:0

    ref_signal_topic: enter the topic that you will use for reference
    ex: '/SensorData'
    '''


    ### Function test section - start
    # workspace_path = os.getcwd()
    # loadDir = workspace_path + '/Ramp_Data/TF16v2/'  ### Put your Cache file directory here
    # loadfileDir = loadDir + '1'
    # target_rawdata = jb.load(loadfileDir)
    #
    # ref_signal_topic = str('/SensorData')
    ### Function test section - end

    if target_data_dict.__class__ == str:
        target_rawdata = jb.load(target_data_dict)
    elif target_data_dict.__class__ == dict:
        target_rawdata = target_data_dict
    else:
        print('Error! Check the input type (Directory or Dictionary)')

    target_sensordata = target_rawdata[ref_signal_topic]
    target_sensor_timeval = target_sensordata['header']

    ##### Align time segment to State
    time_val = target_rawdata['/fsm/State']['header']
    state = target_rawdata['/fsm/State']['state']

    ### Phase indexing
    time_idx_ES = []
    time_idx_LS = []
    time_idx_SE = []
    time_idx_SF = []

    i = 0
    for phase in state:
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

    ### Mode indexing: HS
    time_idx_RAHS = []  ### Ramp Ascend Heel Strike
    time_idx_RDHS = []  ### Ramp Descend Heel Strike
    time_idx_LWHS = []  ### Level Walking Heel Strike

    for j in range(0, len(time_idx_ES)):
        if 'RA' in state[time_idx_ES[j]]:
            time_idx_RAHS.append(time_idx_ES[j])

        if 'RD' in state[time_idx_ES[j]]:
            time_idx_RDHS.append(time_idx_ES[j])

        if 'LW' in state[time_idx_ES[j]]:
            time_idx_LWHS.append(time_idx_ES[j])

    time_val_RAHS = time_val[time_idx_RAHS].array
    time_val_RDHS = time_val[time_idx_RDHS].array
    time_val_LWHS = time_val[time_idx_LWHS].array

    if len(time_val_RDHS) < 1: ### If there is no ramp descent
        return

    ### Mode indexing: SF
    time_idx_RASF = []  ### Ramp Ascend Heel Strike
    time_idx_RDSF = []  ### Ramp Descend Heel Strike
    time_idx_LWSF = []  ### Level Walking Heel Strike

    for j in range(0, len(time_idx_SF)):
        if 'RA' in state[time_idx_SF[j]]:
            time_idx_RASF.append(time_idx_SF[j])

        if 'RD' in state[time_idx_SF[j]]:
            time_idx_RDSF.append(time_idx_SF[j])

        if 'LW' in state[time_idx_SF[j]]:
            time_idx_LWSF.append(time_idx_SF[j])

    time_val_RASF = time_val[time_idx_RASF].array
    time_val_RDSF = time_val[time_idx_RDSF].array
    time_val_LWSF = time_val[time_idx_LWSF].array

    if len(time_val_RDSF) < 1: ### If there is no ramp descent
        return

    ### Labeling
    ground_truth = np.zeros([len(target_sensordata)])

    ### Find Transition time indices (RA<->LW, RD<->LW)
    transitionIdx_LW_to_RA = closest_point(time_val_LWSF, time_val_RASF[0])
    transitionIdx_LW_to_RD = closest_point(time_val_LWSF, time_val_RDSF[0])

    ### Transition timings (RA->LW, RD->LW)
    transition_time_RA_to_LW = time_val_RASF[-1]
    transition_time_RD_to_LW = time_val_RDSF[-1]
    transition_time_LW_to_RA = time_val_LWSF[transitionIdx_LW_to_RA]
    transition_time_LW_to_RD = time_val_LWSF[transitionIdx_LW_to_RD]

    sensortime_transition_timeIdx_LW_to_RA = closest_point(target_sensor_timeval,
                                                           transition_time_LW_to_RA)
    sensortime_transition_timeIdx_RA_to_LW = closest_point(target_sensor_timeval,
                                                           transition_time_RA_to_LW)
    sensortime_transition_timeIdx_LW_to_RD = closest_point(target_sensor_timeval,
                                                           transition_time_LW_to_RD)
    sensortime_transition_timeIdx_RD_to_LW = closest_point(target_sensor_timeval,
                                                           transition_time_RD_to_LW)

    ground_truth[sensortime_transition_timeIdx_LW_to_RA:sensortime_transition_timeIdx_RA_to_LW] = 1
    ground_truth[sensortime_transition_timeIdx_LW_to_RD:sensortime_transition_timeIdx_RD_to_LW] = -1

    return ground_truth

def Ramp_GT_SE(target_data_dict, ref_signal_topic):
    '''
    Interpolate Grount Truth (GT) slope values aligned with sensor timestamp
    * Ascent:1, Descent:-1, LW:0

    ref_signal_topic: enter the topic that you will use for reference
    ex: '/SensorData'
    '''


    ### Function test section - start
    # workspace_path = os.getcwd()
    # loadDir = workspace_path + '/Ramp_Data/TF16v2/'  ### Put your Cache file directory here
    # loadfileDir = loadDir + '1'
    # target_rawdata = jb.load(loadfileDir)
    #
    # ref_signal_topic = str('/SensorData')
    ### Function test section - end

    if target_data_dict.__class__ == str:
        target_rawdata = jb.load(target_data_dict)
    elif target_data_dict.__class__ == dict:
        target_rawdata = target_data_dict
    else:
        print('Error! Check the input type (Directory or Dictionary)')

    target_sensordata = target_rawdata[ref_signal_topic]
    target_sensor_timeval = target_sensordata['header']

    ##### Align time segment to State
    time_val = target_rawdata['/fsm/State']['header']
    state = target_rawdata['/fsm/State']['state']

    ### Phase indexing
    time_idx_ES = []
    time_idx_LS = []
    time_idx_SE = []
    time_idx_SF = []

    i = 0
    for phase in state:
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

    ### Mode indexing: HS
    time_idx_RAHS = []  ### Ramp Ascend Heel Strike
    time_idx_RDHS = []  ### Ramp Descend Heel Strike
    time_idx_LWHS = []  ### Level Walking Heel Strike

    for j in range(0, len(time_idx_ES)):
        if 'RA' in state[time_idx_ES[j]]:
            time_idx_RAHS.append(time_idx_ES[j])

        if 'RD' in state[time_idx_ES[j]]:
            time_idx_RDHS.append(time_idx_ES[j])

        if 'LW' in state[time_idx_ES[j]]:
            time_idx_LWHS.append(time_idx_ES[j])

    time_val_RAHS = time_val[time_idx_RAHS].array
    time_val_RDHS = time_val[time_idx_RDHS].array
    time_val_LWHS = time_val[time_idx_LWHS].array

    if len(time_val_RDHS) < 1: ### If there is no ramp descent
        return

    ### Mode indexing: SF
    time_idx_RASE = []  ### Ramp Ascend Heel Strike
    time_idx_RDSE = []  ### Ramp Descend Heel Strike
    time_idx_LWSE = []  ### Level Walking Heel Strike

    for j in range(0, len(time_idx_SE)):
        if 'RA' in state[time_idx_SE[j]]:
            time_idx_RASE.append(time_idx_SE[j])

        if 'RD' in state[time_idx_SE[j]]:
            time_idx_RDSE.append(time_idx_SE[j])

        if 'LW' in state[time_idx_SE[j]]:
            time_idx_LWSE.append(time_idx_SE[j])

    time_val_RASE = time_val[time_idx_RASE].array
    time_val_RDSE = time_val[time_idx_RDSE].array
    time_val_LWSE = time_val[time_idx_LWSE].array

    if len(time_val_RDSE) < 1: ### If there is no ramp descent
        return

    ### Labeling
    ground_truth = np.zeros([len(target_sensordata)])

    ### Find Transition time indices (RA<->LW, RD<->LW)
    transitionIdx_LW_to_RA = closest_point(time_val_LWSE, time_val_RASE[0])
    transitionIdx_LW_to_RD = closest_point(time_val_LWSE, time_val_RDSE[0])

    ### Transition timings (RA->LW, RD->LW)
    transition_time_RA_to_LW = time_val_RASE[-1]
    transition_time_RD_to_LW = time_val_RDSE[-1]
    transition_time_LW_to_RA = time_val_LWSE[transitionIdx_LW_to_RA]
    transition_time_LW_to_RD = time_val_LWSE[transitionIdx_LW_to_RD]

    sensortime_transition_timeIdx_LW_to_RA = closest_point(target_sensor_timeval,
                                                           transition_time_LW_to_RA)
    sensortime_transition_timeIdx_RA_to_LW = closest_point(target_sensor_timeval,
                                                           transition_time_RA_to_LW)
    sensortime_transition_timeIdx_LW_to_RD = closest_point(target_sensor_timeval,
                                                           transition_time_LW_to_RD)
    sensortime_transition_timeIdx_RD_to_LW = closest_point(target_sensor_timeval,
                                                           transition_time_RD_to_LW)

    ground_truth[sensortime_transition_timeIdx_LW_to_RA:sensortime_transition_timeIdx_RA_to_LW] = 1
    ground_truth[sensortime_transition_timeIdx_LW_to_RD:sensortime_transition_timeIdx_RD_to_LW] = -1

    return ground_truth

def footIMU_slope(target_footIMU_dataframe, target_state_dataframe, ground_truth):
    '''
    target_footIMU_dataframe: DataFrame including orientation_x,y,z,w
    target_state_dataframe: DataFrame including state and timestamp

    Returns (ES-LS) filter
    '''

    # ### Function test
    # loadDir = 'C:/Users/hkim910/Documents/MATLAB/'  ### Put your Cache file directory here
    # loadfileDir = loadDir + 'OSL_Slope_SensorDat_footIMU'
    #
    # rawdatdict = jb.load(loadfileDir)
    #
    # target_footIMU_dataframe = rawdatdict['TF12v2'][7.8]['29']['foot_IMU']
    # target_state_dataframe = rawdatdict['TF12v2'][7.8]['29']['state']
    # ground_truth = rawdatdict['TF12v2'][7.8]['29']['ground_truth']

    '''
    Raw slope estimate from Quaternion information
    '''
    q_x = target_footIMU_dataframe['orientation_x']
    q_y = target_footIMU_dataframe['orientation_y']
    q_z = target_footIMU_dataframe['orientation_z']
    q_w = target_footIMU_dataframe['orientation_w']

    qmat = np.array([q_w, q_x, q_y, q_z]).T

    Rmat = np.zeros([len(qmat), 3, 3])
    rotZmat = np.zeros([len(qmat), 3])
    foot_anglemat = np.zeros([len(qmat)])
    Zvector = np.array([0, 0, 1])

    for i in range(0, len(qmat)):
        Rmat[i, :] = AHRS.q2Rot(qmat[i])
        rotZmat[i, :] = np.matmul(Rmat[i, :], Zvector)

        vlen = rotZmat[i, 0]
        hlen = rotZmat[i, 2]

        foot_anglemat[i] = math.atan2(hlen, vlen) * 180 / np.pi

    state = target_state_dataframe['state']
    time_val_state = target_state_dataframe['header']
    time_val_IMU = target_footIMU_dataframe['header']

    '''
    Extract timings of Early/Late stance
    '''
    time_idx_ES = []
    time_idx_LS = []
    time_idx_SE = []
    time_idx_SF = []

    i = 0
    for phase in state:
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

    time_idx_RALS = []  ### Ramp Ascend Late Stance
    time_idx_RDLS = []  ### Ramp Descend Late Stance
    time_idx_LWLS = []  ### Level Walking Late Stance

    for j in range(0, len(time_idx_LS)):
        if 'RA' in state[time_idx_LS[j]]:
            time_idx_RALS.append(time_idx_LS[j])

        if 'RD' in state[time_idx_LS[j]]:
            time_idx_RDLS.append(time_idx_LS[j])

        if 'LW' in state[time_idx_LS[j]]:
            time_idx_LWLS.append(time_idx_LS[j])

    # if len(time_val_RDHS) < 1: ### If there is no ramp descent
    #     return

    slope_mat = 90 - foot_anglemat
    slope_mat_filt_ES2LS = np.zeros([len(slope_mat)])

    filter_fullContact = np.zeros([len(qmat)])

    for i in range(0, len(time_idx_ES)):
        if i < len(time_idx_LS):

            time_state_ES = time_val_state[time_idx_ES[i]]
            time_state_ES_next = time_val_state[time_idx_ES[i+1]]
            time_state_LS = time_val_state[time_idx_LS[i]]

            if time_idx_LS[i] in time_idx_RALS:
                contact_len = 55
            elif time_idx_LS[i] in time_idx_LWLS:
                contact_len = 50
            elif time_idx_LS[i] in time_idx_RDLS:
                contact_len = 25

            time_idx_ES_IMU = closest_point(time_val_IMU, time_state_ES)
            time_idx_ES_IMU_next = closest_point(time_val_IMU, time_state_ES_next)
            time_idx_LS_IMU = closest_point(time_val_IMU, time_state_LS)

            if i == 0:
                offset_idx = time_idx_LS_IMU

            filter_fullContact[time_idx_LS_IMU-contact_len-10:time_idx_LS_IMU] = 1

            slope_mat_filt_ES2LS[time_idx_ES_IMU:time_idx_ES_IMU_next] = np.mean(slope_mat[time_idx_LS_IMU-contact_len:time_idx_LS_IMU])

        else:
            time_state_ES = time_val_state[time_idx_ES[i]]
            time_idx_ES_IMU = closest_point(time_val_IMU, time_state_ES)
            filter_fullContact[time_idx_ES_IMU:] = 1

            slope_mat_filt_ES2LS[time_idx_ES_IMU:] = np.mean(slope_mat[time_idx_ES_IMU:])

    offset_value = np.min(slope_mat[offset_idx-contact_len:offset_idx-5])
    slope_mat_filt_ES2LS -= offset_value

    start_time_idx_IMU = closest_point(time_val_IMU, time_val_state[time_idx_ES[0]])
    end_time_idx_IMU = closest_point(time_val_IMU, time_val_state[time_idx_ES[-1]])
    slope_mat_filt_ES2LS_mult = slope_mat * filter_fullContact

    ### Crop estimation 1st ES~ last ES
    slope_mat_filt_ES2LS = slope_mat_filt_ES2LS[start_time_idx_IMU:end_time_idx_IMU]
    ground_truth_crop = ground_truth[start_time_idx_IMU:end_time_idx_IMU]
    time_val_IMU = time_val_IMU[start_time_idx_IMU:end_time_idx_IMU]
    slope_mat_filt_ES2LS_mult_crop = slope_mat_filt_ES2LS_mult[start_time_idx_IMU:end_time_idx_IMU]

    # plt.plot(time_val_IMU, slope_mat_filt_ES2LS)
    # plt.plot(time_val_IMU, ground_truth_crop)

    ''' Estimation Scenario
    1. Calculate offset in static (Should start from LW)
    2. Initial slope:0
    3. ES detect(start trigger) -> LS detect(end trigger) -> Calculate slope from Quaternion
    4. Before next ES, slope is the same as right before
    5. Repeat 3, 4
    '''

    return slope_mat_filt_ES2LS, slope_mat_filt_ES2LS_mult_crop, slope_mat_filt_ES2LS_mult, slope_mat, time_val_IMU, ground_truth, ground_truth_crop, offset_value