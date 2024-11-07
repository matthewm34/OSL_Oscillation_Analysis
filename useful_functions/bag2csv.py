import sys
import os

import pandas as pd
from psutil import POSIX

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

# loadfileDir = workspace_path + '/PILOT_HK_2024.Oct.21/'
loadfileDir = workspace_path + '/Raw_Bag_Data_Collection_1/'
# csvDir = workspace_path + loadfileDir + '/OSL_CSVs/'
csvDir = workspace_path 


# Check if directory exists, and if not, create it
if not os.path.exists(csvDir):
    os.makedirs(csvDir)
    print(f"Directory '{csvDir}' created.")
else:
    print(f"Directory '{csvDir}' already exists.")
bagfilelist = os.listdir(loadfileDir)
W = 75
data_load = False
NUM_OBSERVATIONS_PRE_ES1 = 125
NUM_OBSERVATIONS_POST_ES2 = 30

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

    #2. Split data into gait cycles
    FSM = loaded_bag["/fsm/State"]
    FSM_states = FSM['state']
    FSM_headers = FSM['header']


    #3. Collect Early Stances
    FSM_ES_idx = []
    for f in range(0, len(FSM_states)):
        if 'EarlyStance' in FSM_states[f]:
            FSM_ES_idx.append(f)

    #4. Go Through Pairs of Early Stances
    for ff in range(1, len(FSM_ES_idx) - 1):


        prev_ES_mode = FSM_states[FSM_ES_idx[ff] - 1].split("_")[0]
        start_ES_mode = FSM_states[FSM_ES_idx[ff]].split("_")[0]
        end_ES_mode = FSM_states[FSM_ES_idx[ff + 1]].split("_")[0]

        # Get the Previous ES as well as the current and next one
        prev_es_time = FSM_headers[FSM_ES_idx[ff - 1]]
        es_1_time = FSM_headers[FSM_ES_idx[ff]]
        es_2_time = FSM_headers[FSM_ES_idx[ff + 1]]

        #Get the corresponding indices based on timestamp
        prev_idx_SensorData = al.closest_point(loaded_bag['/SensorData']['header'], prev_es_time)
        es_1_idx_SensorData = al.closest_point(loaded_bag['/SensorData']['header'], es_1_time)
        es_2_idx_SensorData = al.closest_point(loaded_bag['/SensorData']['header'], es_2_time)

        ## SensorData: force, kinematics
        print(loaded_bag['/SensorData'].keys())
        # slope = loaded_bag['/fsm/context']['svalue'][0] # COMMENTED OUT BECAUSE JUST WANT SENSOR DATA
        csv_dict = {}

        ''' COMMENTED OUT BECAUSE JUST WANT SENSOR DATA

        #Load Ground Truth Based on LW and Ascent/Descent from Context
        csv_dict["GT"] = []
        if "LW" in prev_ES_mode:
            for i in range(es_1_idx_SensorData - NUM_OBSERVATIONS_PRE_ES1, es_1_idx_SensorData):
                if "D" in end_ES_mode:
                    csv_dict["GT"].append(-slope)
                else:
                    csv_dict["GT"].append(slope)
        else:
            for i in range(es_1_idx_SensorData - NUM_OBSERVATIONS_PRE_ES1, es_1_idx_SensorData):
                csv_dict["GT"].append(slope)

        if "LW" in start_ES_mode:
            for i in range(es_1_idx_SensorData, es_2_idx_SensorData):
                csv_dict["GT"].append(0)
        else:
            for i in range(es_1_idx_SensorData, es_2_idx_SensorData):
                if "D" in end_ES_mode:
                    csv_dict["GT"].append(-slope)
                else:
                    csv_dict["GT"].append(slope)

        if "LW" in end_ES_mode:
            for i in range(es_2_idx_SensorData, es_2_idx_SensorData + NUM_OBSERVATIONS_POST_ES2):
                csv_dict["GT"].append(0)
        else:
            for i in range(es_2_idx_SensorData, es_2_idx_SensorData + NUM_OBSERVATIONS_POST_ES2):
                if "D" in end_ES_mode:
                    csv_dict["GT"].append(-slope)
                else:
                    csv_dict["GT"].append(slope)
        '''
        #Build the Pandas Dataframe
        for key in loaded_bag['/SensorData'].keys():
            csv_dict[key] = loaded_bag['/SensorData'][key][
                             es_1_idx_SensorData - NUM_OBSERVATIONS_PRE_ES1:es_2_idx_SensorData + NUM_OBSERVATIONS_POST_ES2]

            print(len(csv_dict[key]))

        df = pd.DataFrame.from_dict(csv_dict)
        df.reset_index()
        df.to_csv(csvDir + "/" + bagfilelist[b][:-4] + "_" + str(ff) + ".csv")
