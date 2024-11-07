import math
import numpy as np
from scipy.interpolate import interp1d as itpd
import useful_functions.align_sensor2ES_v2 as al

def calcArea(input, x_interval, window_length, overlap_length):
    '''

    '''
    num_increment = math.floor( len(input) / (window_length-overlap_length) )
    windowed_Area = [0] * num_increment

    for i in range(0, num_increment):
        if len(input) - ((window_length) + i * (window_length-overlap_length)) < 0 :
            break

        windowed_Area[i] = np.sum(input[i * (window_length - overlap_length) : window_length + i * (window_length - overlap_length)]) * x_interval

    windowed_Area = windowed_Area[:i]
    return windowed_Area

def rawsignal_windowed(input, window_length, overlap_length):
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

def normalize_maxmin(input):
    '''
    Normalizes input signal by max and min
    '''
    input = np.array(input)

    a = input - np.min(input)
    b = np.max(input) - np.min(input)

    if b == 0:
        normalized_signal = np.zeros([len(input)]) ### When input is just flat signal
    else:
        normalized_signal = a/b

    return normalized_signal

def normalize_stat(input):
    '''
    Normalizes input signal by mean and std
    '''
    input = np.array(input)

    mean = np.mean(input)
    std = np.std(input)

    if std == 0:
        normalized_signal = np.zeros([len(input)]) ### When input is just flat signal
    else:
        normalized_signal = (input-mean)/std

    return normalized_signal

def RMSE(input1, input2):
    '''
    Computes RMSE between two distributions
    input1: vector
    input2: vector
    '''

    rmse = np.sqrt(np.mean(np.square(input1-input2)))

    return rmse

def MAE(input1, input2):
    '''
    Computes MAE between two distributions
    input1: vector
    input2: vector
    '''

    mae = np.mean(np.abs(input1-input2))

    return mae

def MAFilter(input, av_length):
    '''
    Moving Average Filter
    '''

    MAFiltered_signal = np.zeros([len(input) - av_length+1])

    for i in range(0, len(MAFiltered_signal)):
        MAFiltered_signal[i] = np.mean(input[i:i+av_length+1])

    return MAFiltered_signal

def MAFilter_zeroPhase(input_array, av_length):
    '''
    Moving Average Filter (Zero-Phase, av_length should be odd)
    '''

    pad_left = np.zeros([int((av_length-1)/2)]) + input_array[0]
    pad_right = np.zeros([int((av_length-1)/2)]) + input_array[-1]

    padded_signal = np.concatenate([pad_left, input_array, pad_right], axis = 0)

    MAFiltered_signal = np.zeros([len(input_array)])

    for i in range(0, len(input_array)):
        MAFiltered_signal[i] = np.mean(padded_signal[i:i+av_length+1])

    return MAFiltered_signal

def DCoffset(input):
    '''
    Zero-mean normalization (Works as high-pass filter)
    '''

    output = input-np.mean(input)

    return output

def kalmanFilter(predictions: np.ndarray, process_noise=1e-2, measurement_var=0.1) -> np.ndarray:
    '''
    Inputs:
        - Context predictions (e.g. slope, walking speed, etc.)
        - Process noise for predictions
        - Measurement uncertainty (in the form of variance)

    Output:
        - Updated estimates of context
    '''

    estimates = []

    # Initialize
    prior_estimate = 0
    prior_var = 0.1

    for i in range(len(predictions)):
        slope_measurement = np.float64(predictions[i])

        # Update
        kalman_gain = prior_var / (prior_var + measurement_var)  # Kn
        estimate = prior_estimate + kalman_gain * (slope_measurement - prior_estimate)  # Xnn
        estimates.append(estimate)
        estimate_var = (1 - kalman_gain) * prior_var  # Pnn

        # Dynamics
        prior_estimate = estimate
        prior_var = estimate_var + process_noise

    estimates = np.array(estimates)

    return estimates

def align_nparrays(reference_signal, aligning_target, ref_header = None, target_header = None):

    ### Test
    # workspace_path = os.getcwd()
    # Dir_TF = workspace_path + '/Ramp_Data_Raw/'  # This gives us a str with the path and the file we are using
    # TFlist = os.listdir(Dir_TF)
    # TF = 'TF02v2'
    # target_TF_dir = Dir_TF + TF + '/'
    # filelist = os.listdir(target_TF_dir)
    # file = '23.bag'
    # topics = rsbg.checkTopics(target_TF_dir + file)
    #
    # try:
    #     bagread = rsbg.read2var(target_TF_dir + file, topics_include=topics)
    # except:
    #     bagread = rsbg.read2var2(target_TF_dir + file, topics_include=topics)
    #
    # knee_k = bagread['/knee/scaled_params']['k'].__array__()
    # knee_b = bagread['/knee/scaled_params']['b'].__array__()
    # knee_theta_eq = bagread['/knee/scaled_params']['theta_eq'].__array__()
    # params_header = bagread['/knee/scaled_params']['header'].__array__()
    #
    # knee_theta = bagread['/knee/joint_state']['theta'].__array__()
    # knee_theta_dot = bagread['/knee/joint_state']['theta_dot'].__array__()
    # joint_header = bagread['/knee/joint_state']['header'].__array__()
    #
    # reference_signal = knee_k
    # aligning_target = knee_theta
    # ref_header = params_header
    # target_header = joint_header
    ### Test

    try:
        start_header = np.max([ref_header[0], target_header[0]])
        end_header = np.min([ref_header[-1], target_header[-1]])

        start_idx_ref = al.closest_point(ref_header, start_header)
        end_idx_ref = al.closest_point(ref_header, end_header)

        start_idx_target = al.closest_point(target_header, start_header)
        end_idx_target = al.closest_point(target_header, end_header)

        cropped_ref = reference_signal[start_idx_ref:end_idx_ref]
        cropped_ref_header = ref_header[start_idx_ref:end_idx_ref]

        cropped_target = aligning_target[start_idx_target:end_idx_target]
        cropped_target_header = target_header[start_idx_target:end_idx_target]

        x_new = np.linspace(0, len(cropped_target), num=len(cropped_ref))
        x_original = np.linspace(0, len(cropped_target), num=len(cropped_target))

        f_aligning_target = itpd(x_original, cropped_target)
        f_aligning_target_header = itpd(x_original, cropped_target_header)

        aligned_target = f_aligning_target(x_new)
        aligned_target_header = f_aligning_target_header(x_new)
        aligned_reference = cropped_ref  # Cropped

        return aligned_reference, aligned_target, cropped_ref_header, aligned_target_header

    except:

        x_new = np.linspace(0, len(aligning_target), num=len(reference_signal))
        x_original = np.linspace(0, len(aligning_target), num = len(aligning_target))

        f_aligning_target = itpd(x_original, aligning_target)
        aligned_target = f_aligning_target(x_new)
        aligned_reference = reference_signal # Just same

        return aligned_reference, aligned_target, 0, 0

def hex2rgb(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))





