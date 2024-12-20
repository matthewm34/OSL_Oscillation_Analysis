%% ME 6404 TEAM 6: OSL Prosthesis Dynamics Analysis
% Date: November 6, 2024
% 
% NOTE: Make sure to run from the OSL_Oscillation_Analysis folder
%       Run Code in independent sections

%% ME 6404 TEAM 6: OSL Prosthesis Dynamics Analysis
clear all; close all; clc
fprintf("Begin Plotting Script... \n");

parent_folder_name = 'Converted_CSV_Data_Collection_2'; %______________ Filtered Dataset
% parent_folder_name = 'Converted_CSV_Data_Collection_3'; %SwingExtension Filtered Dataset
% parent_folder_name = 'Converted_CSV_Data_Collection_4'; %Cropped For only swing phase: from swing flexion index to early stance index 
% parent_folder_name = 'Converted_CSV_Data_Collection_5'; %Cropped with ZERO OFFSET GOOD Set

len_foot_accelX_vec = [];
folders_struct = dir(parent_folder_name);
for i = 3:length({folders_struct.name}) % loop through folders
    cur_folder_name = folders_struct(i).name;
    filename_struct = dir([parent_folder_name '/' cur_folder_name]);

    for j = 3:length(filename_struct) % loop through files
        cur_filename = filename_struct(j).name;

        % Read data as a table
        dataTable = readtable([parent_folder_name  '/'  cur_folder_name '/' cur_filename]);
    
        % Extract relevant columns
        foot_accelX = dataTable.foot_accelX;
        foot_accelY = dataTable.foot_accelY;
        foot_accelZ = dataTable.foot_accelZ;
        knee_theta = dataTable.knee_theta;
        knee_thetadot = dataTable.knee_thetadot;
    
        % Plot entire signal trial as segmented by the bag2csv.py code
        % figure();
        % plot([0:1:length(foot_accelX)-1],foot_accelX)
        
        len_foot_accelX_vec = [len_foot_accelX_vec, length(foot_accelX)];
        

        % Plot only residual oscillations
        %   Note: segment signal for residual oscillations between: 100 < data <300
        figure();
        % timesteps_mask = 100:300;                               % mask for residual oscillations in Converted_CSV_Data_Collection_1
        % time_trimmed_ms = [0:1:length(timesteps_mask)-1] * 10 ; % time steps are sampled every 10 ms
        time_trimmed_ms = [0:1:length(foot_accelX(210:end))-1] * 10 ; % time steps are sampled every 10 ms
        time_trimmed_s = time_trimmed_ms * 1/1000;              
        % signal_trimmed = foot_accelX(timesteps_mask);           % choose which signal to plot
        accel_signal = foot_accelX(210:end);
        
        accel_signal_detrend = detrend(accel_signal);             % detrend the data, get rid of y axis offset
        vel_signal = cumtrapz(time_trimmed_s, accel_signal);
        vel_signal_detrend = detrend(vel_signal);                 % detrend the data, get rid of y axis offset
        pos_signal = cumtrapz(time_trimmed_s, vel_signal);
        pos_signal_detrend = detrend(pos_signal);

        plot(time_trimmed_s,pos_signal_detrend)
        xlabel('Time (s)'); ylabel('pos_signal_detrend');
        title(['Residual Oscillations: ', cur_filename], 'Interpreter', 'none');



        %compute Fast Fourier Transform (FFT) to Find the Frequency of Oscillation
        Yf = fft(pos_signal_detrend);
        % Define the frequency axis
        N = length(Yf);  % Number of data points
        dt = .01;       % Sampling interval s
        df = 1/(N*dt);  % Frequency resolution
        nf = floor(N/2)+1;
        f = (0:nf-1)'*df;
        Yf = Yf(1:nf);
        acceleration_magnitude = abs(Yf); % Compute the magnitude of the FFT (convert to acceleration)
   
        figure();
        plot(f,acceleration_magnitude) % plot FFT
        title([cur_filename, "  Position vs. Frequency"], 'Interpreter', 'none');
        xlabel("Frequency (Hz)")
        ylabel("POs (m)")
    end
    fprintf('Plotted all files in: %s \n', cur_folder_name);
    fprintf("PAUSED: Press any to key close all plots and generate the next set of plots \n");
    pause;      
    close all;  % close all trial plots for this set of k and b params
end


%% Determine Frequency From Acceleration Signal
% Date: November 6, 2024
% 
% NOTE: Make sure to run from the OSL_Oscillation_Analysis folder
clear all; close all; clc
fprintf("Begin Plotting Script... \n");

parent_folder_name = 'Converted_CSV_Data_Collection_1'; %______________ Filtered Dataset for analyzing the residual oscillations only

len_foot_accelX_vec = [];
folders_struct = dir(parent_folder_name);
for i = 3:length(folders_struct) % loop through folders
    cur_folder_name = folders_struct(i).name;
    filename_struct = dir([parent_folder_name '/' cur_folder_name]);

    for j = 3:length(filename_struct) % loop through files
        cur_filename = filename_struct(j).name;

        % Read data as a table
        dataTable = readtable([parent_folder_name  '/'  cur_folder_name '/' cur_filename]);
    
        % Extract relevant columns
        foot_accelX = dataTable.foot_accelX;
        foot_accelY = dataTable.foot_accelY;
        foot_accelZ = dataTable.foot_accelZ;
        knee_theta = dataTable.knee_theta;
        knee_thetadot = dataTable.knee_thetadot;
        
        % Plot entire signal trial as segmented by the bag2csv.py code
        % figure();
        % plot([0:1:length(foot_accelX)-1],foot_accelX)
        
        % len_foot_accelX_vec = [len_foot_accelX_vec, length(foot_accelX)];
        
        % Plot only residual oscillations
        %   Note: segment signal for residual oscillations between: 100 < data <300
        figure();
        timesteps_mask = 100:300;                               % mask for residual oscillations in Converted_CSV_Data_Collection_1
        time_trimmed_ms = [0:1:length(timesteps_mask)-1] * 10 ; % time steps are sampled every 10 ms
        time_trimmed_s = time_trimmed_ms * 1/1000;              
        signal_trimmed = foot_accelX(timesteps_mask);           % choose which signal to plot
        accel_signal_detrend = detrend(signal_trimmed);             % detrend the data, get rid of y axis offset

        plot(time_trimmed_s,accel_signal_detrend)
        xlabel('Time (s)'); ylabel('accel_signal_X');
        title(['Residual Oscillations: ', cur_filename], 'Interpreter', 'none');

        %compute Fast Fourier Transform (FFT) to Find the Frequency of Oscillation
        Yf = fft(accel_signal_detrend);
        % Define the frequency axis
        N = length(Yf);  % Number of data points
        dt = .01;       % Sampling interval s
        df = 1/(N*dt);  % Frequency resolution
        nf = floor(N/2)+1;
        f = (0:nf-1)'*df;
        Yf = Yf(1:nf);
        acceleration_magnitude = abs(Yf); % Compute the magnitude of the FFT (convert to acceleration)
   
        figure();
        plot(f,acceleration_magnitude) % plot FFT
        title([cur_filename, "  Position vs. Frequency"], 'Interpreter', 'none');
        xlabel("Frequency (Hz)")
        ylabel("POs (m)")
    end
    fprintf('Plotted all files in: %s \n', cur_folder_name);
    fprintf("PAUSED: Press any to key close all plots and generate the next set of plots \n");
    pause;      
    close all;  % close all trial plots for this set of k and b params
end

%% Plotting raw signal channels
% Date: November 6, 2024
% 
% NOTE: Make sure to run from the OSL_Oscillation_Analysis folder
%
% plotting the raw signal channels
%
%
%

clear all; close all; clc
fprintf("Begin Plotting Script... \n");

parent_folder_name = 'Converted_CSV_Data_Collection_5'; %______________ Filtered Dataset for analyzing the residual oscillations only

len_foot_accelX_vec = [];
folders_struct = dir(parent_folder_name);
for i = 3:length(folders_struct) % loop through folders
    cur_folder_name = folders_struct(i).name;
    filename_struct = dir([parent_folder_name '/' cur_folder_name]);

    for j = 3:length(filename_struct) % loop through files
        cur_filename = filename_struct(j).name;

        % Read data as a table
        dataTable = readtable([parent_folder_name  '/'  cur_folder_name '/' cur_filename]);
    
        % Extract relevant columns
        foot_accelX = dataTable.foot_accelX;
        foot_accelY = dataTable.foot_accelY;
        foot_accelZ = dataTable.foot_accelZ;
        knee_theta = dataTable.knee_theta;
        knee_thetadot = dataTable.knee_thetadot;
        
        % Plot entire signal trial as segmented by the bag2csv.py code
        %plot accel x
        figure();
        time_ms = [0:1:length(foot_accelX)-1] * 10 ; % time steps are sampled every 10 ms
        time_s = time_ms * 1/1000;  
        plot(time_s,foot_accelX)
        xlabel('Time (s)'); ylabel('accel_signal_X', 'Interpreter', 'none');
        title(['Entire Stride: accel_signal_X ', cur_filename], 'Interpreter', 'none');

        %plot knee_theta
        figure();
        plot(time_s,knee_theta)
        xlabel('Time (s)'); ylabel('knee_theta', 'Interpreter', 'none');
        title(['Entire Stride: knee_theta ', cur_filename], 'Interpreter', 'none');
       
        %plot knee_thetadot
        figure();
        plot(time_s,knee_thetadot)
        xlabel('Time (s)'); ylabel('knee_thetadot', 'Interpreter', 'none');
        title(['Entire Stride: knee_thetadot ', cur_filename], 'Interpreter', 'none');

        test = 0;

    end
    fprintf('Plotted all files in: %s \n', cur_folder_name);
    fprintf("PAUSED: Press any to key close all plots and generate the next set of plots \n");
    pause;      
    close all;  % close all trial plots for this set of k and b params
end


%% Find the Shaper Length
% Date: November 6, 2024
% 
% Output: Signal_Lengths cell array where cells are parameters sets  {'kneeK1.5B0.2'}    {'kneeK2.0B0.1'}    {'kneeK2.0B0.15'}    {'knee_K2.0B0.2'}    {'knee_K2.5B0.2'}
%   and within each cell is a matrix 3 by num_trials where...
%       First Row is the number of time steps during flexion (B->A)
%       Second Row is the number of time steps during extension (A->B)
%       Last Row is the number of time steps afterwards of no movement 
%       Each Column is the trial number

clear all; close all; clc
fprintf("Begin Plotting Script... \n");

parent_folder_name = 'Converted_CSV_Data_Collection_5'; %______________ Filtered Dataset for analyzing the residual oscillations only
name_vec = [num2str(1:5)];
len_foot_accelX_vec = [];
folders_struct = dir(parent_folder_name);
num_timesteps_to_max_flexion_vec = [];
num_timesteps_to_max_extension_vec = [];
num_timesteps_after_max_extension_vec = [];

for i = 3:length(folders_struct) % loop through folders
    cur_folder_name = folders_struct(i).name;
    filename_struct = dir([parent_folder_name '/' cur_folder_name]);
    len_foot_accelX_vec = [];
    num_timesteps_to_max_flexion_vec = [];
    num_timesteps_to_max_extension_vec = [];
    num_timesteps_after_max_extension_vec = [];

    for j = 3:length(filename_struct) % loop through files
        cur_filename = filename_struct(j).name;

        % Read data as a table
        dataTable = readtable([parent_folder_name  '/'  cur_folder_name '/' cur_filename]);
    
        % Extract relevant columns
        foot_accelX = dataTable.foot_accelX;
        knee_thetadot = dataTable.knee_thetadot;
        
        %find when knee_thetadot turns negative
        time_ms = [0:1:length(foot_accelX)-1] * 10 ; % time steps are sampled every 10 ms
        time_s = time_ms * 1/1000;  

        % knee_thetadot_flexion = knee_thetadot(5:end);
        knee_thetadot_flexion = knee_thetadot;
        [flexion_indexes, ~]= find(knee_thetadot < 0);
        indx_peak_flexion = flexion_indexes(1) -1;
        knee_thetadot_flexion = knee_thetadot(1:indx_peak_flexion);
        

        % Find relevant locations
        knee_thetadot_extension = knee_thetadot(indx_peak_flexion+1:end);
        [extension_indexes, ~]= find(knee_thetadot_extension > 0);
        indx_peak_extension = extension_indexes(1)-1;
        knee_thetadot_extension = knee_thetadot(indx_peak_flexion+1:indx_peak_extension+indx_peak_flexion);

        extra_indices_at_end = knee_thetadot(indx_peak_flexion+indx_peak_extension+1:end);
        num_indices_at_end = length(extra_indices_at_end);

        % Find important indices into vector
        num_timesteps_to_max_flexion_vec = [num_timesteps_to_max_flexion_vec, indx_peak_flexion]
        num_timesteps_to_max_extension_vec = [num_timesteps_to_max_extension_vec, indx_peak_flexion]
        num_timesteps_after_max_extension_vec = [num_timesteps_after_max_extension_vec, num_indices_at_end]

    end
    %cell array in order of {'kneeK1.5B0.2'}    {'kneeK2.0B0.1'}    {'kneeK2.0B0.15'}    {'knee_K2.0B0.2'}    {'knee_K2.5B0.2'}
    Signal_Lengths{i-2} = [num_timesteps_to_max_flexion_vec;num_timesteps_to_max_extension_vec;num_timesteps_after_max_extension_vec];
   
    fprintf('Calculated lengths all signal length in: %s \n', cur_folder_name);
    close all;  % close all trial plots for this set of k and b params
end

%% Determine Damping Ratio
% Date: November 6, 2024
% 
% NOTE: Make sure to run from the OSL_Oscillation_Analysis folder
clear all; close all; clc
fprintf("Begin Plotting Script... \n");

parent_folder_name = 'Converted_CSV_Data_Collection_1'; %______________ Filtered Dataset for analyzing the residual oscillations only

len_foot_accelX_vec = [];
folders_struct = dir(parent_folder_name);
for i = 3:length(folders_struct) % loop through folders
    cur_folder_name = folders_struct(i).name;
    filename_struct = dir([parent_folder_name '/' cur_folder_name]);

    for j = 3:length(filename_struct) % loop through files
        cur_filename = filename_struct(j).name;

        % Read data as a table
        dataTable = readtable([parent_folder_name  '/'  cur_folder_name '/' cur_filename]);
    
        % Extract relevant columns
        foot_accelX = dataTable.foot_accelX;
        foot_accelY = dataTable.foot_accelY;
        foot_accelZ = dataTable.foot_accelZ;
        knee_theta = dataTable.knee_theta;
        knee_thetadot = dataTable.knee_thetadot;
        
        % Plot entire signal trial as segmented by the bag2csv.py code
        % figure();
        % plot([0:1:length(foot_accelX)-1],foot_accelX)
        
        % len_foot_accelX_vec = [len_foot_accelX_vec, length(foot_accelX)];
        
        % Plot only residual oscillations
        %   Note: segment signal for residual oscillations between: 100 < data <300
        figure();
        timesteps_mask = 100:300;                               % mask for residual oscillations in Converted_CSV_Data_Collection_1
        time_trimmed_ms = [0:1:length(timesteps_mask)-1] * 10 ; % time steps are sampled every 10 ms
        time_trimmed_s = time_trimmed_ms * 1/1000;              
        signal_trimmed = foot_accelX(timesteps_mask);           % choose which signal to plot
        accel_signal_detrend = detrend(signal_trimmed);             % detrend the data, get rid of y axis offset
        
        plot(time_trimmed_s,accel_signal_detrend)
        xlabel('Time (s)'); ylabel('accel_signal_detrend');
        title(['Residual Oscillations: ', cur_filename], 'Interpreter', 'none');




        % accel_signal_detrend = detrend(accel_signal);             % detrend the data, get rid of y axis offset
        vel_signal = cumtrapz(time_trimmed_s, accel_signal_detrend);
        vel_signal_detrend = detrend(vel_signal);                 % detrend the data, get rid of y axis offset
        pos_signal = cumtrapz(time_trimmed_s, vel_signal_detrend);
        pos_signal_detrend = detrend(pos_signal);
        
        figure();
        plot(time_trimmed_s,vel_signal)
        xlabel('Time (s)'); ylabel('vel_signal');
        title(['Residual Oscillations: ', cur_filename], 'Interpreter', 'none');
        
        figure();
        plot(time_trimmed_s,vel_signal_detrend)
        xlabel('Time (s)'); ylabel('vel_signal_detrend');
        title(['Residual Oscillations: ', cur_filename], 'Interpreter', 'none');
        
        figure();
        plot(time_trimmed_s,pos_signal)
        xlabel('Time (s)'); ylabel('pos_signal');
        title(['Residual Oscillations: ', cur_filename], 'Interpreter', 'none');

        figure();
        plot(time_trimmed_s,pos_signal_detrend)
        xlabel('Time (s)'); ylabel('pos_signal');
        title(['Residual Oscillations: ', cur_filename], 'Interpreter', 'none');
    
        test = 0;

        close all

        % 
        % %compute Fast Fourier Transform (FFT) to Find the Frequency of Oscillation
        % Yf = fft(accel_signal_detrend);
        % % Define the frequency axis
        % N = length(Yf);  % Number of data points
        % dt = .01;       % Sampling interval s
        % df = 1/(N*dt);  % Frequency resolution
        % nf = floor(N/2)+1;
        % f = (0:nf-1)'*df;
        % Yf = Yf(1:nf);
        % acceleration_magnitude = abs(Yf); % Compute the magnitude of the FFT (convert to acceleration)
        % 
        % figure();
        % plot(f,acceleration_magnitude) % plot FFT
        % title([cur_filename, "  Position vs. Frequency"], 'Interpreter', 'none');
        % xlabel("Frequency (Hz)")
        % ylabel("POs (m)")
    end
    fprintf('Plotted all files in: %s \n', cur_folder_name);
    fprintf("PAUSED: Press any to key close all plots and generate the next set of plots \n");
    % pause;      
    close all;  % close all trial plots for this set of k and b params
end



%% Get FFT Accel Amplitude for each K,B set and plot

%% Determine Frequency From Acceleration Signal
% Date: November 6, 2024
% 
% NOTE: Make sure to run from the OSL_Oscillation_Analysis folder
clear all; close all; clc
fprintf("Begin Plotting Script... \n");

parent_folder_name = 'Converted_CSV_Data_Collection_1'; %______________ Filtered Dataset for analyzing the residual oscillations only
avg_accel_mags_vec = [];
len_foot_accelX_vec = [];
folders_struct = dir(parent_folder_name);
for i = 3:length(folders_struct) % loop through folders
    cur_folder_name = folders_struct(i).name;
    filename_struct = dir([parent_folder_name '/' cur_folder_name]);
    
    max_accel_vec_fft = [];

    for j = 3:length(filename_struct) % loop through files
        cur_filename = filename_struct(j).name;

        % Read data as a table
        dataTable = readtable([parent_folder_name  '/'  cur_folder_name '/' cur_filename]);
    
        % Extract relevant columns
        foot_accelX = dataTable.foot_accelX;
        foot_accelY = dataTable.foot_accelY;
        foot_accelZ = dataTable.foot_accelZ;
        knee_theta = dataTable.knee_theta;
        knee_thetadot = dataTable.knee_thetadot;
        
        % Plot entire signal trial as segmented by the bag2csv.py code
        % figure();
        % plot([0:1:length(foot_accelX)-1],foot_accelX)
        
        % len_foot_accelX_vec = [len_foot_accelX_vec, length(foot_accelX)];
        
        % Plot only residual oscillations
        %   Note: segment signal for residual oscillations between: 100 < data <300
        % figure();
        timesteps_mask = 100:300;                               % mask for residual oscillations in Converted_CSV_Data_Collection_1
        time_trimmed_ms = [0:1:length(timesteps_mask)-1] * 10 ; % time steps are sampled every 10 ms
        time_trimmed_s = time_trimmed_ms * 1/1000;              
        signal_trimmed = foot_accelX(timesteps_mask);           % choose which signal to plot
        accel_signal_detrend = detrend(signal_trimmed);             % detrend the data, get rid of y axis offset

        % plot(time_trimmed_s,accel_signal_detrend)
        % xlabel('Time (s)'); ylabel('accel_signal_X');
        % title(['Residual Oscillations: ', cur_filename], 'Interpreter', 'none');

        %compute Fast Fourier Transform (FFT) to Find the Frequency of Oscillation
        Yf = fft(accel_signal_detrend);
        % Define the frequency axis
        N = length(Yf);  % Number of data points
        dt = .01;       % Sampling interval s
        df = 1/(N*dt);  % Frequency resolution
        nf = floor(N/2)+1;
        f = (0:nf-1)'*df;
        Yf = Yf(1:nf);
        acceleration_magnitude = abs(Yf); % Compute the magnitude of the FFT (convert to acceleration)
        test = 0;
        cur_max_accel_fft = max(acceleration_magnitude);
        max_accel_vec_fft = [max_accel_vec_fft,cur_max_accel_fft ]
        
        figure();
        plot(f,acceleration_magnitude) % plot FFT
        title([cur_filename, "  Accel vs. Frequency"], 'Interpreter', 'none');
        xlabel("Frequency (Hz)")
        ylabel("Accel (m/s^2)")

        
    end
    cur_avg_max_accel_vec_fft = mean(max_accel_vec_fft)
    %cell array in order of {'kneeK1.5B0.2'}    {'kneeK2.0B0.1'}    {'kneeK2.0B0.15'}    {'knee_K2.0B0.2'}    {'knee_K2.5B0.2'}
    accel_mags{i-2} = [max_accel_vec_fft];
    avg_accel_mags{i-2} = [cur_avg_max_accel_vec_fft];
    avg_accel_mags_vec = [avg_accel_mags_vec cur_avg_max_accel_vec_fft]
    
    fprintf('Plotted all files in: %s \n', cur_folder_name);
    fprintf("PAUSED: Press any to key close all plots and generate the next set of plots \n");
    % pause;      
    close all;  % close all trial plots for this set of k and b params
end
K_vec = [1.5 2.0 2.0 2.0 2.5]
B_vec = [.2 .1 .15 .2 .2]
mags_vec = avg_accel_mags_vec;

%plot mag of accecl vs B
B_vec_plot = B_vec(2:4)
mags_vec_B_plot = mags_vec(2:4)
plot(B_vec_plot,mags_vec_B_plot);
title("Constant K: Avg FFT Accel vs Time"); xlabel('B'); ylabel('accel')

%plot mag of accecl vs K
K_vec_plot = [K_vec(1),K_vec(4),K_vec(5)]
mags_vec_K_plot = [mags_vec(1),mags_vec(4),mags_vec(5)]
plot(K_vec_plot,mags_vec_K_plot);
title("Constant B: Avg FFT Accel vs Time"); xlabel('K'); ylabel('accel')

test = 0;