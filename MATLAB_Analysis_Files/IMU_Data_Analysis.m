%% ME 6404 TEAM 6: OSL Prosthesis Dynamics Analysis
% Date: November 6, 2024
% 
% NOTE: Make sure to run from the OSL_Oscillation_Analysis folder
clear all; close all; clc
fprintf("Begin Plotting Script... \n");

parent_folder_name = 'Converted_CSV_Data_Collection_1';

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

        % Plot only residual oscillations
        %   Note: segment signal for residual oscillations between: 100 < data <300
        figure();
        timesteps_mask = 100:300;                               % mask for residual oscillations
        time_trimmed_ms = [0:1:length(timesteps_mask)-1] * 10 ; % time steps are sampled every 10 ms
        time_trimmed_s = time_trimmed_ms * 1/1000;              
        signal_trimmed = foot_accelY(timesteps_mask);           % choose which signal to plot
        signal_trimmed = detrend(signal_trimmed);               % detrend the data, get rid of y axis offset
        plot(time_trimmed_s,signal_trimmed)
        xlabel('Time (s)'); ylabel('foot_accelY');
        title(['Residual Oscillations: ', cur_filename], 'Interpreter', 'none');

        %compute Fast Fourier Transform (FFT) to Find the Frequency of Oscillation
        Yf = fft(signal_trimmed);
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
        title([cur_filename, "  Acceleration vs. Frequency"], 'Interpreter', 'none');
        xlabel("Frequency (Hz)")
        ylabel("Acceleration (m/s)")
    end
    fprintf('Plotted all files in: %s \n', cur_folder_name);
    fprintf("PAUSED: Press any key close all plots and generate the next set of plots \n");
    pause;      
    close all;  % close all trial plots for this set of k and b params
end