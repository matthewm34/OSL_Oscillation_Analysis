%% ME 6404 TEAM 6: OSL Prosthesis Dynamics Analysis
% Date: November 6, 2024
% 
% NOTE: Make sure to run from the OSL_Oscillation_Analysis folder
%       Run Code in independent sections


%% Plots subplot acceleration residual oscillations vs time
% Date: November 6, 2024
% 
% NOTE: Make sure to run from the OSL_Oscillation_Analysis folder
clear all; close all; clc
fprintf("Begin Plotting Script... \n");

parent_folder_name = 'Shaped_Converted_CSV_Data_Collection_1'; %______________ Filtered Dataset for analyzing shaped ZV, unshaped, EI and ZVD
% parent_folder_name = 'Shaped_Converted_CSV_Data_Collection_2'; %______________ Filtered Dataset for analyzing shaped EI and ZVD

subplot_name = {'Unshaped','ZV Shaped','ZVD Shaped','EI Shaped'};

len_foot_accelX_vec = [];
folders_struct = dir(parent_folder_name);
for i = 3:length(folders_struct) % loop through folders
    cur_folder_name = folders_struct(i).name;
    filename_struct = dir([parent_folder_name '/' cur_folder_name]);
    hold on
    subplot(2,2,i-2)
    hold on
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
        % figure(i*2-3);
        % plot([0:1:length(knee_thetadot)-1],knee_theta)
        % title([cur_filename, "  knee_theta vs. time"], 'Interpreter', 'none');

        len_foot_accelX_vec = [len_foot_accelX_vec, length(foot_accelX)];

          % Plot only residual oscillations
          % Note: segment signal for residual oscillations between: 100 < data <300
          % Altered to be: segment signal for residual oscillations between: 200 < data <300
        hold on
        % figure(i*3-2);
        % hold on
        % figure();
        timesteps_mask = 200:300;                               % mask for residual oscillations in Converted_CSV_Data_Collection_1
        time_trimmed_ms = [0:1:length(timesteps_mask)-1] * 10 ; % time steps are sampled every 10 ms
        time_trimmed_s = time_trimmed_ms * 1/1000;              
        signal_trimmed = foot_accelX(timesteps_mask);           % choose which signal to plot
        accel_signal_detrend = detrend(signal_trimmed);             % detrend the data, get rid of y axis offset

        plot(time_trimmed_s,accel_signal_detrend)
        
        grid on
        ylim([-5,3])
        xlabel('Time (s)'); ylabel('Acceleration Signal (m/s^2)');
        % title(['Residual Oscillations: ', cur_filename], 'Interpreter', 'none');
        title([subplot_name(i-2), ' Residual Oscillations']);


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
        % hold on
        % figure(i-2);
        % hold on
        % % figure();
        % plot(f,acceleration_magnitude) % plot FFT
        % ylim([0,40])
        % grid on
        % title([cur_filename, "  Accel vs. Frequency"], 'Interpreter', 'none');
        % xlabel("Frequency (Hz)")
        % ylabel("Accel (m/s^2)")
        % hold on 
    end
    hold off
    fprintf('Plotted all files in: %s \n', cur_folder_name);
    fprintf("PAUSED: Press any to key close all plots and generate the next set of plots \n");
    % pause;      
    % close all;  % close all trial plots for this set of k and b params
end

%% Plot Knee Theta vs Time
% Date: November 6, 2024
% 
% NOTE: Make sure to run from the OSL_Oscillation_Analysis folder
clear all; close all; clc
fprintf("Begin Plotting Script... \n");

parent_folder_name = 'Shaped_Converted_CSV_Data_Collection_1'; %______________ Filtered Dataset for analyzing shaped ZV, unshaped, EI and ZVD
% parent_folder_name = 'Shaped_Converted_CSV_Data_Collection_2'; %______________ Filtered Dataset for analyzing shaped EI and ZVD

len_foot_accelX_vec = [];
color_vec = ['r', 'b', 'g', 'k'];
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

        hold on
        % figure(i*2-3);
        figure(1);
        hold on

        % plot([0:1:length(knee_theta)-1],knee_theta(0:300), color_vec(i-2))
        timesteps_mask = 50:200;                               % mask for residual oscillations in Converted_CSV_Data_Collection_1
        time_trimmed_ms = [0:1:length(timesteps_mask)-1] * 10 ; % time steps are sampled every 10 ms
        time_trimmed_s = time_trimmed_ms * 1/1000;              
        signal_trimmed = knee_theta(timesteps_mask);           
        plot(time_trimmed_s,signal_trimmed, color_vec(i-2))
        grid on
        title([cur_filename, "Knee Theta for each "], 'Interpreter', 'none');
        xlabel('Time (s)'); ylabel('Knee Theta (degrees)')
        len_foot_accelX_vec = [len_foot_accelX_vec, length(foot_accelX)];
        
          % Plot only residual oscillations
          % Note: segment signal for residual oscillations between: 100 < data <300
          % Altered to be: segment signal for residual oscillations between: 200 < data <300
        % hold on
        % figure(i*3-2);
        % hold on
        % % figure();
        % timesteps_mask = 200:300;                               % mask for residual oscillations in Converted_CSV_Data_Collection_1
        % time_trimmed_ms = [0:1:length(timesteps_mask)-1] * 10 ; % time steps are sampled every 10 ms
        % time_trimmed_s = time_trimmed_ms * 1/1000;              
        % signal_trimmed = foot_accelX(timesteps_mask);           % choose which signal to plot
        % accel_signal_detrend = detrend(signal_trimmed);             % detrend the data, get rid of y axis offset
        % 
        % plot(time_trimmed_s,accel_signal_detrend)
        % grid on
        % ylim([-5,5])
        % xlabel('Time (s)'); ylabel('accel_signal_X');
        % title(['Residual Oscillations: ', cur_filename], 'Interpreter', 'none');
        % 
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
        % hold on
        % figure(i-2);
        % hold on
        % % figure();
        % plot(f,acceleration_magnitude) % plot FFT
        % ylim([0,40])
        % grid on
        % title([cur_filename, "  Accel vs. Frequency"], 'Interpreter', 'none');
        % xlabel("Frequency (Hz)")
        % ylabel("Accel (m/s^2)")
        % hold on 
    end
    % handlevec = [0,4,7,11];
    legend('Unshaped','','','ZV Shaped','','','','ZVD Shaped','','','','EI Shaped')
    hold off
    fprintf('Plotted all files in: %s \n', cur_folder_name);
    fprintf("PAUSED: Press any to key close all plots and generate the next set of plots \n");
    % pause;      
    % close all;  % close all trial plots for this set of k and b params
end



%% Plotting residual oscillation accleration vs. Time 

% Date: November 6, 2024
% 
% NOTE: Make sure to run from the OSL_Oscillation_Analysis folder
clear all; close all; clc
fprintf("Begin Plotting Script... \n");

parent_folder_name = 'Shaped_Converted_CSV_Data_Collection_1'; %______________ Filtered Dataset for analyzing shaped ZV, unshaped, EI and ZVD
% parent_folder_name = 'Shaped_Converted_CSV_Data_Collection_2'; %______________ Filtered Dataset for analyzing shaped EI and ZVD

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
        % figure(i*2-3);
        % plot([0:1:length(knee_thetadot)-1],knee_theta)
        % title([cur_filename, "  knee_theta vs. time"], 'Interpreter', 'none');

        % len_foot_accelX_vec = [len_foot_accelX_vec, length(foot_accelX)];
        
          % Plot only residual oscillations
          % Note: segment signal for residual oscillations between: 100 < data <300
          % Altered to be: segment signal for residual oscillations between: 200 < data <300
        hold on
        figure(i*3-2);
        hold on
        % figure();
        timesteps_mask = 200:300;                               % mask for residual oscillations in Converted_CSV_Data_Collection_1
        time_trimmed_ms = [0:1:length(timesteps_mask)-1] * 10 ; % time steps are sampled every 10 ms
        time_trimmed_s = time_trimmed_ms * 1/1000;              
        signal_trimmed = foot_accelX(timesteps_mask);           % choose which signal to plot
        accel_signal_detrend = detrend(signal_trimmed);             % detrend the data, get rid of y axis offset
       
        vel_signal = cumtrapz(time_trimmed_s, accel_signal_detrend);
        vel_signal_detrend = detrend(vel_signal);                 % detrend the data, get rid of y axis offset
        pos_signal = cumtrapz(time_trimmed_s, vel_signal);
        pos_signal_detrend = detrend(pos_signal);

        plot(time_trimmed_s,vel_signal)
        grid on
        ylim([-.2,.15])
        xlabel('Time (s)'); ylabel('vel_signal');
        title(['Residual Oscillations: ', cur_filename], 'Interpreter', 'none');
        hold on 
        % 
        % %compute Fast Fourier Transform (FFT) to Find the Frequency of Oscillation
        Yf = fft(pos_signal_detrend);
        % Define the frequency axis
        N = length(Yf);  % Number of data points
        dt = .01;       % Sampling interval s
        df = 1/(N*dt);  % Frequency resolution
        nf = floor(N/2)+1;
        f = (0:nf-1)'*df;
        Yf = Yf(1:nf);
        acceleration_magnitude = abs(Yf); % Compute the magnitude of the FFT (convert to acceleration)
        hold on
        figure(i-2);
        hold on
        % figure();
        plot(f,acceleration_magnitude) % plot FFT
        % ylim([0,40])
        grid on
        title([cur_filename, "  Pos. vs. Frequency"], 'Interpreter', 'none');
        xlabel("Frequency (Hz)")
        ylabel("pos (m)")
        hold on 
    end
    hold off
    fprintf('Plotted all files in: %s \n', cur_folder_name);
    fprintf("PAUSED: Press any to key close all plots and generate the next set of plots \n");
    % pause;      
    % close all;  % close all trial plots for this set of k and b params
end

        
        % figure();
        % plot(time_trimmed_s,vel_signal)
        % xlabel('Time (s)'); ylabel('vel_signal');
        % title(['Residual Oscillations: ', cur_filename], 'Interpreter', 'none');
        % 
        % figure();
        % plot(time_trimmed_s,vel_signal_detrend)
        % xlabel('Time (s)'); ylabel('vel_signal_detrend');
        % title(['Residual Oscillations: ', cur_filename], 'Interpreter', 'none');
        % 
        % figure();
        % plot(time_trimmed_s,pos_signal)
        % xlabel('Time (s)'); ylabel('pos_signal');
        % title(['Residual Oscillations: ', cur_filename], 'Interpreter', 'none');




%% subplots Plots position residual oscillations vs time
% Date: November 6, 2024
% 
% NOTE: Make sure to run from the OSL_Oscillation_Analysis folder
clear all; close all; clc
fprintf("Begin Plotting Script... \n");

parent_folder_name = 'Shaped_Converted_CSV_Data_Collection_1'; %______________ Filtered Dataset for analyzing shaped ZV, unshaped, EI and ZVD
% parent_folder_name = 'Shaped_Converted_CSV_Data_Collection_2'; %______________ Filtered Dataset for analyzing shaped EI and ZVD

subplot_name = {'Unshaped','ZV Shaped','ZVD Shaped','EI Shaped'};

len_foot_accelX_vec = [];
folders_struct = dir(parent_folder_name);
for i = 3:length(folders_struct) % loop through folders
    cur_folder_name = folders_struct(i).name;
    filename_struct = dir([parent_folder_name '/' cur_folder_name]);
    hold on
    subplot(2,2,i-2)
    hold on
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
        % figure(i*2-3);
        % plot([0:1:length(knee_thetadot)-1],knee_theta)
        % title([cur_filename, "  knee_theta vs. time"], 'Interpreter', 'none');

        len_foot_accelX_vec = [len_foot_accelX_vec, length(foot_accelX)];

          % Plot only residual oscillations
          % Note: segment signal for residual oscillations between: 100 < data <300
          % Altered to be: segment signal for residual oscillations between: 200 < data <300
        hold on
        % figure(i*3-2);
        % hold on
        % figure();
        timesteps_mask = 200:300;                               % mask for residual oscillations in Converted_CSV_Data_Collection_1
        time_trimmed_ms = [0:1:length(timesteps_mask)-1] * 10 ; % time steps are sampled every 10 ms
        time_trimmed_s = time_trimmed_ms * 1/1000;              
        signal_trimmed = foot_accelX(timesteps_mask);           % choose which signal to plot
        accel_signal_detrend = detrend(signal_trimmed);             % detrend the data, get rid of y axis offset

        vel_signal = cumtrapz(time_trimmed_s, accel_signal_detrend);
        vel_signal_detrend = detrend(vel_signal);                 % detrend the data, get rid of y axis offset
        pos_signal = cumtrapz(time_trimmed_s, vel_signal);
        pos_signal_detrend = detrend(pos_signal);

        pos_signal = pos_signal * 1000;
        plot(time_trimmed_s,pos_signal)
        
        grid on
        ylim([-25,15])
        xlabel('Time (s)'); ylabel('Position Signal (mm)');
        % title(['Residual Oscillations: ', cur_filename], 'Interpreter', 'none');
        title(['Residual Oscillations: ', subplot_name(i-2)]);


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
        % hold on
        % figure(i-2);
        % hold on
        % % figure();
        % plot(f,acceleration_magnitude) % plot FFT
        % ylim([0,40])
        % grid on
        % title([cur_filename, "  Accel vs. Frequency"], 'Interpreter', 'none');
        % xlabel("Frequency (Hz)")
        % ylabel("Accel (m/s^2)")
        % hold on 
    end
    hold off
    fprintf('Plotted all files in: %s \n', cur_folder_name);
    fprintf("PAUSED: Press any to key close all plots and generate the next set of plots \n");
    % pause;      
    % close all;  % close all trial plots for this set of k and b params
end


%% subplot vel residual oscillations vs time
% Date: November 6, 2024
% 
% NOTE: Make sure to run from the OSL_Oscillation_Analysis folder
clear all; close all; clc
fprintf("Begin Plotting Script... \n");

parent_folder_name = 'Shaped_Converted_CSV_Data_Collection_1'; %______________ Filtered Dataset for analyzing shaped ZV, unshaped, EI and ZVD
% parent_folder_name = 'Shaped_Converted_CSV_Data_Collection_2'; %______________ Filtered Dataset for analyzing shaped EI and ZVD

subplot_name = {'Unshaped','ZV Shaped','ZVD Shaped','EI Shaped'};

len_foot_accelX_vec = [];
folders_struct = dir(parent_folder_name);
for i = 3:length(folders_struct) % loop through folders
    cur_folder_name = folders_struct(i).name;
    filename_struct = dir([parent_folder_name '/' cur_folder_name]);
    hold on
    subplot(2,2,i-2)
    hold on
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

        
        grid on
        % Plot entire signal trial as segmented by the bag2csv.py code
        % figure();
        % plot([0:1:length(foot_accelX)-1],foot_accelX)
        % figure(i*2-3);
        % plot([0:1:length(knee_thetadot)-1],knee_theta)
        % title([cur_filename, "  knee_theta vs. time"], 'Interpreter', 'none');

        % len_foot_accelX_vec = [len_foot_accelX_vec, length(foot_accelX)];

          % Plot only residual oscillations
          % Note: segment signal for residual oscillations between: 100 < data <300
          % Altered to be: segment signal for residual oscillations between: 200 < data <300
        hold on
        % figure(i*3-2);
        % hold on
        % figure();
        timesteps_mask = 200:300;                               % mask for residual oscillations in Converted_CSV_Data_Collection_1
        time_trimmed_ms = [0:1:length(timesteps_mask)-1] * 10 ; % time steps are sampled every 10 ms
        time_trimmed_s = time_trimmed_ms * 1/1000;              
        signal_trimmed = foot_accelX(timesteps_mask);           % choose which signal to plot
        accel_signal_detrend = detrend(signal_trimmed);             % detrend the data, get rid of y axis offset


        vel_signal = cumtrapz(time_trimmed_s, accel_signal_detrend);
        vel_signal_detrend = detrend(vel_signal);                 % detrend the data, get rid of y axis offset
        pos_signal = cumtrapz(time_trimmed_s, vel_signal);
        pos_signal_detrend = detrend(pos_signal);

        pos_signal = pos_signal * 1000;
        % plot(time_trimmed_s,vel_signal)

        plot(time_trimmed_s,pos_signal)
        
        grid on
        % ylim([-5,3])
        xlabel('Time (s)'); ylabel('Position Signal (m)');
        % title(['Residual Oscillations: ', cur_filename], 'Interpreter', 'none');
        title([subplot_name(i-2), ' Residual Oscillations']);


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
        % hold on
        % figure(i-2);
        % hold on
        % % figure();
        % plot(f,acceleration_magnitude) % plot FFT
        % ylim([0,40])
        % grid on
        % title([cur_filename, "  Accel vs. Frequency"], 'Interpreter', 'none');
        % xlabel("Frequency (Hz)")
        % ylabel("Accel (m/s^2)")
        % hold on 
    end
    hold off
    fprintf('Plotted all files in: %s \n', cur_folder_name);
    fprintf("PAUSED: Press any to key close all plots and generate the next set of plots \n");
    % pause;      
    % close all;  % close all trial plots for this set of k and b params
end
