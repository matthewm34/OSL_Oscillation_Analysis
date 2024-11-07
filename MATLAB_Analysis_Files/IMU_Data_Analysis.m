%% ME 6404 TEAM 6: OSL Prosthesis Dynamics Analysis
% Date: November 6, 2024
% 
% NOTE: Make sure to run from the OSL_Oscillation_Analysis folder
clear all; close all;

parent_folder_name = 'Converted_CSV_Data_Collection_1';

folders_struct = dir(parent_folder_name);
for i = 3:length(folders_struct) %loop through folders
    cur_folder_name = folders_struct(i).name;
    filename_struct = dir([parent_folder_name '/' cur_folder_name]);

    for j = 3:length(filename_struct) %loop through files
        cur_filename = filename_struct(j).name;

        % Read data as a table
        dataTable = readtable([parent_folder_name  '/'  cur_folder_name '/' cur_filename]);
    
        % Extract relevant columns
        foot_accelX = dataTable.foot_accelX;
        foot_accelY = dataTable.foot_accelY;
        foot_accelZ = dataTable.foot_accelZ;
        knee_theta = dataTable.knee_theta;
        knee_thetadot = dataTable.knee_thetadot;
    
        % Plot Sensor Channel
        figure();
        plot([0:1:length(foot_accelY)-1],foot_accelY)
        xlabel('Digital Time Steps (k)'); ylabel('foot_accelY');
        title([cur_filename]);
        test = 0;
    end
    printf('Press any key to plot next data set');pause; % analyze plots wait until keypress to continue
    close all;
end