function [result] = ShapedInput(vel,dt)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

count = 0;

for i = 1:length(vel)

    if vel(i,2) == round(count * dt,4)
        count = count+1;
        
    else

        vel(i,2) = count * dt;
    end
end


% Step 1: Get unique y values and indices
[unique_y, ~, idx] = unique(round(vel(:,2),4));  % Unique y values and index map

% Step 2: Sum the x values for each unique y
summed_x = accumarray(idx, vel(:,1));  % Sum x-values for each group of y

% Step 3: Combine the unique y values with their summed x values
result = [summed_x, unique_y];  % nx2 matrix where n <= 1400
end