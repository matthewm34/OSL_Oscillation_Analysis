% ME 6404
% Fall 2024
% Team 1
% Lab 5
% Bridge Crane Shapers

clear;
clc;
close all;

%Load data - CHANGE THIS TO THE FILENAME
lengthVec = 35;
k = 1.5;
b = 0.2;

vecOnes = [ones(lengthVec,1)];

% Creates Time Vector - times steps of 10 ms (dt) for # (lengthVec) of steps
dt = 0.01; %10 ms
t = 0:dt:dt*(lengthVec-1);
t=round(t,4);
t=t';


% calculate k based on given damping ratio
d_ratio = 0;
K = exp((-d_ratio*pi)/sqrt(1-d_ratio^2));

% calculate shapers
freq = 5;
T = 1/5;

%ZV Shaper
ZV_shaper = [1/(1+K) K/(1+K);0 .5*T]

%EI Shaper 
% % V_tol = .05;
% % EI_shaper = [(1+V_tol)/4, (1-V_tol)/2, (1+V_tol)/4; 0, .5*T, T];

%ZVD Shaper
% % ZVD_shaper = [1/(1+K)^2 2*K/(1+K)^2 K^2/(1+K)^2;0 .5*T T];


vec_unshaped = [vecOnes, t]';
vecShaped = combine_shapers(vec_unshaped, ZV_shaper);
[~,h] = sort(vecShaped(2,:));
vecShaped = vecShaped(:,h)';
vecShaped = round(vecShaped,4);


shaperVec = ShapedInput(vecShaped,dt);
shaperVecAmplitudes = shaperVec(:,1);
shaperVecTimes = shaperVec(:,2);
shaperVec = [shaperVecTimes, shaperVecAmplitudes]


kShaped = [shaperVecTimes, k * shaperVecAmplitudes]
bShaped = [shaperVecTimes, b * shaperVecAmplitudes]

plot(shaperVecTimes, shaperVecAmplitudes)


% %% Shaper Length Exceeded
% %initialization
% shaperIndCount = 0;
% 
% % during iterations of k and b / torque calcs
% shaperInd = shaperInd++;
% shaperLength = lengthVec * 2;
% if(shaperIndCount > shaperLength){
%     shaperVal = shaperVal(end);
%     }
% 

