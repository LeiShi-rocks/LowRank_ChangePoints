% Report large MC simulation results
clear;clc;
load('Record_largeMC.mat');

%% error
mean_error = mean(Record_largeMC.error, 3);
std_error  = std(Record_largeMC.error, 0, 3); 

%% rank
mean_rank = mean(Record_largeMC.rank, 3);
std_rank  = std(Record_largeMC.rank, 0, 3); 

%% time
mean_timeTuning = mean(Record_largeMC.timeTuning, 3);
mean_timeSolving = mean(Record_largeMC.timeSolving, 3);

%% iterations
mean_numIter = mean(Record_largeMC.numIter, 3);