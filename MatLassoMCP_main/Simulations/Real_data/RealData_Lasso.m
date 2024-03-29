%% Import data from text file
clear;clc;

% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 18);

% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["No", "year", "month", "day", "hour", "PM25", "PM10", "SO2", "NO2", "CO", "O3", "Var12", "Var13", "Var14", "Var15", "Var16", "Var17", "Var18"];
opts.SelectedVariableNames = ["No", "year", "month", "day", "hour", "PM25", "PM10", "SO2", "NO2", "CO", "O3"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "string", "string", "string", "string", "string", "string", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["Var12", "Var13", "Var14", "Var15", "Var16", "Var17", "Var18"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Var12", "Var13", "Var14", "Var15", "Var16", "Var17", "Var18"], "EmptyFieldRule", "auto");

% Import the data (go to the data folder first)
PRSADataAotizhongxin = readtable("PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv", opts);
PRSADataChangping = readtable("PRSA_Data_20130301-20170228/PRSA_Data_Changping_20130301-20170228.csv", opts);
PRSADataDingling = readtable("PRSA_Data_20130301-20170228/PRSA_Data_Dingling_20130301-20170228.csv", opts);
PRSADataDongsi = readtable("PRSA_Data_20130301-20170228/PRSA_Data_Dongsi_20130301-20170228.csv", opts);
PRSADataGuanyuan = readtable("PRSA_Data_20130301-20170228/PRSA_Data_Guanyuan_20130301-20170228.csv", opts);
PRSADataGucheng = readtable("PRSA_Data_20130301-20170228/PRSA_Data_Gucheng_20130301-20170228.csv", opts);
PRSADataHuairou = readtable("PRSA_Data_20130301-20170228/PRSA_Data_Huairou_20130301-20170228.csv", opts);
PRSADataNongzhanguan = readtable("PRSA_Data_20130301-20170228/PRSA_Data_Nongzhanguan_20130301-20170228.csv", opts);
PRSADataShunyi = readtable("PRSA_Data_20130301-20170228/PRSA_Data_Shunyi_20130301-20170228.csv", opts);
PRSADataTiantan = readtable("PRSA_Data_20130301-20170228/PRSA_Data_Tiantan_20130301-20170228.csv", opts);
PRSADataWanliu = readtable("PRSA_Data_20130301-20170228/PRSA_Data_Wanliu_20130301-20170228.csv", opts);
PRSADataWanshouxigong = readtable("PRSA_Data_20130301-20170228/PRSA_Data_Wanshouxigong_20130301-20170228.csv", opts);

% Clear temporary variables
clear opts






%% Preprocessing 

PRSAgroupmean = @(T) groupsummary(T, {'year', 'month', 'day'}, 'mean', {'PM25', 'PM10', 'SO2', 'NO2',  'CO', 'O3'}, 'IncludeMissingGroups', false);

PRSADataAotizhongxin1    =  PRSAgroupmean(PRSADataAotizhongxin);
PRSADataChangping1       =  PRSAgroupmean(PRSADataChangping);
PRSADataDingling1        =  PRSAgroupmean(PRSADataDingling);
PRSADataDongsi1          =  PRSAgroupmean(PRSADataDongsi);
PRSADataGuanyuan1        =  PRSAgroupmean(PRSADataGuanyuan);
PRSADataGucheng1         =  PRSAgroupmean(PRSADataGucheng);
PRSADataHuairou1         =  PRSAgroupmean(PRSADataHuairou);
PRSADataNongzhanguan1    =  PRSAgroupmean(PRSADataNongzhanguan);
PRSADataShunyi1          =  PRSAgroupmean(PRSADataShunyi);
PRSADataTiantan1         =  PRSAgroupmean(PRSADataTiantan);
PRSADataWanliu1          =  PRSAgroupmean(PRSADataWanliu);
PRSADataWanshouxigong1   =  PRSAgroupmean(PRSADataWanshouxigong);


% y = [PRSADataAotizhongxin1.mean_PM25, ...
%     PRSADataChangping1.mean_PM25, ...
%     PRSADataDingling1.mean_PM25, ...
%     PRSADataDongsi1.mean_PM25, ...
%     PRSADataGuanyuan1.mean_PM25, ...
%     PRSADataGucheng1.mean_PM25, ...
%     PRSADataHuairou1.mean_PM25, ...
%     PRSADataNongzhanguan1.mean_PM25, ...
%     PRSADataShunyi1.mean_PM25, ...
%     PRSADataTiantan1.mean_PM25, ...
%     PRSADataWanliu1.mean_PM25, ...
%     PRSADataWanshouxigong1.mean_PM25]';

y = [PRSADataAotizhongxin1{:, {'mean_PM25', 'mean_PM10'}}, ...
    PRSADataChangping1{:, {'mean_PM25', 'mean_PM10'}}, ...
    PRSADataDingling1{:, {'mean_PM25', 'mean_PM10'}}, ...
    PRSADataDongsi1{:, {'mean_PM25', 'mean_PM10'}}, ...
    PRSADataGuanyuan1{:, {'mean_PM25', 'mean_PM10'}}, ...
    PRSADataGucheng1{:, {'mean_PM25', 'mean_PM10'}}, ...
    PRSADataHuairou1{:, {'mean_PM25', 'mean_PM10'}}, ...
    PRSADataNongzhanguan1{:, {'mean_PM25', 'mean_PM10'}}, ...
    PRSADataShunyi1{:, {'mean_PM25', 'mean_PM10'}}, ...
    PRSADataTiantan1{:, {'mean_PM25', 'mean_PM10'}}, ...
    PRSADataWanliu1{:, {'mean_PM25', 'mean_PM10'}}, ...
    PRSADataWanshouxigong1{:, {'mean_PM25', 'mean_PM10'}}]';


X = [PRSADataAotizhongxin1{:, {'mean_SO2', 'mean_NO2', 'mean_CO', 'mean_O3'}},...
    PRSADataChangping1{:, {'mean_SO2', 'mean_NO2', 'mean_CO', 'mean_O3'}},...
    PRSADataDingling1{:, {'mean_SO2', 'mean_NO2', 'mean_CO', 'mean_O3'}},...
    PRSADataDongsi1{:, {'mean_SO2', 'mean_NO2', 'mean_CO', 'mean_O3'}},...
    PRSADataGuanyuan1{:, {'mean_SO2', 'mean_NO2', 'mean_CO', 'mean_O3'}},...
    PRSADataGucheng1{:, {'mean_SO2', 'mean_NO2', 'mean_CO', 'mean_O3'}},...
    PRSADataHuairou1{:, {'mean_SO2', 'mean_NO2', 'mean_CO', 'mean_O3'}},...
    PRSADataNongzhanguan1{:, {'mean_SO2', 'mean_NO2', 'mean_CO', 'mean_O3'}},...
    PRSADataShunyi1{:, {'mean_SO2', 'mean_NO2', 'mean_CO', 'mean_O3'}},...
    PRSADataTiantan1{:, {'mean_SO2', 'mean_NO2', 'mean_CO', 'mean_O3'}},...
    PRSADataWanliu1{:, {'mean_SO2', 'mean_NO2', 'mean_CO', 'mean_O3'}},...
    PRSADataWanshouxigong1{:, {'mean_SO2', 'mean_NO2', 'mean_CO', 'mean_O3'}}]';

dates = PRSADataAotizhongxin1{:, {'year', 'month', 'day'}};

% delete missing columns

nr = size(y, 1);
nc = size(X, 1);
N  = size(y, 2);

[row_y, col_y] = ind2sub([nr, N], find(isnan(y)));
[row_X, col_X] = ind2sub([nc, N], find(isnan(X)));
missing_col    = unique([col_y;col_X]);

y(:, missing_col) = [];
X(:, missing_col) = [];
dates(missing_col, :) = [];

N  = size(y, 2);
% subset of the data
% y = y(:, 1:50);
% X = X(:, 1:50);

% normalizing the data 
y = normalize(y,2);
X = normalize(X,2);

% split the data for training and testing
test_ind  = 2:5:N;  % 2 is a good split
train_ind = 1:N;
train_ind(test_ind) = [];

y_train = y(:, train_ind);
X_train = X(:, train_ind);
y_test  = y(:, test_ind);
X_test  = X(:, test_ind);

%% Train model

nr = size(y_train, 1);
nc = size(X_train, 1);
N_train  = size(y_train, 2);
N_test   = size(y_test,  2);

%% OLS (with no penalty)
Theta_hat_0   = y_train * X_train' / (X_train * X_train');
Pred_0        = Theta_hat_0 * X_test;
Pred_error_0  = sum(sum((y_test - Pred_0).^2))/(nr * N_test);
tPred_0       = Theta_hat_0 * X_train;
train_error_0 = sum(sum((y_train - tPred_0).^2))/(nr* N_train);
rank_0        = sum(svd(Theta_hat_0) > 1e-7);

%% Lasso without change points
if 1
    type = struct(...
        'name', 'L2',...
        'eta', 0.8,...
        'Lf', 1e5);
    Clambda = 0.02;
    tol = 1e-4;
    maxiter = 4e2;
    Theta_init = zeros(nr, nc);
    
    [Theta_hat_00, rank_00] = MVAPG_MCP(y_train, X_train, type, Clambda, tol, maxiter, Theta_init, 'l1');
    Pred_00       = Theta_hat_00 * X_test;
    Pred_error_00 = sum(sum((y_test - Pred_00).^2))/(nr*N_test);
    tPred_00       = Theta_hat_00 * X_train;
    train_error_00 = sum(sum((y_train - tPred_00).^2))/(nr*N_train);
end

%% Lasso with a single change point
if 0
    threshold_var = (1:N_train)/N_train;
    APG_opts = struct(...
        'type', type,...
        'Clambda', 0.05,...
        'tol', 1e-4,...
        'maxiter', 4e2,...
        'Theta_init', zeros(nr, 2*nc));
    [Theta_Delta_hat, tau_hat, obj_path, Delta_path] = LassoSCP(y_train, X_train, threshold_var, 0.05, [0.1, 0.9], 50, APG_opts);
    
    N_hat        = floor(tau_hat*N_test);
    Theta_hat_31 = Theta_Delta_hat(:,1:nc);
    Theta_hat_32 = Theta_Delta_hat(:,1:nc) + Theta_Delta_hat(:,(nc+1):(2*nc));
    Pred_31      = Theta_hat_31 * X_test(:, 1:N_hat);
    Pred_32      = Theta_hat_32 * X_test(:, (N_hat+1):end);
    Pred_3       = [Pred_31, Pred_32];
    Pred_error_3 = sum(sum((y_test - Pred_3).^2))/(nr*N_test);
end


%% global tuning parameters
Clambda_base  = [0.05, 0.05];
Clambda = 0.05;
%% Matrix Lasso with multiple change points (cutoff = 30, 1 change point)
Clambda = 0.10;
threshold_var = (1:N_train)/N_train;
type = struct(...
    'name', 'L2',...
    'eta', 0.8,...
    'Lf', 1e5);
%Clambda_base  = [0.15, 0.15];
%Clambda_base  = [0.15, 0.15];
window_length = 0.10;
num_windows   = 30;
cutoff        = 30; 

APG_opts_1 = struct(...
    "type", type,...
    "tol", 1e-4,...
    "maxiter", 3e2);
SCP_args_1 = struct(...
    "kappa", 0.1,...
    "resolution_In", 25,...
    "APG_opts", APG_opts_1);

APG_opts_2 = struct(...
    "type", type,...
    "tol", 1e-4,...
    "maxiter", 3e2);
SCP_args_2 = struct(...
    "kappa", 0.05,...
    "resolution_In", 50,...
    "APG_opts", APG_opts_2);

post_APG_args = struct(...
    "type", type,...
    "Clambda", 0.05,...
    "tol", 1e-4,...
    "maxiter", 3e2);
    
MCP_opts      = struct();


[post_Theta_hat_1, post_tau_hat_1, MCP_outInfo_1] = ...
    LassoMCP(y_train, X_train, threshold_var, Clambda_base,...
    window_length, num_windows, cutoff,...
    SCP_args_1, SCP_args_2, post_APG_args, MCP_opts);

% post_tau_hat = 0.3939;

Theta_hat_1   = zeros(nr,nc,length(post_tau_hat_1)+1);
% rank_1 = zeros(1,length(post_tau_hat_1)+1);
N_train_hat   = [1, floor(post_tau_hat_1 * N_train), N_train + 1];
N_test_hat    = [1, floor(post_tau_hat_1 * N_test),  N_test  + 1];
Pred_1        = [];
tPred_1       = [];
for seg_ind = 1:(length(N_test_hat) - 1)
    [Theta_hat_1(:,:,seg_ind), ~] = MVAPG_MCP(y_train(:,N_train_hat(seg_ind):(N_train_hat(seg_ind + 1) - 1)), ...
        X_train(:,N_train_hat(seg_ind):(N_train_hat(seg_ind + 1) - 1)), ...
        type, Clambda, tol, maxiter, Theta_init, 'l1');
    Pred_1     = [Pred_1,  Theta_hat_1(:,:,seg_ind) * X_test(:, N_test_hat(seg_ind):(N_test_hat(seg_ind + 1) - 1))];
    tPred_1    = [tPred_1, Theta_hat_1(:,:,seg_ind) * X_train(:, N_train_hat(seg_ind):(N_train_hat(seg_ind + 1) - 1))];
end

Pred_error_1   = sum(sum((y_test - Pred_1).^2))/(nr * N_test);
train_error_1  = sum(sum((y_train - tPred_1).^2))/(nr * N_train);

%% Matrix Lasso with multiple change points (cutoff = 25, 2 change point)
Clambda = 0.10;
threshold_var = (1:N_train)/N_train;
type = struct(...
    'name', 'L2',...
    'eta', 0.8,...
    'Lf', 1e5);
% Clambda_base  = [0.15, 0.15];
% Clambda_base  = [0.15, 0.15];
window_length = 0.10;
num_windows   = 30;
cutoff        = 25; 

APG_opts_1 = struct(...
    "type", type,...
    "tol", 1e-4,...
    "maxiter", 3e2);
SCP_args_1 = struct(...
    "kappa", 0.1,...
    "resolution_In", 25,...
    "APG_opts", APG_opts_1);

APG_opts_2 = struct(...
    "type", type,...
    "tol", 1e-4,...
    "maxiter", 3e2);
SCP_args_2 = struct(...
    "kappa", 0.05,...
    "resolution_In", 50,...
    "APG_opts", APG_opts_2);

post_APG_args = struct(...
    "type", type,...
    "Clambda", 0.05,...
    "tol", 1e-4,...
    "maxiter", 3e2);
    
MCP_opts      = struct();


[post_Theta_hat_2, post_tau_hat_2, MCP_outInfo_2] = ...
   LassoMCP(y_train, X_train, threshold_var, Clambda_base,...
   window_length, num_windows, cutoff,...
   SCP_args_1, SCP_args_2, post_APG_args, MCP_opts);

% post_tau_hat = [0.3947    0.9219];

Theta_hat_2 = zeros(nr,nc,length(post_tau_hat_2)+1);
rank_2 = zeros(1,length(post_tau_hat_2)+1);
N_train_hat  = [1, floor(post_tau_hat_2 * N_train), N_train + 1];
N_test_hat   = [1, floor(post_tau_hat_2 * N_test),  N_test  + 1];
Pred_2       = [];
tPred_2      = [];
for seg_ind = 1:(length(N_test_hat) - 1)
    [Theta_hat_2(:,:,seg_ind), ~] = MVAPG_MCP(y_train(:,N_train_hat(seg_ind):(N_train_hat(seg_ind + 1) - 1)), ...
        X_train(:,N_train_hat(seg_ind):(N_train_hat(seg_ind + 1) - 1)), ...
        type, Clambda, tol, maxiter, Theta_init, 'l1');
    Pred_2     = [Pred_2, Theta_hat_2(:,:,seg_ind) * X_test(:, N_test_hat(seg_ind):(N_test_hat(seg_ind + 1) - 1))];
    tPred_2    = [tPred_2, Theta_hat_2(:,:,seg_ind) * X_train(:, N_train_hat(seg_ind):(N_train_hat(seg_ind + 1) - 1))];
end



Pred_error_2 = sum(sum((y_test - Pred_2).^2))/(nr * N_test);
train_error_2 = sum(sum((y_train - tPred_2).^2))/(nr * N_train);

% post_tau_hat = [0.3945, 0.9220];
% N_train_hat  = floor(post_tau_hat * N_train);
% N_test_hat   = floor(post_tau_hat * N_test);
% 
% 
% type = struct(...
%     'name', 'L2',...
%     'eta', 0.8,...
%     'Lf', 1e5);
% Clambda = 0.15;
% tol = 1e-4;
% maxiter = 3e2;
% Theta_init = zeros(nr, nc);
% 
% [Theta_hat_51, rank] = MVAPG_MCP(y_train(:,1:347), X_train(:,1:347), type, Clambda, tol, maxiter, Theta_init);
% Pred_51       = Theta_hat_51 * X_test(:, 1:86);
% 
% 
% [Theta_hat_52, rank] = MVAPG_MCP(y_train(:,348:811), X_train(:,348:811), type, Clambda, tol, maxiter, Theta_init);
% Pred_52       = Theta_hat_52 * X_test(:, 87:202);
% 
% 
% [Theta_hat_53, rank] = MVAPG_MCP(y_train(:,812:end), X_train(:,812:end), type, Clambda, tol, maxiter, Theta_init);
% Pred_53       = Theta_hat_53 * X_test(:, 203:end);
% 
% Pred_5 = [Pred_51, Pred_52, Pred_53];
% 
% Pred_error_5 = sum(sum((y_test - Pred_5).^2))/(nr * N_test);


%% Matrix Lasso with multiple change points (cutoff = 15, 3 change point)
Clambda = 0.10;
threshold_var = (1:N_train)/N_train;
type = struct(...
    'name', 'L2',...
    'eta', 0.8,...
    'Lf', 1e5);
%Clambda_base  = [0.15, 0.15];
%Clambda_base  = [0.15, 0.15];
window_length = 0.10;
num_windows   = 30;
cutoff        = 20; 

APG_opts_1 = struct(...
    "type", type,...
    "tol", 1e-4,...
    "maxiter", 3e2);
SCP_args_1 = struct(...
    "kappa", 0.1,...
    "resolution_In", 25,...
    "APG_opts", APG_opts_1);

APG_opts_2 = struct(...
    "type", type,...
    "tol", 1e-4,...
    "maxiter", 3e2);
SCP_args_2 = struct(...
    "kappa", 0.05,...
    "resolution_In", 50,...
    "APG_opts", APG_opts_2);

post_APG_args = struct(...
    "type", type,...
    "Clambda", 0.05,...
    "tol", 1e-4,...
    "maxiter", 3e2);
    
MCP_opts      = struct();


[post_Theta_hat_3, post_tau_hat_3, MCP_outInfo_3] = ...
    LassoMCP(y_train, X_train, threshold_var, Clambda_base,...
    window_length, num_windows, cutoff,...
    SCP_args_1, SCP_args_2, post_APG_args, MCP_opts);

% post_tau_hat = [0.2096    0.3947    0.9219];

Theta_hat_3 = zeros(nr,nc,length(post_tau_hat_3)+1);
rank_3 = zeros(1,length(post_tau_hat_3)+1);
N_train_hat  = [1, floor(post_tau_hat_3 * N_train), N_train + 1];
N_test_hat   = [1, floor(post_tau_hat_3 * N_test),  N_test  + 1];
Pred_3       = [];
tPred_3      = [];
for seg_ind = 1:(length(N_test_hat) - 1)
    [Theta_hat_3(:,:,seg_ind), ~] = MVAPG_MCP(y_train(:,N_train_hat(seg_ind):(N_train_hat(seg_ind + 1) - 1)), ...
        X_train(:,N_train_hat(seg_ind):(N_train_hat(seg_ind + 1) - 1)), ...
        type, Clambda, tol, maxiter, Theta_init, 'l1');
    Pred_3     = [Pred_3,  Theta_hat_3(:,:,seg_ind) * X_test(:, N_test_hat(seg_ind):(N_test_hat(seg_ind + 1) - 1))];
    tPred_3    = [tPred_3, Theta_hat_3(:,:,seg_ind) * X_train(:, N_train_hat(seg_ind):(N_train_hat(seg_ind + 1) - 1))];
end



Pred_error_3 = sum(sum((y_test - Pred_3).^2))/(nr * N_test);
train_error_3 = sum(sum((y_train - tPred_3).^2))/(nr * N_train);


% post_tau_hat = [0.1880, 0.4298, 0.7592];
% N_train_hat  = floor(post_tau_hat * N_train);
% N_test_hat   = floor(post_tau_hat * N_test);
% 
% 
% type = struct(...
%     'name', 'L2',...
%     'eta', 0.8,...
%     'Lf', 1e5);
% Clambda = 0.15;
% tol = 1e-4;
% maxiter = 3e2;
% Theta_init = zeros(nr, nc);
% 
% 
% [Theta_hat_61, rank] = MVAPG_MCP(y_train(:,1:N_train_hat(1)), X_train(:,1:N_train_hat(1)), type, Clambda, tol, maxiter, Theta_init);
% % [Theta_hat_61, rank] = MVAPG_MCP(y_train(:,1:N_train_hat(2)), X_train(:,1:N_train_hat(2)), type, Clambda, tol, maxiter, Theta_init);
% Pred_61       = Theta_hat_61 * X_test(:, 1:N_test_hat(1));
% 
% [Theta_hat_62, rank] = MVAPG_MCP(y_train(:,(N_train_hat(1)+1):N_train_hat(2)), X_train(:,(N_train_hat(1)+1):N_train_hat(2)), type, Clambda, tol, maxiter, Theta_init);
% % [Theta_hat_62, rank] = MVAPG_MCP(y_train(:,1:N_train_hat(2)), X_train(:,1:N_train_hat(2)), type, Clambda, tol, maxiter, Theta_init);
% Pred_62       = Theta_hat_62 * X_test(:, (N_test_hat(1)+1):N_test_hat(2));
% 
% [Theta_hat_63, rank] = MVAPG_MCP(y_train(:,(N_train_hat(2)+1):N_train_hat(3)), X_train(:,(N_train_hat(2)+1):N_train_hat(3)), type, Clambda, tol, maxiter, Theta_init);
% Pred_63       = Theta_hat_63 * X_test(:, (N_test_hat(2)+1):N_test_hat(3));
% 
% [Theta_hat_64, rank] = MVAPG_MCP(y_train(:,(N_train_hat(3)+1):end), X_train(:,(N_train_hat(3)+1):end), type, Clambda, tol, maxiter, Theta_init);
% Pred_64       = Theta_hat_64 * X_test(:, (N_test_hat(3)+1):end);
% 
% Pred_6 = [Pred_61, Pred_62, Pred_63, Pred_64];
% 
% Pred_error_6 = sum(sum((y_test - Pred_6).^2))/(nr * N_test);


%% Matrix Lasso with multiple change points (cutoff = 10, 4 change point)
Clambda = 0.10;
threshold_var = (1:N_train)/N_train;
type = struct(...
    'name', 'L2',...
    'eta', 0.8,...
    'Lf', 1e5);
%Clambda_base  = [0.15, 0.15];
%Clambda_base  = [0.15, 0.15];
window_length = 0.10;
num_windows   = 30;
cutoff        = 10; 

APG_opts_1 = struct(...
    "type", type,...
    "tol", 1e-4,...
    "maxiter", 3e2);
SCP_args_1 = struct(...
    "kappa", 0.1,...
    "resolution_In", 25,...
    "APG_opts", APG_opts_1);

APG_opts_2 = struct(...
    "type", type,...
    "tol", 1e-4,...
    "maxiter", 3e2);
SCP_args_2 = struct(...
    "kappa", 0.05,...
    "resolution_In", 50,...
    "APG_opts", APG_opts_2);

post_APG_args = struct(...
    "type", type,...
    "Clambda", 0.05,...
    "tol", 1e-4,...
    "maxiter", 3e2);
    
MCP_opts      = struct();


[post_Theta_hat_4, post_tau_hat_4, MCP_outInfo_4] = ...
    LassoMCP(y_train, X_train, threshold_var, Clambda_base,...
    window_length, num_windows, cutoff,...
    SCP_args_1, SCP_args_2, post_APG_args, MCP_opts);

% post_tau_hat = [0.2096    0.3947    0.9219];

Theta_hat_4 = zeros(nr,nc,length(post_tau_hat_4)+1);
% rank_4 = zeros(1,length(post_tau_hat_4)+1);
N_train_hat  = [1, floor(post_tau_hat_4 * N_train), N_train + 1];
N_test_hat   = [1, floor(post_tau_hat_4 * N_test),  N_test  + 1];
Pred_4       = [];
tPred_4      = [];
for seg_ind = 1:(length(N_test_hat) - 1)
    [Theta_hat_4(:,:,seg_ind), ~] = MVAPG_MCP(y_train(:,N_train_hat(seg_ind):(N_train_hat(seg_ind + 1) - 1)), ...
        X_train(:,N_train_hat(seg_ind):(N_train_hat(seg_ind + 1) - 1)), ...
        type, Clambda, tol, maxiter, Theta_init, 'l1');
    Pred_4     = [Pred_4, Theta_hat_4(:,:,seg_ind) * X_test(:, N_test_hat(seg_ind):(N_test_hat(seg_ind + 1) - 1))];
    tPred_4    = [tPred_4, Theta_hat_4(:,:,seg_ind) * X_train(:, N_train_hat(seg_ind):(N_train_hat(seg_ind + 1) - 1))];
end



Pred_error_4  = sum(sum((y_test - Pred_4).^2))/(nr * N_test);
train_error_4 = sum(sum((y_train - tPred_4).^2))/(nr * N_train);


%%


real_data_lasso_record = struct(...
    'Theta_hat_0', Theta_hat_0, ...
    'Theta_hat_00', Theta_hat_00, ...
    'Theta_hat_1', Theta_hat_1, ...
    'Theta_hat_2', Theta_hat_2, ...
    'Theta_hat_3', Theta_hat_3, ...
    'Theta_hat_4', Theta_hat_4, ...
    'post_tau_hat_1', post_tau_hat_1, ...
    'post_tau_hat_2', post_tau_hat_2, ...
    'post_tau_hat_3', post_tau_hat_3, ...
    'post_tau_hat_4', post_tau_hat_4); 

save_flag = 1;
if save_flag
    save('real_data_lasso_result.mat', 'real_data_lasso_record');
end



%% 
Pred_error  = [Pred_error_0 Pred_error_00 Pred_error_1 Pred_error_2 Pred_error_3 Pred_error_4]';
train_error = [train_error_0 train_error_00 train_error_1 train_error_2 train_error_3 train_error_4]';

figure;
plot(0:4, Pred_error(2:end), 'o-', 'Linewidth', 6, 'MarkerSize', 14); 
hold on;
plot(0:4, train_error(2:end), '*--', 'Linewidth', 6, 'MarkerSize', 14);
legend('Test', 'Train');
ylabel('Error');
xlabel('Number of change points selected');



