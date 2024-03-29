clear;clc;
nr = 40;
nc = 40;
N = 2000;
r = 5;

noise = struct(...
    'type', 'Gaussian', ...
    'scale', 1.0, ...
    'para', 0.1);
problem = 'MR';
signal_size = 1;
cp_opts = struct(...
    'num_seg', 4,...
    'pos_seg', [0, 0.25, 0.50, 0.75]);

design = struct(...
    'type', 'AR',...
    'para', 0);


num_iter = 100;

record    = cell(5, num_iter); % post_Theta_hat, post_tau_hat, post_rank, MCP_outInfo, Theta_star
save_flag = 1;

rng(2022);

for iter = 1:num_iter

    fprintf('\n');
    fprintf('Running iter %d \n\n', iter);
    
    [y, X, outInfo] = DataGenMCP(nr, nc, N, r, noise, problem, design, cp_opts);
    Theta_star = outInfo.Theta_star;
    threshold_var = outInfo.threshold_var;
    
    %%
    % type = struct(...
    %     'name', 'L2',...
    %     'eta', 0.8,...
    %     'Lf', 1e4);
    % [Theta_hat, rank, outInfo] = MVAPG_MCP(y(:,1:200), X(:,1:200), type, 0.5, 1e-4, 1e2, zeros(nr, nc));
    % disp(sum(sum((Theta_hat - Theta_star(:,:,1)).^2)));
    
    %%
    % post_APG_args = struct(...
    %     "type", type,...
    %     "Clambda", 0.15,...
    %     "tol", 1e-4,...
    %     "maxiter", 1e2);
    % [Theta_Delta_hat, tau_hat, obj_path, Delta_path, SCP_outInfo] = MatLassoSCP(y, X, threshold_var, 0.10, [0.7,0.8], 25, post_APG_args);
    
    
    %% MatLassoMCP
    type = struct(...
        'name', 'L2',...
        'eta', 0.8,...
        'Lf', 1e4);
    Clambda_base  = [0.15, 0.15];
    window_length = 0.10;
    num_windows   = 20;
    cutoff        = 0.8;
    
    APG_opts_1 = struct(...
        "type", type,...
        "tol", 1e-4,...
        "maxiter", 2e2);
    SCP_args_1 = struct(...
        "kappa", 0.1,...
        "resolution_In", 25,...
        "APG_opts", APG_opts_1);
    
    APG_opts_2 = struct(...
        "type", type,...
        "tol", 1e-4,...
        "maxiter", 2e2);
    SCP_args_2 = struct(...
        "kappa", 0.1,...
        "resolution_In", 200,...
        "APG_opts", APG_opts_2);
    
    post_APG_args = struct(...
        "type", type,...
        "Clambda", 0.7,...
        "tol", 1e-4,...
        "maxiter", 2e2);
    
    MCP_opts   = struct('only_plot_flag', 0);
    
    
    [post_Theta_hat, post_tau_hat, post_rank, MCP_outInfo] = ...
        MatLassoMCP(y, X, threshold_var, Clambda_base,...
        window_length, num_windows, cutoff,...
        SCP_args_1, SCP_args_2, post_APG_args, MCP_opts);
    
    % record information
    record{1, iter} = post_Theta_hat;
    record{2, iter} = post_tau_hat;
    record{3, iter} = post_rank;
    record{4, iter} = MCP_outInfo;
    record{5, iter} = Theta_star;
    
end

if save_flag
    save('MR_MCP_large_signal.mat', 'record');
end

