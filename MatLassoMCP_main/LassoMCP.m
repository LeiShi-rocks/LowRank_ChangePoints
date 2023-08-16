function [post_Theta_hat, post_tau_hat, MCP_outInfo] = LassoMCP(y, X, threshold_var, Clambda_base, window_length, num_windows, cutoff,...
    SCP_args_1, SCP_args_2, post_APG_args, MCP_opts)
% MATLASSOMCP Summary of this function goes here
%   SCP_arg_1: for Step 1
%        - kappa
%        - resolution_In
%        - APG_opts = type + tol + maxiter + Theta_init
%   SCP_arg_2: for Step 2
%   post_APG_args =

runningFlag = [1,1,1,0];

% Initialization
post_Theta_hat = [];
post_tau_hat   = [];
% post_rank      = [];
MCP_outInfo    = [];
pre_tau_hat    = [];


step_length = (1 - window_length) / num_windows;
left_end = 0;
right_end = window_length;
best_obj_path = zeros(1,num_windows);
best_Delta_path = zeros(1,num_windows);

only_plot_flag = setOpts(MCP_opts, 'only_plot_flag', 0);

if size(X,3) == 1
    design_type = "V"; % vector
    nc = size(X, 1);
    N  = size(X, 2);
    nr = size(y, 1);
else
    design_type = "M"; % matrix
    nr = size(X, 1);
    nc = size(X, 2);
    N  = size(X, 3);
end


kappa_1         = setOpts(SCP_args_1, "kappa", 0.1);
resolution_In_1 = setOpts(SCP_args_1, "resolution_In", 50);
APG_opts_1      = setOpts(SCP_args_1, "APG_opts", []);


kappa_2         = setOpts(SCP_args_2, "kappa", 0.1);
resolution_In_2 = setOpts(SCP_args_2, "resolution_In", 50);
APG_opts_2      = setOpts(SCP_args_2, "APG_opts", []);


verbose = 1;


tau_hat_path = zeros(1,num_windows);
best_obj_path = zeros(1,num_windows);
best_Delta_Fnormsq_path = zeros(1,num_windows);


% Step 1: rolling windows for consistent number of change points

if runningFlag(1)
    if verbose
        fprintf("\n Running Step 1... \n");
    end
    
    for current_window = 1:num_windows
        % update interval endpoints
        
        left_end    =  (current_window - 1) * step_length;
        right_end   =  left_end + window_length;
        
        APG_opts_1.Clambda = Clambda_base(1);
        running_window = [left_end, right_end];
        [~, tau_hat, ~, ~, SCP_outInfo] = LassoSCP(y, X, threshold_var, kappa_1, running_window, resolution_In_1, APG_opts_1);
        
        % record the best obj and best Delta Fnorm square
        
        tau_hat_path(current_window)  =  tau_hat;
        best_obj_path(current_window) =  SCP_outInfo.best_obj;
        best_Delta_Fnormsq_path(current_window) = SCP_outInfo.best_Delta_Fnormsq;
        
        % display
        if verbose
            fprintf(" ");
            fprintf("Current: %d ; tau_hat: %d ; best_obj: %d ; best_Delta_Fnormsq: %d \n ", ...
                current_window, tau_hat, SCP_outInfo.best_obj, SCP_outInfo.best_Delta_Fnormsq);
            fprintf(" ");
        end
    end
end



if ~only_plot_flag
    
    % Step 2: Selection by pruning
    if runningFlag(2)
        if verbose
            fprintf("\n Running Step 2... \n");
        end
        
        cutting_span = floor(window_length/step_length);
        running_Delta_path = best_Delta_Fnormsq_path;
        record_max_ind = [];
        record_max_Delta_Fnormsq = [];
        while any(running_Delta_path > cutoff)
            [max_Delta_Fnormsq, max_ind] = max(running_Delta_path);
            max_ind = max_ind(1);
            record_max_ind = [record_max_ind, max_ind];
            record_max_Delta_Fnormsq = [record_max_Delta_Fnormsq, max_Delta_Fnormsq];
            prune_left  = max([1, max_ind - cutting_span]);
            prune_right =  min([num_windows, max_ind + cutting_span]);
            running_Delta_path(prune_left:prune_right) = -Inf;
        end
        
        pre_tau_hat = sort(tau_hat_path(record_max_ind));
        
        if verbose
            fprintf("\n pre_tau_hat: %.4f \n", pre_tau_hat);
        end
    end
    
    
    
    % Step 3: Selection using pre selected positions
    
    if runningFlag(3)
        if verbose
            fprintf("\n Running Step 3... \n");
        end
        
        
        num_cps = length(pre_tau_hat);
        
        left_ends  = [0, 0.25*pre_tau_hat(2:end) + 0.75*pre_tau_hat(1:(end-1))];
        right_ends = [0.75*pre_tau_hat(2:end) + 0.25*pre_tau_hat(1:(end-1)), 1];
        
        %mid_points = 0.5*(pre_tau_hat(2:end) + pre_tau_hat(1:(end-1)));
        %mid_points = [0, mid_points, 1];
        post_tau_hat = zeros(1, num_cps);
        
        
        for ind = 1:num_cps
            %left_end  = mid_points(ind);
            %right_end = mid_points(ind+1);
            left_end = left_ends(ind);
            right_end = right_ends(ind);
            running_window = [left_end, right_end];
            %APG_opts_2.Clambda = Clambda_base;
            APG_opts_2.Clambda = Clambda_base(2);
            [Theta_Delta_hat, tau_hat, ~, ~, ~] = LassoSCP(y, X, threshold_var, kappa_2, running_window, resolution_In_2, APG_opts_2);
            post_tau_hat(ind) = tau_hat;
            
            if verbose
                fprintf("Detecting change point %d ... ; %d in total. \n", ind, num_cps);
            end
        end
        
        if verbose
            fprintf('\n post_tau_hat: %.4f\n', post_tau_hat);
        end
        
    end
    
    if runningFlag(4)
        % Step 4: matrix estimation using the selected change points
        end_points = [0, post_tau_hat, 1];
        num_seg    = num_cps + 1;
        post_Theta_hat  = zeros(nr, nc, num_seg);
%        post_rank  = zeros(1,num_seg);
        
        
        type         = setOpts(post_APG_args, "type", []);
        Clambda      = setOpts(post_APG_args, "Clambda", 1.0);
        tol          = setOpts(post_APG_args, "tol", 1e-4);
        maxiter      = setOpts(post_APG_args, "maxiter", 1e2);
        Theta_init   = setOpts(post_APG_args, "Theta_init", zeros(nr, nc));
        
        
        for ind = 1:num_seg
            left_end    =  end_points(ind);
            right_end   =  end_points(ind+1);
            running_ind =  find(threshold_var > left_end & threshold_var <= right_end);
            
            switch design_type
                case "V"
                    y_running = y(:, running_ind);
                    X_running = X(:, running_ind);
                    [Theta_hat, ~, ~] = MVAPG_MCP(y_running, X_running, type, Clambda, tol, maxiter, Theta_init);
                    post_Theta_hat(:,:,ind) = Theta_hat;
%                    post_rank(ind) = rank;
                case "M"
                    y_running = y(running_ind);
                    X_running = X(:,:,running_ind);
                    [Theta_hat, ~, ~] = MMAPG_MCP(y_running, X_running, type, Clambda, tol, maxiter, Theta_init);
                    post_Theta_hat(:,:,ind) = Theta_hat;
%                    post_rank(ind) = rank;
            end
        end
    end
    
end

MCP_outInfo.tau_hat_path  = tau_hat_path;
MCP_outInfo.best_obj_path = best_obj_path;
MCP_outInfo.best_Delta_Fnormsq_path = best_Delta_Fnormsq_path;
MCP_outInfo.pre_tau_hat   = pre_tau_hat;

