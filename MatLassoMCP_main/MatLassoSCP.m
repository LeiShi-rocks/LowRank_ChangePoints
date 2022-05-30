function [Theta_Delta_hat, tau_hat, obj_path, Delta_path, SCP_outInfo] = MatLassoSCP(y, X, threshold_var, kappa, running_window, resolution_In, APG_opts)
%MATLASSOSCP Matrix lasso for a single change point
%   APG_opts: the arguments for APG

% initiailzation
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


left_end = running_window(1);
right_end = running_window(2);
window_length = right_end - left_end;


running_ind = find(threshold_var > left_end & threshold_var <= right_end);
threshold_var_running = threshold_var(running_ind);
N_running   = length(running_ind);


if resolution_In >= N_running*(1-2*kappa)
    resolution = floor(N_running*(1-2*kappa));
else
    resolution = floor(resolution_In);
end



switch design_type
    case "V"
        y_running = y(:, running_ind);
        X_running = X(:, running_ind);
    case "M"
        y_running = y(running_ind);
        X_running = X(:,:,running_ind);
end


test_pos = linspace(left_end + kappa * window_length,...
    right_end - kappa * window_length, resolution);


type       =  setOpts(APG_opts, "type", "L2");
Clambda    =  setOpts(APG_opts, "Clambda", 1.0);
tol        =  setOpts(APG_opts, "tol", 1e-4);
maxiter    =  setOpts(APG_opts, "maxiter", 2e2);


% disp([Clambda, tol, maxiter]);


obj_path   =  zeros(1, resolution);
Delta_path =  zeros(1, resolution);


verbose    =  1;

if verbose
    fprintf(' ');
    fprintf('Running MatLassoSCP... \n');
    fprintf(' ');
end

switch design_type
    case "V"
        X_new = zeros(2*nc, N_running);
        best_obj = Inf;
        Theta_init =  setOpts(APG_opts, "Theta_init", zeros(nr, 2*nc));
        for ind = 1:resolution
            tau = test_pos(ind);
            for i = 1:N_running
                X_new(:,i) = [X_running(:,i); X_running(:,i) .* (threshold_var_running(i) > tau)];
            end
            [Theta_Delta, ~, outInfo] = MVAPG_MCP(y_running, X_new, type, Clambda, tol, maxiter, Theta_init);
            Delta_path(ind) = sum(sum(Theta_Delta(:,(nc+1):(2*nc)).^2));
            obj_path(ind) = outInfo.obj;
            if outInfo.obj < best_obj
                Theta_Delta_hat = Theta_Delta;
                tau_hat = tau;
                best_obj = outInfo.obj;
                best_Delta_Fnormsq = sum(sum((Theta_Delta(:,(nc+1):(2*nc))).^2));
            end
            if verbose
                fprintf(' ');
                fprintf('ind: %d |', ind);
                fprintf('tau: %d |', tau);
                fprintf('obj: %d |', outInfo.obj);
                fprintf('best_obj: %d |\n', best_obj);
                fprintf(' ');
            end
        end
        
    case "M"
        X_new = zeros(2*nr, nc, N_running);
        best_obj = Inf;
        Theta_init =  setOpts(APG_opts, "Theta_init", zeros(2*nr, nc));
        for ind = 1:resolution
            tau = test_pos(ind);
            for i = 1:N_running
                X_new(:,:,i) = [X_running(:,:,i); X_running(:,:,i) .* (threshold_var_running(i) > tau)];
            end
            [Theta_Delta, ~, outInfo] = MMAPG_MCP(y_running, X_new, type, Clambda, tol, maxiter, Theta_init);
            obj_path(ind) = outInfo.obj;
            Delta_path(ind) = sum(sum(Theta_Delta((nr+1):(2*nr),:).^2));
            if outInfo.obj < best_obj
                Theta_Delta_hat = Theta_Delta;
                tau_hat = tau;
                best_obj = outInfo.obj;
                best_Delta_Fnormsq = sum(sum((Theta_Delta((nr+1):(2*nr),:)).^2));
            end
            if verbose
                fprintf('ind: %d \n', ind);
                fprintf('tau: %d \n', tau);
                fprintf('obj: %d \n', outInfo.obj);
                fprintf('best_obj: %d \n', best_obj);
            end
        end
            
    otherwise
        error("No such design_type defined!");
end

SCP_outInfo.best_obj = best_obj;
SCP_outInfo.best_Delta_Fnormsq = best_Delta_Fnormsq;

end


