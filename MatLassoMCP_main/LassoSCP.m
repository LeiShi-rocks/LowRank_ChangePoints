function [Theta_Delta_hat, tau_hat, obj_path, Delta_path, SCP_outInfo] = ...
    LassoSCP(y, X, threshold_var, kappa, running_window, resolution_In, Clambda)
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
        y_running_vec = reshape(y_running, [N_running*nr, 1]);
        X_running_vec = zeros(nr*nc, N_running*nr);
        threshold_var_vec = repelem(threshold_var_running, nr);
        fill_ind = 0;
        e_ind = zeros(nr, 1);
        for N_iter = 1:N_running
            for row_iter = 1:nr
                fill_ind = fill_ind + 1;
                e_ind(row_iter) = 1;
                X_running_vec(:, fill_ind) = reshape(e_ind * (X_running(:,N_iter)'), [nr*nc,1]);
            end
        end
        N_running = N_running * nr;
    case "M"
        y_running = y(running_ind);
        X_running = X(:,:,running_ind);
        y_running_vec = reshape(y_running, [N_running, 1]);
        X_running_vec = reshape(X_running, [nr*nc, N_running]);
        threshold_var_vec = threshold_var_running;
end


test_pos = linspace(left_end + kappa * window_length,...
    right_end - kappa * window_length, resolution);


% disp([Clambda, tol, maxiter]);


obj_path   =  zeros(1, resolution);
Delta_path =  zeros(1, resolution);


verbose    =  1;

if verbose
    fprintf(' ');
    fprintf('Running MatLassoSCP... \n');
    fprintf(' ');
end


X_new_vec = zeros(2*nr*nc, N_running);
best_obj = Inf;

for ind = 1:resolution
    tau = test_pos(ind);
    for i = 1:(N_running)
        X_new_vec(:,i) = [X_running_vec(:,i); X_running_vec(:,i) .* (threshold_var_vec(i) > tau)];
    end
    fit = glmnet(X_new_vec', y_running_vec, 'gaussian');
    Theta_Delta = glmnetCoef(fit, Clambda*sqrt(log(nr*nc)/N_running));
    Delta_path(ind) = sum(sum(Theta_Delta((nr*nc+1):(2*nr*nc)).^2));
    pred = glmnetPredict(fit, X_new_vec', Clambda*sqrt(log(nr*nc)/N_running), 'link');
    obj_path(ind) = sum(sum((y_running_vec - pred).^2))/(2*N_running)...
        + Clambda*sqrt(log(nr*nc)/N_running) * sum(abs(Theta_Delta));
    if obj_path(ind) < best_obj
        Theta_Delta_hat = Theta_Delta;
        tau_hat = tau;
        best_obj = obj_path(ind);
        best_Delta_Fnormsq = sum(sum((Theta_Delta(:,(nc+1):(2*nc))).^2));
    end
    if verbose
        fprintf(' ');
        fprintf('ind: %d |', ind);
        fprintf('tau: %d |', tau);
        fprintf('obj: %d |', obj_path(ind));
        fprintf('best_obj: %d |\n', best_obj);
        fprintf(' ');
    end
end


SCP_outInfo.best_obj = best_obj;
SCP_outInfo.best_Delta_Fnormsq = best_Delta_Fnormsq;






